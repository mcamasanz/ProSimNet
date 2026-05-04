"""
RHS del ODE para el reactor tubular 1D (tubo vacío o lecho catalítico).

Layout del vector de estado (determinado por has_catalyst y shell_tube):

    Sin catalizador, sin shell_tube:   [C(nc,N), Hg(N)]               → (nc+1)*N
    Sin catalizador, con shell_tube:   [C(nc,N), Hg(N), Tw(N)]        → (nc+2)*N
    Con catalizador, sin shell_tube:   [C(nc,N), Hg(N), Ts(N)]        → (nc+2)*N
    Con catalizador, con shell_tube:   [C(nc,N), Hg(N), Ts(N), Tw(N)] → (nc+3)*N

Orden de cálculo (12 pasos):
    1.  Lectura de params
    2.  Desempaquetado del sv → C, Hg, Ts, Tw, Tg, P, y
    3.  Condiciones de contorno → v_in, v_out, C_in, T_in
    4.  Propiedades de mezcla del gas → rho_g, mu_g, k_g
    5.  Velocidades y coeficientes de transporte
        - Ergun (lecho catalítico) o continuidad (tubo vacío)
        - compute_transfer_coefficients → h_bed, h_wall, D_disp
    6.  Cinética — iteración sobre reactions_config
        - Acumula source_gas (nc, N) [mol/m³_gas/s]
        - Acumula Q_rxn_solid (N,) [W/m³_bed] para hetero → va a dTsdt
        - Acumula q_rxn_gas  (N,) [W/m³_bed] para homo  → va a dHgdt
    7.  [No aplica al reactor — sin conveyor ni sólido consumido]
    8.  Balance de especies gaseosas  dC_i/dt  (convección + dispersión + fuentes)
    9.  [No aplica — rho_catalyst es constante, no se integra]
    10. Balances de energía  dHg/dt  [+ dTs/dt si has_catalyst]
        - dHg/dt = −epsi·div(Fh_conv) − div(qg_diff) ± q_gs ± q_wall + q_rxn_gas
        - dTs/dt = (q_gs + Q_rxn_solid + q_induction) / Cs_vol
        - Sin q_masstransfer (no hay masa sólido→gas)
        - Sin thermal_mass_correction (rho_cat = cte → drho·H = 0)
    11. ODE de pared  dTw/dt  (solo si shell_tube)
    12. Empaquetado del RHS

Convenciones de unidades:
    C_i         mol/m³_gas      (por volumen de vacíos)
    Hg          J/m³_bed        = epsi * Σ C_i * h_i(Tg)
    Ts          K               (variable primaria, integrada directamente)
    reactions_config → rate_fn devuelve [mol/m³_bed/s]  para ambos tipos
    source_gas  mol/m³_gas/s    = rate / epsi
    Q_rxn       W/m³_bed        = (−ΔH_rxn) * rate
"""

from __future__ import annotations

import numpy as np

from src.boundary_conditions.reactor_boundary import get_reactor_boundary
from src.discretization.fluxes import (
    convective_flux,
    diffusive_flux,
    gas_diffusive_heat_flux,
    gas_enthalpy_convective_flux,
)
from src.physics.mixture_gas import compute_gas_mixture_properties
from src.physics.momentum.darcy_weisbach import continuity_face_velocity
from src.physics.momentum.ergun import ergun_face_velocity
from src.physics.thermal.wall_heat_flux import wall_heat_flux
from src.physics.thermal.wall_ode import wall_exterior_q, wall_axial_q, wall_ode_rhs
from src.physics.thermodynamics.solid_props import eval_solid_property
from src.physics.transport.transfer_coefficients import compute_transfer_coefficients
from src.units.reactor.state import unpack_state_vector
from src.utils.profiling import profiled

R_GAS = 8.31446261815324   # [J/mol/K]


@profiled
def core_rhs(t: float, sv: np.ndarray, params: dict) -> np.ndarray:
    """
    Evalúa el RHS del sistema de EDOs del reactor 1D.

    Firma idéntica a los demás equipos de ProSimNet para intercambiabilidad
    en el runner.

    Parameters
    ----------
    t      : float
    sv     : ndarray — vector de estado (tamaño según has_catalyst y shell_tube)
    params : dict    — parámetros completos validados por el runner

    Returns
    -------
    dydt : ndarray — mismo tamaño que sv
    """
    # ── 1. Lectura de params ───────────────────────────────────────────────────
    nc   = int(params["n_comp"])
    nn   = int(params["N"])
    dz   = float(params["dz"])
    epsi = float(params["epsi"])
    Ai   = float(params["Ai"])
    Di   = float(params["Di"])
    Pi   = float(params["Pi"])
    Po   = float(params["Po"])

    prop_gas      = params["prop_gas"]
    gas_T_ref     = float(params["gas_T_ref"])
    species       = list(params["species"])

    bc_config        = params["bc_config"]
    thermal_bc_cfg   = params["thermal_bc_config"]
    trans_config     = params["trans_config"]
    reactions_config = params.get("reactions_config", [])
    catalyst_config  = params.get("catalyst_config")     # None → tubo vacío
    induction_config = params.get("induction_config")    # None → sin inducción
    wall_config      = params.get("wall_config")
    energy           = bool(params.get("energy", True))

    has_catalyst = catalyst_config is not None
    shell_tube   = wall_config is not None

    prop_update_mode  = params.get("prop_update_mode",  "always")
    trans_update_mode = params.get("trans_update_mode", "always")

    cache  = params.setdefault("_cache", {})
    sp_idx = params.setdefault("_sp_idx",
                               {sp: i for i, sp in enumerate(species)})

    # ── 2. Desempaquetado del sv ───────────────────────────────────────────────
    Tg_guess = cache.get("Tg_last", np.full(nn, 700.0))

    state = unpack_state_vector(
        sv=sv, n_comp=nc, N=nn,
        has_catalyst=has_catalyst, shell_tube=shell_tube,
        prop_gas=prop_gas, epsi=epsi,
        Tg_guess=Tg_guess, gas_T_ref=gas_T_ref,
    )

    # Clip defensivo: BDF perturba sv para estimar el Jacobiano; deltas pequeños
    # pueden producir C<0 → Ctot<0 → P<0 → rho_g<0 → NaN en propiedades.
    C_mat  = np.maximum(state["C"], 0.0)    # (nc, N)
    Tg_arr = state["Tg"]                    # (N,) [K]
    Ts_arr = state["Ts"]                    # (N,) [K]  o None
    Tw_arr = state["Tw"]                    # (N,) [K]  o None

    Ctot_arr   = np.sum(C_mat, axis=0)                        # (N,) [mol/m³_gas]
    _Ctot_safe = np.maximum(Ctot_arr, 1.0e-300)
    y_mat      = C_mat / _Ctot_safe[None, :]                  # (nc, N)
    x_mat      = y_mat.T                                       # (N, nc) para Wilke
    P_bar      = np.maximum(Ctot_arr * R_GAS * Tg_arr / 1.0e5,
                            1.0e-6)                            # (N,) [bar]
    P_Pa       = P_bar * 1.0e5                                 # (N,) [Pa]

    cache["Tg_last"] = Tg_arr.copy()

    # ── 3. Condiciones de contorno ─────────────────────────────────────────────
    bc       = get_reactor_boundary(t=t, P_cell=P_bar, Ctot_cell=Ctot_arr,
                                    bc_config=bc_config, n_comp=nc)
    v_in     = float(bc["inlet"]["v_m_s"])
    v_out    = float(bc["outlet"]["v_m_s"])
    C_in     = bc["inlet"]["C_mol_m3"]    # ndarray(nc,) o None (batch)
    T_in     = bc["inlet"]["T_K"]         # float o None

    has_inlet = C_in is not None

    # ── 4. Propiedades de mezcla del gas ──────────────────────────────────────
    if prop_update_mode == "frozen" and "gas_props" in cache:
        gas_props = cache["gas_props"]
    else:
        gas_props = compute_gas_mixture_properties(
            P_Pa=P_Pa, Tg=Tg_arr, x=x_mat,
            prop_gas=prop_gas, n_comp=nc, N=nn,
        )
        if prop_update_mode == "frozen":
            cache["gas_props"] = gas_props

    rho_g_arr = gas_props["rho"]    # (N,) [kg/m³_gas]
    mu_g_arr  = gas_props["mu"]     # (N,) [Pa·s]
    k_g_arr   = gas_props["k"]      # (N,) [W/m/K]

    # ── 5. Velocidades y coeficientes de transporte ───────────────────────────
    # Hidráulica: Ergun para lecho catalítico, continuidad para tubo vacío.
    if has_catalyst and nn > 1:
        dp = float(catalyst_config["dp"])    # [m] partícula de catalizador
        v_face = ergun_face_velocity(
            P=P_Pa, rho_g=rho_g_arr, mu_g=mu_g_arr,
            epsi=epsi, dp=dp, dz=dz,
            v_in=v_in, v_out=v_out,
        )
    else:
        # N=1 (cstr/batch) o tubo vacío: velocidad de caras desde continuidad
        v_face = continuity_face_velocity(
            rho_g=rho_g_arr, v_in=v_in, v_out=v_out, N=nn,
        )
    v_cell = 0.5 * (v_face[:-1] + v_face[1:])                # (N,) [m/s]

    # Coeficientes de transporte
    if trans_update_mode == "frozen" and "trans_props" in cache:
        trans_props = cache["trans_props"]
    else:
        # Ts_for_trans: Ts si hay catalizador, Tg si no (h_bed no se usa sin catalizador)
        Ts_for_trans = Ts_arr if has_catalyst else Tg_arr
        Tw_for_trans = Tw_arr if shell_tube else None
        prop_lecho   = ({"D_p": np.full(nn, catalyst_config["dp"]),
                         "a_surf": np.full(nn, catalyst_config["a_p"])}
                        if has_catalyst else {"D_p": None, "a_surf": None})
        trans_props = compute_transfer_coefficients(
            Tg=Tg_arr, Ts=Ts_for_trans, x=x_mat,
            gas_props=gas_props, u_rel=np.abs(v_cell),
            prop_gas=prop_gas, prop_lecho=prop_lecho,
            Di=Di, trans_config=trans_config,
            n_comp=nc, N=nn,
            Tw=Tw_for_trans, L=dz * nn,
        )
        if trans_update_mode == "frozen":
            cache["trans_props"] = trans_props

    h_bed_arr  = trans_props["h_bed"]    # (N,) [W/m²/K] — relevante solo con catalizador
    h_wall_arr = trans_props["h_wall"]   # (N,) [W/m²/K]
    D_disp_mat = trans_props["D_disp"]   # (nc, N) [m²/s] o None

    # ── 6. Cinética ────────────────────────────────────────────────────────────
    epsi_safe    = max(float(epsi), 1.0e-10)
    source_gas   = np.zeros((nc, nn), dtype=float)  # [mol/m³_gas/s]
    Q_rxn_solid  = np.zeros(nn, dtype=float)         # [W/m³_bed] → sólido (hetero)
    q_rxn_gas    = np.zeros(nn, dtype=float)         # [W/m³_bed] → gas    (homo)

    for rxn in reactions_config:
        # rate_fn(C, Tg, Ts, P_Pa, params) → (N,) [mol/m³_bed/s]
        rate = rxn["rate_fn"](C_mat, Tg_arr, Ts_arr, P_Pa, params)   # (N,)
        rate = np.maximum(rate, 0.0)          # cinéticas nunca producen tasa negativa

        # ΔH_rxn: float constante o callable(T) → float [J/mol_rxn]
        dH_rxn_raw = rxn["dH_rxn"]
        if callable(dH_rxn_raw):
            T_rxn  = Ts_arr if rxn["type"] == "heterogeneous" else Tg_arr
            dH_val = np.asarray(dH_rxn_raw(T_rxn), dtype=float)   # (N,) [J/mol_rxn]
        else:
            dH_val = float(dH_rxn_raw)                             # escalar

        # Fuentes de especies gaseosas: mismo tratamiento para homo y hetero
        # rate está en [mol/m³_bed/s] → /epsi → [mol/m³_gas/s]
        stoich = rxn["stoich"]
        for sp, coeff in stoich.items():
            if sp in sp_idx:
                source_gas[sp_idx[sp]] += coeff * rate / epsi_safe

        # Calor: hetero → sólido; homo → gas
        q_rxn = (-dH_val) * rate                                   # [W/m³_bed]
        if rxn["type"] == "heterogeneous":
            Q_rxn_solid += q_rxn
        else:
            q_rxn_gas += q_rxn

    # ── 7. [Sin conveyor — no aplica al reactor] ──────────────────────────────

    # ── 8. Balance de especies gaseosas   dC_i/dt [mol/m³_gas/s] ─────────────
    bc_in = "dirichlet" if has_inlet else "neumann"

    dCdt_mat = np.zeros((nc, nn), dtype=float)
    for i in range(nc):
        C_in_i = None if C_in is None else float(C_in[i])
        F_conv = convective_flux(
            phi_cell=C_mat[i], v_face=v_face,
            phi_in=C_in_i, phi_out=None,
        )
        if D_disp_mat is not None:
            F_diff = diffusive_flux(
                phi_cell=C_mat[i], Gamma=D_disp_mat[i], dz=dz,
                phi_in=C_in_i, phi_out=None,
                bc_in=bc_in, bc_out="neumann",
                face_method="arithmetic",
            )
            F_tot = F_conv + F_diff
        else:
            F_tot = F_conv
        dCdt_mat[i] = -(F_tot[1:] - F_tot[:-1]) / dz + source_gas[i]

    # ── 9. [Sin balance de sólido — rho_catalyst constante] ───────────────────

    # ── 10. Balances de energía ────────────────────────────────────────────────
    if not energy:
        dHgdt_arr = np.zeros(nn, dtype=float)
        dTsdt_arr = np.zeros(nn, dtype=float) if has_catalyst else None
    else:
        # Flujo de calor gas↔sólido [W/m³_bed]
        # Solo activo con catalizador; sin catalizador no hay fase sólida.
        if has_catalyst:
            a_p       = float(catalyst_config["a_p"])   # [m²/m³_bed] superficie específica
            q_gs_vol  = h_bed_arr * a_p * (Tg_arr - Ts_arr)   # >0 si gas cede calor al sólido
        else:
            q_gs_vol = np.zeros(nn, dtype=float)

        # Flujo de calor gas↔pared [W/m³_bed]
        if shell_tube:
            qwall_vol = h_wall_arr * (Pi / Ai) * (Tw_arr - Tg_arr)
        else:
            qwall_vol, _, _ = wall_heat_flux(
                Tg=Tg_arr, h_wall=h_wall_arr,
                thermal_bc_config=thermal_bc_cfg,
                N=nn, Ai=Ai, Pi=Pi, Po=Po, dz=dz,
            )

        # ── Energía del gas (Hg) ──────────────────────────────────────────────
        T_in_d = T_in if has_inlet else None
        C_in_d = C_in if has_inlet else None

        Fh_conv = gas_enthalpy_convective_flux(
            Tg_cell=Tg_arr, C_cell=C_mat, v_face=v_face,
            prop_gas=prop_gas, n_comp=nc, gas_T_ref=gas_T_ref,
            T_in=T_in_d, C_in=C_in_d,
        )
        qg_diff = gas_diffusive_heat_flux(
            Tg_cell=Tg_arr, k_g=k_g_arr, dz=dz,
            T_in=T_in_d, T_out=None,
            bc_in=bc_in, bc_out="neumann",
            face_method="arithmetic",
        )

        div_Fh_conv = (Fh_conv[1:] - Fh_conv[:-1]) / dz    # (N,) [W/m³_bed] si sin epsi
        div_qg_diff = (qg_diff[1:] - qg_diff[:-1]) / dz    # (N,) [W/m³_bed]

        dHgdt_arr = (-epsi * div_Fh_conv
                     - div_qg_diff
                     - q_gs_vol
                     + qwall_vol
                     + q_rxn_gas)          # q_rxn_gas ≠ 0 solo para reacciones homogéneas

        # ── Energía del sólido (Ts) — solo con catalizador ───────────────────
        if has_catalyst:
            rho_cat = float(catalyst_config["rho_bulk"])    # [kg/m³_bed]
            Cp_cat  = np.asarray(
                catalyst_config["Cp_fn"](Ts_arr), dtype=float
            )                                               # (N,) [J/kg/K]
            Cs_vol  = np.maximum(rho_cat * Cp_cat, 1.0e-6) # (N,) [J/m³_bed/K]

            # q_induction: fuente volumétrica de calor sobre el sólido/catalizador
            # Modela el acoplamiento electromagnético del calentamiento por inducción.
            # La energía se deposita en el catalizador (material ferromagnético o conductor)
            # y se transfiere al gas por convección a través de q_gs.
            if induction_config is not None:
                if induction_config.get("mode") == "profile":
                    z_cells    = (np.arange(nn) + 0.5) * dz
                    q_induction = np.asarray(
                        induction_config["q_fn"](z_cells, t), dtype=float
                    )                                       # (N,) [W/m³_bed]
                else:
                    q_induction = np.full(nn,
                                          float(induction_config["q_vol"]),
                                          dtype=float)
            else:
                q_induction = np.zeros(nn, dtype=float)

            # dTs/dt = (q_gs→sólido + Q_rxn_solid + q_induction) / Cs_vol
            # Sin thermal_mass_correction: rho_cat es constante → drho·H = 0
            # Sin q_masstransfer: no hay masa sólido→gas en un reactor catalítico
            dTsdt_arr = (q_gs_vol + Q_rxn_solid + q_induction) / Cs_vol
        else:
            dTsdt_arr = None

    # ── 11. ODE de pared   dTw/dt  [solo si shell_tube] ──────────────────────
    if shell_tube:
        A_w       = float(wall_config["A_w"])
        mat       = wall_config["material"]
        rho_w_arr = eval_solid_property(mat["rho"], Tw_arr)  # (N,) [kg/m³]
        cp_w_arr  = eval_solid_property(mat["cp"],  Tw_arr)  # (N,) [J/kg/K]
        k_w_arr   = eval_solid_property(mat["k"],   Tw_arr)  # (N,) [W/m/K]

        Q_gw_cell  = h_wall_arr * Pi * dz * (Tg_arr - Tw_arr)
        Q_ext_cell = wall_exterior_q(
            Tw_arr=Tw_arr, thermal_bc_cfg=thermal_bc_cfg,
            k_w_arr=k_w_arr, Pi=Pi, Po=Po, dz=dz, N=nn,
        )
        Q_ax_cell  = wall_axial_q(Tw_arr=Tw_arr, k_w_arr=k_w_arr, A_w=A_w, dz=dz)
        dTwdt_arr  = wall_ode_rhs(
            Q_gw_cell=Q_gw_cell, Q_ext_cell=Q_ext_cell, Q_ax_cell=Q_ax_cell,
            rho_w_arr=rho_w_arr, cp_w_arr=cp_w_arr, A_w=A_w, dz=dz,
        )

    # ── 12. Empaquetado del RHS ────────────────────────────────────────────────
    parts = [dCdt_mat[i] for i in range(nc)]
    parts.append(dHgdt_arr)
    if has_catalyst:
        parts.append(dTsdt_arr)
    if shell_tube:
        parts.append(dTwdt_arr)
    return np.concatenate(parts)
