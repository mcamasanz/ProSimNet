"""
@module rhs.rhs_gasifier
@brief RHS del ODE para el modelo 1D del gasificador (batch 0D, CSTR, updraft, conveyor).

@details
Maneja 9 especies gaseosas + 3 densidades másicas del sólido + Hg + Ts
[+ Tw si shell-tube activo].

Layout del vector de estado:
    Sin shell-tube:
        sv = [C_CO, C_CO2, C_H2O, C_H2, C_O2, C_CH4, C_C2H4, C_tar, C_N2,
              rho_biomass, rho_char, rho_moisture, Hg, Ts]
        tamaño = 14 × N

    Con shell-tube (wall_config presente en params):
        sv = [...igual que arriba..., Tw]
        tamaño = 15 × N

Orden de cálculo (12 pasos):
    1.  Lectura de params   → nc, nn, dz, epsi_r, dp0, ...
    2.  Desempaquetado      → C, rho_solid, Hg, Ts [, Tw], Tg, P, y
    3.  Contornos           → v_in, v_out, C_in, T_in, vs_face, rho_s_inlet
    4.  Propiedades de mezcla del gas → rho_g, mu_g, k_g
    5.  Geometría de partícula, velocidades y coeficientes de transporte
        - dp (SCM), a_p (superficie específica)
        - v_face (Ergun en 1D, perfil lineal en 0D)
        - compute_transfer_coefficients → h_bed, h_wall, D_disp
    6.  Tasas de reacción   → r_dry, r_pyr, r_ox, r_CO2, r_H2O
    7.  Sólido calculado [sólo inlet_mode="computed"]
        - Cálculo de rho_solid_in desde balance global de masa sólida
        - Actualización de caché para el siguiente paso (explicit/implicit)
    8.  Balance de especies gaseosas  dC_i/dt  (convección + dispersión axial + fuentes)
    9.  Balance de densidades sólidas d(rho_s_i)/dt  (convección sólida + reacciones)
    10. Balances de energía  dHg/dt, dTs/dt
        - dHg/dt incluye q_masstransfer = epsi_r·Σᵢ src_gas[i]·h_i(Ts)
          (entalpía portada por las nuevas moléculas al aparecer a temperatura Ts)
    11. ODE de pared [sólo shell_tube=True]  dTw/dt
    12. Empaquetado del RHS

Convenciones de unidades (iguales a PSA/heater):
    C_i         mol/m³_gas   (por volumen de vacíos)
    rho_s_i     kg/m³_bed    (por volumen de lecho)
    Hg          J/m³_bed     = epsi_r × Σ C_i × h_i(Tg)
    Ts          K            (variable primaria — integrada directamente)
    Fuentes de reacción en mol/m³_bed o kg/m³_bed,
    convertidas a mol/m³_gas dividiendo por epsi_r antes de añadirlas a dC/dt.
"""

from __future__ import annotations

import numpy as np

from src.boundary_conditions.gasifier_boundary import get_gasifier_boundary
from src.discretization.fluxes import (
    convective_flux,
    diffusive_flux,
    solid_convective_flux,
    gas_diffusive_heat_flux,
    gas_enthalpy_convective_flux,
)
from src.physics.mixture_gas import compute_gas_mixture_properties
from src.physics.momentum.ergun import ergun_face_velocity
from src.physics.reactions.char_conversion import (
    char_gas_sources,
    char_het_rates,
    char_reaction_heat,
    particle_diameter,
    specific_surface_area,
)
from src.physics.reactions.drying import (
    drying_enthalpy_sink,
    drying_gas_source,
    drying_rate,
)
from src.physics.reactions.pyrolysis import (
    pyrolysis_enthalpy_sink,
    pyrolysis_rate,
    pyrolysis_sources,
)
from src.physics.thermal.wall_heat_flux import wall_heat_flux
from src.physics.thermal.wall_ode import wall_exterior_q, wall_axial_q, wall_ode_rhs
from src.physics.thermodynamics.enthalpy import calc_species_enthalpy
from src.physics.thermodynamics.solid_props import eval_solid_property
from src.physics.transport.transfer_coefficients import compute_transfer_coefficients
from src.units.gasifier.state import unpack_state_vector
from src.utils.profiling import profiled

R_GAS = 8.31446261815324   # [J/mol/K]
_IDX  = {
    "CO": 0, "CO2": 1, "H2O": 2, "H2": 3, "O2": 4,
    "CH4": 5, "C2H4": 6, "tar": 7, "N2": 8,
}


@profiled
def core_rhs(t: float, sv: np.ndarray, params: dict) -> np.ndarray:
    """
    @brief
    Evalúa el RHS del sistema de EDOs del gasificador 1D.

    @details
    Firma idéntica al RHS del adsorbedor y del heater para que el runner
    pueda intercambiar equipos sin cambiar su código.

    Parameters
    ----------
    t      : float
    sv     : ndarray (14*N,) o (15*N,)  vector de estado actual
    params : dict   parámetros completos del modelo (ver runner_gasifier.py)

    Returns
    -------
    dydt : ndarray (14*N,) o (15*N,)
    """
    # =========================================================
    # 1. Lectura de params
    # =========================================================
    nc        = int(params["n_comp"])
    nn        = int(params["N"])
    dz        = float(params["dz"])
    epsi_r    = float(params["epsi_r"])
    dp0       = float(params["dp0"])
    Ai        = float(params["Ai"])
    Di        = float(params["Di"])
    Pi        = float(params["Pi"])
    Po        = float(params["Po"])

    prop_gas      = params["prop_gas"]
    MW_arr        = np.asarray(params["MW"], dtype=float)   # (nc,)
    gas_T_ref     = float(params["gas_T_ref"])
    species       = list(params["species"])

    fuel_config    = params["fuel_config"]
    solid_config   = params["solid_config"]
    bc_config      = params["bc_config"]
    thermal_bc_cfg = params["thermal_bc_config"]
    trans_config   = params["trans_config"]
    energy         = bool(params.get("energy", True))

    rho_char0  = float(params.get("rho_char0", 1.0e-12))
    dH_pyr     = float(params.get("dH_pyr", 0.0))          # [J/kg_biomass]

    prop_update_mode  = params.get("prop_update_mode",  "always")
    trans_update_mode = params.get("trans_update_mode", "always")

    wall_config = params.get("wall_config")
    shell_tube  = wall_config is not None

    cache = params.get("_cache", {})

    # =========================================================
    # 2. Desempaquetado del estado
    # =========================================================
    Tg_guess = cache.get("Tg_last", np.full(nn, 700.0))

    state = unpack_state_vector(
        sv=sv, n_comp=nc, N=nn, prop_gas=prop_gas,
        epsi_r=epsi_r, Tg_guess=Tg_guess,
        gas_T_ref=gas_T_ref, shell_tube=shell_tube,
    )
    # Tg y Ts vienen del desempaquetado (Newton warm-start para Tg)
    Tg_arr = state["Tg"]    # (N,) [K]
    Ts_arr = state["Ts"]    # (N,) [K]
    Tw_arr = state["Tw"]    # (N,) [K] o None

    # Clip defensivo: BDF perturba el estado con deltas pequeños para estimar el
    # Jacobiano; esas perturbaciones pueden hacer C<0 o rho<0.
    #   C<0  → Ctot<0 → P<0 → rho_g<0 → Re<0 → Re**0.6 = NaN
    #   rho<0 → dp=(rho/rho0)^(1/3) = NaN
    # Clipear antes de cualquier cálculo físico garantiza resultados finitos.
    C_mat     = np.maximum(state["C"],         0.0)   # (nc, N)
    rho_solid = np.maximum(state["rho_solid"], 0.0)   # (3, N)

    # Derivar Ctot, fracciones molares y presión desde C_mat ya clipado
    Ctot_arr   = np.sum(C_mat, axis=0)                             # (N,) [mol/m³_gas]
    _Ctot_safe = np.maximum(Ctot_arr, 1.0e-300)
    y_mat      = C_mat / _Ctot_safe[None, :]                       # (nc, N)
    x_mat      = y_mat.T                                           # (N, nc) para Wilke
    P_bar      = np.maximum(Ctot_arr * R_GAS * Tg_arr / 1.0e5,
                            1.0e-6)                                # (N,) [bar]
    P_Pa       = P_bar * 1.0e5

    rho_biomass  = rho_solid[0]     # (N,) [kg/m³_bed]
    rho_char     = rho_solid[1]     # (N,) [kg/m³_bed]
    rho_moisture = rho_solid[2]     # (N,) [kg/m³_bed]

    cache["Tg_last"] = Tg_arr.copy()

    # =========================================================
    # 3. Contornos
    # =========================================================
    bc = get_gasifier_boundary(
        t=t, P_cell=P_bar, Ctot_cell=Ctot_arr,
        bc_config=bc_config, n_comp=nc,
    )
    v_in  = float(bc["inlet"]["v_m_s"])
    v_out = float(bc["outlet"]["v_m_s"])
    C_in  = bc["inlet"]["C_mol_m3"]     # ndarray(nc,) o None
    T_in  = bc["inlet"]["T_K"]          # float o None

    has_inlet = C_in is not None and T_in is not None

    # Velocidad del sólido — magnitud desde BC, signo desde direction
    # Opción 1 (actual): vs constante a lo largo del lecho.
    # NOTE (futuro — Opción 2): vs variable desde continuidad del sólido:
    #   ∂(Σ ρs,i)/∂t + ∂(vs·Σ ρs,i)/∂z = Σ Ss,i
    vs_mag      = float(bc["solid_inlet"]["v_m_s"])  # magnitud [m/s] ≥ 0
    rho_s_inlet = bc["solid_inlet"]["rho_solid"]     # ndarray(3,) o None

    # Signo de vs: updraft → sólido sube (−), downdraft → sólido baja (+), None → estático
    _dir = bc["solid_inlet"].get("direction")        # "updraft" | "downdraft" | None
    if _dir == "updraft":
        vs_signed = -vs_mag
    elif _dir == "downdraft":
        vs_signed = +vs_mag
    else:
        vs_signed = 0.0              # sólido estático (v_solid=0)

    vs_face = np.full(nn + 1, vs_signed, dtype=float)  # (N+1,) perfil constante

    # Sólido calculado (conveyor): precarga desde caché de la llamada anterior
    inlet_mode_cv   = str(bc_config.get("inlet_mode", "prescribed"))
    inlet_method_cv = str(bc_config.get("inlet_method", "explicit"))
    if inlet_mode_cv == "computed" and inlet_method_cv == "explicit":
        _cached_rsi = cache.get("rho_solid_in_conveyor")
        if _cached_rsi is not None:
            rho_s_inlet = np.asarray(_cached_rsi, dtype=float)
        else:
            # Primera llamada: arranque en frío desde densidad del sólido fresco
            _rho_f = float(bc_config.get("rho_solid_fresh_total", 0.0))
            _mc    = float(bc_config.get("mc_wb", 0.0))
            rho_s_inlet = np.array(
                [_rho_f * (1.0 - _mc), 0.0, _rho_f * _mc], dtype=float,
            )

    # =========================================================
    # 4. Propiedades de mezcla del gas
    # =========================================================
    if prop_update_mode == "frozen" and "gas_props" in cache:
        gas_props = cache["gas_props"]
    else:
        gas_props = compute_gas_mixture_properties(
            P_Pa=P_Pa, Tg=Tg_arr, x=x_mat,
            prop_gas=prop_gas, n_comp=nc, N=nn,
        )
        if prop_update_mode == "frozen":
            cache["gas_props"] = gas_props

    rho_g_arr = gas_props["rho"]      # (N,) [kg/m³_gas]
    mu_g_arr  = gas_props["mu"]       # (N,) [Pa·s]
    k_g_arr   = gas_props["k"]        # (N,) [W/m/K]

    # =========================================================
    # 5. Geometría de partícula, velocidades y coeficientes de transporte
    # =========================================================
    # Diámetro de partícula SCM: dp = dp0·(rho_char/rho_char0)^(1/3)
    # Calculado una sola vez; reutilizado en el paso 6 (cinéticas) y en el paso 10.
    dp_arr = particle_diameter(rho_char, rho_char0, dp0)
    dp_eff = np.maximum(dp_arr, dp0 * 1.0e-4)          # evitar dp → 0
    a_p    = specific_surface_area(dp_eff, epsi_r)      # (N,) [m²/m³_bed]

    # Velocidades del gas en caras
    if nn == 1:
        # 0D: perfil lineal entre inlet y outlet (Ergun no aplica)
        v_face = np.array([v_in, v_out], dtype=float)   # (2,)
    else:
        v_face = ergun_face_velocity(
            P=P_Pa, rho_g=rho_g_arr, mu_g=mu_g_arr,
            epsi=epsi_r, dp=float(np.mean(dp_eff)), dz=dz,
            v_in=v_in, v_out=v_out,
        )
    v_cell = 0.5 * (v_face[:-1] + v_face[1:])          # (N,) [m/s]

    # Coeficientes de transporte:
    #   modo constant  → h_bed, h_wall desde trans_config
    #   modo correlation → Ranz-Marshall (h_bed) + Dittus-Boelter/Nu=3.66 (h_wall)
    #                      propiedades de película a Tfilm=0.5·(Tg+Ts); Ra si Tw disponible
    # prop_lecho_dyn lleva el dp dinámico (SCM) y la superficie específica.
    if trans_update_mode == "frozen" and "trans_props" in cache:
        trans_props = cache["trans_props"]
    else:
        prop_lecho_dyn = {"D_p": dp_eff, "a_surf": a_p}
        Tw_for_trans   = Tw_arr if shell_tube else None
        trans_props = compute_transfer_coefficients(
            Tg=Tg_arr, Ts=Ts_arr, x=x_mat,
            gas_props=gas_props, u_rel=np.abs(v_cell),
            prop_gas=prop_gas, prop_lecho=prop_lecho_dyn,
            Di=Di, trans_config=trans_config,
            n_comp=nc, N=nn,
            Tw=Tw_for_trans,
            L=dz * nn,
        )
        if trans_update_mode == "frozen":
            cache["trans_props"] = trans_props

    h_bed_arr  = trans_props["h_bed"]   # (N,)      [W/m²/K]
    h_wall_arr = trans_props["h_wall"]  # (N,)      [W/m²/K]
    D_disp_mat = trans_props["D_disp"]  # (nc, N)   [m²/s] o None (plug-flow)

    # =========================================================
    # 6. Tasas de reacción
    # =========================================================
    kinetics  = fuel_config["kinetics"]
    char_comp = fuel_config["char_composition"]
    co_co2    = fuel_config["co_co2_ratio"]
    yields    = fuel_config["pyrolysis_yields"]
    hv        = fuel_config["heating_values"]

    # Secado
    r_dry = drying_rate(rho_moisture, Ts_arr, kinetics["drying"])   # (N,) [kg/m³_bed/s]

    # Pirólisis
    r_pyr = pyrolysis_rate(rho_biomass, Ts_arr, kinetics["pyrolysis"])  # (N,)

    # Reacciones heterogéneas del char (SCM + Ranz-Marshall masa)
    params_rxn = {
        "kinetics":         kinetics,
        "char_composition": char_comp,
        "co_co2_ratio":     co_co2,
        "dp0":              dp0,
        "rho_char0":        rho_char0,
    }
    r_ox, r_CO2, r_H2O = char_het_rates(
        rho_char=rho_char, C_gas=C_mat, Ts=Ts_arr,
        v_cell=v_cell, rho_g=rho_g_arr, mu_g=mu_g_arr,
        Tg=Tg_arr, P_Pa=P_Pa,
        prop_gas=prop_gas, fuel_config=fuel_config,
        params_rxn=params_rxn, epsi_r=epsi_r, species=species,
    )   # cada uno (N,) [kg_char/m³_bed/s]

    # =========================================================
    # 7. Conveyor — cálculo de rho_solid_in desde balance de masa sólida
    #
    # rho_solid_in_total = rho_solid_out_total + (1/vs) · ∫ṁ_s→g dz
    #
    # ṁ_s→g = r_dry + r_pyr·(1−yield_char) + r_char_total   [kg/m³_bed/s]
    #
    # Se ejecuta tanto en modo "implicit" (usa reacciones del paso actual)
    # como en modo "explicit" (actualiza la caché para la siguiente llamada).
    # =========================================================
    if inlet_mode_cv == "computed":
        _yield_char   = float(yields.get("char", 0.0))
        _r_char_total = r_ox + r_CO2 + r_H2O                       # (N,)
        _m_s2g        = r_dry + r_pyr * (1.0 - _yield_char) + _r_char_total  # (N,)
        _m_s2g_int    = float(np.sum(_m_s2g) * dz)                 # [kg/m²/s]

        # _dir calculado en el paso 3; updraft → salida en z=0, downdraft → salida en z=L
        if _dir == "updraft":
            _rho_out = float(np.sum(rho_solid[:, 0]))               # salida z=0
        else:
            _rho_out = float(np.sum(rho_solid[:, -1]))              # salida z=L

        _vs_abs      = max(abs(vs_signed), 1.0e-12)
        _rho_in_tot  = max(_rho_out + _m_s2g_int / _vs_abs, 0.0)

        _mc_cv               = float(bc_config.get("mc_wb", 0.0))
        _rho_solid_in_cv     = np.array(
            [_rho_in_tot * (1.0 - _mc_cv), 0.0, _rho_in_tot * _mc_cv],
            dtype=float,
        )
        cache["rho_solid_in_conveyor"] = _rho_solid_in_cv          # actualizar siempre

        if inlet_method_cv == "implicit" or rho_s_inlet is None:
            rho_s_inlet = _rho_solid_in_cv

    # =========================================================
    # 8. Balance de especies gaseosas   dC_i/dt [mol/m³_gas/s]
    # =========================================================
    bc_in = "dirichlet" if has_inlet else "neumann"

    # Fuentes de reacción [mol/m³_bed/s → mol/m³_gas/s vía /epsi_r]
    src_dry_H2O     = drying_gas_source(r_dry, MW_H2O=float(MW_arr[_IDX["H2O"]]))  # (N,)
    src_pyr_gas, _  = pyrolysis_sources(
        r_pyr=r_pyr, yields=yields, MW_gas=MW_arr, species=species,
    )   # (nc, N) [mol/m³_bed/s]
    src_char_gas    = char_gas_sources(
        r_ox=r_ox, r_CO2=r_CO2, r_H2O=r_H2O,
        Ts=Ts_arr, char_comp=char_comp, co_co2_ratio=co_co2,
        MW_gas=MW_arr, species=species,
    )   # (nc, N) [mol/m³_bed/s]

    epsi_safe  = max(float(epsi_r), 1.0e-10)
    source_gas = np.zeros((nc, nn), dtype=float)
    for j, sp in enumerate(species):
        if sp == "H2O":
            source_gas[j] += src_dry_H2O / epsi_safe
        source_gas[j] += src_pyr_gas[j] / epsi_safe
        source_gas[j] += src_char_gas[j] / epsi_safe

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

    # =========================================================
    # 9. Balance de densidades sólidas   d(rho_s_i)/dt [kg/m³_bed/s]
    #    ∂ρs,i/∂t + ∂(vs · ρs,i)/∂z = Ss,i
    # =========================================================
    _, src_char_from_pyr = pyrolysis_sources(
        r_pyr=r_pyr, yields=yields, MW_gas=MW_arr, species=species,
    )   # (N,) [kg_char/m³_bed/s]

    r_char_total = r_ox + r_CO2 + r_H2O                 # (N,) consumo total de char

    src_s = np.array([
        -r_pyr,                             # biomasa: perdida por pirólisis
        src_char_from_pyr - r_char_total,   # char: ganado de pir., perdido por rxns
        -r_dry,                             # humedad: perdida por secado
    ])   # (3, N) [kg/m³_bed/s]

    # Transporte convectivo del sólido (activo sólo en 1D con vs ≠ 0)
    if vs_signed != 0.0 and nn > 1:
        conv_s = np.zeros((3, nn), dtype=float)
        for i in range(3):
            rho_in_i = (float(rho_s_inlet[i])
                        if rho_s_inlet is not None else None)
            F_s = solid_convective_flux(
                rho_cell=rho_solid[i],
                vs_face=vs_face,
                rho_solid_in=rho_in_i,
            )
            conv_s[i] = -(F_s[1:] - F_s[:-1]) / dz
    else:
        conv_s = np.zeros((3, nn), dtype=float)

    d_rho_biomass  = conv_s[0] + src_s[0]
    d_rho_char     = conv_s[1] + src_s[1]
    d_rho_moisture = conv_s[2] + src_s[2]

    # Clip: evitar que tasas negativas empujen una variable ya nula hacia valores negativos.
    # Umbral 1e-6 kg/m³_bed (≈1 μg/m³): físicamente cero. Umbral más alto que la
    # precisión de máquina para evitar que BDF necesite pasos < eps*t en sim largas.
    _EPS_RHO = 1.0e-6
    d_rho_biomass  = np.where(rho_biomass  < _EPS_RHO, np.maximum(d_rho_biomass,  0.0), d_rho_biomass)
    d_rho_char     = np.where(rho_char     < _EPS_RHO, np.maximum(d_rho_char,     0.0), d_rho_char)
    d_rho_moisture = np.where(rho_moisture < _EPS_RHO, np.maximum(d_rho_moisture, 0.0), d_rho_moisture)

    # =========================================================
    # 10. Balances de energía
    # =========================================================
    if not energy:
        dHgdt_arr     = np.zeros(nn, dtype=float)
        dTsdt_arr     = np.zeros(nn, dtype=float)
        dQ_mt_acc_dt  = np.zeros(nn, dtype=float)
        dQ_rxn_acc_dt = np.zeros(nn, dtype=float)
    else:
        # Máscara de sólido presente: sin sólido no hay superficie de intercambio.
        # Evita a_p→∞ (SCM) cuando rho_char→0 y produce dTsdt→∞ dividida por Cs_vol→0.
        solid_present = (rho_biomass + rho_char + rho_moisture) > _EPS_RHO   # (N,) bool

        # Intercambio de calor gas-sólido [W/m³_bed]
        q_gs_vol = h_bed_arr * a_p * (Tg_arr - Ts_arr)   # >0 cuando gas calienta sólido
        q_gs_vol = np.where(solid_present, q_gs_vol, 0.0) # cero si no hay sólido

        # Flujo de calor gas-pared [W/m³_bed]
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

        div_h_conv  = (Fh_conv[1:] - Fh_conv[:-1]) / dz
        div_qg_diff = (qg_diff[1:] - qg_diff[:-1]) / dz

        # Entalpía portada por las nuevas especies que entran al gas desde el sólido.
        # Las moléculas aparecen a Ts (temperatura del sólido), no a Tg.
        # source_gas [mol/m³_gas/s] × epsi_r → [mol/m³_bed/s] × h_i(Ts) [J/mol]
        # = [J/m³_bed/s]. Término distinto a q_gs_vol (que es HT convectivo superficial).
        h_i_Ts        = calc_species_enthalpy(Ts_arr, prop_gas, nc, gas_T_ref)  # (nc, N)
        q_masstransfer = epsi_r * np.sum(source_gas * h_i_Ts, axis=0)            # (N,) [J/m³_bed/s]

        dHgdt_arr = (-epsi_r * div_h_conv
                     - div_qg_diff
                     - q_gs_vol
                     + qwall_vol
                     + q_masstransfer)

        # ── Energía del sólido (Ts) ───────────────────────────────────────────
        Q_dry  = -drying_enthalpy_sink(r_dry)
        Q_pyr  = -pyrolysis_enthalpy_sink(r_pyr, dH_pyr)
        Q_char = char_reaction_heat(
            r_ox=r_ox, r_CO2=r_CO2, r_H2O=r_H2O,
            Ts=Ts_arr, heating_values=hv,
            co_co2_ratio=co_co2, char_comp=char_comp,
        )
        Q_rxn_vol = Q_dry + Q_pyr + Q_char

        Cp_fns = solid_config["Cp_fns"]
        h_fns  = solid_config["h_fns"]   # H_j(T) = ∫_{273}^T Cp_j dT [J/kg]
        Cp0 = np.asarray(Cp_fns[0](Ts_arr), dtype=float)   # (N,) [J/kg/K]
        Cp1 = np.asarray(Cp_fns[1](Ts_arr), dtype=float)
        Cp2 = np.asarray(Cp_fns[2](Ts_arr), dtype=float)
        Cs_vol = (rho_biomass * Cp0 + rho_char * Cp1 + rho_moisture * Cp2)
        Cs_vol = np.maximum(Cs_vol, 1.0e-6)

        # Corrección de masa: d(Σ_j ρ_j·H_j(Ts))/dt = Q_rxn + q_gs
        # → Cs·dTs/dt = Q_rxn + q_gs − Σ_j H_j(Ts)·(dρ_j/dt)
        # Usando H_j = ∫Cp_j dT (integral exacta), no Cp_j·Ts (que introduce error cuando
        # Cp depende de T). La corrección -Cp_j·Ts solo es exacta para Cp constante.
        H0 = np.asarray(h_fns[0](Ts_arr), dtype=float)  # (N,) [J/kg]
        H1 = np.asarray(h_fns[1](Ts_arr), dtype=float)
        H2 = np.asarray(h_fns[2](Ts_arr), dtype=float)
        thermal_mass_correction = -(
            H0 * (src_s[0] + conv_s[0]) +
            H1 * (src_s[1] + conv_s[1]) +
            H2 * (src_s[2] + conv_s[2])
        )                                                    # (N,) [J/m³_bed/s]

        # dTsdt = 0 donde no hay sólido: Ts es irrelevante y no debe producir rigidez
        dTsdt_arr = np.where(solid_present,
                             (Q_rxn_vol + q_gs_vol + thermal_mass_correction) / Cs_vol,
                             0.0)

        # Acumuladores energéticos: d/dt(acc) = tasa instantánea
        # Integrados por BDF con el mismo control de error → cierre energético exacto
        #
        # Q_rxn_acc = ∫Q_rxn_vol dt  (SOLO calor de reacciones, SIN corrección térmica)
        # La thermal_mass_correction es interna al ODE: hace que d(Cs·Ts)/dt = Q_rxn + q_gs,
        # de modo que ΔHs = Q_rxn_acc + Q_gs en el post-proceso. Si se incluye en el
        # acumulador, el cierre del sólido falla: ΔHs − Q_gs − Q_rxn_acc = −∫thermal_correction.
        dQ_mt_acc_dt  = q_masstransfer   # (N,) [J/m³_bed/s]
        dQ_rxn_acc_dt = Q_rxn_vol        # (N,) [J/m³_bed/s]  — sin thermal_mass_correction

    # =========================================================
    # 11. ODE de pared   dTw/dt  [sólo si shell_tube activo]
    # =========================================================
    if shell_tube:
        A_w       = float(wall_config["A_w"])
        mat       = wall_config["material"]
        rho_w_arr = eval_solid_property(mat["rho"], Tw_arr)  # (N,) [kg/m³]
        cp_w_arr  = eval_solid_property(mat["cp"],  Tw_arr)  # (N,) [J/kg/K]
        k_w_arr   = eval_solid_property(mat["k"],   Tw_arr)  # (N,) [W/m/K]

        Q_gw_cell  = h_wall_arr * Pi * dz * (Tg_arr - Tw_arr)  # [W/celda] positivo si Tg > Tw
        Q_ext_cell = wall_exterior_q(
            Tw_arr=Tw_arr, thermal_bc_cfg=thermal_bc_cfg,
            k_w_arr=k_w_arr, Pi=Pi, Po=Po, dz=dz, N=nn,
        )
        Q_ax_cell  = wall_axial_q(Tw_arr=Tw_arr, k_w_arr=k_w_arr, A_w=A_w, dz=dz)
        dTwdt_arr  = wall_ode_rhs(
            Q_gw_cell=Q_gw_cell, Q_ext_cell=Q_ext_cell, Q_ax_cell=Q_ax_cell,
            rho_w_arr=rho_w_arr, cp_w_arr=cp_w_arr, A_w=A_w, dz=dz,
        )

    # =========================================================
    # 12. Empaquetado del RHS
    # =========================================================
    parts = (
        [dCdt_mat[i] for i in range(nc)]
        + [d_rho_biomass, d_rho_char, d_rho_moisture]
        + [dHgdt_arr, dTsdt_arr]
    )
    if shell_tube:
        parts.append(dTwdt_arr)
    # Acumuladores: siempre al final (misma posición que en pack_state_vector)
    parts += [dQ_mt_acc_dt, dQ_rxn_acc_dt]
    return np.concatenate(parts)
