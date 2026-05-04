"""
@module rhs_adsorption
@brief Función RHS del ODE para la simulación de columna de adsorción 1D.

@details
Implementa el lado derecho (RHS) del sistema de EDOs que el integrador temporal
(scipy.integrate.solve_ivp / odeint) evalúa en cada paso de tiempo.

Este módulo es el núcleo físico de cualquier proceso de adsorción en columna
empaquetada: PSA (Pressure Swing Adsorption), TSA (Temperature Swing Adsorption),
VSA (Vacuum Swing Adsorption) y sus variantes. La física de adsorción está
siempre presente; lo que cambia entre procesos son las condiciones de contorno,
gestionadas por `get_step_boundary` a través de `bc_config`.

Layout del vector de estado:

    Sin shell-tube (wall_config ausente o None):
        sv = [C.reshape(-1), q.reshape(-1), Hg, Ts]
        shape: (2*nc*N + 2*N,)

    Con shell-tube (wall_config presente):
        sv = [C.reshape(-1), q.reshape(-1), Hg, Ts, Tw]
        shape: (2*nc*N + 3*N,)
        Tw — temperatura de la pared por celda [K], shape (N,)

El RHS ensambla en orden:
    1. Desempaquetado → C, q, Hg, Ts, Tg, P, y [, Tw si shell_tube]
    2. Contornos del paso → v_in, v_out, C_in, T_in, y_in
    3. Propiedades de mezcla → rho, mu, k, h_i, Dim
    4. Velocidades Ergun en caras → v_face
    5. Coeficientes de transporte → k_mtc, D_disp, h_bed, h_wall
    6. Cinética LDF → dqdt = k_mtc*(q_eq - q)
    7. Flujos de especie (convección + difusión)
    8. Balance de energía del gas (Hg) y del sólido (Ts)
       8.4 Fuente de pared:
           - Sin shell-tube: wall_heat_flux() con BC térmica exterior
           - Con shell-tube: qwall_vol = h_wall * Pi/Ai * (Tw − Tg)
       8.5 ODE de pared (solo shell-tube): dTw/dt
           = (Q_gw + Q_ext + Q_ax) / (rho_w * cp_w * A_w * dz)
    9. Empaquetado del RHS [dC/dt, dq/dt, dHg/dt, dTs/dt, dTw/dt si shell_tube]

El dict `params` que se pasa al RHS contiene todos los parámetros físicos y
de configuración. El campo `params["_cache"]` es un dict mutable que actúa
como caché de propiedades entre evaluaciones (optimización para modo "frozen").

Estructura mínima de params (ver runner_adsorption.py para la lista validada):
    n_comp, N, dz                          — malla
    prop_gas                               — propiedades puras del gas
    MW                                     — masas molares [kg/mol], shape (nc,)
    gas_T_ref                              — temperatura de referencia entálpica [K]
    iso_fn                                 — callable: isoterma multicomponente
    epsi                                   — porosidad del lecho [-]
    rho_s                                  — densidad del sólido [kg/m³]
    Cp_s                                   — Cp del sólido [J/kg/K]
    k_s                                    — conductividad del sólido [W/m/K]
    dH                                     — entalpía de adsorción [J/mol], (nc, N)
    prop_lecho                             — dict {D_p, a_surf, tau, r_pore, eps}
    bc_config                              — dict de contornos (de build_boundary_c_config)
    Ai, Di, Pi, Po                         — geometría de la sección transversal
    trans_config                           — dict de transporte
    thermal_bc_config                      — dict de contorno térmico
    Tg_init                                — temperatura inicial [K] (para energy=False)
    step                                   — str: paso activo (inyectado por runner)
    energy                                 — bool: activar balance de energía
    prop_update_mode                       — "always" o "frozen"
    trans_update_mode                      — "always" o "frozen"
    _cache                                 — dict mutable (gestionado por runner)
    wall_config     — (opcional) dict de pared dinámica (de build_wall_config);
                      si ausente o None, el modelo opera sin pared dinámica

Unidades: SI.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from src.units.adsorber.state import unpack_state_vector, pack_state_vector
from src.boundary_conditions.adsorber_boundary import get_step_boundary
from src.physics.mixture_gas import compute_gas_mixture_properties
from src.physics.momentum.ergun import ergun_face_velocity
from src.physics.transport.transfer_coefficients import compute_transfer_coefficients
from src.physics.transport.nusselt import h_wall_tube
from src.physics.thermal.wall_heat_flux import wall_heat_flux
from src.physics.thermal.wall_ode import wall_exterior_q, wall_axial_q, wall_ode_rhs
from src.physics.thermodynamics.solid_props import eval_solid_property
from src.discretization.fluxes import (
    convective_flux,
    diffusive_flux,
    gas_enthalpy_convective_flux,
    gas_diffusive_heat_flux,
    solid_diffusive_heat_flux,
)
from src.utils.profiling import profiled


@profiled
def core_rhs(
    t: float,
    sv: np.ndarray,
    params: Dict[str, Any],
) -> np.ndarray:
    """
    @brief
    Evalúa el RHS del sistema de EDOs para un paso de adsorción activo.

    Parameters
    ----------
    t      : float — tiempo actual [s]
    sv     : ndarray, shape (2*nc*N + 2*N,) — vector de estado del integrador
    params : dict — todos los parámetros físicos y de configuración del modelo

    Returns
    -------
    dydt : ndarray, shape (2*nc*N + 2*N,) — derivada temporal del vector de estado
    """
    nc       = int(params["n_comp"])
    nn       = int(params["N"])
    dz       = float(params["dz"])
    epsi     = float(params["epsi"])
    rho_s    = float(params["rho_s"])
    Cp_s     = float(params["Cp_s"])
    k_s      = float(params["k_s"])
    dH_mat   = params["dH"]                   # (nc, N) [J/mol]
    prop_lecho = params["prop_lecho"]
    iso_fn   = params["iso_fn"]
    prop_gas = params["prop_gas"]
    MW_arr   = np.asarray(params["MW"], dtype=float).reshape(-1)
    gas_T_ref = float(params["gas_T_ref"])
    bc_config = params["bc_config"]
    Ai       = float(params["Ai"])
    Di       = float(params["Di"])
    Pi       = float(params["Pi"])
    Po       = float(params["Po"])
    trans_config    = params["trans_config"]
    thermal_bc_cfg  = params["thermal_bc_config"]
    step     = params["step"]
    energy   = bool(params["energy"])
    prop_update_mode  = params.get("prop_update_mode",  "always")
    trans_update_mode = params.get("trans_update_mode", "always")
    cache    = params["_cache"]

    wall_config = params.get("wall_config")
    shell_tube  = wall_config is not None

    # =========================================================
    # 1. Desempaquetado del estado
    # =========================================================
    state = unpack_state_vector(
        sv=sv, n_comp=nc, N=nn,
        prop_gas=prop_gas, epsi=epsi,
        Tg_guess=cache.get("Tg_last", np.full(nn, 300.0)),
        gas_T_ref=gas_T_ref,
        shell_tube=shell_tube,
    )
    C_mat    = state["C"]       # (nc, N) [mol/m³]
    q_mat    = state["q"]       # (nc, N) [mol/kg]
    Hg_arr   = state["Hg"]     # (N,) [J/m³]
    Ts_arr   = state["Ts"]     # (N,) [K]
    Tg_arr   = state["Tg"]     # (N,) [K]
    P_bar    = state["P"]       # (N,) [bar]
    y_mat    = state["y"]       # (nc, N) [-]
    Tw_arr   = state["Tw"] if shell_tube else None   # (N,) [K] o None
    P_Pa     = P_bar * 1.0e5
    x_mat    = y_mat.T          # (N, nc) — node-first para funciones de mezcla

    # Guardar Tg como guess para la próxima iteración de Newton
    cache["Tg_last"] = Tg_arr

    # =========================================================
    # 2. Contornos del paso activo
    # =========================================================
    boundary = get_step_boundary(
        t=t, P_cell=P_bar, Tg_cell=Tg_arr, y_cell=y_mat,
        step=step, bc_config=bc_config, n_comp=nc,
        MW=MW_arr, epsi=epsi, Ai=Ai,
    )
    T_in   = boundary["inlet"]["T_K"]
    C_in   = boundary["inlet"]["C_mol_m3"]     # (nc,) o None
    v_in   = boundary["inlet"]["v_m_s"]
    v_out  = boundary["outlet"]["v_m_s"]

    # =========================================================
    # 3. Propiedades de mezcla
    # =========================================================
    if prop_update_mode == "frozen" and cache.get("gas_props") is not None:
        gas_props = cache["gas_props"]
    else:
        gas_props = compute_gas_mixture_properties(
            P_Pa=P_Pa, Tg=Tg_arr, x=x_mat,
            prop_gas=prop_gas, n_comp=nc, N=nn,
        )
        cache["gas_props"] = gas_props

    rho_arr  = gas_props["rho"]    # (N,) [kg/m³]
    mu_arr   = gas_props["mu"]     # (N,) [Pa·s]
    k_g_arr  = gas_props["k"]      # (N,) [W/m/K]
    h_i_mat  = gas_props["h_i"].T  # (nc, N) [J/mol]

    # =========================================================
    # 4. Velocidades Ergun en caras
    # =========================================================
    v_face = ergun_face_velocity(
        P=P_Pa, rho_g=rho_arr, mu_g=mu_arr,
        epsi=epsi,
        dp=float(prop_lecho["D_p"][0]),
        dz=dz,
        v_in=v_in, v_out=v_out,
    )
    v_cell = 0.5 * (v_face[:-1] + v_face[1:])  # (N,) [m/s]
    u_rel  = np.abs(v_cell)

    # =========================================================
    # 5. Coeficientes de transporte
    # =========================================================
    if trans_update_mode == "frozen" and cache.get("trans_props") is not None:
        trans_props = cache["trans_props"]
    else:
        # Determinar Tw para el cómputo de Ra dentro de compute_transfer_coefficients.
        # shell_tube: Tw es variable de estado, exactamente conocida.
        # fixed_twall: Tw es constante de configuración, exactamente conocida.
        # ambient_htc: Tw_int estimada en un paso explícito (cadena de resistencias)
        #   usando h_wall forzado como semilla; las propiedades se evalúan a Tg bulk.
        # adiabatic / heatfluxwall / h_wall_fixed: Ra=None → solo convección forzada.
        _twall_mode_ads = thermal_bc_cfg["mode"]
        if shell_tube:
            Tw_for_trans = Tw_arr
        elif _twall_mode_ads == "fixed_twall" and trans_config.get("h_wall_fixed") is None:
            Tw_for_trans = np.full(nn, float(thermal_bc_cfg["T_wall"]), dtype=float)
        elif _twall_mode_ads == "ambient_htc" and trans_config.get("h_wall_fixed") is None:
            _Cp_m   = gas_props["Cp_mass"]
            _Re0    = rho_arr * u_rel * Di / np.maximum(mu_arr, 1.0e-30)
            _Pr0    = _Cp_m * mu_arr / np.maximum(k_g_arr, 1.0e-30)
            _h0     = h_wall_tube(Re=_Re0, Pr=_Pr0, D=Di, k_film=k_g_arr, Ra=None)
            _Do     = Po / np.pi
            _Ri     = 1.0 / np.maximum(_h0 * Pi, 1.0e-30)
            _Rc     = np.log(_Do / Di) / (2.0 * np.pi * float(thermal_bc_cfg["k_wall"]))
            _Rx     = 1.0 / (float(thermal_bc_cfg["h_ambi"]) * Po)
            _Rt     = _Ri + _Rc + _Rx
            Tw_for_trans = Tg_arr - (Tg_arr - float(thermal_bc_cfg["T_ambi"])) * _Ri / np.maximum(_Rt, 1.0e-30)
        else:
            Tw_for_trans = None
        trans_props = compute_transfer_coefficients(
            Tg=Tg_arr, Ts=Ts_arr, x=x_mat,
            gas_props=gas_props, u_rel=u_rel,
            prop_gas=prop_gas, prop_lecho=prop_lecho,
            Di=Di, trans_config=trans_config,
            n_comp=nc, N=nn,
            Tw=Tw_for_trans,
            L=dz * nn,
        )
        cache["trans_props"] = trans_props

    k_mtc_mat  = trans_props["k_mtc"]   # (nc, N) [1/s]
    D_disp_mat = trans_props["D_disp"]  # (nc, N) [m²/s]
    h_bed_arr  = trans_props["h_bed"]   # (N,) [W/m²/K]
    h_wall_arr = trans_props["h_wall"]  # (N,) [W/m²/K]

    # =========================================================
    # 6. Adsorción: equilibrio y cinética LDF
    # =========================================================
    # Presiones parciales: p_i = C_i * R * Tg / 1e5  [bar]
    R_GAS = 8.31446261815324
    P_part_list = [C_mat[i, :] * R_GAS * Tg_arr / 1.0e5 for i in range(nc)]
    q_eq_mat = np.asarray(iso_fn(P_part_list, Ts_arr), dtype=float)   # (nc, N) [mol/kg]

    dqdt_mat = k_mtc_mat * (q_eq_mat - q_mat)                          # (nc, N) [mol/kg/s]
    source_ads_mat = ((1.0 - epsi) / epsi) * rho_s * dqdt_mat          # (nc, N) [mol/m³/s]

    # =========================================================
    # 7. Tipo de BC en inlet para especies y energía del gas
    # =========================================================
    # Dirichlet si hay flujo entrante; Neumann (gradiente libre) si no
    if step in ("ads", "purge", "pr_feed"):
        bc_in_species = "dirichlet"
        bc_in_T       = "dirichlet"
    else:
        bc_in_species = "neumann"
        bc_in_T       = "neumann"
    bc_out_species = "neumann"
    bc_out_T       = "neumann"

    # =========================================================
    # 8. Balance de especies (gas-phase)  dC_i/dt
    # =========================================================
    dCdt_mat = np.zeros((nc, nn), dtype=float)

    for i in range(nc):
        C_in_i = None if C_in is None else float(C_in[i])
        F_conv = convective_flux(
            phi_cell=C_mat[i, :], v_face=v_face,
            phi_in=C_in_i, phi_out=None,
        )
        F_diff = diffusive_flux(
            phi_cell=C_mat[i, :], Gamma=D_disp_mat[i, :], dz=dz,
            phi_in=C_in_i, phi_out=None,
            bc_in=bc_in_species, bc_out=bc_out_species,
            face_method="arithmetic",
        )
        F_tot = F_conv + F_diff
        dCdt_mat[i, :] = -(F_tot[1:] - F_tot[:-1]) / dz - source_ads_mat[i, :]

    # =========================================================
    # 9. Balance de energía
    # =========================================================
    if not energy:
        dHgdt_arr = np.zeros(nn, dtype=float)
        dTsdt_arr = np.zeros(nn, dtype=float)
    else:
        a_surf_arr = np.asarray(prop_lecho["a_surf"], dtype=float).reshape(-1)

        # 9.1 Flujo convectivo de entalpía gaseosa [W/m²]
        Fh_conv = gas_enthalpy_convective_flux(
            Tg_cell=Tg_arr, C_cell=C_mat, v_face=v_face,
            prop_gas=prop_gas, n_comp=nc, gas_T_ref=gas_T_ref,
            T_in=T_in if bc_in_T == "dirichlet" else None,
            C_in=C_in if bc_in_T == "dirichlet" else None,
        )

        # 9.2 Flujo difusivo de calor del gas [W/m²]
        qg_diff = gas_diffusive_heat_flux(
            Tg_cell=Tg_arr, k_g=k_g_arr, dz=dz,
            T_in=T_in if bc_in_T == "dirichlet" else None,
            T_out=None,
            bc_in=bc_in_T, bc_out=bc_out_T,
            face_method="arithmetic",
        )

        # 9.3 Intercambio gas-sólido [W/m³]
        q_gs_vol = h_bed_arr * a_surf_arr * (Tg_arr - Ts_arr)

        # 9.4 Fuente térmica de pared [W/m³]
        if shell_tube:
            # Pared dinámica: Tw es variable de estado; acoplamiento directo gas↔pared
            qwall_vol = h_wall_arr * (Pi / Ai) * (Tw_arr - Tg_arr)
        else:
            qwall_vol, _, _ = wall_heat_flux(
                Tg=Tg_arr, h_wall=h_wall_arr,
                thermal_bc_config=thermal_bc_cfg,
                N=nn, Ai=Ai, Pi=Pi, Po=Po, dz=dz,
            )

        # 9.5 Entalpía sensible retirada del gas por adsorción [W/m³]
        # h_i_mat (nc, N) · dqdt_mat (nc, N), sum axis=0 → (N,)
        q_ads_sens_gas_vol = (1.0 - epsi) * rho_s * np.sum(dqdt_mat * h_i_mat, axis=0)

        # 9.6 Balance de Hg (vectorizado)
        div_h_conv  = (Fh_conv[1:]  - Fh_conv[:-1])  / dz   # (N,)
        div_qg_diff = (qg_diff[1:]  - qg_diff[:-1])  / dz   # (N,)
        dHgdt_arr = (-epsi * div_h_conv
                     - div_qg_diff
                     - q_gs_vol
                     + qwall_vol
                     - q_ads_sens_gas_vol)

        # 9.7 Balance de Ts
        qs_diff = solid_diffusive_heat_flux(
            Ts_cell=Ts_arr, k_s=k_s, dz=dz,
            T_in=None, T_out=None,
            bc_in="neumann", bc_out="neumann",
            face_method="arithmetic",
        )
        # Calor de adsorción liberado al sólido: dH (nc, N) · dqdt (nc, N), sum → (N,)
        q_ads_vol = (1.0 - epsi) * rho_s * np.sum(dH_mat * dqdt_mat, axis=0)
        Cs_vol    = (1.0 - epsi) * rho_s * Cp_s * np.ones(nn, dtype=float)

        div_qs_diff = (qs_diff[1:] - qs_diff[:-1]) / dz      # (N,)
        dTsdt_arr = (-div_qs_diff + q_gs_vol + q_ads_vol) / np.maximum(Cs_vol, 1.0e-30)

    # =========================================================
    # 9.8 ODE de pared — dTw/dt  (solo si shell_tube activo)
    # =========================================================
    if shell_tube:
        A_w = float(wall_config["A_w"])
        mat = wall_config["material"]
        rho_w_arr = eval_solid_property(mat["rho"], Tw_arr)   # (N,) [kg/m³]
        cp_w_arr  = eval_solid_property(mat["cp"],  Tw_arr)   # (N,) [J/kg/K]
        k_w_arr   = eval_solid_property(mat["k"],   Tw_arr)   # (N,) [W/m/K]

        # Acoplamiento gas → pared [W/celda], positivo si Tg > Tw (gas calienta la pared)
        Q_gw_cell = h_wall_arr * Pi * dz * (Tg_arr - Tw_arr)

        Q_ext_cell = wall_exterior_q(
            Tw_arr=Tw_arr, thermal_bc_cfg=thermal_bc_cfg,
            k_w_arr=k_w_arr, Pi=Pi, Po=Po, dz=dz, N=nn,
        )
        Q_ax_cell = wall_axial_q(Tw_arr=Tw_arr, k_w_arr=k_w_arr, A_w=A_w, dz=dz)

        dTwdt_arr = wall_ode_rhs(
            Q_gw_cell=Q_gw_cell, Q_ext_cell=Q_ext_cell, Q_ax_cell=Q_ax_cell,
            rho_w_arr=rho_w_arr, cp_w_arr=cp_w_arr, A_w=A_w, dz=dz,
        )

    # =========================================================
    # 10. Empaquetado del RHS
    # =========================================================
    parts = (
        [dCdt_mat[i, :] for i in range(nc)] +
        [dqdt_mat[i, :] for i in range(nc)] +
        [dHgdt_arr, dTsdt_arr]
    )
    if shell_tube:
        parts.append(dTwdt_arr)
    return np.concatenate(parts)
