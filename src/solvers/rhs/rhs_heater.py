"""
@module rhs_heater
@brief Función RHS del ODE para la simulación de calentador de gas 1D.

@details
Implementa el lado derecho (RHS) del sistema de EDOs que el integrador temporal
(scipy.integrate.solve_ivp) evalúa en cada paso de tiempo.

El calentador es un tubo vacío de sección circular que transfiere calor al gas
mediante su pared. No hay lecho empaquetado, adsorción ni fase sólida.

Layout del vector de estado:

    Sin shell-tube (wall_config ausente o None):
        sv = [C.reshape(-1), Hg]
        shape: (nc + 1) * N

    Con shell-tube (wall_config presente):
        sv = [C.reshape(-1), Hg, Tw]
        shape: (nc + 2) * N
        Tw — temperatura de la pared por celda [K], shape (N,)

El RHS ensambla en orden:
    1. Desempaquetado → C, Hg, Tg, P, y, Ctot [, Tw si shell_tube]
    2. Condiciones de contorno → v_in, T_in, C_in, v_out
    3. Propiedades de mezcla → rho, mu, k_g, h_i, Cp_mass
    4. Velocidades Darcy-Weisbach en caras → v_face, v_cell
    5. Coeficiente h_wall → constante o correlación (forzada + natural mixta)
    6. Balance de especies (convección pura): dC_i/dt
    7. Balance de energía del gas (Hg): dHg/dt
       7.3 Término gas↔pared:
           - Sin shell-tube: wall_heat_flux() con BC térmica exterior
           - Con shell-tube: q_gw_vol = h_wall * Pi/Ai * (Tw − Tg)
       7.5 ODE de pared (solo shell-tube): dTw/dt
           = (Q_gw + Q_ext + Q_ax) / (rho_w * cp_w * A_w * dz)
    8. Empaquetado del RHS [dC/dt, dHg/dt, dTw/dt si shell_tube]

Hipótesis:
    - Tubo vacío: porosidad efectiva = 1.0 (todo el volumen es gas)
    - Sin dispersión axial de especies (Peclet de masa >> 1 en tubo vacío)
    - Conducción axial de calor en el gas incluida (Peclet térmico puede ser ~1)
    - Propiedades del gas para la correlación Nusselt:
        · shell-tube o fixed_twall: μ, k, Cp a T_film = 0.5*(Tg+Tw); ρ a bulk
        · ambient_htc sin shell-tube: propiedades a Tg bulk; Tw_int estimada en un
          paso explícito desde la cadena de resistencias para calcular Ra
        · adiabatic / heatfluxwall: propiedades a Tg bulk; Ra=None (solo forzada)
    - Propiedades del sólido (rho_w, cp_w, k_w) evaluadas a Tw local en cada llamada

Estructura de params:
    n_comp          — número de componentes
    N               — número de celdas
    dz              — tamaño de celda [m]
    prop_gas        — propiedades puras del gas (de build_gas_prop_config)
    MW              — masas molares [kg/mol], shape (nc,)
    gas_T_ref       — temperatura de referencia entálpica [K]
    Di              — diámetro interno del tubo [m]
    Ai              — área interna de sección transversal [m²]
    Pi              — perímetro interno [m]
    Po              — perímetro externo [m]
    bc_config       — dict de contornos (de build_boundary_c_config del heater)
    trans_config    — dict de transporte (de build_transport_config del heater)
    thermal_bc_config — dict de contorno térmico (de build_thermal_bc_config)
    energy          — bool: activar balance de energía
    prop_update_mode  — "always" o "frozen"
    _cache          — dict mutable (gestionado por runner)
    wall_config     — (opcional) dict de pared dinámica (de build_wall_config);
                      si ausente o None, el modelo opera sin pared dinámica

Unidades: SI.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from src.units.heater.state import unpack_state_vector
from src.boundary_conditions.heater_boundary import get_heater_boundary
from src.physics.mixture_gas import compute_gas_mixture_properties
from src.physics.momentum.darcy_weisbach import continuity_face_velocity
from src.physics.transport.nusselt import h_wall_tube, compute_Ra_Di
from src.physics.thermal.wall_heat_flux import wall_heat_flux
from src.physics.thermal.wall_ode import wall_exterior_q, wall_axial_q, wall_ode_rhs
from src.physics.thermodynamics.solid_props import eval_solid_property
from src.discretization.fluxes import (
    convective_flux,
    gas_enthalpy_convective_flux,
    gas_diffusive_heat_flux,
)
from src.utils.profiling import profiled

_EPSI_TUBE = 1.0   # tubo vacío: porosidad efectiva = 1


@profiled
def core_rhs(
    t: float,
    sv: np.ndarray,
    params: Dict[str, Any],
) -> np.ndarray:
    """
    @brief
    Evalúa el RHS del sistema de EDOs del calentador de gas 1D.

    Parameters
    ----------
    t      : float — tiempo actual [s]
    sv     : ndarray, shape ((nc + 1) * N,) — vector de estado del integrador
    params : dict — todos los parámetros físicos y de configuración del modelo

    Returns
    -------
    dydt : ndarray, shape ((nc + 1) * N,) — derivada temporal del vector de estado
    """
    nc       = int(params["n_comp"])
    nn       = int(params["N"])
    dz       = float(params["dz"])
    prop_gas = params["prop_gas"]
    gas_T_ref = float(params["gas_T_ref"])
    Di_v     = float(params["Di"])
    Ai       = float(params["Ai"])
    Pi       = float(params["Pi"])
    Po       = float(params["Po"])
    bc_config       = params["bc_config"]
    trans_config    = params["trans_config"]
    thermal_bc_cfg  = params["thermal_bc_config"]
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
        prop_gas=prop_gas,
        Tg_guess=cache.get("Tg_last", np.full(nn, 300.0)),
        gas_T_ref=gas_T_ref,
        shell_tube=shell_tube,
    )
    C_mat  = state["C"]     # (nc, N) [mol/m³]
    Hg_arr = state["Hg"]   # (N,) [J/m³]  — usado por Newton (implícito en unpack)
    Tg_arr = state["Tg"]   # (N,) [K]
    P_bar  = state["P"]    # (N,) [bar]
    y_mat  = state["y"]    # (nc, N) [-]
    Ctot   = state["Ctot"] # (N,) [mol/m³]
    Tw_arr = state["Tw"]   if shell_tube else None   # (N,) [K]
    P_Pa   = P_bar * 1.0e5
    x_mat  = y_mat.T       # (N, nc) — node-first para compute_gas_mixture_properties

    cache["Tg_last"] = Tg_arr

    # =========================================================
    # 2. Condiciones de contorno
    # =========================================================
    boundary = get_heater_boundary(
        t=t, P_cell=P_bar, Ctot_cell=Ctot,
        bc_config=bc_config, n_comp=nc,
    )
    T_in  = boundary["inlet"]["T_K"]
    C_in  = boundary["inlet"]["C_mol_m3"]   # (nc,)
    v_in  = boundary["inlet"]["v_m_s"]

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

    rho_arr    = gas_props["rho"]      # (N,) [kg/m³]
    mu_arr     = gas_props["mu"]       # (N,) [Pa·s]
    k_g_arr    = gas_props["k"]        # (N,) [W/m/K]
    Cp_mass_arr = gas_props["Cp_mass"] # (N,) [J/kg/K]

    # =========================================================
    # 4. Velocidades en caras — conservación de masa (continuidad)
    # =========================================================
    # Se propaga v_in mediante v·C_tot = cte en lugar de reconstruir
    # v desde el gradiente de presión (DW). A velocidades bajas ΔP es
    # demasiado pequeño para ser resuelto por el ODE (ΔP << RTOL·P),
    # lo que provocaba velocidades interiores ≈ 0 e inestabilidad numérica.
    C_tot_in = float(C_in.sum())
    v_face = continuity_face_velocity(
        C_tot_cell=Ctot,
        C_tot_in=C_tot_in,
        v_in=v_in,
    )
    v_cell = 0.5 * (v_face[:-1] + v_face[1:])   # (N,) [m/s]

    # =========================================================
    # 5. Coeficiente de transferencia de calor gas-pared
    # =========================================================
    if trans_config["mode"] == "constant":
        h_wall_arr = trans_config["h_wall"]   # ndarray(N,) ya construido en config
    elif trans_update_mode == "frozen" and cache.get("h_wall") is not None:
        h_wall_arr = cache["h_wall"]
    else:
        _twall_mode = thermal_bc_cfg["mode"]
        _use_tfilm  = trans_config["h_wall_fixed"] is None and (
            shell_tube or _twall_mode == "fixed_twall"
        )
        if _use_tfilm:
            # Propiedades a temperatura de película T_film = 0.5*(Tg+Tw).
            # ρ permanece a temperatura de bulk (ideal gas, P y Tg reales).
            if shell_tube:
                Tw_film = Tw_arr
            else:  # fixed_twall: T_wall es escalar constante
                Tw_film = np.full(nn, float(thermal_bc_cfg["T_wall"]), dtype=float)
            T_film = 0.5 * (Tg_arr + Tw_film)
            gas_film = compute_gas_mixture_properties(
                P_Pa=P_Pa, Tg=T_film, x=x_mat,
                prop_gas=prop_gas, n_comp=nc, N=nn,
            )
            mu_Nu = gas_film["mu"]
            k_Nu  = gas_film["k"]
            Cp_Nu = gas_film["Cp_mass"]
        else:
            mu_Nu = mu_arr
            k_Nu  = k_g_arr
            Cp_Nu = Cp_mass_arr

        Re_wall = rho_arr * np.abs(v_cell) * Di_v / np.maximum(mu_Nu, 1.0e-30)
        Pr      = Cp_Nu * mu_Nu / np.maximum(k_Nu, 1.0e-30)

        # Ra para convección natural:
        #   · shell_tube / fixed_twall: Tw exactamente conocida; propiedades a T_film.
        #   · ambient_htc: Tw_int estimada en un paso explícito con la cadena de
        #     resistencias (R_int + R_cond + R_ext); propiedades a Tg bulk.
        #   · adiabatic / heatfluxwall / h_wall_fixed: Ra=None → solo forzada.
        if _use_tfilm:
            Tw_for_Ra = Tw_arr if shell_tube else Tw_film
            Ra_wall = compute_Ra_Di(
                Tg=Tg_arr, Tw=Tw_for_Ra,
                rho_Tg=rho_arr, mu_film=mu_Nu, k_film=k_Nu, Cp_film=Cp_Nu,
                Di=Di_v,
            )
        elif _twall_mode == "ambient_htc" and trans_config["h_wall_fixed"] is None:
            h_wall_0   = h_wall_tube(Re=Re_wall, Pr=Pr, D=Di_v, k_film=k_Nu, Ra=None)
            Do_val_amb = Po / np.pi
            R_int_amb  = 1.0 / np.maximum(h_wall_0 * Pi, 1.0e-30)
            R_cond_amb = np.log(Do_val_amb / Di_v) / (2.0 * np.pi * float(thermal_bc_cfg["k_wall"]))
            R_ext_amb  = 1.0 / (float(thermal_bc_cfg["h_ambi"]) * Po)
            R_tot_amb  = R_int_amb + R_cond_amb + R_ext_amb
            Tw_int_est = Tg_arr - (Tg_arr - float(thermal_bc_cfg["T_ambi"])) * R_int_amb / np.maximum(R_tot_amb, 1.0e-30)
            Ra_wall = compute_Ra_Di(
                Tg=Tg_arr, Tw=Tw_int_est,
                rho_Tg=rho_arr, mu_film=mu_Nu, k_film=k_Nu, Cp_film=Cp_Nu,
                Di=Di_v,
            )
        else:
            Ra_wall = None

        h_wall_arr = h_wall_tube(Re=Re_wall, Pr=Pr, D=Di_v, k_film=k_Nu, Ra=Ra_wall)
        # Sobreescribir con valor fijo si se suministró en la config
        if trans_config["h_wall_fixed"] is not None:
            h_wall_arr = trans_config["h_wall_fixed"] * np.ones(nn, dtype=float)
        cache["h_wall"] = h_wall_arr

    # =========================================================
    # 6. Balance de especies — dC_i/dt  (convección pura)
    # =========================================================
    # BC: Dirichlet en inlet (flujo siempre entrante), Neumann en outlet
    dCdt_mat = np.zeros((nc, nn), dtype=float)

    for i in range(nc):
        F_conv = convective_flux(
            phi_cell=C_mat[i, :], v_face=v_face,
            phi_in=float(C_in[i]), phi_out=None,
        )
        dCdt_mat[i, :] = -(F_conv[1:] - F_conv[:-1]) / dz

    # =========================================================
    # 7. Balance de energía del gas — dHg/dt
    # =========================================================
    if not energy:
        dHgdt_arr = np.zeros(nn, dtype=float)
    else:
        # 7.1 Flujo convectivo de entalpía [W/m²]
        Fh_conv = gas_enthalpy_convective_flux(
            Tg_cell=Tg_arr, C_cell=C_mat, v_face=v_face,
            prop_gas=prop_gas, n_comp=nc, gas_T_ref=gas_T_ref,
            T_in=T_in,
            C_in=C_in,
        )

        # 7.2 Conducción axial del gas [W/m²]
        # Con v=0 el inlet es una pared cerrada: BC Neumann (flujo nulo).
        # Con v>0 existe un depósito aguas arriba a T_in: BC Dirichlet.
        _bc_in_diff = "neumann" if v_in == 0.0 else "dirichlet"
        qg_diff = gas_diffusive_heat_flux(
            Tg_cell=Tg_arr, k_g=k_g_arr, dz=dz,
            T_in=T_in if _bc_in_diff == "dirichlet" else None,
            T_out=None,
            bc_in=_bc_in_diff, bc_out="neumann",
            face_method="arithmetic",
        )

        # 7.3 Fuente de calor de pared [W/m³ de gas]
        if shell_tube:
            # Pared dinámica: T_w es variable de estado; acoplamiento directo gas↔pared
            qwall_vol = h_wall_arr * (Pi / Ai) * (Tw_arr - Tg_arr)
        else:
            qwall_vol, _, _ = wall_heat_flux(
                Tg=Tg_arr, h_wall=h_wall_arr,
                thermal_bc_config=thermal_bc_cfg,
                N=nn, Ai=Ai, Pi=Pi, Po=Po, dz=dz,
            )

        # 7.4 Balance (tubo vacío: epsi = 1, sin términos de sólido ni adsorción)
        div_h_conv  = (Fh_conv[1:] - Fh_conv[:-1]) / dz   # (N,)
        div_qg_diff = (qg_diff[1:] - qg_diff[:-1])  / dz   # (N,)
        dHgdt_arr = -_EPSI_TUBE * div_h_conv - div_qg_diff + qwall_vol

    # =========================================================
    # 7.5 ODE de pared — dTw/dt  (solo si shell_tube activo)
    # =========================================================
    if shell_tube:
        A_w = float(wall_config["A_w"])
        mat = wall_config["material"]
        rho_w_arr = eval_solid_property(mat["rho"], Tw_arr)   # (N,) [kg/m³]
        cp_w_arr  = eval_solid_property(mat["cp"],  Tw_arr)   # (N,) [J/kg/K]
        k_w_arr   = eval_solid_property(mat["k"],   Tw_arr)   # (N,) [W/m/K]

        # Acoplamiento gas → pared [W/celda], positivo si Tg > Tw (gas calienta la pared)
        Q_gw_cell = h_wall_arr * Pi * dz * (Tg_arr - Tw_arr)

        # Condición de contorno exterior [W/celda]
        Q_ext_cell = wall_exterior_q(
            Tw_arr=Tw_arr, thermal_bc_cfg=thermal_bc_cfg,
            k_w_arr=k_w_arr, Pi=Pi, Po=Po, dz=dz, N=nn,
        )

        # Conducción axial a través de la pared [W/celda] — Neumann en ambos extremos
        Q_ax_cell = wall_axial_q(Tw_arr=Tw_arr, k_w_arr=k_w_arr, A_w=A_w, dz=dz)

        dTwdt_arr = wall_ode_rhs(
            Q_gw_cell=Q_gw_cell, Q_ext_cell=Q_ext_cell, Q_ax_cell=Q_ax_cell,
            rho_w_arr=rho_w_arr, cp_w_arr=cp_w_arr, A_w=A_w, dz=dz,
        )

    # =========================================================
    # 8. Empaquetado del RHS
    # =========================================================
    parts = [dCdt_mat[i, :] for i in range(nc)] + [dHgdt_arr]
    if shell_tube:
        parts.append(dTwdt_arr)
    return np.concatenate(parts)
