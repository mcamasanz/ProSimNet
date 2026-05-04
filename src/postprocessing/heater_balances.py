"""
@module postprocessing.heater_balances
@brief Comprobación de balances de conservación para simulaciones del calentador 1D.

@details
Calcula los balances de conservación comparando la acumulación en el tubo con
los flujos integrados en los contornos y los términos fuente internos.

El heater es un tubo vacío: sin fase sólida, sin adsorción.

Balances disponibles:

1. **Balance molar por especie y total**:
   Acumulación en fase gas vs. flujo convectivo neto en caras de contorno.
   No hay fase adsorbida ni dispersión axial (Peclet de masa >> 1 en tubo vacío).
   La velocidad se reconstruye con Darcy-Weisbach exactamente como en core_rhs.

2. **Balance de energía del gas**:
       Ein − Eout + Qcond_g_net + Qwall = ΔE_gas

   donde:
     Ein, Eout     — entalpía convectiva en caras de contorno [J]
     Qcond_g_net   — conducción térmica neta del gas en contornos [J]
     Qwall         — calor total intercambiado con la pared/ambiente [J]
     ΔE_gas        — acumulación de energía en el gas [J]

   No hay Qgs (sin sólido), ni Eads (sin adsorción).
   Todos los términos se integran en el tiempo con la regla trapezoidal.

El objeto `heater` debe ser el devuelto por `run_step` (SimpleNamespace con
_v_in_results, _v_out_results, _C_in_results, _T_in_results, etc.).

Parámetros requeridos en `params`:
    n_comp, N, dz, Ai, Di, Pi, Po, gas_T_ref, prop_gas, MW
    trans_config, thermal_bc_config

Unidades: SI.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

from src.physics.mixture_gas import compute_gas_mixture_properties
from src.physics.momentum.darcy_weisbach import continuity_face_velocity
from src.physics.transport.nusselt import h_wall_tube, compute_Ra_Di
from src.physics.thermal.wall_heat_flux import wall_heat_flux
from src.physics.thermodynamics.solid_props import eval_solid_property
from src.discretization.fluxes import (
    convective_flux,
    gas_enthalpy_convective_flux,
    gas_diffusive_heat_flux,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Auxiliar privado — potencia exterior sobre la pared (shell_tube)
# ═══════════════════════════════════════════════════════════════════════════════

def _ext_q_dot(
    Tw_arr: np.ndarray,
    thermal_bc_cfg: dict,
    k_w_arr: np.ndarray,
    Pi: float,
    Po: float,
    dz: float,
    N: int,
) -> np.ndarray:
    """
    Calcula la potencia [W] que entra a la pared desde el exterior, por celda.
    Replica la lógica de rhs_heater._wall_exterior_q para el post-proceso.
    Devuelve ceros para modos no aplicables al balance (adiabatic).
    """
    mode = thermal_bc_cfg["mode"]
    nn   = int(N)

    if mode in ("adiabatic", "fixed_twall"):
        return np.zeros(nn, dtype=float)

    if mode == "heatfluxwall":
        Qwall = thermal_bc_cfg["Qwall"]
        if np.ndim(Qwall) == 0:
            return np.full(nn, float(Qwall) / nn, dtype=float)
        else:
            Qwall_arr = np.asarray(Qwall, dtype=float)
            if Qwall_arr.shape != (nn,):
                raise ValueError(f"Qwall array shape {Qwall_arr.shape} != ({nn},)")
            return Qwall_arr.copy()

    if mode == "ambient_htc":
        h_ambi  = float(thermal_bc_cfg["h_ambi"])
        T_ambi  = float(thermal_bc_cfg["T_ambi"])
        Di_wall = Pi / np.pi
        Do_wall = Po / np.pi
        R_cond  = np.log(Do_wall / Di_wall) / (
            2.0 * np.pi * np.maximum(k_w_arr, 1.0e-30) * dz
        )
        R_ext   = 1.0 / max(h_ambi * Po * dz, 1.0e-30)
        return (T_ambi - Tw_arr) / np.maximum(R_cond + R_ext, 1.0e-30)

    return np.zeros(nn, dtype=float)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Balance molar
# ═══════════════════════════════════════════════════════════════════════════════

def molar_balance(heater, params: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    @brief
    Calcula el balance molar por especie y total del calentador.

    @details
    **Acumulación** (tubo vacío, epsi = 1):
        ΔN_gas_i = Ai · dz · Σ_k (C_i,k,final − C_i,k,inicial)  [mol]

    **Flujo neto en contornos** (integrado con regla trapezoidal):
        J_in_i(t)  = Ai · F_conv_i(t, cara 0)   [mol/s]
        J_out_i(t) = Ai · F_conv_i(t, cara N)   [mol/s]
        ΔN_flux_i  = ∫ (J_in_i − J_out_i) dt    [mol]

    Entrada siempre Dirichlet (flujo entrante), salida Neumann.
    Sin dispersión axial (Peclet de masa >> 1 en tubo vacío).

    **Error**:
        ε_abs = ΔN_gas − ΔN_flux   [mol]
        ε_rel = ε_abs / max(|ΔN_gas|, |ΔN_flux|, ε_min)   [−]

    Parameters
    ----------
    heater : SimpleNamespace devuelto por run_step.
             Requiere: _t_results, _C_results, _Tg_results, _P_results,
             _y_results, _v_in_results, _v_out_results, _C_in_results.
    params : dict del RHS. Requiere: n_comp, N, dz, Ai, Di, prop_gas,
             prop_update_mode (opcional).

    Returns
    -------
    dict con claves (todos ndarray shape (nc,), salvo "species"):
        "species"     — list[str] nombres de componentes
        "delta_gas"   — acumulación en fase gas por especie [mol]
        "delta_total" — igual a delta_gas (no hay fase adsorbida) [mol]
        "flux_in"     — moles que entraron por el contorno de entrada [mol]
        "flux_out"    — moles que salieron por el contorno de salida [mol]
        "net_flux"    — flujo neto entrante (flux_in − flux_out) [mol]
        "error_abs"   — delta_total − net_flux [mol]
        "error_rel"   — error relativo [−]
    """
    nc          = int(params["n_comp"])
    dz          = float(params["dz"])
    Ai          = float(params["Ai"])

    t         = np.asarray(heater._t_results,   dtype=float)   # (n_t,)
    C         = np.asarray(heater._C_results,   dtype=float)   # (n_t, nc, N)
    v_in      = np.asarray(heater._v_in_results, dtype=float)  # (n_t,)
    C_in_hist = np.asarray(heater._C_in_results, dtype=float)  # (n_t, nc)

    species = list(getattr(heater, "_species", None) or [f"comp_{i}" for i in range(nc)])

    n_t   = t.shape[0]
    J_in  = np.zeros((n_t, nc), dtype=float)
    J_out = np.zeros((n_t, nc), dtype=float)

    for k in range(n_t):
        C_k    = C[k]           # (nc, N)
        v_in_k = float(v_in[k])
        C_in_k = C_in_hist[k]  # (nc,)

        v_face_k = continuity_face_velocity(
            C_tot_cell=C_k.sum(axis=0),
            C_tot_in=float(C_in_k.sum()),
            v_in=v_in_k,
        )

        for i in range(nc):
            F_conv = convective_flux(
                phi_cell=C_k[i], v_face=v_face_k,
                phi_in=float(C_in_k[i]), phi_out=None,
            )
            # epsi_tube = 1 → sin factor de porosidad
            J_in[k, i]  = Ai * F_conv[0]
            J_out[k, i] = Ai * F_conv[-1]

    # Integración temporal
    flux_in  = np.trapz(J_in,  t, axis=0)   # (nc,) [mol]
    flux_out = np.trapz(J_out, t, axis=0)   # (nc,) [mol]
    net_flux = flux_in - flux_out

    # Acumulación (epsi_tube = 1)
    dV = Ai * dz
    delta_gas   = (C[-1] - C[0]).sum(axis=1) * dV   # (nc,) [mol]
    delta_total = delta_gas.copy()

    error_abs = delta_total - net_flux

    # Referencia para error relativo: máximo entre el inventario inicial de gas,
    # el flujo total que entró (continuo) y la acumulación (batch).
    # Evita división por cero cuando net_flux ≈ 0 (flujo continuo quasi-estacionario
    # donde J_in ≈ J_out → net_flux ≈ 0 aunque el throughput sea grande).
    N_inventory = C[0].sum(axis=1) * dV   # moles iniciales por especie (nc,)
    ref = np.maximum(flux_in, np.abs(net_flux))
    ref = np.maximum(ref, np.maximum(np.abs(delta_total), N_inventory))
    ref = np.maximum(ref, 1.0e-15)
    error_rel = error_abs / ref

    return {
        "species":     species,
        "delta_gas":   delta_gas,
        "delta_total": delta_total,
        "flux_in":     flux_in,
        "flux_out":    flux_out,
        "net_flux":    net_flux,
        "error_abs":   error_abs,
        "error_rel":   error_rel,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Balance de energía
# ═══════════════════════════════════════════════════════════════════════════════

def energy_balance(heater, params: Dict[str, Any]) -> Dict[str, float]:
    """
    @brief
    Calcula el balance de energía del gas en el calentador.

    @details
    Replica exactamente la metodología de core_rhs para cada instante temporal:
    reconstruye el campo de velocidades (Darcy-Weisbach), las propiedades de
    mezcla y todos los términos de flujo y fuente, integrando con la regla
    trapezoidal.

    **Balance del gas** (tubo vacío: sin sólido, sin adsorción):
        Ein − Eout + Qcond_g_net + Qwall = ΔE_gas

    donde:
      Ein, Eout     — entalpía convectiva en caras de contorno [J]
      Qcond_g_net   — conducción térmica neta del gas en contornos [J]
      Qwall         — calor total intercambiado con la pared/ambiente [J]
      ΔE_gas        — acumulación de energía en el gas [J]

    Parameters
    ----------
    heater : SimpleNamespace devuelto por run_step.
             Requiere: _t_results, _Hg_results, _Tg_results, _C_results,
             _P_results, _y_results, _v_in_results, _v_out_results,
             _C_in_results, _T_in_results.
    params : dict del RHS. Requiere: n_comp, N, dz, Ai, Di, Pi, Po,
             gas_T_ref, prop_gas, trans_config, thermal_bc_config.
             Opcionales: prop_update_mode, trans_update_mode.

    Returns
    -------
    dict con las siguientes claves (todos float, en Joules salvo _rel):
        "delta_gas_J"     — ΔE_gas (acumulación en gas) [J]
        "Ein_J"           — entalpía convectiva entrante en cara 0 [J]
        "Eout_J"          — entalpía convectiva saliente en cara N [J]
        "Qcond_g_net_J"   — conducción térmica neta del gas en contornos [J]
        "Qwall_J"         — calor total intercambiado con la pared [J]
        "err_gas_J"       — error del balance de gas [J]
        "err_gas_rel"     — error relativo de gas [−]

    Notes
    -----
    - Si params["energy"] = False la simulación fue isotermal; Hg es nulo
      y el balance de gas no tiene significado. Se devuelven todos los campos
      a cero con err_gas_rel = 0.
    """
    if not bool(params.get("energy", True)):
        return {k: 0.0 for k in (
            "delta_gas_J", "Ein_J", "Eout_J",
            "Qcond_g_net_J", "Qwall_J",
            "err_gas_J", "err_gas_rel",
            "delta_wall_J", "Qext_J",
            "err_sys_J", "err_sys_rel",
        )}

    nc          = int(params["n_comp"])
    nn          = int(params["N"])
    dz          = float(params["dz"])
    Ai          = float(params["Ai"])
    Di_v        = float(params["Di"])
    Pi          = float(params["Pi"])
    Po          = float(params["Po"])
    gas_T_ref   = float(params["gas_T_ref"])
    prop_gas    = params["prop_gas"]
    trans_config    = params["trans_config"]
    thermal_bc_cfg  = params["thermal_bc_config"]
    freeze_gas   = (str(params.get("prop_update_mode",  "always")).strip().lower() == "frozen")
    freeze_trans = (str(params.get("trans_update_mode", "always")).strip().lower() == "frozen")

    wall_config = params.get("wall_config")
    shell_tube  = wall_config is not None
    if shell_tube:
        Tw_all = np.asarray(heater._Tw_results, dtype=float)   # (n_t, N)
        mat    = wall_config["material"]
        A_w    = float(wall_config["A_w"])

    t         = np.asarray(heater._t_results,    dtype=float)   # (n_t,)
    Hg        = np.asarray(heater._Hg_results,   dtype=float)   # (n_t, N)
    Tg        = np.asarray(heater._Tg_results,   dtype=float)   # (n_t, N)
    C         = np.asarray(heater._C_results,    dtype=float)   # (n_t, nc, N)
    P         = np.asarray(heater._P_results,    dtype=float)   # (n_t, N) [bar]
    y3d       = np.asarray(heater._y_results,    dtype=float)   # (n_t, nc, N)
    v_in      = np.asarray(heater._v_in_results,  dtype=float)  # (n_t,)
    C_in_hist = np.asarray(heater._C_in_results,  dtype=float)  # (n_t, nc)
    T_in_hist = np.asarray(heater._T_in_results,  dtype=float)  # (n_t,)

    n_t = t.shape[0]

    # ── Acumulación de energía (epsi_tube = 1) ────────────────────────────────
    dV_col     = Ai * dz
    delta_gas  = float((Hg[-1] - Hg[0]).sum() * dV_col)

    # ── Propiedades de referencia para modo frozen ────────────────────────────
    _gas_ref     = None
    _h_wall_ref  = None

    if freeze_gas or freeze_trans:
        _gas_ref = compute_gas_mixture_properties(
            P_Pa=P[0] * 1.0e5, Tg=Tg[0], x=y3d[0].T,
            prop_gas=prop_gas, n_comp=nc, N=nn,
        )
        _vface0 = continuity_face_velocity(
            C_tot_cell=C[0].sum(axis=0),
            C_tot_in=float(C_in_hist[0].sum()),
            v_in=float(v_in[0]),
        )
        _vcell0 = 0.5 * (_vface0[:-1] + _vface0[1:])

    if freeze_trans and trans_config["mode"] == "correlation":
        _twall_mode0 = thermal_bc_cfg["mode"]
        _use_tfilm0  = trans_config["h_wall_fixed"] is None and (
            shell_tube or _twall_mode0 == "fixed_twall"
        )
        if _use_tfilm0:
            if shell_tube:
                Tw_film0 = np.asarray(heater._Tw_results, dtype=float)[0]
            else:
                Tw_film0 = np.full(nn, float(thermal_bc_cfg["T_wall"]), dtype=float)
            T_film0 = 0.5 * (Tg[0] + Tw_film0)
            _gas_film0 = compute_gas_mixture_properties(
                P_Pa=P[0] * 1.0e5, Tg=T_film0, x=y3d[0].T,
                prop_gas=prop_gas, n_comp=nc, N=nn,
            )
            Re0 = _gas_ref["rho"] * np.abs(_vcell0) * Di_v / np.maximum(_gas_film0["mu"], 1.0e-30)
            Pr0 = _gas_film0["Cp_mass"] * _gas_film0["mu"] / np.maximum(_gas_film0["k"], 1.0e-30)
            # Ra: válido para shell_tube y fixed_twall (ambos cubren _use_tfilm0)
            Ra0 = compute_Ra_Di(
                Tg=Tg[0], Tw=Tw_film0,
                rho_Tg=_gas_ref["rho"], mu_film=_gas_film0["mu"],
                k_film=_gas_film0["k"], Cp_film=_gas_film0["Cp_mass"],
                Di=Di_v,
            )
            _h_wall_ref = h_wall_tube(Re=Re0, Pr=Pr0, D=Di_v, k_film=_gas_film0["k"], Ra=Ra0)
        elif _twall_mode0 == "ambient_htc":
            Re0  = _gas_ref["rho"] * np.abs(_vcell0) * Di_v / np.maximum(_gas_ref["mu"], 1.0e-30)
            Pr0  = _gas_ref["Cp_mass"] * _gas_ref["mu"] / np.maximum(_gas_ref["k"], 1.0e-30)
            h0   = h_wall_tube(Re=Re0, Pr=Pr0, D=Di_v, k_film=_gas_ref["k"], Ra=None)
            Do0  = Po / np.pi
            Ri0  = 1.0 / np.maximum(h0 * Pi, 1.0e-30)
            Rc0  = np.log(Do0 / Di_v) / (2.0 * np.pi * float(thermal_bc_cfg["k_wall"]))
            Rx0  = 1.0 / (float(thermal_bc_cfg["h_ambi"]) * Po)
            Rt0  = Ri0 + Rc0 + Rx0
            Tw0_est = Tg[0] - (Tg[0] - float(thermal_bc_cfg["T_ambi"])) * Ri0 / np.maximum(Rt0, 1.0e-30)
            Ra0 = compute_Ra_Di(
                Tg=Tg[0], Tw=Tw0_est,
                rho_Tg=_gas_ref["rho"], mu_film=_gas_ref["mu"],
                k_film=_gas_ref["k"], Cp_film=_gas_ref["Cp_mass"],
                Di=Di_v,
            )
            _h_wall_ref = h_wall_tube(Re=Re0, Pr=Pr0, D=Di_v, k_film=_gas_ref["k"], Ra=Ra0)
        else:
            Re0 = _gas_ref["rho"] * np.abs(_vcell0) * Di_v / np.maximum(_gas_ref["mu"], 1.0e-30)
            Pr0 = _gas_ref["Cp_mass"] * _gas_ref["mu"] / np.maximum(_gas_ref["k"], 1.0e-30)
            _h_wall_ref = h_wall_tube(Re=Re0, Pr=Pr0, D=Di_v, k_film=_gas_ref["k"])
        if trans_config["h_wall_fixed"] is not None:
            _h_wall_ref = trans_config["h_wall_fixed"] * np.ones(nn, dtype=float)

    # ── Arrays de potencias instantáneas [W] ─────────────────────────────────
    Ein_dot         = np.zeros(n_t, dtype=float)
    Eout_dot        = np.zeros(n_t, dtype=float)
    Qcond_g_net_dot = np.zeros(n_t, dtype=float)
    Qwall_dot       = np.zeros(n_t, dtype=float)
    Qext_dot        = np.zeros(n_t, dtype=float)   # potencia exterior sobre pared

    for k in range(n_t):
        Tg_k    = Tg[k]              # (N,)
        C_k     = C[k]              # (nc, N)
        v_in_k  = float(v_in[k])
        C_in_k  = C_in_hist[k]     # (nc,)
        T_in_k  = float(T_in_hist[k])

        # Propiedades de mezcla
        if freeze_gas:
            gas = _gas_ref
        else:
            gas = compute_gas_mixture_properties(
                P_Pa=P[k] * 1.0e5, Tg=Tg_k, x=y3d[k].T,
                prop_gas=prop_gas, n_comp=nc, N=nn,
            )

        # Velocidades — continuidad: v·C_tot = cte desde inlet
        v_face_k = continuity_face_velocity(
            C_tot_cell=C_k.sum(axis=0),
            C_tot_in=float(C_in_k.sum()),
            v_in=v_in_k,
        )
        v_cell_k = 0.5 * (v_face_k[:-1] + v_face_k[1:])

        # Coeficiente h_wall (replica la lógica de rhs_heater paso 5)
        if trans_config["mode"] == "constant":
            h_wall_k = trans_config["h_wall"]
        elif freeze_trans:
            h_wall_k = _h_wall_ref
        else:
            _twall_mode_k = thermal_bc_cfg["mode"]
            _use_tfilm_k  = trans_config["h_wall_fixed"] is None and (
                shell_tube or _twall_mode_k == "fixed_twall"
            )
            if _use_tfilm_k:
                if shell_tube:
                    Tw_film_k = Tw_all[k]
                else:
                    Tw_film_k = np.full(nn, float(thermal_bc_cfg["T_wall"]), dtype=float)
                T_film_k = 0.5 * (Tg_k + Tw_film_k)
                gas_film_k = compute_gas_mixture_properties(
                    P_Pa=P[k] * 1.0e5, Tg=T_film_k, x=y3d[k].T,
                    prop_gas=prop_gas, n_comp=nc, N=nn,
                )
                mu_Nu_k = gas_film_k["mu"]
                k_Nu_k  = gas_film_k["k"]
                Cp_Nu_k = gas_film_k["Cp_mass"]
            else:
                mu_Nu_k = gas["mu"]
                k_Nu_k  = gas["k"]
                Cp_Nu_k = gas["Cp_mass"]
            Re_k = gas["rho"] * np.abs(v_cell_k) * Di_v / np.maximum(mu_Nu_k, 1.0e-30)
            Pr_k = Cp_Nu_k * mu_Nu_k / np.maximum(k_Nu_k, 1.0e-30)
            if _use_tfilm_k:
                # shell_tube o fixed_twall: Tw conocida, propiedades a T_film
                Ra_k = compute_Ra_Di(
                    Tg=Tg_k, Tw=Tw_film_k,
                    rho_Tg=gas["rho"], mu_film=mu_Nu_k,
                    k_film=k_Nu_k, Cp_film=Cp_Nu_k,
                    Di=Di_v,
                )
            elif _twall_mode_k == "ambient_htc" and trans_config["h_wall_fixed"] is None:
                h_wall_0_k = h_wall_tube(Re=Re_k, Pr=Pr_k, D=Di_v, k_film=k_Nu_k, Ra=None)
                Do_bal     = Po / np.pi
                R_int_k    = 1.0 / np.maximum(h_wall_0_k * Pi, 1.0e-30)
                R_cond_k   = np.log(Do_bal / Di_v) / (2.0 * np.pi * float(thermal_bc_cfg["k_wall"]))
                R_ext_k    = 1.0 / (float(thermal_bc_cfg["h_ambi"]) * Po)
                R_tot_k    = R_int_k + R_cond_k + R_ext_k
                Tw_int_k   = Tg_k - (Tg_k - float(thermal_bc_cfg["T_ambi"])) * R_int_k / np.maximum(R_tot_k, 1.0e-30)
                Ra_k = compute_Ra_Di(
                    Tg=Tg_k, Tw=Tw_int_k,
                    rho_Tg=gas["rho"], mu_film=mu_Nu_k,
                    k_film=k_Nu_k, Cp_film=Cp_Nu_k,
                    Di=Di_v,
                )
            else:
                Ra_k = None
            h_wall_k = h_wall_tube(Re=Re_k, Pr=Pr_k, D=Di_v, k_film=k_Nu_k, Ra=Ra_k)
            if trans_config["h_wall_fixed"] is not None:
                h_wall_k = trans_config["h_wall_fixed"] * np.ones(nn, dtype=float)

        # 1. Flujo convectivo de entalpía → Ein, Eout [W]
        #    epsi_tube = 1 → sin factor de porosidad en contorno
        Fh_conv_k = gas_enthalpy_convective_flux(
            Tg_cell=Tg_k, C_cell=C_k, v_face=v_face_k,
            prop_gas=prop_gas, n_comp=nc, gas_T_ref=gas_T_ref,
            T_in=T_in_k,
            C_in=C_in_k,
        )
        Ein_dot[k]  = Ai * Fh_conv_k[0]
        Eout_dot[k] = Ai * Fh_conv_k[-1]

        # 2. Conducción térmica del gas en contornos [W]
        #    La suma telescópica de div sobre todas las celdas = Ai*(qg[0] - qg[N])
        qg_diff_k = gas_diffusive_heat_flux(
            Tg_cell=Tg_k, k_g=gas["k"], dz=dz,
            T_in=T_in_k, T_out=None,
            bc_in="dirichlet", bc_out="neumann",
            face_method="arithmetic",
        )
        Qcond_g_net_dot[k] = Ai * (qg_diff_k[0] - qg_diff_k[-1])

        # 3. Calor de pared [W]
        if shell_tube:
            Tw_k = Tw_all[k]
            Qwall_dot[k] = float(np.sum(h_wall_k * Pi * dz * (Tw_k - Tg_k)))
            k_w_k = eval_solid_property(mat["k"], Tw_k)
            Qext_dot[k] = float(np.sum(_ext_q_dot(Tw_k, thermal_bc_cfg, k_w_k, Pi, Po, dz, nn)))
        else:
            _, Qwall_k, _ = wall_heat_flux(
                Tg=Tg_k, h_wall=h_wall_k,
                thermal_bc_config=thermal_bc_cfg,
                N=nn, Ai=Ai, Pi=Pi, Po=Po, dz=dz,
            )
            Qwall_dot[k] = Qwall_k

    # ── Integración temporal (regla trapezoidal) ──────────────────────────────
    Ein         = float(np.trapz(Ein_dot,         t))
    Eout        = float(np.trapz(Eout_dot,        t))
    Qcond_g_net = float(np.trapz(Qcond_g_net_dot, t))
    Qwall       = float(np.trapz(Qwall_dot,       t))
    Qext        = float(np.trapz(Qext_dot,        t)) if shell_tube else 0.0

    # ── ΔE_wall (solo shell_tube) ─────────────────────────────────────────────
    delta_wall = 0.0
    if shell_tube:
        dV_wall  = A_w * dz
        Tw_init  = Tw_all[0]
        Tw_final = Tw_all[-1]
        Tw_mean  = 0.5 * (Tw_init + Tw_final)
        rho_w    = eval_solid_property(mat["rho"], Tw_mean)
        cp_w     = eval_solid_property(mat["cp"],  Tw_mean)
        delta_wall = float(np.sum(rho_w * cp_w * dV_wall * (Tw_final - Tw_init)))

    # ── Error de balance del gas ──────────────────────────────────────────────
    predicted_gas = Ein - Eout + Qcond_g_net + Qwall
    err_gas       = predicted_gas - delta_gas
    ref_gas       = max(abs(Ein), abs(Eout), abs(delta_gas), 1.0e-6)

    # ── Error de balance del sistema gas+pared ────────────────────────────────
    if shell_tube:
        delta_sys     = delta_gas + delta_wall
        predicted_sys = Ein - Eout + Qcond_g_net + Qext
        err_sys       = predicted_sys - delta_sys
        ref_sys       = max(abs(delta_sys), abs(predicted_sys), 1.0e-6)
    else:
        delta_sys, err_sys, ref_sys = delta_gas, err_gas, ref_gas

    return {
        "delta_gas_J":    delta_gas,
        "Ein_J":          Ein,
        "Eout_J":         Eout,
        "Qcond_g_net_J":  Qcond_g_net,
        "Qwall_J":        Qwall,
        "err_gas_J":      err_gas,
        "err_gas_rel":    err_gas / ref_gas,
        "delta_wall_J":   delta_wall,
        "Qext_J":         Qext,
        "err_sys_J":      err_sys,
        "err_sys_rel":    err_sys / ref_sys,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Informe de balance completo
# ═══════════════════════════════════════════════════════════════════════════════

def balance_report(
    heater,
    params: Dict[str, Any],
    include_energy: bool = True,
    print_report: bool = True,
) -> pd.DataFrame:
    """
    @brief
    Genera un informe tabular de los balances de masa y energía del calentador.

    @details
    Calcula y, opcionalmente imprime, los balances molar (por especie y total)
    y de energía (con desglose de todos los términos). Devuelve un DataFrame
    con todas las filas del informe.

    Parameters
    ----------
    heater         : SimpleNamespace devuelto por run_step.
    params         : dict del RHS.
    include_energy : bool — incluir balance de energía. Default True.
    print_report   : bool — imprimir el informe por pantalla. Default True.

    Returns
    -------
    pd.DataFrame
        Columnas: "Acumulación", "Flujo neto BC", "Error abs", "Error rel [%]".
        Filas: una por especie, fila total molar y (si include_energy=True)
        fila del balance de energía del gas. Unidades: [mol] y [kJ].
    """
    mb      = molar_balance(heater, params)
    species = mb["species"]

    rows = []

    # ── Balance molar por especie ─────────────────────────────────────────────
    for i, sp in enumerate(species):
        rows.append({
            "Variable":      f"N_{sp} [mol]",
            "Acumulación":   mb["delta_total"][i],
            "Flujo neto BC": mb["net_flux"][i],
            "Error abs":     mb["error_abs"][i],
            "Error rel [%]": mb["error_rel"][i] * 100.0,
        })

    # ── Balance molar total ───────────────────────────────────────────────────
    net_tot     = mb["net_flux"].sum()
    acc_tot     = mb["delta_total"].sum()
    err_tot_abs = mb["error_abs"].sum()
    ref_tot     = max(mb["flux_in"].sum(), abs(net_tot),
                      abs(acc_tot), 1.0e-15)
    rows.append({
        "Variable":      "N_total [mol]",
        "Acumulación":   acc_tot,
        "Flujo neto BC": net_tot,
        "Error abs":     err_tot_abs,
        "Error rel [%]": err_tot_abs / ref_tot * 100.0,
    })

    # ── Balance de energía ────────────────────────────────────────────────────
    shell_tube = params.get("wall_config") is not None
    eb = None
    if include_energy:
        try:
            eb  = energy_balance(heater, params)
            kJ  = 1.0e-3
            predicted_gas_kJ = (eb["Ein_J"] - eb["Eout_J"]
                                + eb["Qcond_g_net_J"] + eb["Qwall_J"]) * kJ
            rows.append({
                "Variable":      "E_gas [kJ]",
                "Acumulación":   eb["delta_gas_J"] * kJ,
                "Flujo neto BC": predicted_gas_kJ,
                "Error abs":     eb["err_gas_J"] * kJ,
                "Error rel [%]": eb["err_gas_rel"] * 100.0,
            })
            if shell_tube:
                _wall_rhs_kJ = (eb["Qext_J"] - eb["Qwall_J"]) * kJ
                _wall_err_kJ = eb["delta_wall_J"] * kJ - _wall_rhs_kJ
                _wall_ref    = max(abs(eb["delta_wall_J"]), abs(eb["Qext_J"] - eb["Qwall_J"]), 1.0e-6)
                rows.append({
                    "Variable":      "E_wall [kJ]",
                    "Acumulación":   eb["delta_wall_J"] * kJ,
                    "Flujo neto BC": _wall_rhs_kJ,
                    "Error abs":     _wall_err_kJ,
                    "Error rel [%]": _wall_err_kJ * 1.0e3 / _wall_ref * 100.0,
                })
                predicted_sys_kJ = (eb["Ein_J"] - eb["Eout_J"]
                                    + eb["Qcond_g_net_J"] + eb["Qext_J"]) * kJ
                rows.append({
                    "Variable":      "E_sys [kJ]",
                    "Acumulación":   (eb["delta_gas_J"] + eb["delta_wall_J"]) * kJ,
                    "Flujo neto BC": predicted_sys_kJ,
                    "Error abs":     eb["err_sys_J"] * kJ,
                    "Error rel [%]": eb["err_sys_rel"] * 100.0,
                })
        except Exception as exc:
            rows.append({
                "Variable":      "E_gas [kJ]",
                "Acumulación":   float("nan"),
                "Flujo neto BC": float("nan"),
                "Error abs":     float("nan"),
                "Error rel [%]": float("nan"),
            })
            if print_report:
                print(f"  [balance_report] Balance de energía no disponible: {exc}")

    df = pd.DataFrame(rows).set_index("Variable")

    if print_report:
        sep = "─" * 82
        print(sep)
        print("  INFORME DE BALANCE DE CONSERVACIÓN — HEATER")
        print(sep)

        pd.set_option("display.float_format", "{:+.5g}".format)
        print(df.to_string())
        print(sep)

        # Semáforo de calidad
        max_err = df["Error rel [%]"].abs().max()
        if max_err < 0.1:
            label = "EXCELENTE  (<0.1 %)"
        elif max_err < 1.0:
            label = "BUENO      (<1 %)"
        elif max_err < 5.0:
            label = "ACEPTABLE  (<5 %)"
        else:
            label = "REVISAR    (>5 %)"
        print(f"  Error máximo: {max_err:.3f} %  →  {label}")

        # Desglose detallado de la energía
        if include_energy and eb is not None:
            kJ = 1.0e-3
            print()
            print("  Desglose de términos energéticos [kJ]:")
            print(f"    Ein             = {eb['Ein_J']*kJ:+.4g}")
            print(f"    Eout            = {eb['Eout_J']*kJ:+.4g}")
            print(f"    Qcond_g_net     = {eb['Qcond_g_net_J']*kJ:+.4g}"
                  "  (conducción gas en contornos)")
            print(f"    Qwall           = {eb['Qwall_J']*kJ:+.4g}"
                  "  (calor gas↔pared)")
            print(f"    ΔE_gas          = {eb['delta_gas_J']*kJ:+.4g}")
            if shell_tube:
                print(f"    Qext            = {eb['Qext_J']*kJ:+.4g}"
                      "  (calor exterior→pared)")
                print(f"    ΔE_wall         = {eb['delta_wall_J']*kJ:+.4g}")
                print(f"    ΔE_sys          = {(eb['delta_gas_J']+eb['delta_wall_J'])*kJ:+.4g}"
                      "  (gas + pared)")

        print(sep)

    return df
