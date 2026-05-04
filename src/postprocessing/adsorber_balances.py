"""
@module postprocessing.balances
@brief Comprobación de balances de conservación para simulaciones de columna PSA 1D.

@details
Calcula los balances de conservación comparando la acumulación en la columna
con los flujos integrados en los contornos y los términos fuente internos.
Los resultados son diagnósticos: un balance cerrado indica que la integración
numérica es coherente con las ecuaciones físicas implementadas.

Balances disponibles:

1. **Balance molar por especie y total**:
   Acumulación (gas + adsorbida) vs. flujo neto neto en caras de contorno.
   El flujo en la cara incluye convección (upwind) y dispersión axial (Fick),
   exactamente igual que en el núcleo del RHS. Requiere recalcular los
   coeficientes de dispersión en cada instante temporal.

2. **Balance de energía** (tres sub-balances):

   Gas:
       Ein − Eout + Qcond_g_net + Qwall − Qgs − Eads_sens_gas = ΔE_gas

   Sólido (con Neumann en ambos extremos, sin flujo de conducción exterior):
       Qgs + Eads = ΔE_solid

   Total (Qgs cancela al sumar gas+sólido):
       Ein − Eout + Qcond_g_net + Qwall − Eads_sens_gas + Eads = ΔE_total

   donde:
     Ein, Eout        — entalpía convectiva en caras de contorno [J]
     Qcond_g_net      — conducción térmica neta del gas en contornos [J]
     Qwall            — calor intercambiado con la pared/ambiente [J]
     Qgs              — calor transferido del gas al sólido [J]
     Eads             — calor de adsorción liberado al sólido [J]
     Eads_sens_gas    — entalpía sensible retirada del gas por adsorción [J]

   Todos los términos se integran en el tiempo con la regla trapezoidal.
   dq/dt se obtiene de la cinética LDF exacta (k_mtc * (q_eq − q)),
   re-evaluando la isoterma en cada instante almacenado.

El objeto `col` debe ser el devuelto por `run_step` (versión que incluye los
atributos `_v_in_results`, `_v_out_results`, `_C_in_results`, `_T_in_results`
y `_Hg_results`).

Parámetros requeridos en `params`:
    n_comp, N, dz, epsi, rho_s, Cp_s, Ai, Di, Pi, Po
    prop_gas, MW, gas_T_ref
    prop_lecho (D_p, a_surf)
    trans_config, thermal_bc_config
    dH    — entalpía de adsorción (nc, N) [J/mol]
    step  — nombre del paso PSA activo

Unidades: SI.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

R_GAS = 8.31446261815324   # [J/mol/K]

from src.physics.mixture_gas import compute_gas_mixture_properties
from src.physics.momentum.ergun import ergun_face_velocity
from src.physics.transport.transfer_coefficients import compute_transfer_coefficients
from src.physics.thermal.wall_heat_flux import wall_heat_flux
from src.physics.thermal.wall_ode import wall_exterior_q
from src.physics.thermodynamics.solid_props import eval_solid_property
from src.discretization.fluxes import (
    convective_flux,
    diffusive_flux,
    gas_enthalpy_convective_flux,
    gas_diffusive_heat_flux,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper interno
# ═══════════════════════════════════════════════════════════════════════════════

def _step_bc_type(step: str) -> str:
    """
    Devuelve el tipo de condición de contorno en la cara de entrada para
    especies y temperatura, siguiendo la misma lógica que core_rhs.

    Steps con Dirichlet en inlet (flujo entrante positivo):
        "ads", "purge", "pr_feed"
    Resto (blowdown, wait, etc.): Neumann (flujo difusivo nulo en contorno).
    """
    return "dirichlet" if step in ("ads", "purge", "pr_feed") else "neumann"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Balance molar
# ═══════════════════════════════════════════════════════════════════════════════

def molar_balance(col, params: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    @brief
    Calcula el balance molar por especie y total.

    @details
    **Acumulación**:
        ΔN_gas_i   = ε · Ai · dz · Σ_k (C_i,k,final − C_i,k,inicial)  [mol]
        ΔN_ads_i   = (1−ε) · ρs · Ai · dz · Σ_k (q_i,k,final − q_i,k,inicial)  [mol]
        ΔN_total_i = ΔN_gas_i + ΔN_ads_i  [mol]

    **Flujo neto en contornos** (integrado con regla trapezoidal):
        J_in_i(t)  = ε · Ai · [F_conv_i(t, cara 0) + F_diff_i(t, cara 0)]  [mol/s]
        J_out_i(t) = ε · Ai · [F_conv_i(t, cara N) + F_diff_i(t, cara N)]  [mol/s]
        ΔN_flux_i  = ∫ (J_in_i − J_out_i) dt   [mol]

    El flujo en la cara incluye el término difusivo calculado con los
    coeficientes de dispersión del paso activo (igual que en core_rhs).
    En la cara de salida, la BC es Neumann → F_diff = 0.

    **Error**:
        ε_abs = ΔN_total − ΔN_flux   [mol]
        ε_rel = ε_abs / max(|ΔN_total|, |ΔN_flux|, ε_min)   [−]

    Parameters
    ----------
    col    : SimpleNamespace devuelto por run_step.
             Requiere: _t_results, _C_results, _q_results, _Tg_results,
             _Ts_results, _P_results, _y_results, _v_in_results,
             _v_out_results, _C_in_results.
    params : dict del RHS. Requiere: n_comp, N, dz, epsi, rho_s, Ai, Di,
             prop_gas, prop_lecho, trans_config, step.

    Returns
    -------
    dict con claves (todos ndarray shape (nc,), salvo "species"):
        "species"     — list[str] nombres de componentes
        "delta_gas"   — acumulación en fase gas por especie [mol]
        "delta_ads"   — acumulación en fase adsorbida [mol]
        "delta_total" — acumulación total [mol]
        "flux_in"     — moles que entraron por el contorno de entrada [mol]
        "flux_out"    — moles que salieron por el contorno de salida [mol]
        "net_flux"    — flujo neto entrante (flux_in − flux_out) [mol]
        "error_abs"   — delta_total − net_flux [mol]
        "error_rel"   — error relativo [−]
    """
    nc        = int(params["n_comp"])
    nn        = int(params["N"])
    dz        = float(params["dz"])
    epsi      = float(params["epsi"])
    rho_s     = float(params["rho_s"])
    Ai        = float(params["Ai"])
    Di        = float(params["Di"])
    prop_gas  = params["prop_gas"]
    prop_lecho = params["prop_lecho"]
    trans_config = params["trans_config"]
    step      = str(params["step"]).strip().lower()
    # Modos de actualización: deben coincidir con los del RHS para que el
    # balance sea consistente con la trayectoria realmente integrada.
    freeze_gas   = (str(params.get("prop_update_mode",  "always")).strip().lower() == "frozen")
    freeze_trans = (str(params.get("trans_update_mode", "always")).strip().lower() == "frozen")

    dp     = float(np.mean(np.asarray(prop_lecho["D_p"], dtype=float).reshape(-1)))
    bc_in  = _step_bc_type(step)

    t         = np.asarray(col._t_results,      dtype=float)   # (n_t,)
    C         = np.asarray(col._C_results,      dtype=float)   # (n_t, nc, N)
    q         = np.asarray(col._q_results,      dtype=float)   # (n_t, nc, N)
    Tg        = np.asarray(col._Tg_results,     dtype=float)   # (n_t, N)
    Ts        = np.asarray(col._Ts_results,     dtype=float)   # (n_t, N)
    P         = np.asarray(col._P_results,      dtype=float)   # (n_t, N) [bar]
    y3d       = np.asarray(col._y_results,      dtype=float)   # (n_t, nc, N)
    v_in      = np.asarray(col._v_in_results,   dtype=float)   # (n_t,)
    v_out     = np.asarray(col._v_out_results,  dtype=float)   # (n_t,)
    C_in_hist = np.asarray(col._C_in_results,   dtype=float)   # (n_t, nc)

    species = list(getattr(col, "_species", None) or [f"comp_{i}" for i in range(nc)])

    n_t   = t.shape[0]
    J_in  = np.zeros((n_t, nc), dtype=float)   # [mol/s] flujo entrante por especie
    J_out = np.zeros((n_t, nc), dtype=float)   # [mol/s] flujo saliente por especie

    # ── Propiedades de referencia (modo frozen: se calculan en k=0 y se fijan) ─
    if freeze_gas or freeze_trans:
        _gas_ref = compute_gas_mixture_properties(
            P_Pa=P[0] * 1.0e5, Tg=Tg[0], x=y3d[0].T,
            prop_gas=prop_gas, n_comp=nc, N=nn,
        )
        _vface_ref = ergun_face_velocity(
            P=P[0] * 1.0e5, rho_g=_gas_ref["rho"], mu_g=_gas_ref["mu"],
            epsi=epsi, dp=dp, dz=dz,
            v_in=float(v_in[0]), v_out=float(v_out[0]),
        )
        _vcell_ref = 0.5 * (_vface_ref[:-1] + _vface_ref[1:])
    if freeze_trans:
        _trans_ref = compute_transfer_coefficients(
            Tg=Tg[0], Ts=Ts[0], x=y3d[0].T,
            gas_props=_gas_ref, u_rel=np.abs(_vcell_ref),
            prop_gas=prop_gas, prop_lecho=prop_lecho,
            Di=Di, trans_config=trans_config,
            n_comp=nc, N=nn,
        )

    for k in range(n_t):
        Tg_k     = Tg[k]                          # (N,)
        x_k      = y3d[k].T                       # (N, nc)
        C_k      = C[k]                           # (nc, N)
        v_in_k   = float(v_in[k])
        v_out_k  = float(v_out[k])
        C_in_k   = C_in_hist[k]                  # (nc,)

        # Propiedades de mezcla — frozen usa el estado inicial
        if freeze_gas:
            gas = _gas_ref
        else:
            gas = compute_gas_mixture_properties(
                P_Pa=P[k] * 1.0e5, Tg=Tg_k, x=x_k,
                prop_gas=prop_gas, n_comp=nc, N=nn,
            )

        # Campo de velocidades en caras: rho/mu frozen o actuales; P siempre actual
        v_face_k = ergun_face_velocity(
            P=P[k] * 1.0e5, rho_g=gas["rho"], mu_g=gas["mu"],
            epsi=epsi, dp=dp, dz=dz,
            v_in=v_in_k, v_out=v_out_k,
        )
        v_cell_k = 0.5 * (v_face_k[:-1] + v_face_k[1:])

        # Coeficientes de transporte — frozen usa los del estado inicial
        if freeze_trans:
            trans = _trans_ref
        else:
            trans = compute_transfer_coefficients(
                Tg=Tg_k, Ts=Ts[k], x=x_k,
                gas_props=gas, u_rel=np.abs(v_cell_k),
                prop_gas=prop_gas, prop_lecho=prop_lecho,
                Di=Di, trans_config=trans_config,
                n_comp=nc, N=nn,
            )
        D_disp_k = trans["D_disp"]   # (nc, N)

        for i in range(nc):
            # Valor de contorno en la cara de entrada
            C_in_i = float(C_in_k[i]) if bc_in == "dirichlet" else None

            F_conv = convective_flux(
                phi_cell=C_k[i], v_face=v_face_k,
                phi_in=C_in_i, phi_out=None,
            )
            F_diff = diffusive_flux(
                phi_cell=C_k[i], Gamma=D_disp_k[i], dz=dz,
                phi_in=C_in_i, phi_out=None,
                bc_in=bc_in, bc_out="neumann",
            )
            F_tot = F_conv + F_diff   # (N+1,) [mol/m²/s]

            # Flujo molar en cada contorno [mol/s] = ε·Ai·F_tot en la cara
            J_in[k, i]  = epsi * Ai * F_tot[0]
            J_out[k, i] = epsi * Ai * F_tot[-1]

    # ── Integración temporal ──────────────────────────────────────────────────
    flux_in  = np.trapz(J_in,  t, axis=0)   # (nc,) [mol]
    flux_out = np.trapz(J_out, t, axis=0)   # (nc,) [mol]
    net_flux = flux_in - flux_out            # (nc,) [mol]

    # ── Acumulación ───────────────────────────────────────────────────────────
    # C[-1], C[0]: shape (nc, N). .sum(axis=1) suma sobre N nodos → (nc,)
    dV_gas = epsi * Ai * dz          # volumen de vacío por celda [m³]
    dV_sol = (1.0 - epsi) * Ai * dz  # volumen de sólido por celda [m³]

    delta_gas   = (C[-1] - C[0]).sum(axis=1) * dV_gas           # (nc,) [mol]
    delta_ads   = (q[-1] - q[0]).sum(axis=1) * rho_s * dV_sol   # (nc,) [mol]
    delta_total = delta_gas + delta_ads                          # (nc,) [mol]

    # ── Error de balance ──────────────────────────────────────────────────────
    error_abs = delta_total - net_flux

    # Referencia para error relativo: máximo entre inventario inicial (gas+ads),
    # throughput total (flux_in) y la acumulación.
    # Evita división por cero cuando net_flux ≈ 0 (p. ej. pasos de ecualización
    # donde el flujo neto entrante es pequeño aunque el throughput sea finito).
    N_inv_gas = C[0].sum(axis=1) * dV_gas
    N_inv_ads = q[0].sum(axis=1) * rho_s * dV_sol
    N_inventory = N_inv_gas + N_inv_ads
    ref = np.maximum(flux_in, np.abs(net_flux))
    ref = np.maximum(ref, np.maximum(np.abs(delta_total), N_inventory))
    ref = np.maximum(ref, 1.0e-15)
    error_rel = error_abs / ref

    return {
        "species":     species,
        "delta_gas":   delta_gas,
        "delta_ads":   delta_ads,
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

def energy_balance(col, params: Dict[str, Any]) -> Dict[str, float]:
    """
    @brief
    Calcula el balance de energía completo de la columna (gas + sólido).

    @details
    Replica exactamente la metodología del núcleo RHS para cada instante temporal:
    reconstruye el campo de velocidades (Ergun), las propiedades de mezcla
    (Wilke), y todos los términos de flujo y fuente, integrando con la regla
    trapezoidal.

    **Balance del gas**:
        Ein − Eout + Qcond_g_net + Qwall − Qgs − Eads_sens_gas = ΔE_gas

    **Balance del sólido** (Neumann en contornos → conducción axial interna):
        Qgs + Eads = ΔE_solid

    **Balance total** (Qgs se cancela):
        Ein − Eout + Qcond_g_net + Qwall − Eads_sens_gas + Eads = ΔE_total

    Los términos que involucran dq/dt (Eads, Eads_sens_gas) se calculan
    usando la cinética LDF exacta: dqdt = k_mtc * (q_eq − q), re-evaluando
    la isoterma en cada instante almacenado (igual que core_rhs).

    Parameters
    ----------
    col    : SimpleNamespace devuelto por run_step.
             Requiere: _t_results, _Hg_results, _Ts_results, _Tg_results,
             _C_results, _q_results, _P_results, _y_results, _v_in_results,
             _v_out_results, _C_in_results, _T_in_results.
    params : dict del RHS. Requiere: n_comp, N, dz, epsi, rho_s, Cp_s,
             Ai, Di, Pi, Po, gas_T_ref, prop_gas, prop_lecho, trans_config,
             thermal_bc_config, dH, iso_fn, step.

    Returns
    -------
    dict con las siguientes claves (todos float, en Joules salvo las _rel):
        "delta_gas_J"       — ΔE_gas (acumulación en gas)
        "delta_solid_J"     — ΔE_solid (acumulación en sólido)
        "delta_total_J"     — ΔE_total = ΔE_gas + ΔE_solid
        "Ein_J"             — entalpía convectiva entrante en cara 0
        "Eout_J"            — entalpía convectiva saliente en cara N
        "Qcond_g_net_J"     — conducción térmica neta del gas en contornos
        "Qwall_J"           — calor total intercambiado con la pared
        "Qgs_J"             — calor transferido del gas al sólido
        "Eads_J"            — calor de adsorción liberado al sólido
        "Eads_sens_gas_J"   — entalpía sensible retirada del gas por adsorción
        "err_gas_J"         — error del balance de gas [J]
        "err_solid_J"       — error del balance de sólido [J]
        "err_total_J"       — error del balance total [J]
        "err_gas_rel"       — error relativo de gas [−]
        "err_solid_rel"     — error relativo de sólido [−]
        "err_total_rel"     — error relativo total [−]

    Notes
    -----
    - Si params["energy"] = False la simulación fue isotermal; Hg es cero
      y el balance de gas no tiene significado. Se devuelven todos los campos
      a cero con error_rel = 0.
    - La conducción axial del sólido (ks_diff) tiene BC Neumann en ambos
      extremos, por lo que no aporta flujo neto al balance total.
    """
    wall_config = params.get("wall_config")
    shell_tube  = wall_config is not None

    if not bool(params.get("energy", True)):
        _zero = {k: 0.0 for k in (
            "delta_gas_J", "delta_solid_J", "delta_total_J",
            "Ein_J", "Eout_J", "Qcond_g_net_J", "Qwall_J",
            "Qgs_J", "Eads_J", "Eads_sens_gas_J",
            "err_gas_J", "err_solid_J", "err_total_J",
            "err_gas_rel", "err_solid_rel", "err_total_rel",
        )}
        if shell_tube:
            _zero.update({
                "delta_wall_J": 0.0, "Qext_J": 0.0,
                "delta_sys_J": 0.0, "err_sys_J": 0.0, "err_sys_rel": 0.0,
            })
        return _zero

    nc          = int(params["n_comp"])
    nn          = int(params["N"])
    dz          = float(params["dz"])
    epsi        = float(params["epsi"])
    rho_s       = float(params["rho_s"])
    Cp_s        = float(params["Cp_s"])
    Ai          = float(params["Ai"])
    Di          = float(params["Di"])
    Pi          = float(params["Pi"])
    Po          = float(params["Po"])
    gas_T_ref   = float(params["gas_T_ref"])
    prop_gas    = params["prop_gas"]
    prop_lecho  = params["prop_lecho"]
    trans_config    = params["trans_config"]
    thermal_bc_cfg  = params["thermal_bc_config"]
    dH_mat      = np.asarray(params["dH"], dtype=float)   # (nc, N) [J/mol]
    iso_fn      = params["iso_fn"]                         # callable: isoterma multicomponente
    step        = str(params["step"]).strip().lower()
    # Modos de actualización: deben coincidir con los del RHS para coherencia
    # con la trayectoria integrada. En modo frozen, el RHS fija gas_props y
    # trans_props en el primer instante y los usa sin actualizar durante todo
    # el paso — el balance debe replicar exactamente eso.
    freeze_gas   = (str(params.get("prop_update_mode",  "always")).strip().lower() == "frozen")
    freeze_trans = (str(params.get("trans_update_mode", "always")).strip().lower() == "frozen")

    dp     = float(np.mean(np.asarray(prop_lecho["D_p"], dtype=float).reshape(-1)))
    a_surf = np.asarray(prop_lecho["a_surf"], dtype=float).reshape(-1)   # (N,)
    bc_in_T = _step_bc_type(step)

    t         = np.asarray(col._t_results,      dtype=float)   # (n_t,)
    Hg        = np.asarray(col._Hg_results,     dtype=float)   # (n_t, N) [J/m³ col]
    Ts        = np.asarray(col._Ts_results,     dtype=float)   # (n_t, N) [K]
    Tg        = np.asarray(col._Tg_results,     dtype=float)   # (n_t, N) [K]
    C         = np.asarray(col._C_results,      dtype=float)   # (n_t, nc, N)
    q         = np.asarray(col._q_results,      dtype=float)   # (n_t, nc, N)
    P         = np.asarray(col._P_results,      dtype=float)   # (n_t, N) [bar]
    y3d       = np.asarray(col._y_results,      dtype=float)   # (n_t, nc, N)
    v_in      = np.asarray(col._v_in_results,   dtype=float)   # (n_t,)
    v_out     = np.asarray(col._v_out_results,  dtype=float)   # (n_t,)
    C_in_hist = np.asarray(col._C_in_results,   dtype=float)   # (n_t, nc)
    T_in_hist = np.asarray(col._T_in_results,   dtype=float)   # (n_t,) NaN si sin entrada

    n_t = t.shape[0]

    if shell_tube:
        Tw_hist = np.asarray(col._Tw_results, dtype=float)   # (n_t, N)

    # ── Acumulación de energía ────────────────────────────────────────────────
    # Hg [J/m³ de columna total] → integral sobre volumen de cada celda
    dV_col = Ai * dz   # m³ por celda (total, sin descontar porosidad)

    delta_gas   = float((Hg[-1]  - Hg[0]).sum()  * dV_col)
    delta_solid = float((Ts[-1] - Ts[0]).sum() * (1.0 - epsi) * rho_s * Cp_s * dV_col)
    delta_total = delta_gas + delta_solid

    # ── Propiedades de referencia (modo frozen) ───────────────────────────────
    # El RHS en frozen mode fija gas_props y trans_props al primer instante.
    # El balance debe usar esas mismas propiedades para ser coherente.
    if freeze_gas or freeze_trans:
        _gas_ref = compute_gas_mixture_properties(
            P_Pa=P[0] * 1.0e5, Tg=Tg[0], x=y3d[0].T,
            prop_gas=prop_gas, n_comp=nc, N=nn,
        )
        _vface_ref = ergun_face_velocity(
            P=P[0] * 1.0e5, rho_g=_gas_ref["rho"], mu_g=_gas_ref["mu"],
            epsi=epsi, dp=dp, dz=dz,
            v_in=float(v_in[0]), v_out=float(v_out[0]),
        )
        _vcell_ref = 0.5 * (_vface_ref[:-1] + _vface_ref[1:])
    if freeze_trans:
        _trans_ref = compute_transfer_coefficients(
            Tg=Tg[0], Ts=Ts[0], x=y3d[0].T,
            gas_props=_gas_ref, u_rel=np.abs(_vcell_ref),
            prop_gas=prop_gas, prop_lecho=prop_lecho,
            Di=Di, trans_config=trans_config,
            n_comp=nc, N=nn,
        )

    # ── Arrays de potencias instantáneas [W] ─────────────────────────────────
    Ein_dot          = np.zeros(n_t, dtype=float)
    Eout_dot         = np.zeros(n_t, dtype=float)
    Qcond_g_net_dot  = np.zeros(n_t, dtype=float)
    Qwall_dot        = np.zeros(n_t, dtype=float)
    Qgs_dot          = np.zeros(n_t, dtype=float)
    Eads_dot         = np.zeros(n_t, dtype=float)
    Eads_sens_gas_dot = np.zeros(n_t, dtype=float)
    Qext_dot         = np.zeros(n_t, dtype=float) if shell_tube else None

    for k in range(n_t):
        Tg_k    = Tg[k]              # (N,)
        Ts_k    = Ts[k]              # (N,)
        x_k     = y3d[k].T          # (N, nc)
        C_k     = C[k]              # (nc, N)
        v_in_k  = float(v_in[k])
        v_out_k = float(v_out[k])
        C_in_k  = C_in_hist[k]     # (nc,)
        T_in_k  = float(T_in_hist[k])
        T_in_val = T_in_k if np.isfinite(T_in_k) else None
        C_in_val = C_in_k if T_in_val is not None else None

        # Propiedades de mezcla — frozen usa estado inicial; always usa estado actual
        if freeze_gas:
            gas = _gas_ref
        else:
            gas = compute_gas_mixture_properties(
                P_Pa=P[k] * 1.0e5, Tg=Tg_k, x=x_k,
                prop_gas=prop_gas, n_comp=nc, N=nn,
            )

        # Ergun: rho/mu frozen o actuales; P siempre actual (es el driver dinámico)
        v_face_k = ergun_face_velocity(
            P=P[k] * 1.0e5, rho_g=gas["rho"], mu_g=gas["mu"],
            epsi=epsi, dp=dp, dz=dz,
            v_in=v_in_k, v_out=v_out_k,
        )
        v_cell_k = 0.5 * (v_face_k[:-1] + v_face_k[1:])

        # Coeficientes de transporte — frozen usa estado inicial; always usa actual
        if freeze_trans:
            trans = _trans_ref
        else:
            trans = compute_transfer_coefficients(
                Tg=Tg_k, Ts=Ts_k, x=x_k,
                gas_props=gas, u_rel=np.abs(v_cell_k),
                prop_gas=prop_gas, prop_lecho=prop_lecho,
                Di=Di, trans_config=trans_config,
                n_comp=nc, N=nn,
            )

        # 1. Flujo convectivo de entalpía → Ein, Eout [W]
        use_inlet = (bc_in_T == "dirichlet") and (T_in_val is not None)
        Fh_conv_k = gas_enthalpy_convective_flux(
            Tg_cell=Tg_k, C_cell=C_k, v_face=v_face_k,
            prop_gas=prop_gas, n_comp=nc, gas_T_ref=gas_T_ref,
            T_in=T_in_val if use_inlet else None,
            C_in=C_in_val if use_inlet else None,
        )
        Ein_dot[k]  = epsi * Ai * Fh_conv_k[0]
        Eout_dot[k] = epsi * Ai * Fh_conv_k[-1]

        # 2. Conducción térmica del gas en contornos → Qcond_g_net [W]
        #    La suma telescópica de div(qg_diff) sobre todas las celdas da
        #    exactamente Ai * (qg_diff[0] - qg_diff[N]).
        qg_diff_k = gas_diffusive_heat_flux(
            Tg_cell=Tg_k, k_g=gas["k"], dz=dz,
            T_in=T_in_val if use_inlet else None,
            T_out=None,
            bc_in=bc_in_T, bc_out="neumann",
        )
        Qcond_g_net_dot[k] = Ai * (qg_diff_k[0] - qg_diff_k[-1])

        # 3. Calor de pared [W]
        if shell_tube:
            Tw_k = Tw_hist[k]   # (N,)
            # Heat from wall to gas (positive when Tw > Tg), matching rhs_adsorption step 9.4
            Qwall_dot[k] = float(np.sum(trans["h_wall"] * Pi * dz * (Tw_k - Tg_k)))
            # Exterior heat entering the wall
            k_w_arr = eval_solid_property(wall_config["material"]["k"], Tw_k)
            Q_ext_cell = wall_exterior_q(
                Tw_arr=Tw_k, thermal_bc_cfg=thermal_bc_cfg,
                k_w_arr=k_w_arr, Pi=Pi, Po=Po, dz=dz, N=nn,
            )
            Qext_dot[k] = float(np.sum(Q_ext_cell))
        else:
            _, Qwall_k, _ = wall_heat_flux(
                Tg=Tg_k, h_wall=trans["h_wall"],
                thermal_bc_config=thermal_bc_cfg,
                N=nn, Ai=Ai, Pi=Pi, Po=Po, dz=dz,
            )
            Qwall_dot[k] = Qwall_k

        # 4. Intercambio gas-sólido [W]: positivo = gas transfiere calor al sólido
        q_gs_vol_k = trans["h_bed"] * a_surf * (Tg_k - Ts_k)   # (N,) [W/m³]
        Qgs_dot[k] = float(Ai * dz * np.sum(q_gs_vol_k))

        # 5. Términos de adsorción — dqdt por cinética LDF exacta (igual que core_rhs)
        #    p_i = C_i * R * Tg / 1e5  [bar]
        P_part_k = [C_k[i, :] * R_GAS * Tg_k / 1.0e5 for i in range(nc)]
        q_eq_k   = np.asarray(iso_fn(P_part_k, Ts_k), dtype=float)   # (nc, N) [mol/kg]
        dqdt_k   = trans["k_mtc"] * (q_eq_k - q[k])                   # (nc, N) [mol/kg/s]
        h_i_k    = gas["h_i"].T                                        # (nc, N) [J/mol]

        # Calor de adsorción al sólido [W]
        q_ads_vol_k = (1.0 - epsi) * rho_s * np.sum(dH_mat * dqdt_k, axis=0)   # (N,)
        Eads_dot[k] = float(Ai * dz * np.sum(q_ads_vol_k))

        # Entalpía sensible retirada del gas al adsorber [W]
        q_ads_sens_k = (1.0 - epsi) * rho_s * np.sum(h_i_k * dqdt_k, axis=0)  # (N,)
        Eads_sens_gas_dot[k] = float(Ai * dz * np.sum(q_ads_sens_k))

    # ── Integración temporal (regla trapezoidal) ──────────────────────────────
    Ein           = float(np.trapz(Ein_dot,           t))
    Eout          = float(np.trapz(Eout_dot,          t))
    Qcond_g_net   = float(np.trapz(Qcond_g_net_dot,   t))
    Qwall         = float(np.trapz(Qwall_dot,         t))
    Qgs           = float(np.trapz(Qgs_dot,           t))
    Eads          = float(np.trapz(Eads_dot,          t))
    Eads_sens_gas = float(np.trapz(Eads_sens_gas_dot, t))

    def _ref(a: float, b: float) -> float:
        return max(abs(a), abs(b), 1.0e-6)

    # ── Errores de balance gas + sólido ──────────────────────────────────────
    err_gas   = Ein - Eout + Qcond_g_net + Qwall - Qgs - Eads_sens_gas - delta_gas
    err_solid = Qgs + Eads - delta_solid
    err_total = Ein - Eout + Qcond_g_net + Qwall - Eads_sens_gas + Eads - delta_total

    # Para E_gas se usa max(|Ein|, |Eout|) como referencia, no delta_gas.
    # El gas balance tiene cancelación masiva (Ein ≈ 7 MJ, Eout ≈ 5 MJ,
    # ΔE_gas ≈ 3 kJ), por lo que el error de integración trapezoidal del
    # postproceso (~84 J) aparece como ~2 % de ΔE_gas pero es sólo
    # 0.001 % del throughput — la referencia correcta es la escala dominante.
    ref_gas   = max(abs(Ein), abs(Eout), abs(delta_gas), 1.0e-6)
    ref_solid = _ref(delta_solid, Qgs + Eads)
    ref_total = _ref(delta_total, Ein - Eout + Qcond_g_net + Qwall - Eads_sens_gas + Eads)

    result = {
        "delta_gas_J":       delta_gas,
        "delta_solid_J":     delta_solid,
        "delta_total_J":     delta_total,
        "Ein_J":             Ein,
        "Eout_J":            Eout,
        "Qcond_g_net_J":     Qcond_g_net,
        "Qwall_J":           Qwall,
        "Qgs_J":             Qgs,
        "Eads_J":            Eads,
        "Eads_sens_gas_J":   Eads_sens_gas,
        "err_gas_J":         err_gas,
        "err_solid_J":       err_solid,
        "err_total_J":       err_total,
        "err_gas_rel":       err_gas   / ref_gas,
        "err_solid_rel":     err_solid / ref_solid,
        "err_total_rel":     err_total / ref_total,
    }

    # ── Balance de pared y sistema (solo shell_tube) ──────────────────────────
    if shell_tube:
        Qext_J = float(np.trapz(Qext_dot, t))

        # ΔE_wall: energía térmica acumulada en la pared metálica.
        # Se usa la regla del punto medio: cp(Tmean)·ΔT, consistente con
        # la formulación del ODE (rho·cp·dTw/dt = Q). La fórmula cp(Tf)·Tf - cp(T0)·T0
        # sobreestima en T·dcp/dT, lo que produce errores de varios % para aceros.
        mat    = wall_config["material"]
        A_w    = float(wall_config["A_w"])
        Tw_0   = Tw_hist[0]
        Tw_f   = Tw_hist[-1]
        Tw_mid = 0.5 * (Tw_0 + Tw_f)
        rho_w  = eval_solid_property(mat["rho"], Tw_mid)
        cp_w   = eval_solid_property(mat["cp"],  Tw_mid)
        delta_wall_J = float(np.sum(rho_w * cp_w * (Tw_f - Tw_0)) * A_w * dz)

        # Sistema completo: gas + sólido adsorbente + pared metálica
        # Qwall cancels (gas loses what wall gains from gas side):
        #   E_sys_rhs = Ein − Eout + Qcond_g_net + Qext − Eads_sens_gas + Eads
        delta_sys_J  = delta_gas + delta_solid + delta_wall_J
        E_sys_rhs    = Ein - Eout + Qcond_g_net + Qext_J - Eads_sens_gas + Eads
        err_sys_J    = delta_sys_J - E_sys_rhs
        ref_sys      = _ref(delta_sys_J, E_sys_rhs)

        result.update({
            "delta_wall_J": delta_wall_J,
            "Qext_J":       Qext_J,
            "delta_sys_J":  delta_sys_J,
            "err_sys_J":    err_sys_J,
            "err_sys_rel":  err_sys_J / ref_sys,
        })

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Informe de balance completo
# ═══════════════════════════════════════════════════════════════════════════════

def balance_report(
    col,
    params: Dict[str, Any],
    include_energy: bool = True,
    print_report: bool = True,
) -> pd.DataFrame:
    """
    @brief
    Genera un informe tabular de los balances de masa y energía.

    @details
    Calcula y, opcionalmente imprime, los balances molar (por especie y total)
    y de energía (con desglose completo de todos los términos). Devuelve un
    DataFrame con todas las filas del informe.

    Parameters
    ----------
    col            : SimpleNamespace devuelto por run_step.
    params         : dict del RHS.
    include_energy : bool — incluir balance de energía. Default True.
    print_report   : bool — imprimir el informe por pantalla. Default True.

    Returns
    -------
    pd.DataFrame
        Columnas: "Acumulación", "Flujo neto BC", "Error abs", "Error rel [%]".
        Las filas incluyen una fila por especie, la fila total molar, y (si
        include_energy=True) las filas del balance de energía completo.
        Las unidades de las filas son [mol] para el balance molar y [kJ] para
        el balance de energía.
    """
    mb = molar_balance(col, params)
    species = mb["species"]
    shell_tube = params.get("wall_config") is not None

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
    net_tot = mb["net_flux"].sum()
    acc_tot = mb["delta_total"].sum()
    err_tot_abs = mb["error_abs"].sum()
    ref_tot = max(mb["flux_in"].sum(), abs(net_tot),
                  abs(acc_tot), 1.0e-15)
    rows.append({
        "Variable":      "N_total [mol]",
        "Acumulación":   acc_tot,
        "Flujo neto BC": net_tot,
        "Error abs":     err_tot_abs,
        "Error rel [%]": err_tot_abs / ref_tot * 100.0,
    })

    # ── Balance de energía ────────────────────────────────────────────────────
    if include_energy:
        try:
            eb = energy_balance(col, params)

            kJ = 1.0e-3   # J → kJ

            # Fila de energía del gas
            rows.append({
                "Variable":      "E_gas [kJ]",
                "Acumulación":   eb["delta_gas_J"] * kJ,
                "Flujo neto BC": (eb["Ein_J"] - eb["Eout_J"]) * kJ,
                "Error abs":     eb["err_gas_J"] * kJ,
                "Error rel [%]": eb["err_gas_rel"] * 100.0,
            })
            # Fila de energía del sólido
            rows.append({
                "Variable":      "E_solid [kJ]",
                "Acumulación":   eb["delta_solid_J"] * kJ,
                "Flujo neto BC": (eb["Qgs_J"] + eb["Eads_J"]) * kJ,
                "Error abs":     eb["err_solid_J"] * kJ,
                "Error rel [%]": eb["err_solid_rel"] * 100.0,
            })
            # Fila de energía total
            rows.append({
                "Variable":      "E_total [kJ]",
                "Acumulación":   eb["delta_total_J"] * kJ,
                "Flujo neto BC": (eb["Ein_J"] - eb["Eout_J"] + eb["Qcond_g_net_J"]
                                  + eb["Qwall_J"] - eb["Eads_sens_gas_J"]
                                  + eb["Eads_J"]) * kJ,
                "Error abs":     eb["err_total_J"] * kJ,
                "Error rel [%]": eb["err_total_rel"] * 100.0,
            })
            # Filas adicionales solo con shell_tube
            if shell_tube and "delta_wall_J" in eb:
                rows.append({
                    "Variable":      "E_wall [kJ]",
                    "Acumulación":   eb["delta_wall_J"] * kJ,
                    "Flujo neto BC": (eb["Qext_J"] - eb["Qwall_J"]) * kJ,
                    "Error abs":     (eb["delta_wall_J"] - (eb["Qext_J"] - eb["Qwall_J"])) * kJ,
                    "Error rel [%]": (
                        (eb["delta_wall_J"] - (eb["Qext_J"] - eb["Qwall_J"]))
                        / max(abs(eb["delta_wall_J"]), abs(eb["Qext_J"] - eb["Qwall_J"]), 1.0e-6)
                        * 100.0
                    ),
                })
                rows.append({
                    "Variable":      "E_sistema [kJ]",
                    "Acumulación":   eb["delta_sys_J"] * kJ,
                    "Flujo neto BC": (eb["Ein_J"] - eb["Eout_J"] + eb["Qcond_g_net_J"]
                                      + eb["Qext_J"] - eb["Eads_sens_gas_J"]
                                      + eb["Eads_J"]) * kJ,
                    "Error abs":     eb["err_sys_J"] * kJ,
                    "Error rel [%]": eb["err_sys_rel"] * 100.0,
                })

        except Exception as exc:
            for label in ("E_gas [kJ]", "E_solid [kJ]", "E_total [kJ]"):
                rows.append({
                    "Variable":      label,
                    "Acumulación":   float("nan"),
                    "Flujo neto BC": float("nan"),
                    "Error abs":     float("nan"),
                    "Error rel [%]": float("nan"),
                })
            if print_report:
                print(f"  [balance_report] Balance de energía no disponible: {exc}")
            eb = None

    df = pd.DataFrame(rows).set_index("Variable")

    if print_report:
        sep = "─" * 82
        print(sep)
        print("  INFORME DE BALANCE DE CONSERVACIÓN")
        print(sep)

        pd.set_option("display.float_format", "{:+.5g}".format)
        print(df.to_string())
        print(sep)

        # ── Semáforo de calidad ───────────────────────────────────────────────
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

        # ── Desglose detallado de la energía ─────────────────────────────────
        if include_energy and eb is not None:
            kJ = 1.0e-3
            print()
            print("  Desglose de términos energéticos [kJ]:")
            print(f"    Ein               = {eb['Ein_J']*kJ:+.4g}")
            print(f"    Eout              = {eb['Eout_J']*kJ:+.4g}")
            print(f"    Qcond_g_net       = {eb['Qcond_g_net_J']*kJ:+.4g}"
                  "  (conducción gas en contornos)")
            print(f"    Qwall             = {eb['Qwall_J']*kJ:+.4g}"
                  "  (calor de pared)")
            print(f"    Qgs               = {eb['Qgs_J']*kJ:+.4g}"
                  "  (gas→sólido, interno)")
            print(f"    Eads              = {eb['Eads_J']*kJ:+.4g}"
                  "  (calor de adsorción al sólido)")
            print(f"    Eads_sens_gas     = {eb['Eads_sens_gas_J']*kJ:+.4g}"
                  "  (entalpía sensible retirada del gas)")
            print(f"    ΔE_gas            = {eb['delta_gas_J']*kJ:+.4g}")
            print(f"    ΔE_solid          = {eb['delta_solid_J']*kJ:+.4g}")
            if shell_tube and "delta_wall_J" in eb:
                print(f"    Qext              = {eb['Qext_J']*kJ:+.4g}"
                      "  (calor exterior → pared)")
                print(f"    ΔE_wall           = {eb['delta_wall_J']*kJ:+.4g}"
                      "  (acumulación en pared metálica)")
                print(f"    ΔE_sistema        = {eb['delta_sys_J']*kJ:+.4g}"
                      "  (gas + sólido + pared)")

        print(sep)

    return df
