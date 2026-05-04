"""
@module boundary_conditions.heater_boundary
@brief Condiciones de contorno del heater de gas 1D.

@details
El heater opera en régimen de flujo continuo con una única dirección de paso
y sin lógica de ciclo. Las condiciones de contorno son:

    Cara de entrada (cara 0):
        v_in   — velocidad superficial impuesta [m/s]
        T_in   — temperatura del gas en inlet [K]
        y_in   — fracción molar del gas en inlet [-], shape (nc,)
        C_in   — concentración molar derivada de (y_in, P_cell[0], T_in) [mol/m³]

    Cara de salida (cara N):
        v_out  — velocidad derivada por conservación de flujo molar [m/s]
        P_out  — presión de salida impuesta [bar] (referencia de nivel de presión)

La velocidad de salida se calcula imponiendo conservación de flujo molar total
entre la cara de entrada y la cara de salida:

        v_in · Ctot_in = v_out · Ctot_out
        ↓
        v_out = v_in · Ctot_in / Ctot_out

donde:
    Ctot_in  = P_cell[0] · 1e5 / (R · T_in)   — concentración total en el inlet
                                                  (se usa P de la primera celda como
                                                   aproximación de la presión en cara 0)
    Ctot_out = Ctot_cell[-1]                   — concentración total en la última celda

Esta condición garantiza que la masa total de gas se conserva globalmente en
estado quasi-estacionario; el transitorio dinámico de las concentraciones es
gestionado por los balances del RHS.

Las entradas v_in, T_in e y_in pueden ser estáticas (escalar/array) o dinámicas
(callable f(t: float) → valor), para permitir transitorios de arranque y parada.

Convención de signo:
    v_in  > 0 — gas entra en la cara 0 en dirección +z
    v_out > 0 — gas sale por la cara N en dirección +z

Unidades: SI.
  v    [m/s]
  T    [K]
  P    [bar]
  C    [mol/m³]
  y    [-]
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from src.utils.profiling import profiled

R_GAS = 8.31446261815324   # [J/mol/K]


@profiled
def get_heater_boundary(
    t: float,
    P_cell: np.ndarray,
    Ctot_cell: np.ndarray,
    bc_config: dict,
    n_comp: int,
) -> Dict[str, Any]:
    """
    @brief
    Construye el dict de condiciones de contorno del heater para el instante t.

    @details
    Evalúa los valores de inlet (v_in, T_in, y_in, C_in) y outlet (v_out)
    a partir de la configuración de contorno y el estado actual del dominio.

    La concentración de inlet C_in se calcula usando la presión de la primera
    celda (aproximación del nivel de presión en la cara de entrada):

        C_in[i] = y_in[i] · P_cell[0] · 1e5 / (R · T_in)

    La velocidad de outlet se obtiene por conservación de flujo molar:

        v_out = v_in · Ctot_in / max(Ctot_cell[-1], ε)

    Parameters
    ----------
    t          : float — tiempo actual [s] (necesario para callables en bc_config)
    P_cell     : np.ndarray, shape (N,) — presión por celda [bar]
    Ctot_cell  : np.ndarray, shape (N,) — concentración molar total por celda [mol/m³]
    bc_config  : dict — resultado de `build_boundary_c_config` del heater
    n_comp     : int  — número de componentes

    Returns
    -------
    dict con claves:
        "inlet" : dict con claves T_K, y (nc,), C_mol_m3 (nc,), v_m_s
        "outlet": dict con claves P_out_bar, v_m_s
    """
    nc = int(n_comp)

    P_arr     = np.asarray(P_cell,    dtype=float).reshape(-1)
    Ctot_arr  = np.asarray(Ctot_cell, dtype=float).reshape(-1)

    # ── Evaluar condiciones de inlet (estáticas o dinámicas) ──────────────────
    v_raw = bc_config["v_in"]
    T_raw = bc_config["T_in"]
    y_raw = bc_config["y_in"]

    v_in = float(v_raw(t)) if callable(v_raw) else float(v_raw)
    T_in = float(T_raw(t)) if callable(T_raw) else float(T_raw)

    y_raw_val = y_raw(t) if callable(y_raw) else y_raw
    y_in = np.asarray(y_raw_val, dtype=float).reshape(-1)

    if y_in.shape != (nc,):
        raise ValueError(f"y_in must have length {nc}, got {y_in.shape}")

    # ── Concentración de inlet: gas ideal con presión de la primera celda ─────
    P_inlet_Pa = float(P_arr[0]) * 1.0e5
    T_in_safe  = max(T_in, 1.0e-12)
    C_in = y_in * P_inlet_Pa / (R_GAS * T_in_safe)   # [mol/m³], shape (nc,)

    # ── Velocidad de outlet: conservación de flujo molar total ───────────────
    # v_in · Ctot_in = v_out · Ctot_out
    Ctot_in  = P_inlet_Pa / (R_GAS * T_in_safe)       # [mol/m³]
    Ctot_out = max(float(Ctot_arr[-1]), 1.0e-30)
    v_out = v_in * Ctot_in / Ctot_out                 # [m/s]

    return {
        "inlet": {
            "T_K":      T_in,
            "y":        y_in.copy(),
            "C_mol_m3": C_in.copy(),
            "v_m_s":    v_in,
        },
        "outlet": {
            "P_out_bar": float(bc_config["P_out"]),
            "v_m_s":     v_out,
        },
    }
