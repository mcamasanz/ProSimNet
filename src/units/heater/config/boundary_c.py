"""
@module heater.config.boundary_c
@brief Validación y construcción del bloque de condiciones de contorno del heater.

@details
El heater opera en régimen de flujo continuo con una sola dirección de paso.
Las condiciones de contorno son:

    Entrada (cara 0):
        v_in  — velocidad superficial [m/s]         (escalar o callable f(t))
        T_in  — temperatura del gas   [K]            (escalar o callable f(t))
        y_in  — fracción molar        [-], shape (nc,)  (array o callable f(t) → array(nc,))

    Salida (cara N):
        P_out — presión en la cara de salida [bar]  (escalar positivo)
                Ancla el nivel de presión del dominio. La velocidad en la cara de
                salida se deriva por conservación de flujo molar en la función de contorno.

Las condiciones de entrada pueden ser estáticas (escalar / array) o dinámicas
(callable f(t)) para permitir transitorios de arranque y parada.

Unidades: SI.
  v_in  [m/s]
  T_in  [K]
  y_in  [-]
  P_out [bar]
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.utils.profiling import profiled


@profiled
def build_boundary_c_config(
    n_comp: int,
    v_in,
    T_in,
    y_in,
    P_out: float,
) -> dict:
    """
    @brief
    Valida los parámetros de contorno del heater y devuelve un dict de configuración.

    @details
    Acepta valores estáticos o callables para v_in, T_in e y_in. Los callables
    deben tener la firma f(t: float) → valor, donde t es el tiempo en segundos.

    Parameters
    ----------
    n_comp : int   — número de componentes
    v_in   : float o callable(t) — velocidad superficial en inlet [m/s] (>= 0)
    T_in   : float o callable(t) — temperatura en inlet [K] (> 0)
    y_in   : array(nc,) o callable(t) → array(nc,) — fracción molar en inlet [-]
    P_out  : float — presión en la cara de salida [bar] (> 0)

    Returns
    -------
    dict con claves:
        v_in  : float o callable
        T_in  : float o callable
        y_in  : ndarray(nc,) o callable
        P_out : float
    """
    nc = int(n_comp)

    # ── v_in ─────────────────────────────────────────────────────────────────
    if callable(v_in):
        v_in_val = v_in
    else:
        fval = float(v_in)
        if not np.isfinite(fval) or fval < 0.0:
            raise ValueError("v_in must be a finite scalar >= 0 [m/s]")
        v_in_val = fval

    # ── T_in ─────────────────────────────────────────────────────────────────
    if callable(T_in):
        T_in_val = T_in
    else:
        fval = float(T_in)
        if not np.isfinite(fval) or fval <= 0.0:
            raise ValueError("T_in must be a finite scalar > 0 [K]")
        T_in_val = fval

    # ── y_in ─────────────────────────────────────────────────────────────────
    if callable(y_in):
        y_in_val = y_in
    else:
        y_arr = np.asarray(y_in, dtype=float).reshape(-1)
        if y_arr.shape != (nc,):
            raise ValueError(f"y_in must have length {nc}")
        if not np.all(np.isfinite(y_arr)):
            raise ValueError("y_in contains nan or inf")
        if np.any(y_arr < 0.0):
            raise ValueError("y_in contains negative values")
        if np.abs(np.sum(y_arr) - 1.0) > 1.0e-6:
            raise ValueError("y_in must sum to 1 within tolerance 1e-6")
        y_in_val = y_arr.copy()

    # ── P_out ────────────────────────────────────────────────────────────────
    P_out_val = float(P_out)
    if not np.isfinite(P_out_val) or P_out_val <= 0.0:
        raise ValueError("P_out must be a finite scalar > 0 [bar]")

    return {
        "v_in":  v_in_val,
        "T_in":  T_in_val,
        "y_in":  y_in_val,
        "P_out": P_out_val,
    }
