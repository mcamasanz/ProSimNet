"""
@module config.boundary_c
@brief Validación y construcción del bloque de condiciones de contorno del ciclo PSA.

@details
Este módulo centraliza la validación de todos los parámetros de contorno de la
columna PSA: caudales, presiones, temperaturas y fracciones molares de entrada/salida
para cada paso del ciclo (adsorption, purge, pressurization with feed, blowdown,
equalization), así como los coeficientes de válvula (Cv) que controlan los flujos.

Parámetros obligatorios para una configuración single-column completa:
    Q_in_feed, P_in_feed, T_in_feed, y_in_feed
    P_out_ads, Cv_out_ads
    Q_in_purge, P_in_purge, P_out_purge, Cv_out_purge
    P_high, Cv_in_prfeed
    P_low, Cv_out_blowdown

Parámetros opcionales:
    T_in_purge, y_in_purge  — pueden ser escalares, arrays o callables f(t)
    Cv_eq_lm, Cv_eq_mh      — equalization entre columnas
    forward_flow_direction   — True (inlet=cara 0) / False (inlet=cara N, flujo invertido)

Unidades:
    Q  [m³/s]   — caudal volumétrico
    P  [bar]    — presión (convención interna del modelo)
    T  [K]      — temperatura
    y  [-]      — fracción molar, suma = 1 por celda
    Cv [-]      — coeficiente de válvula adimensional
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.utils.profiling import profiled


@profiled
def build_boundary_c_config(
    n_comp: int,
    N: int,
    forward_flow_direction: bool = True,
    Q_in_feed: Optional[float] = None,
    P_in_feed: Optional[float] = None,
    T_in_feed: Optional[float] = None,
    y_in_feed=None,
    P_out_ads: Optional[float] = None,
    Cv_out_ads: Optional[float] = None,
    Q_in_purge: Optional[float] = None,
    P_in_purge: Optional[float] = None,
    T_in_purge=None,
    y_in_purge=None,
    P_out_purge: Optional[float] = None,
    Cv_out_purge: Optional[float] = None,
    P_high: Optional[float] = None,
    Cv_in_prfeed: Optional[float] = None,
    P_low: Optional[float] = None,
    Cv_out_blowdown: Optional[float] = None,
    Cv_eq_lm: Optional[float] = None,
    Cv_eq_mh: Optional[float] = None,
) -> dict:
    """
    @brief
    Valida todos los parámetros de contorno PSA y devuelve un dict de configuración.

    @details
    La función valida:
    - Escalares no negativos (Cv, Q)
    - Escalares positivos (P, T escalares)
    - y_in_feed: array(nc,) que suma 1
    - T_in_purge y y_in_purge: escalar/array o callable
    - forward_flow_direction: bool

    El campo "is_complete" del dict devuelto indica si los 14 campos obligatorios
    para un ciclo PSA single-column están presentes.

    Parameters
    ----------
    n_comp                 : int — número de componentes
    N                      : int — número de celdas (no se usa para validación aquí,
                                   pero se pasa por coherencia con los otros builders)
    forward_flow_direction : bool — True = flujo de cara 0 a cara N (positivo)
    (resto)                : ver docstring del módulo

    Returns
    -------
    dict con todos los parámetros de contorno validados + campo "is_complete" bool
    + campo "flow_direction" str ("forward" / "backward")
    """
    nc = int(n_comp)

    if not isinstance(forward_flow_direction, bool):
        raise TypeError("forward_flow_direction must be bool")

    # --- escalares no negativos ---
    nonneg_scalars = {
        "Q_in_feed":       Q_in_feed,
        "Cv_out_ads":      Cv_out_ads,
        "Q_in_purge":      Q_in_purge,
        "Cv_out_purge":    Cv_out_purge,
        "Cv_in_prfeed":    Cv_in_prfeed,
        "Cv_out_blowdown": Cv_out_blowdown,
        "Cv_eq_lm":        Cv_eq_lm,
        "Cv_eq_mh":        Cv_eq_mh,
    }
    parsed_nonneg = {}
    for name, val in nonneg_scalars.items():
        if val is None:
            parsed_nonneg[name] = None
            continue
        fval = float(val)
        if not np.isfinite(fval) or fval < 0.0:
            raise ValueError(f"{name} must be a finite scalar >= 0")
        parsed_nonneg[name] = fval

    # --- escalares positivos ---
    pos_scalars = {
        "P_in_feed":  P_in_feed,
        "T_in_feed":  T_in_feed,
        "P_out_ads":  P_out_ads,
        "P_in_purge": P_in_purge,
        "P_out_purge": P_out_purge,
        "P_high":     P_high,
        "P_low":      P_low,
    }
    parsed_pos = {}
    for name, val in pos_scalars.items():
        if val is None:
            parsed_pos[name] = None
            continue
        fval = float(val)
        if not np.isfinite(fval) or fval <= 0.0:
            raise ValueError(f"{name} must be a finite scalar > 0")
        parsed_pos[name] = fval

    # --- y_in_feed: array(nc,), suma = 1 ---
    y_in_feed_arr = None
    if y_in_feed is not None:
        y_in_feed_arr = np.asarray(y_in_feed, dtype=float).reshape(-1)
        if y_in_feed_arr.shape != (nc,):
            raise ValueError(f"y_in_feed must have length {nc}")
        if not np.all(np.isfinite(y_in_feed_arr)):
            raise ValueError("y_in_feed contains nan or inf")
        if np.any(y_in_feed_arr < 0.0):
            raise ValueError("y_in_feed contains negative values")
        if np.sum(y_in_feed_arr) <= 0.0:
            raise ValueError("y_in_feed must have positive sum")
        if np.abs(np.sum(y_in_feed_arr) - 1.0) > 1.0e-6:
            raise ValueError("y_in_feed must sum to 1 within tolerance 1e-6")
        y_in_feed_arr = y_in_feed_arr.copy()

    # --- T_in_purge: escalar > 0 o callable ---
    T_in_purge_val = T_in_purge
    if T_in_purge_val is not None and not callable(T_in_purge_val):
        fval = float(T_in_purge_val)
        if not np.isfinite(fval) or fval <= 0.0:
            raise ValueError("T_in_purge must be a finite scalar > 0 [K] or callable")
        T_in_purge_val = fval

    # --- y_in_purge: array(nc,) o callable ---
    y_in_purge_val = y_in_purge
    if y_in_purge_val is not None and not callable(y_in_purge_val):
        y_purge_arr = np.asarray(y_in_purge_val, dtype=float).reshape(-1)
        if y_purge_arr.shape != (nc,):
            raise ValueError(f"y_in_purge must have length {nc}")
        if not np.all(np.isfinite(y_purge_arr)):
            raise ValueError("y_in_purge contains nan or inf")
        if np.any(y_purge_arr < 0.0):
            raise ValueError("y_in_purge contains negative values")
        if np.sum(y_purge_arr) <= 0.0:
            raise ValueError("y_in_purge must have positive sum")
        if np.abs(np.sum(y_purge_arr) - 1.0) > 1.0e-6:
            raise ValueError("y_in_purge must sum to 1 within tolerance 1e-6")
        y_in_purge_val = y_purge_arr.copy()

    # --- completeness check ---
    mandatory_values = [
        parsed_nonneg["Q_in_feed"],
        parsed_pos["P_in_feed"],
        parsed_pos["T_in_feed"],
        y_in_feed_arr,
        parsed_pos["P_out_ads"],
        parsed_nonneg["Cv_out_ads"],
        parsed_nonneg["Q_in_purge"],
        parsed_pos["P_in_purge"],
        parsed_pos["P_out_purge"],
        parsed_nonneg["Cv_out_purge"],
        parsed_pos["P_high"],
        parsed_nonneg["Cv_in_prfeed"],
        parsed_pos["P_low"],
        parsed_nonneg["Cv_out_blowdown"],
    ]
    is_complete = all(v is not None for v in mandatory_values)

    return {
        "flow_direction":   "forward" if forward_flow_direction else "backward",
        "Q_in_feed":        parsed_nonneg["Q_in_feed"],
        "P_in_feed":        parsed_pos["P_in_feed"],
        "T_in_feed":        parsed_pos["T_in_feed"],
        "y_in_feed":        y_in_feed_arr,
        "P_out_ads":        parsed_pos["P_out_ads"],
        "Cv_out_ads":       parsed_nonneg["Cv_out_ads"],
        "Q_in_purge":       parsed_nonneg["Q_in_purge"],
        "P_in_purge":       parsed_pos["P_in_purge"],
        "T_in_purge":       T_in_purge_val,
        "y_in_purge":       y_in_purge_val,
        "P_out_purge":      parsed_pos["P_out_purge"],
        "Cv_out_purge":     parsed_nonneg["Cv_out_purge"],
        "P_high":           parsed_pos["P_high"],
        "Cv_in_prfeed":     parsed_nonneg["Cv_in_prfeed"],
        "Cv_out_prfeed":    0.0,          # siempre 0 (válvula de salida cerrada en PR-feed)
        "P_low":            parsed_pos["P_low"],
        "Cv_in_blowdown":   0.0,          # siempre 0 (válvula de entrada cerrada en blowdown)
        "Cv_out_blowdown":  parsed_nonneg["Cv_out_blowdown"],
        "Cv_eq_lm":         parsed_nonneg["Cv_eq_lm"],
        "Cv_eq_mh":         parsed_nonneg["Cv_eq_mh"],
        "is_complete":      is_complete,
    }
