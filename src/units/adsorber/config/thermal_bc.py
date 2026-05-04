"""
@module config.thermal_bc
@brief Validación del bloque de condición de contorno térmica exterior.

@details
Cuatro modos de contorno térmico:

    "adiabatic"    — no hay intercambio de calor con el exterior.
                     Ningún parámetro adicional es necesario.

    "ambient_htc"  — resistencia térmica pared + convección exterior:
                     requiere h_ambi [W/m²/K], T_ambi [K], k_wall [W/m/K].
                     rho_wall y Cp_wall son opcionales (solo para balance de
                     energía en la pared si se resuelve dinámicamente).

    "fixed_twall"  — temperatura de pared impuesta:
                     requiere T_wall [K].
                     k_wall, rho_wall, Cp_wall son opcionales.

    "heatfluxwall" — flujo de calor total impuesto a la columna:
                     requiere Qwall [W] (puede ser negativo para extracción).
                     k_wall, rho_wall, Cp_wall son opcionales.

La geometría de la columna (Di, Do, e_wall) es necesaria para validar
coherencia dimensional pero no se almacena en el dict devuelto — pertenece
al bloque de geometría del dominio.

Unidades: SI.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.utils.profiling import profiled

_ALLOWED_MODES = ("adiabatic", "ambient_htc", "fixed_twall", "heatfluxwall")


@profiled
def build_thermal_bc_config(
    mode: str,
    Di: float,
    Do: float,
    e_wall: float,
    h_ambi: Optional[float] = None,
    T_ambi: Optional[float] = None,
    T_wall: Optional[float] = None,
    Qwall:  Optional[float] = None,
    k_wall: Optional[float] = None,
    rho_wall: Optional[float] = None,
    Cp_wall:  Optional[float] = None,
) -> dict:
    """
    @brief
    Valida los parámetros de contorno térmico y devuelve un dict de configuración.

    Parameters
    ----------
    mode     : str — uno de "adiabatic", "ambient_htc", "fixed_twall", "heatfluxwall"
    Di       : float — diámetro interno de la columna [m] (> 0)
    Do       : float — diámetro externo de la columna [m] (> Di)
    e_wall   : float — espesor de pared [m] (> 0)
    h_ambi   : float, opcional — htc convección exterior pared-ambiente [W/m²/K]
    T_ambi   : float, opcional — temperatura ambiente exterior [K]
    T_wall   : float, opcional — temperatura de pared impuesta [K]
    Qwall    : float, opcional — potencia térmica total a la columna [W]
    k_wall   : float, opcional — conductividad térmica de la pared [W/m/K]
    rho_wall : float, opcional — densidad del material de pared [kg/m³]
    Cp_wall  : float, opcional — calor específico del material de pared [J/kg/K]

    Returns
    -------
    dict con claves:
        mode, h_ambi, T_ambi, T_wall, Qwall, k_wall, rho_wall, Cp_wall
        (los campos no aplicables al modo elegido se devuelven como None)
    """
    mode_str = str(mode).strip().lower()
    if mode_str not in _ALLOWED_MODES:
        raise ValueError(f"mode must be one of {_ALLOWED_MODES}")

    # --- geometría ---
    if not np.isfinite(float(Di)) or float(Di) <= 0.0:
        raise ValueError("Di must be a finite scalar > 0 [m]")
    if not np.isfinite(float(Do)) or float(Do) <= 0.0:
        raise ValueError("Do must be a finite scalar > 0 [m]")
    if float(Do) <= float(Di):
        raise ValueError("Do must satisfy Do > Di")
    if not np.isfinite(float(e_wall)) or float(e_wall) <= 0.0:
        raise ValueError("e_wall must be a finite scalar > 0 [m]")

    # --- conversión a float ---
    def _opt_float(val, name):
        return None if val is None else float(val)

    h_ambi_val   = _opt_float(h_ambi,   "h_ambi")
    T_ambi_val   = _opt_float(T_ambi,   "T_ambi")
    T_wall_val   = _opt_float(T_wall,   "T_wall")
    k_wall_val   = _opt_float(k_wall,   "k_wall")
    rho_wall_val = _opt_float(rho_wall, "rho_wall")
    Cp_wall_val  = _opt_float(Cp_wall,  "Cp_wall")

    # Qwall: scalar (total [W] distribuido uniformemente) o 1D array (potencia por celda [W])
    if Qwall is None:
        Qwall_val = None
    else:
        _q = np.asarray(Qwall, dtype=float)
        if _q.ndim == 0:
            Qwall_val = float(_q)
        elif _q.ndim == 1:
            Qwall_val = _q
        else:
            raise ValueError("Qwall debe ser un escalar [W] o un array 1D de potencias por celda [W]")

    # --- helper de validación ---
    def _check_pos(val, name):
        if val is not None and (not np.isfinite(val) or val <= 0.0):
            raise ValueError(f"{name} must be a finite scalar > 0")

    # --- validaciones por modo ---
    if mode_str == "ambient_htc":
        if h_ambi_val is None:
            raise ValueError("h_ambi must be provided when mode='ambient_htc'")
        if T_ambi_val is None:
            raise ValueError("T_ambi must be provided when mode='ambient_htc'")
        if k_wall_val is None:
            raise ValueError("k_wall must be provided when mode='ambient_htc'")
        _check_pos(h_ambi_val,   "h_ambi")
        _check_pos(T_ambi_val,   "T_ambi")
        _check_pos(k_wall_val,   "k_wall")
        _check_pos(rho_wall_val, "rho_wall")
        _check_pos(Cp_wall_val,  "Cp_wall")

    elif mode_str == "fixed_twall":
        if T_wall_val is None:
            raise ValueError("T_wall must be provided when mode='fixed_twall'")
        _check_pos(T_wall_val,   "T_wall")
        _check_pos(k_wall_val,   "k_wall")
        _check_pos(rho_wall_val, "rho_wall")
        _check_pos(Cp_wall_val,  "Cp_wall")

    elif mode_str == "heatfluxwall":
        if Qwall_val is None:
            raise ValueError("Qwall must be provided when mode='heatfluxwall'")
        if np.ndim(Qwall_val) == 0:
            if not np.isfinite(float(Qwall_val)):
                raise ValueError("Qwall must be a finite scalar [W]")
        else:
            if not np.all(np.isfinite(Qwall_val)):
                raise ValueError("All Qwall values must be finite [W]")
        _check_pos(k_wall_val,   "k_wall")
        _check_pos(rho_wall_val, "rho_wall")
        _check_pos(Cp_wall_val,  "Cp_wall")

    # En adiabatic no se necesita ningún parámetro adicional

    # --- construir dict por modo (los campos no aplicables van a None) ---
    if mode_str == "adiabatic":
        return {
            "mode":      mode_str,
            "h_ambi":    None,
            "T_ambi":    None,
            "T_wall":    None,
            "Qwall":     None,
            "k_wall":    None,
            "rho_wall":  None,
            "Cp_wall":   None,
        }
    elif mode_str == "ambient_htc":
        return {
            "mode":      mode_str,
            "h_ambi":    h_ambi_val,
            "T_ambi":    T_ambi_val,
            "T_wall":    None,
            "Qwall":     None,
            "k_wall":    k_wall_val,
            "rho_wall":  rho_wall_val,
            "Cp_wall":   Cp_wall_val,
        }
    elif mode_str == "fixed_twall":
        return {
            "mode":      mode_str,
            "h_ambi":    None,
            "T_ambi":    None,
            "T_wall":    T_wall_val,
            "Qwall":     None,
            "k_wall":    k_wall_val,
            "rho_wall":  rho_wall_val,
            "Cp_wall":   Cp_wall_val,
        }
    else:  # heatfluxwall
        return {
            "mode":      mode_str,
            "h_ambi":    None,
            "T_ambi":    None,
            "T_wall":    None,
            "Qwall":     Qwall_val,
            "k_wall":    k_wall_val,
            "rho_wall":  rho_wall_val,
            "Cp_wall":   Cp_wall_val,
        }
