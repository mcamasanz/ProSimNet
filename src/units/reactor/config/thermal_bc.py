"""
Thermal boundary condition configuration for the reactor.

Four modes, identical in semantics to the gasifier and heater:

    "adiabatic"    — no lateral heat exchange.
    "ambient_htc"  — series resistance: wall conduction + external convection.
    "fixed_twall"  — wall at constant prescribed temperature.
    "heatfluxwall" — prescribed total heat input [W] (scalar or per-cell array).

Self-contained — no dependency on other equipment modules.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.utils.profiling import profiled

_ALLOWED_MODES = ("adiabatic", "ambient_htc", "fixed_twall", "heatfluxwall")


@profiled
def build_thermal_bc_config(
    mode:     str,
    Di:       float,
    Do:       float,
    e_wall:   float,
    h_ambi:   Optional[float] = None,
    T_ambi:   Optional[float] = None,
    T_wall:   Optional[float] = None,
    Qwall:                     None = None,
    k_wall:   Optional[float] = None,
    rho_wall: Optional[float] = None,
    Cp_wall:  Optional[float] = None,
) -> dict:
    """
    Validate thermal boundary parameters and return a configuration dict.

    Parameters
    ----------
    mode     : str    one of "adiabatic", "ambient_htc", "fixed_twall", "heatfluxwall"
    Di       : float  inner diameter [m]
    Do       : float  outer diameter [m]  (must be > Di)
    e_wall   : float  wall thickness [m]
    h_ambi   : float, optional  external convection HTC [W/m²/K]
    T_ambi   : float, optional  ambient temperature [K]
    T_wall   : float, optional  prescribed wall temperature [K]
    Qwall    : float or ndarray(N,), optional  total heat input [W]
    k_wall   : float, optional  wall thermal conductivity [W/m/K]
    rho_wall : float, optional  wall material density [kg/m³]
    Cp_wall  : float, optional  wall material heat capacity [J/kg/K]

    Returns
    -------
    dict with keys: mode, h_ambi, T_ambi, T_wall, Qwall, k_wall, rho_wall, Cp_wall
        Fields not applicable to the chosen mode are returned as None.
    """
    mode_str = str(mode).strip().lower()
    if mode_str not in _ALLOWED_MODES:
        raise ValueError(f"mode must be one of {_ALLOWED_MODES}, got '{mode_str}'")

    if float(Di) <= 0.0:
        raise ValueError(f"Di must be > 0, got {Di}")
    if float(Do) <= float(Di):
        raise ValueError(f"Do must be > Di, got Do={Do}, Di={Di}")
    if float(e_wall) <= 0.0:
        raise ValueError(f"e_wall must be > 0, got {e_wall}")

    def _f(v):
        return None if v is None else float(v)

    def _pos(v, name):
        if v is not None and (not np.isfinite(float(v)) or float(v) <= 0.0):
            raise ValueError(f"{name} must be finite and > 0")

    h_ambi_val   = _f(h_ambi)
    T_ambi_val   = _f(T_ambi)
    T_wall_val   = _f(T_wall)
    k_wall_val   = _f(k_wall)
    rho_wall_val = _f(rho_wall)
    Cp_wall_val  = _f(Cp_wall)

    if Qwall is None:
        Qwall_val = None
    else:
        _q = np.asarray(Qwall, dtype=float)
        Qwall_val = float(_q) if _q.ndim == 0 else _q
        if not np.all(np.isfinite(np.asarray(Qwall_val))):
            raise ValueError("Qwall must contain finite values [W]")

    if mode_str == "ambient_htc":
        if h_ambi_val is None: raise ValueError("h_ambi required for 'ambient_htc'")
        if T_ambi_val is None: raise ValueError("T_ambi required for 'ambient_htc'")
        if k_wall_val is None: raise ValueError("k_wall required for 'ambient_htc'")
        _pos(h_ambi_val, "h_ambi"); _pos(T_ambi_val, "T_ambi"); _pos(k_wall_val, "k_wall")
    elif mode_str == "fixed_twall":
        if T_wall_val is None: raise ValueError("T_wall required for 'fixed_twall'")
        _pos(T_wall_val, "T_wall")
    elif mode_str == "heatfluxwall":
        if Qwall_val is None: raise ValueError("Qwall required for 'heatfluxwall'")

    base = {"h_ambi": None, "T_ambi": None, "T_wall": None, "Qwall": None,
            "k_wall": None, "rho_wall": None, "Cp_wall": None}
    out = {"mode": mode_str, **base}

    if mode_str == "ambient_htc":
        out.update(h_ambi=h_ambi_val, T_ambi=T_ambi_val,
                   k_wall=k_wall_val, rho_wall=rho_wall_val, Cp_wall=Cp_wall_val)
    elif mode_str == "fixed_twall":
        out.update(T_wall=T_wall_val,
                   k_wall=k_wall_val, rho_wall=rho_wall_val, Cp_wall=Cp_wall_val)
    elif mode_str == "heatfluxwall":
        out.update(Qwall=Qwall_val,
                   k_wall=k_wall_val, rho_wall=rho_wall_val, Cp_wall=Cp_wall_val)

    return out
