"""
@module gasifier.config.thermal_bc
@brief Thermal boundary condition configuration for the gasifier.

@details
Four thermal boundary modes, identical in semantics to the adsorber and heater.
The implementation is self-contained here — no dependency on other equipment
modules — so that the gasifier remains functional even if heater or adsorber
are refactored or removed.

    "adiabatic"    — no lateral heat exchange.

    "ambient_htc"  — series resistance: wall conduction + external convection.
                     Requires: h_ambi [W/m²/K], T_ambi [K], k_wall [W/m/K].

    "fixed_twall"  — wall at constant prescribed temperature.
                     Requires: T_wall [K].

    "heatfluxwall" — prescribed total heat input (e.g. external furnace).
                     Requires: Qwall [W]  (scalar = uniform; array = per cell).

Geometry (Di, Do, e_wall) is validated here but NOT stored in the returned dict;
it belongs to the domain geometry block.

Units: SI.
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
    Validate thermal boundary parameters and return a configuration dict.

    Parameters
    ----------
    mode     : str   one of "adiabatic", "ambient_htc", "fixed_twall", "heatfluxwall"
    Di       : float inner diameter [m]
    Do       : float outer diameter [m]  (must be > Di)
    e_wall   : float wall thickness [m]
    h_ambi   : float or callable, optional  external convection HTC [W/m²/K]
    T_ambi   : float or callable, optional  ambient temperature [K]
    T_wall   : float or callable, optional  prescribed wall temperature [K]
    Qwall    : float, ndarray(N,), or callable, optional  total heat input [W]
    k_wall   : float, optional  wall thermal conductivity [W/m/K]
    rho_wall : float, optional  wall material density [kg/m³]
    Cp_wall  : float, optional  wall material heat capacity [J/kg/K]

    Returns
    -------
    dict with keys:
        mode, h_ambi, T_ambi, T_wall, Qwall, k_wall, rho_wall, Cp_wall
        (fields not applicable to the chosen mode are returned as None)
    """
    mode_str = str(mode).strip().lower()
    if mode_str not in _ALLOWED_MODES:
        raise ValueError(f"mode must be one of {_ALLOWED_MODES}")

    # Geometry validation
    if not np.isfinite(float(Di)) or float(Di) <= 0.0:
        raise ValueError("Di must be a finite scalar > 0 [m]")
    if not np.isfinite(float(Do)) or float(Do) <= 0.0:
        raise ValueError("Do must be a finite scalar > 0 [m]")
    if float(Do) <= float(Di):
        raise ValueError("Do must satisfy Do > Di")
    if not np.isfinite(float(e_wall)) or float(e_wall) <= 0.0:
        raise ValueError("e_wall must be a finite scalar > 0 [m]")

    def _opt_val(val):
        # Callables (time-dependent or state-feedback signals) pass through unchanged.
        # They will be resolved by the RHS via src.utils.signals.resolve().
        if val is None:
            return None
        if callable(val):
            return val
        return float(val)

    h_ambi_val   = _opt_val(h_ambi)
    T_ambi_val   = _opt_val(T_ambi)
    T_wall_val   = _opt_val(T_wall)
    k_wall_val   = _opt_val(k_wall)
    rho_wall_val = _opt_val(rho_wall)
    Cp_wall_val  = _opt_val(Cp_wall)

    # Qwall: scalar [W], 1D array [W per cell], or callable(t) / callable(t, snap)
    if Qwall is None:
        Qwall_val = None
    elif callable(Qwall):
        Qwall_val = Qwall
    else:
        _q = np.asarray(Qwall, dtype=float)
        if _q.ndim == 0:
            Qwall_val = float(_q)
        elif _q.ndim == 1:
            Qwall_val = _q
        else:
            raise ValueError("Qwall must be a scalar [W], 1D array [W/cell], or callable")

    def _check_pos(val, name):
        # Callables are resolved at runtime — skip static validation.
        if val is None or callable(val):
            return
        if not np.isfinite(val) or val <= 0.0:
            raise ValueError(f"{name} must be a finite scalar > 0")

    # Mode-specific validation
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
        if not callable(Qwall_val):
            if np.ndim(Qwall_val) == 0:
                if not np.isfinite(float(Qwall_val)):
                    raise ValueError("Qwall must be a finite scalar [W]")
            else:
                if not np.all(np.isfinite(Qwall_val)):
                    raise ValueError("All Qwall values must be finite [W]")
        _check_pos(k_wall_val,   "k_wall")
        _check_pos(rho_wall_val, "rho_wall")
        _check_pos(Cp_wall_val,  "Cp_wall")

    # adiabatic: no additional parameters required

    # Build output dict (fields not applicable to the mode are None)
    if mode_str == "adiabatic":
        return {
            "mode":     mode_str,
            "h_ambi":   None,
            "T_ambi":   None,
            "T_wall":   None,
            "Qwall":    None,
            "k_wall":   None,
            "rho_wall": None,
            "Cp_wall":  None,
        }
    elif mode_str == "ambient_htc":
        return {
            "mode":     mode_str,
            "h_ambi":   h_ambi_val,
            "T_ambi":   T_ambi_val,
            "T_wall":   None,
            "Qwall":    None,
            "k_wall":   k_wall_val,
            "rho_wall": rho_wall_val,
            "Cp_wall":  Cp_wall_val,
        }
    elif mode_str == "fixed_twall":
        return {
            "mode":     mode_str,
            "h_ambi":   None,
            "T_ambi":   None,
            "T_wall":   T_wall_val,
            "Qwall":    None,
            "k_wall":   k_wall_val,
            "rho_wall": rho_wall_val,
            "Cp_wall":  Cp_wall_val,
        }
    else:  # heatfluxwall
        return {
            "mode":     mode_str,
            "h_ambi":   None,
            "T_ambi":   None,
            "T_wall":   None,
            "Qwall":    Qwall_val,
            "k_wall":   k_wall_val,
            "rho_wall": rho_wall_val,
            "Cp_wall":  Cp_wall_val,
        }
