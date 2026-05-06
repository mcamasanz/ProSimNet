"""
Time-variable signal primitives for ProSimNet BC parameters and setpoints.

All functions return callables compatible with ``resolve(signal, t, snap=None)``,
i.e. they are ``callable(t) -> float`` and accept only ``t`` as argument.
They can be used directly as values in ``bc_config`` or ``thermal_bc_config``::

    from src.control.signals import ramp, step, pulse, piecewise, sine

    tbc_cfg["T_wall"] = ramp(t_start=0.0, slope=0.5, value_init=700.0,
                             value_max=1000.0)      # 700 → 1000 K en 600 s

    bc_cfg["v_gas_in"] = step(t_step=600.0, value_before=0.05,
                              value_after=0.0)       # corte de flujo a t=600 s

Units: all time arguments in [s]; output units depend on the BC parameter.
"""
from __future__ import annotations

import numpy as np
from typing import Callable


def ramp(
    t_start: float,
    slope: float,
    value_init: float = 0.0,
    value_max: float | None = None,
    value_min: float | None = None,
) -> Callable[[float], float]:
    """
    Linear ramp starting at t_start.

    Parameters
    ----------
    t_start    : float         Start time [s]; signal is value_init for t < t_start.
    slope      : float         Rate of change [units/s]; can be negative.
    value_init : float         Value at t <= t_start (default 0).
    value_max  : float or None Upper saturation limit (optional).
    value_min  : float or None Lower saturation limit (optional).

    Returns
    -------
    callable(t) -> float
    """
    def _ramp(t: float) -> float:
        val = value_init + slope * max(0.0, float(t) - float(t_start))
        if value_max is not None:
            val = min(val, float(value_max))
        if value_min is not None:
            val = max(val, float(value_min))
        return val
    return _ramp


def step(
    t_step: float,
    value_before: float,
    value_after: float,
) -> Callable[[float], float]:
    """
    Instantaneous step at t_step.

    Parameters
    ----------
    t_step       : float   Time of step [s].
    value_before : float   Value for t < t_step.
    value_after  : float   Value for t >= t_step.

    Returns
    -------
    callable(t) -> float
    """
    _tb = float(t_step)
    _va = float(value_after)
    _vb = float(value_before)

    def _step(t: float) -> float:
        return _va if float(t) >= _tb else _vb
    return _step


def pulse(
    t_start: float,
    t_end: float,
    value_on: float,
    value_off: float = 0.0,
) -> Callable[[float], float]:
    """
    Rectangular pulse active in [t_start, t_end).

    Parameters
    ----------
    t_start   : float   Start time [s] (inclusive).
    t_end     : float   End time [s] (exclusive).
    value_on  : float   Value during the pulse.
    value_off : float   Value outside the pulse (default 0).

    Returns
    -------
    callable(t) -> float
    """
    _ts = float(t_start)
    _te = float(t_end)
    _on  = float(value_on)
    _off = float(value_off)

    def _pulse(t: float) -> float:
        return _on if _ts <= float(t) < _te else _off
    return _pulse


def piecewise(
    t_breakpoints: list[float] | np.ndarray,
    values: list[float] | np.ndarray,
) -> Callable[[float], float]:
    """
    Piecewise-linear interpolation between breakpoints.

    Uses linear interpolation between breakpoints and clamps at the end values
    outside the range.

    Parameters
    ----------
    t_breakpoints : array-like of float   Times [s], strictly increasing.
    values        : array-like of float   Values at each breakpoint.

    Returns
    -------
    callable(t) -> float

    Raises
    ------
    ValueError
        If t_breakpoints and values have different lengths.
    """
    t_arr = np.asarray(t_breakpoints, dtype=float)
    v_arr = np.asarray(values, dtype=float)
    if len(t_arr) != len(v_arr):
        raise ValueError("t_breakpoints and values must have the same length")
    if len(t_arr) < 2:
        raise ValueError("piecewise requires at least 2 breakpoints")

    def _pw(t: float) -> float:
        return float(np.interp(float(t), t_arr, v_arr))
    return _pw


def sine(
    amplitude: float,
    frequency_Hz: float,
    offset: float = 0.0,
    phase_rad: float = 0.0,
) -> Callable[[float], float]:
    """
    Sinusoidal signal: offset + amplitude * sin(2π·f·t + phase).

    Parameters
    ----------
    amplitude    : float   Amplitude [units].
    frequency_Hz : float   Frequency [Hz].
    offset       : float   DC offset (default 0).
    phase_rad    : float   Phase shift [rad] (default 0).

    Returns
    -------
    callable(t) -> float
    """
    _omega = 2.0 * np.pi * float(frequency_Hz)
    _amp   = float(amplitude)
    _off   = float(offset)
    _phi   = float(phase_rad)

    def _sine(t: float) -> float:
        return _off + _amp * np.sin(_omega * float(t) + _phi)
    return _sine


def constant(value: float) -> Callable[[float], float]:
    """
    Wrap a constant as a callable — useful for dynamic composition.

    Parameters
    ----------
    value : float   Constant value to return.

    Returns
    -------
    callable(t) -> float
    """
    _v = float(value)

    def _const(t: float) -> float:
        return _v
    return _const
