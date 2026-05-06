"""
Feedback and feedforward controllers for ProSimNet BC signals.

Controllers return callables compatible with ``resolve(signal, t, snap)``,
specifically of the form ``callable(t, snap) -> float``.  They can be used
directly as BC parameter values in ``bc_config`` or ``thermal_bc_config``.

All controllers here are **stateless** (no integral accumulation), which makes
them safe to call inside the RHS — including during Jacobian estimation calls
where the same t appears multiple times.  For PID with integral action, the
accumulation must happen in the runner between n_sec sub-intervals.

Example::

    from src.control.controllers import proportional, onoff

    # Maintain Ts_mean at 900 K by adjusting T_wall
    tbc_cfg["T_wall"] = proportional(
        setpoint=900.0,
        gain=5.0,
        channel_in=lambda snap: snap.get("Ts_mean", 900.0),
        output_bias=1073.15,
        output_min=700.0,
        output_max=1300.0,
    )

    # Switch air flow on/off based on char bed temperature
    bc_cfg["v_gas_in"] = onoff(
        setpoint=850.0, band=50.0,
        channel_in=lambda snap: snap.get("Ts_mean", 800.0),
        output_on=0.05, output_off=0.0,
    )

The snap dict keys available for the gasifier are:
    "t"          float       current time [s]
    "Tg"         ndarray(N,) gas temperature [K]
    "Ts"         ndarray(N,) solid temperature [K]
    "Tg_out"     float       outlet gas temperature [K]
    "Ts_mean"    float       volume-averaged solid temperature [K]
    "P_bar"      ndarray(N,) pressure per cell [bar]
    "P_out_bar"  float       outlet pressure [bar]
    "C"          ndarray(nc,N) concentrations [mol/m³_gas]
    "rho_solid"  ndarray(3,N) solid densities [kg/m³_bed]
    "rho_bio"    ndarray(N,) biomass density [kg/m³_bed]
    "rho_char"   ndarray(N,) char density [kg/m³_bed]
    "rho_moi"    ndarray(N,) moisture density [kg/m³_bed]
"""
from __future__ import annotations

from typing import Any, Callable


def onoff(
    setpoint: float,
    band: float,
    channel_in: Callable[[dict], float],
    output_on: float,
    output_off: float,
) -> Callable[[float, dict], float]:
    """
    On-off controller with dead-band hysteresis.

    Inside the band, output is linearly interpolated to avoid chattering.
    This is stateless — no history of the previous output is kept.

    Parameters
    ----------
    setpoint   : float                    Target process variable value.
    band       : float                    Hysteresis band width (symmetric around setpoint).
    channel_in : callable(snap) -> float  Extracts the process variable from snap.
    output_on  : float                    Output when PV < setpoint - band/2.
    output_off : float                    Output when PV > setpoint + band/2.

    Returns
    -------
    callable(t, snap) -> float
    """
    _sp   = float(setpoint)
    _half = float(band) / 2.0
    _on   = float(output_on)
    _off  = float(output_off)

    def _ctrl(t: float, snap: dict) -> float:
        pv = float(channel_in(snap))
        if pv < _sp - _half:
            return _on
        if pv > _sp + _half:
            return _off
        frac = (pv - (_sp - _half)) / max(float(band), 1.0e-12)
        return _on + frac * (_off - _on)
    return _ctrl


def proportional(
    setpoint: float,
    gain: float,
    channel_in: Callable[[dict], float],
    output_bias: float,
    output_min: float | None = None,
    output_max: float | None = None,
) -> Callable[[float, dict], float]:
    """
    Proportional controller: output = bias + gain × (setpoint − PV).

    Parameters
    ----------
    setpoint    : float                    Target process variable value.
    gain        : float                    Controller gain [output_units / PV_units].
    channel_in  : callable(snap) -> float  Extracts the process variable from snap.
    output_bias : float                    Output at zero error (nominal operating point).
    output_min  : float or None            Output clamp floor (optional).
    output_max  : float or None            Output clamp ceiling (optional).

    Returns
    -------
    callable(t, snap) -> float
    """
    _sp   = float(setpoint)
    _gain = float(gain)
    _bias = float(output_bias)
    _lo   = float(output_min) if output_min is not None else None
    _hi   = float(output_max) if output_max is not None else None

    def _ctrl(t: float, snap: dict) -> float:
        pv    = float(channel_in(snap))
        out   = _bias + _gain * (_sp - pv)
        if _lo is not None:
            out = max(out, _lo)
        if _hi is not None:
            out = min(out, _hi)
        return out
    return _ctrl


def feedforward(
    base_signal: Any,
    ff_channel: Callable[[dict], float],
    ff_gain: float = 1.0,
) -> Callable[[float, dict], float]:
    """
    Feedforward: output = base_signal(t) + ff_gain × ff_channel(snap).

    base_signal can be a float constant or any callable(t) signal primitive.

    Parameters
    ----------
    base_signal : float or callable(t)       Nominal output (e.g. a ramp or step).
    ff_channel  : callable(snap) -> float    Feedforward input from equipment state.
    ff_gain     : float                      Feedforward gain (default 1.0).

    Returns
    -------
    callable(t, snap) -> float
    """
    from src.utils.signals import resolve as _resolve
    _gain = float(ff_gain)

    def _ctrl(t: float, snap: dict) -> float:
        base = _resolve(base_signal, t, snap)
        return float(base) + _gain * float(ff_channel(snap))
    return _ctrl
