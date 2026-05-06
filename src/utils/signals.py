"""
Universal BC signal resolver for ProSimNet.

A "signal" can be any of:

  - float / int        : constant — returned as float, no evaluation needed.
  - callable(t)        : time-dependent — receives simulation time t [s].
  - callable(t, snap)  : state-feedback — receives t and the equipment state snap dict.

The snap dict is built by the RHS each call step from the current unpacked state
(Tg, Ts, P_bar, C, rho_solid, ...).  Three snap sources are supported, in priority order:

    params["_snap_external"]   ← plant coordinator injection  (highest priority)
    params["_snap_runner"]     ← runner injection between n_sec sub-intervals
    cache["_snap"]             ← RHS-internal snap built from the current state (default)

Usage in the RHS::

    from src.utils.signals import resolve, resolve_config_values

    # Resolve a scalar BC parameter at (t, snap):
    T_wall_val = resolve(thermal_bc_cfg["T_wall"], t, snap)

    # Resolve all callable values in a config dict:
    tbc_resolved = resolve_config_values(thermal_bc_cfg, t, snap,
                                         keys=["T_wall", "Qwall", "T_ambi", "h_ambi"])
"""
from __future__ import annotations

import inspect
from typing import Any


def resolve(signal: Any, t: float, snap: dict | None = None) -> Any:
    """
    Evaluate a signal at time t with optional equipment state snap.

    Parameters
    ----------
    signal : float, int, or callable
        The signal to evaluate:
          - float / int         → returned as float (constant)
          - callable(t)         → called with scalar t [s]
          - callable(t, snap)   → called with t and snap dict
    t : float
        Current simulation time [s].
    snap : dict or None
        Equipment state snapshot. Passed to 2-argument callables.
        If None and a 2-arg callable is given, {} is used.

    Returns
    -------
    Any
        Resolved value. For scalar signals this is a float; for array signals
        (e.g. molar fraction vectors) the raw callable return is preserved.

    Raises
    ------
    TypeError
        If signal is not numeric or callable.
    """
    if isinstance(signal, (int, float)):
        return float(signal)
    if callable(signal):
        try:
            n_params = len(inspect.signature(signal).parameters)
        except (ValueError, TypeError):
            n_params = 0
        if n_params >= 2:
            return signal(t, snap if snap is not None else {})
        return signal(t)
    raise TypeError(
        f"signal must be float, int, or callable; got {type(signal).__name__}"
    )


def resolve_config_values(
    cfg: dict,
    t: float,
    snap: dict | None = None,
    keys: list[str] | None = None,
) -> dict:
    """
    Return a copy of cfg where callable values are evaluated at (t, snap).

    Non-callable values pass through unchanged. Useful for resolving a
    thermal_bc_config or bc_config before passing to a function that expects
    plain floats or arrays (e.g. wall_heat_flux).

    Parameters
    ----------
    cfg  : dict           Configuration dict (may contain callable values)
    t    : float          Current time [s]
    snap : dict or None   Equipment state snap (passed to 2-arg callables)
    keys : list or None   If given, only resolve these keys; others pass through.

    Returns
    -------
    dict   Shallow copy of cfg with resolved values.
    """
    resolved = {}
    for k, v in cfg.items():
        if (keys is None or k in keys) and callable(v):
            resolved[k] = resolve(v, t, snap)
        else:
            resolved[k] = v
    return resolved
