"""
Build boundary condition configuration for the reactor.

The operating mode is set explicitly via the `mode` parameter:

    "batch" — closed system, no gas flow (v_in = v_out = 0)
    "cstr"  — continuous flow, perfect mixing (N=1, prescribed inlet, free outlet)
    "prf"   — continuous flow, plug flow      (N>1, prescribed inlet, free outlet)

For "cstr" and "prf" the boundary mathematics are identical: prescribed inlet and
molar-continuity outlet. The behavioral difference (well-mixed vs. gradients) comes
from the number of cells N and the axial dispersion coefficient in transport_config,
not from this BC config.

`v_in`, `T_in`, `y_in` may be float constants or callable(t) → value to allow
time-varying inlet conditions (ramps, step changes).
"""

from __future__ import annotations

import numpy as np


def build_boundary_c_config(
    mode:      str,        # "batch" | "cstr" | "prf"
    n_comp:    int,        # number of gas species
    P_out_bar: float = 1.01325,
    # ── Gas inlet (required for cstr / prf, unused for batch) ────────────────
    v_in=None,             # float or callable(t) → float [m/s]
    T_in=None,             # float or callable(t) → float [K]
    y_in=None,             # ndarray(nc,) or callable(t) → ndarray(nc,)
) -> dict:
    """
    Build and validate the boundary condition configuration dict.

    Parameters
    ----------
    mode      : {"batch", "cstr", "prf"}
        Operating mode. Determines which BC logic is applied in
        get_reactor_boundary().
    n_comp    : int
        Number of gas species.
    P_out_bar : float
        Outlet (reference) pressure [bar]. Used for P_out in all modes.
    v_in : float, callable(t) → float, or None
        Superficial gas velocity at inlet [m/s]. Required for "cstr" and "prf".
    T_in : float, callable(t) → float, or None
        Gas temperature at inlet [K]. Required for "cstr" and "prf".
    y_in : ndarray(nc,), callable(t) → ndarray(nc,), or None
        Molar fractions at inlet [-]. Required for "cstr" and "prf".

    Returns
    -------
    dict with keys: mode, P_out_bar, v_in, T_in, y_in
    """
    mode_str = str(mode).strip().lower()
    if mode_str not in {"batch", "cstr", "prf"}:
        raise ValueError(
            f"mode must be 'batch', 'cstr', or 'prf'; got '{mode_str}'"
        )

    if P_out_bar <= 0.0:
        raise ValueError(f"P_out_bar must be > 0, got {P_out_bar}")

    # batch: inlet parameters deben estar ausentes (o None)
    if mode_str == "batch":
        if v_in is not None or T_in is not None or y_in is not None:
            raise ValueError(
                "mode='batch' does not use v_in, T_in or y_in — pass them as None "
                "or omit them"
            )
        return {
            "mode":      mode_str,
            "P_out_bar": float(P_out_bar),
            "v_in":      None,
            "T_in":      None,
            "y_in":      None,
        }

    # cstr / prf: inlet parameters obligatorios
    if v_in is None:
        raise ValueError(f"v_in is required for mode='{mode_str}'")
    if T_in is None:
        raise ValueError(f"T_in is required for mode='{mode_str}'")
    if y_in is None:
        raise ValueError(f"y_in is required for mode='{mode_str}'")

    if not callable(v_in) and float(v_in) < 0.0:
        raise ValueError(f"v_in must be >= 0, got {v_in}")

    if not callable(T_in) and float(T_in) <= 0.0:
        raise ValueError(f"T_in must be > 0 [K], got {T_in}")

    if not callable(y_in):
        y_arr = np.asarray(y_in, dtype=float).reshape(-1)
        if y_arr.shape != (n_comp,):
            raise ValueError(
                f"y_in must have length n_comp={n_comp}, got shape {y_arr.shape}"
            )
        if abs(float(np.sum(y_arr)) - 1.0) > 1.0e-4:
            raise ValueError(
                f"y_in must sum to 1; got {float(np.sum(y_arr)):.6f}"
            )
        if np.any(y_arr < -1.0e-12):
            raise ValueError("y_in values must be >= 0")
        y_in = y_arr

    return {
        "mode":      mode_str,
        "P_out_bar": float(P_out_bar),
        "v_in":      v_in,
        "T_in":      T_in,
        "y_in":      y_in,
    }
