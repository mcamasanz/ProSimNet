"""
Build boundary condition configuration for the reactor.

The operating mode is determined implicitly from the BC values — there is no
explicit "mode" parameter. The combination of inlet/outlet values defines the
operating regime:

    v_in = None  →  batch     (closed system, no gas flow)
    v_in ≠ None  →  continuous (cstr if N=1 / prf if N>1 — user's choice via N
                                and axial dispersion in trans_config)

This mirrors the gasifier pattern where mode is never an explicit parameter.

`v_in`, `T_in`, `y_in` may be float constants or callable(t) → value to allow
time-varying inlet conditions (ramps, step changes).
"""

from __future__ import annotations

import numpy as np


def build_boundary_c_config(
    n_comp:    int,        # number of gas species
    P_out_bar: float = 1.01325,
    # ── Gas inlet (None = batch / closed system) ──────────────────────────────
    v_in=None,             # float or callable(t) → float [m/s]
    T_in=None,             # float or callable(t) → float [K]
    y_in=None,             # ndarray(nc,) or callable(t) → ndarray(nc,)
) -> dict:
    """
    Build and validate the boundary condition configuration dict.

    The operating regime is inferred from the presence of `v_in`:
        v_in = None  →  batch (no inlet flow; v_in = v_out = 0 at runtime)
        v_in ≠ None  →  continuous (prescribed inlet; outlet from molar continuity)

    Parameters
    ----------
    n_comp    : int
        Number of gas species.
    P_out_bar : float
        Outlet (reference) pressure [bar].
    v_in : float, callable(t) → float, or None
        Superficial gas velocity at inlet [m/s]. None = batch.
    T_in : float, callable(t) → float, or None
        Gas temperature at inlet [K]. Required if v_in is not None.
    y_in : ndarray(nc,), callable(t) → ndarray(nc,), or None
        Molar fractions at inlet [-]. Required if v_in is not None.

    Returns
    -------
    dict with keys: P_out_bar, v_in, T_in, y_in
    """
    if P_out_bar <= 0.0:
        raise ValueError(f"P_out_bar must be > 0, got {P_out_bar}")

    # Batch: inlet parameters deben ser None
    if v_in is None:
        if T_in is not None or y_in is not None:
            raise ValueError(
                "v_in=None (batch) does not use T_in or y_in — pass them as None "
                "or omit them"
            )
        return {
            "P_out_bar": float(P_out_bar),
            "v_in":      None,
            "T_in":      None,
            "y_in":      None,
        }

    # Continuo: inlet obligatorio
    if T_in is None:
        raise ValueError("T_in is required when v_in is not None")
    if y_in is None:
        raise ValueError("y_in is required when v_in is not None")

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
        "P_out_bar": float(P_out_bar),
        "v_in":      v_in,
        "T_in":      T_in,
        "y_in":      y_in,
    }
