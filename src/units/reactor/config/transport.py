"""
Transport coefficient configuration for the reactor.

Two modes:
    "constant"    — fixed h_bed, h_wall, optional D_disp (pre-tiled arrays)
    "correlation" — Ranz-Marshall (h_bed) + Dittus-Boelter (h_wall) at runtime

h_bed is only meaningful when has_catalyst=True. For empty tubes, h_bed from
the returned dict is unused by the RHS (q_gs = 0 when no catalyst).

Output format aligned with compute_transfer_coefficients() expectations.
"""

from __future__ import annotations

import numpy as np


def build_transport_config(
    mode:         str   = "constant",
    N:            int   = 1,
    n_comp:       int   = 1,
    h_bed:        float = 50.0,           # [W/m²/K]  constant mode
    h_wall:       float = 10.0,           # [W/m²/K]  constant mode
    h_bed_fixed:  float | None = None,    # override correlation h_bed
    h_wall_fixed: float | None = None,    # override correlation h_wall
    D_disp_fixed: float | None = None,    # [m²/s] uniform dispersion (all species)
    Pe_particle:  float = 2.0,            # Bodenstein number D_ax = u·dp/Pe_particle
    Pe_plugflow:  float = 100.0,          # Pe_bed > this → plug flow (D_ax = 0)
    Pe_disponly:  float = 0.1,            # Pe_bed < this → dispersion-only
) -> dict:
    """
    Build transport configuration dict for the reactor.

    Parameters
    ----------
    mode         : {"constant", "correlation"}
    N            : int    number of cells
    n_comp       : int    number of gas species
    h_bed        : float  [W/m²/K]  gas-solid HTC (constant mode)
    h_wall       : float  [W/m²/K]  gas-wall HTC  (constant mode)
    h_bed_fixed  : float or None  overrides correlation h_bed
    h_wall_fixed : float or None  overrides correlation h_wall
    D_disp_fixed : float or None  constant axial dispersion [m²/s]; None = no dispersion
    Pe_particle  : float  Bodenstein number (default 2.0, Edwards & Richardson 1968)
    Pe_plugflow  : float  Pe_bed threshold above which D_ax → 0
    Pe_disponly  : float  Pe_bed threshold below which only dispersion is active

    Returns
    -------
    dict with keys:
        mode, h_bed_fixed, h_wall_fixed,
        h_bed   : ndarray(N,) or None   (pre-tiled in constant mode)
        h_wall  : ndarray(N,) or None
        D_disp_fixed, Pe_particle, Pe_plugflow, Pe_disponly,
        D_disp  : ndarray(n_comp, N) or None
    """
    mode_str = str(mode).strip().lower()
    if mode_str not in {"constant", "correlation"}:
        raise ValueError(
            f"mode must be 'constant' or 'correlation', got '{mode_str}'"
        )
    if h_bed <= 0.0:
        raise ValueError(f"h_bed must be > 0, got {h_bed}")
    if h_wall <= 0.0:
        raise ValueError(f"h_wall must be > 0, got {h_wall}")
    if Pe_particle <= 0.0:
        raise ValueError(f"Pe_particle must be > 0, got {Pe_particle}")
    if Pe_plugflow <= Pe_disponly:
        raise ValueError(
            f"Pe_plugflow ({Pe_plugflow}) must be > Pe_disponly ({Pe_disponly})"
        )

    nc, nn = int(n_comp), int(N)

    D_disp_arr = (np.full((nc, nn), float(D_disp_fixed), dtype=float)
                  if (mode_str == "constant" and D_disp_fixed is not None)
                  else None)

    common = {
        "D_disp_fixed": float(D_disp_fixed) if D_disp_fixed is not None else None,
        "Pe_particle":  float(Pe_particle),
        "Pe_plugflow":  float(Pe_plugflow),
        "Pe_disponly":  float(Pe_disponly),
    }

    if mode_str == "constant":
        return {
            "mode":         mode_str,
            "h_bed_fixed":  None,
            "h_wall_fixed": None,
            "h_bed":        np.full(nn, float(h_bed),  dtype=float),
            "h_wall":       np.full(nn, float(h_wall), dtype=float),
            "D_disp":       D_disp_arr,
            **common,
        }

    return {
        "mode":         mode_str,
        "h_bed_fixed":  float(h_bed_fixed)  if h_bed_fixed  is not None else None,
        "h_wall_fixed": float(h_wall_fixed) if h_wall_fixed is not None else None,
        "h_bed":        None,
        "h_wall":       None,
        "D_disp":       None,
        **common,
    }
