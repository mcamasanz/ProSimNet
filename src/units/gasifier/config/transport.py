"""
@module gasifier.config.transport
@brief Transport coefficient configuration for the gasifier.

Configures heat and mass-dispersion coefficients:
  - h_bed  [W/m²/K]: gas-solid (particle surface) heat transfer
  - h_wall [W/m²/K]: gas-wall (reactor tube) heat transfer
  - D_disp [m²/s]:   axial dispersion of gas species (optional; plug-flow if None)

Axial dispersion is characterised by the axial Péclet number at bed scale:

    Pe_bed = u · L / D_ax

Three regimes (analogous to laminar/turbulent for Nusselt):

    Pe_bed > Pe_plugflow  →  plug flow   (D_ax = 0, convection dominates)
    Pe_disponly ≤ Pe_bed ≤ Pe_plugflow  →  full (convection + dispersion)
    Pe_bed < Pe_disponly  →  dispersion-dominated (diffusion dominates)

D_ax is computed from the Bodenstein correlation:
    D_ax = u · d_p / Pe_particle    (Pe_particle ≈ 2, Edwards & Richardson 1968)

Output dict format is aligned with the adsorber's build_transport_config so that
compute_transfer_coefficients() can be called uniformly from any packed-bed RHS.
"""

import numpy as np


def build_transport_config(
    mode: str = "constant",
    N: int = 1,
    n_comp: int = 9,
    # ── heat transfer ────────────────────────────────────────────────────────
    h_bed: float = 50.0,            # [W/m²/K] gas-particle HTC (constant mode)
    h_wall: float = 10.0,           # [W/m²/K] gas-wall HTC    (constant mode)
    h_bed_fixed: float | None = None,    # overrides correlation h_bed
    h_wall_fixed: float | None = None,   # overrides correlation h_wall
    # ── axial dispersion ─────────────────────────────────────────────────────
    D_disp_fixed: float | None = None,   # [m²/s] constant override (all species)
    Pe_particle: float = 2.0,            # Bodenstein number: D_ax = u·d_p/Pe_particle
    Pe_plugflow: float = 100.0,          # Pe_bed threshold: above → plug flow (D_ax=0)
    Pe_disponly: float = 0.1,            # Pe_bed threshold: below → dispersion only
) -> dict:
    """
    Build transport configuration dict for the gasifier.

    Parameters
    ----------
    mode          : {"constant", "correlation"}
    N             : int   number of cells
    n_comp        : int   number of gas species (used to shape D_disp in constant mode)
    h_bed         : float [W/m²/K] gas-solid HTC (constant mode value)
    h_wall        : float [W/m²/K] gas-wall HTC   (constant mode value)
    h_bed_fixed   : float or None  overrides Ranz-Marshall correlation (correlation mode)
    h_wall_fixed  : float or None  overrides Dittus-Boelter correlation (correlation mode)
    D_disp_fixed  : float or None  constant axial dispersion [m²/s], applied to all
                    species.  None → computed from Bodenstein (correlation mode) or
                    no dispersion (constant mode without explicit value).
    Pe_particle   : float  Bodenstein number for Bodenstein correlation (default 2.0)
    Pe_plugflow   : float  Pe_bed threshold above which D_ax → 0 (default 100.0)
    Pe_disponly   : float  Pe_bed threshold below which only dispersion is active
                    (default 0.1; informational — the RHS still adds both terms)

    Returns
    -------
    dict with keys:
        mode          : str
        h_bed_fixed   : float or None
        h_wall_fixed  : float or None
        h_bed         : ndarray(N,)    constant mode → pre-tiled; correlation → None
        h_wall        : ndarray(N,)    constant mode → pre-tiled; correlation → None
        D_disp_fixed  : float or None  user override value
        Pe_particle   : float
        Pe_plugflow   : float
        Pe_disponly   : float
        D_disp        : ndarray(n_comp, N) or None
                        constant mode with D_disp_fixed → pre-tiled array
                        all other cases → None (computed at runtime)
    """
    mode_str = str(mode).strip().lower()
    if mode_str not in {"constant", "correlation"}:
        raise ValueError(
            f"build_transport_config: mode must be 'constant' or 'correlation', "
            f"got '{mode_str}'"
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

    # D_disp pre-tiled only when constant mode AND D_disp_fixed is given
    if mode_str == "constant" and D_disp_fixed is not None:
        D_disp_arr = np.full((nc, nn), float(D_disp_fixed), dtype=float)
    else:
        D_disp_arr = None

    if mode_str == "constant":
        return {
            "mode":         mode_str,
            "h_bed_fixed":  None,
            "h_wall_fixed": None,
            "h_bed":        np.full(nn, float(h_bed),  dtype=float),
            "h_wall":       np.full(nn, float(h_wall), dtype=float),
            "D_disp_fixed": float(D_disp_fixed) if D_disp_fixed is not None else None,
            "Pe_particle":  float(Pe_particle),
            "Pe_plugflow":  float(Pe_plugflow),
            "Pe_disponly":  float(Pe_disponly),
            "D_disp":       D_disp_arr,
        }

    # correlation mode
    return {
        "mode":         mode_str,
        "h_bed_fixed":  float(h_bed_fixed)  if h_bed_fixed  is not None else None,
        "h_wall_fixed": float(h_wall_fixed) if h_wall_fixed is not None else None,
        "h_bed":        None,
        "h_wall":       None,
        "D_disp_fixed": float(D_disp_fixed) if D_disp_fixed is not None else None,
        "Pe_particle":  float(Pe_particle),
        "Pe_plugflow":  float(Pe_plugflow),
        "Pe_disponly":  float(Pe_disponly),
        "D_disp":       None,
    }
