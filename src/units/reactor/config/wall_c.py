"""
Wall structural configuration for the reactor (shell-tube mode).

Re-export of the gasifier's wall_config pattern. Activates the dynamic wall
ODE Tw(z,t) when present in params; adds N degrees of freedom to the state vector.
"""

from __future__ import annotations

import numpy as np

from src.physics.thermodynamics.solid_props import build_solid_prop_config
from src.utils.profiling import profiled


@profiled
def build_wall_config(
    N:             int,
    Di:            float,
    Do:            float,
    T_w_init,
    material_id:   str   = "SS316L",
    material_mode: str   = "polynomial",
    db_path:       str   = "materials/solids/soliddb.txt",
    rho_fixed:     float = None,
    cp_fixed:      float = None,
    k_fixed:       float = None,
    epsilon_wall:  float = 0.85,
) -> dict:
    """
    Build the structural wall configuration for the reactor shell-tube model.

    Parameters
    ----------
    N             : int    number of axial cells
    Di            : float  inner wall diameter [m]
    Do            : float  outer wall diameter [m]  (must be > Di)
    T_w_init      : scalar or array(N,)  initial wall temperature [K]
    material_id   : str    material identifier in soliddb (e.g. "SS316L", "Inconel625")
    material_mode : str    "polynomial" | "constant" | "fixed"
    db_path       : str    path to soliddb.txt
    rho_fixed     : float  density [kg/m³]        (mode="fixed" only)
    cp_fixed      : float  heat capacity [J/kg/K]  (mode="fixed" only)
    k_fixed       : float  conductivity [W/m/K]    (mode="fixed" only)
    epsilon_wall  : float  inner wall surface emissivity [-]

    Returns
    -------
    dict with keys: material, A_w, Di, Do, T_w_init, epsilon_wall
    """
    nn   = int(N)
    Di_f = float(Di)
    Do_f = float(Do)

    if Di_f <= 0.0:
        raise ValueError(f"Di must be > 0, got {Di_f}")
    if Do_f <= Di_f:
        raise ValueError(f"Do must be > Di, got Do={Do_f}, Di={Di_f}")
    if not (0.0 < float(epsilon_wall) <= 1.0):
        raise ValueError(f"epsilon_wall must be in (0, 1], got {epsilon_wall}")

    A_w = np.pi / 4.0 * (Do_f**2 - Di_f**2)

    Tw0 = np.asarray(T_w_init, dtype=float).reshape(-1)
    if Tw0.size == 1:
        Tw0 = np.full(nn, float(Tw0[0]))
    if Tw0.size != nn:
        raise ValueError(f"T_w_init must be scalar or length {nn}, got {Tw0.size}")
    if not np.all(np.isfinite(Tw0)) or np.any(Tw0 <= 0.0):
        raise ValueError("T_w_init must be finite and > 0 [K]")

    material = build_solid_prop_config(
        material_id=material_id,
        mode=material_mode,
        db_path=db_path,
        rho_fixed=rho_fixed,
        cp_fixed=cp_fixed,
        k_fixed=k_fixed,
    )

    return {
        "material":     material,
        "A_w":          float(A_w),
        "Di":           Di_f,
        "Do":           Do_f,
        "T_w_init":     Tw0,
        "epsilon_wall": float(epsilon_wall),
    }
