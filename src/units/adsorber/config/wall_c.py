"""
@module adsorber.config.wall_c
@brief Wall structural configuration for the adsorber (shell-tube mode).

@details
Builds the `wall_config` dict that activates the dynamic wall model
Tw(z, t) in the adsorber ODE. When `wall_config` is present in `params`,
the state vector grows by N degrees of freedom (wall temperature per cell).

Self-contained implementation — no dependency on heater or other equipment
modules, so the adsorber remains functional if those units are refactored
or removed.

Compatibility with thermal_bc_config:
    "adiabatic"    — adiabatic exterior (only gas-wall coupling is active)
    "heatfluxwall" — total power imposed uniformly on the outer wall surface
    "ambient_htc"  — external convection + radial wall conduction in series
    "fixed_twall"  — NOT compatible with shell-tube (inner wall temperature is
                     now a dynamic ODE variable, not a fixed boundary condition)

Output keys in wall_config:
    material     : dict        — build_solid_prop_config output {rho, cp, k}
    A_w          : float       — wall cross-sectional area [m²]
    Di           : float       — inner diameter [m]
    Do           : float       — outer diameter [m]
    T_w_init     : ndarray(N,) — initial wall temperature [K]
    epsilon_wall : float       — inner wall surface emissivity [-]
                   Reserved for future gas-wall radiation terms.

Axial conduction through the wall is always active.

Units: SI.
"""

from __future__ import annotations

import numpy as np

from src.physics.thermodynamics.solid_props import build_solid_prop_config
from src.utils.profiling import profiled


@profiled
def build_wall_config(
    N: int,
    Di: float,
    Do: float,
    T_w_init,
    material_id: str = "SS316L",
    material_mode: str = "polynomial",
    db_path: str = "materials/solids/soliddb.txt",
    rho_fixed: float = None,
    cp_fixed: float = None,
    k_fixed: float = None,
    epsilon_wall: float = 0.85,
) -> dict:
    """
    Build the structural wall configuration for the adsorber shell-tube model.

    Parameters
    ----------
    N             : int    number of axial cells
    Di            : float  inner wall diameter [m]
    Do            : float  outer wall diameter [m]  (must be > Di)
    T_w_init      : scalar or array(N,)  initial wall temperature [K]
    material_id   : str    material identifier in soliddb (e.g. "SS316L")
    material_mode : str    property evaluation mode: "fixed", "constant" or "polynomial"
    db_path       : str    path to soliddb.txt
    rho_fixed     : float  fixed density [kg/m³]        (mode="fixed" only)
    cp_fixed      : float  fixed heat capacity [J/kg/K] (mode="fixed" only)
    k_fixed       : float  fixed conductivity [W/m/K]   (mode="fixed" only)
    epsilon_wall  : float  inner wall surface emissivity [-] (default 0.85).
                   Reserved for future radiation terms (gas-wall).
                   Typical: polished steel ~0.1, oxidised steel ~0.7–0.9.

    Returns
    -------
    dict with keys:
        material     : dict        build_solid_prop_config output
        A_w          : float       wall cross-sectional area [m²]
        Di           : float       inner diameter [m]
        Do           : float       outer diameter [m]
        T_w_init     : ndarray(N,) initial wall temperature [K]
        epsilon_wall : float       inner wall emissivity [-]
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

    A_w = np.pi / 4.0 * (Do_f**2 - Di_f**2)   # [m²]

    Tw0 = np.asarray(T_w_init, dtype=float).reshape(-1)
    if Tw0.size == 1:
        Tw0 = np.full(nn, float(Tw0[0]), dtype=float)
    if Tw0.size != nn:
        raise ValueError(
            f"T_w_init must be a scalar or have length {nn}, got size={Tw0.size}"
        )
    if not np.all(np.isfinite(Tw0)) or np.any(Tw0 <= 0.0):
        raise ValueError("T_w_init must contain finite values > 0 [K]")

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
