"""
@module heater.config.wall_c
@brief Configuración de la pared estructural del calentador (shell-tube mode).

@details
Construye el dict `wall_config` que activa el modelo de pared dinámica
T_w(z, t) en el heater. Cuando `wall_config` está presente en `params`,
el vector de estado se amplía en N grados de libertad adicionales: las
temperaturas de la pared por celda.

Compatibilidad con thermal_bc_config:
    "adiabatic"    — exterior de la pared adiabático (solo acoplamiento gas-pared)
    "heatfluxwall" — potencia total impuesta uniformemente en el exterior de la pared
    "ambient_htc"  — convección exterior + conducción de pared en serie
    "fixed_twall"  — NO compatible con shell-tube (la temperatura interior de la
                     pared es ahora una variable dinámica del ODE)

Parámetros devueltos en wall_config:
    material : dict  — resultado de build_solid_prop_config (rho, cp, k escalares o callables)
    A_w      : float — área de sección transversal de la pared [m²]
    Di       : float — diámetro interior de la pared [m]
    Do       : float — diámetro exterior de la pared [m]
    T_w_init : ndarray(N,) — temperatura inicial de la pared [K]

La conducción axial en la pared siempre está activa (no hay justificación para suprimirla).

Unidades: SI.
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
    @brief
    Construye la configuración de pared estructural para el modelo shell-tube.

    Parameters
    ----------
    N             : int   — número de celdas del dominio axial
    Di            : float — diámetro interior de la pared [m]
    Do            : float — diámetro exterior de la pared [m]
    T_w_init      : scalar o array(N,) — temperatura inicial de la pared [K]
    material_id   : str   — identificador del material en soliddb (p.ej. "SS316L")
    material_mode : str   — modo de evaluación: "fixed", "constant" o "polynomial"
    db_path       : str   — ruta a soliddb.txt
    rho_fixed     : float — densidad fija [kg/m³]   (solo para mode="fixed")
    cp_fixed      : float — calor específico fijo [J/kg/K] (solo para mode="fixed")
    k_fixed       : float — conductividad fija [W/m/K]   (solo para mode="fixed")
    epsilon_wall  : float — emissivity of the inner wall surface [-] (default 0.85).
                    Reserved for future gas-wall or particle-wall radiation terms.
                    Typical values: polished steel ~0.1, oxidised steel ~0.7–0.9.

    Returns
    -------
    dict con claves:
        material     : dict        — build_solid_prop_config output
        A_w          : float       — área de sección de la pared [m²]
        Di           : float       — diámetro interior [m]
        Do           : float       — diámetro exterior [m]
        T_w_init     : ndarray(N,) — temperatura inicial de la pared [K]
        epsilon_wall : float       — inner wall surface emissivity [-]
    """
    nn = int(N)
    Di_f = float(Di)
    Do_f = float(Do)

    if Di_f <= 0.0:
        raise ValueError(f"Di must be > 0, got {Di_f}")
    if Do_f <= Di_f:
        raise ValueError(f"Do must be > Di, got Do={Do_f}, Di={Di_f}")

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

    if not (0.0 < float(epsilon_wall) <= 1.0):
        raise ValueError(f"epsilon_wall must be in (0, 1], got {epsilon_wall}")

    return {
        "material":     material,
        "A_w":          float(A_w),
        "Di":           Di_f,
        "Do":           Do_f,
        "T_w_init":     Tw0,
        "epsilon_wall": float(epsilon_wall),
    }
