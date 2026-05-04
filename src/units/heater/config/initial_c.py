"""
@module heater.config.initial_c
@brief Validación y construcción de las condiciones iniciales del heater.

@details
Recibe los perfiles iniciales (P, Tg, y) del usuario, los valida y calcula
las variables primarias del integrador (C, Hg) usando la ley del gas ideal
y la relación entalpía-temperatura.

No existe Ts (sin sólido) ni q (sin adsorción). La porosidad efectiva es 1.0
(tubo vacío).

Reglas de broadcasting:
    P_init, Tg_init — escalar (→ uniforme en z) o array(N,)
    y_init          — array(nc,) (→ uniforme en z) o array(nc, N)

Unidades:
    P   [bar]
    T   [K]
    y   [-]
    C   [mol/m³]   (derivado de ideal gas)
    Hg  [J/m³]     (derivado de C y Tg)
"""

from __future__ import annotations

import numpy as np

from src.physics.thermodynamics.enthalpy import calc_volumetric_enthalpy
from src.utils.profiling import profiled

R_GAS     = 8.31446261815324   # [J/mol/K]
_EPSI_TUBE = 1.0               # tubo vacío: todo el volumen es gas


@profiled
def build_initial_c_config(
    P_init,
    Tg_init,
    y_init,
    n_comp: int,
    N: int,
    prop_gas: dict,
    gas_T_ref: float,
    wall_config: dict = None,
) -> dict:
    """
    @brief
    Valida las condiciones iniciales del heater, calcula C y Hg y devuelve
    el estado inicial completo.

    Parameters
    ----------
    P_init      : scalar o array(N,)   — presión inicial [bar]
    Tg_init     : scalar o array(N,)   — temperatura inicial del gas [K]
    y_init      : array(nc,) o (nc, N) — fracciones molares iniciales [-]
    n_comp      : int   — número de componentes
    N           : int   — número de celdas
    prop_gas    : dict  — propiedades del gas (de build_gas_prop_config)
    gas_T_ref   : float — temperatura de referencia de entalpías [K]
    wall_config : dict  — resultado de build_wall_config; si se proporciona,
                          se extrae T_w_init y se incluye "Tw_init" en el dict
                          devuelto (requerido cuando shell-tube está activo)

    Returns
    -------
    dict con claves:
        P_init  : ndarray(N,)    [bar]
        Tg_init : ndarray(N,)    [K]
        y_init  : ndarray(nc, N) [-]
        C_init  : ndarray(nc, N) [mol/m³]
        Hg_init : ndarray(N,)    [J/m³]
        Tw_init : ndarray(N,)    [K]  — solo si wall_config no es None
    """
    nc, nn = int(n_comp), int(N)

    def _expand_1d(arr_in, name):
        arr = np.asarray(arr_in, dtype=float).reshape(-1)
        if arr.size == 1:
            return np.full(nn, float(arr[0]), dtype=float)
        if arr.size == nn:
            return arr.copy()
        raise ValueError(f"{name} must be a scalar or have length {nn}")

    P0  = _expand_1d(P_init,  "P_init")
    Tg0 = _expand_1d(Tg_init, "Tg_init")

    # y_init: (nc,) → tile a (nc, N); (nc, N) → copy
    y_arr = np.asarray(y_init, dtype=float)
    if y_arr.ndim == 1:
        if y_arr.size != nc:
            raise ValueError(f"y_init must have length {nc} when given as 1D array")
        y0 = np.tile(y_arr.reshape(nc, 1), (1, nn))
    elif y_arr.ndim == 2:
        if y_arr.shape != (nc, nn):
            raise ValueError(f"y_init must have shape ({nc}, {nn}) when given as 2D array")
        y0 = y_arr.astype(float).copy()
    else:
        raise ValueError("y_init must be a 1D or 2D array")

    # ── Validaciones ─────────────────────────────────────────────────────────
    if not np.all(np.isfinite(P0)) or np.any(P0 <= 0.0):
        raise ValueError("P_init must contain finite values > 0 [bar]")
    if not np.all(np.isfinite(Tg0)) or np.any(Tg0 <= 0.0):
        raise ValueError("Tg_init must contain finite values > 0 [K]")
    if not np.all(np.isfinite(y0)) or np.any(y0 < 0.0):
        raise ValueError("y_init must be finite and >= 0")
    ysum = np.sum(y0, axis=0)
    if np.any(ysum <= 0.0):
        raise ValueError("y_init must have positive sum in every cell")
    if np.any(np.abs(ysum - 1.0) > 1.0e-6):
        raise ValueError("y_init must sum to 1 in every cell within tolerance 1e-6")

    # ── Variables primarias derivadas ─────────────────────────────────────────
    P0_Pa = P0 * 1.0e5
    C0 = y0 * P0_Pa[None, :] / (R_GAS * Tg0[None, :])   # [mol/m³]

    if not np.all(np.isfinite(C0)) or np.any(C0 < 0.0):
        raise ValueError("Computed C_init contains nan/inf or negative values")

    Hg0 = calc_volumetric_enthalpy(
        C=C0, Tg=Tg0, prop_gas=prop_gas,
        n_comp=nc, epsi=_EPSI_TUBE, gas_T_ref=gas_T_ref,
    )

    if not np.all(np.isfinite(Hg0)):
        raise ValueError("Computed Hg_init contains nan or inf")

    result = {
        "P_init":  P0,
        "Tg_init": Tg0,
        "y_init":  y0,
        "C_init":  C0,
        "Hg_init": Hg0,
    }

    if wall_config is not None:
        Tw0 = np.asarray(wall_config["T_w_init"], dtype=float)
        if Tw0.size != nn:
            raise ValueError(
                f"wall_config['T_w_init'] must have length {nn}, got {Tw0.size}"
            )
        result["Tw_init"] = Tw0.copy()

    return result
