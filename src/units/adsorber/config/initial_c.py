"""
@module config.initial_c
@brief Validación y construcción del bloque de condiciones iniciales de la columna.

@details
Recibe los perfiles iniciales (P, Tg, Ts, y, q), los valida, calcula las variables
derivadas (C, Hg) y devuelve un dict de estado inicial completo listo para llamar
a `init_column_state`.

Reglas de broadcasting:
    P_init, Tg_init, Ts_init — escalar (→ uniforme) o array(N,)
    y_init                   — array(nc,) (→ uniforme en z) o array(nc, N)
    q_init                   — None (→ equilibrio isotérmico) o array(nc,) o array(nc, N)

Unidades:
    P   [bar]
    T   [K]
    y   [-]
    q   [mol/kg]
    C   [mol/m³]   (derivado)
    Hg  [J/m³]     (derivado)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.physics.thermodynamics.enthalpy import calc_volumetric_enthalpy
from src.utils.profiling import profiled

R_GAS = 8.31446261815324   # [J/mol/K]


@profiled
def build_initial_c_config(
    P_init,
    Tg_init,
    Ts_init,
    y_init,
    n_comp: int,
    N: int,
    iso_fn,
    prop_gas: dict,
    epsi: float,
    gas_T_ref: float,
    q_init=None,
) -> dict:
    """
    @brief
    Valida las condiciones iniciales, calcula C y Hg y devuelve el estado inicial.

    Parameters
    ----------
    P_init    : scalar o array(N,)    — presión inicial [bar]
    Tg_init   : scalar o array(N,)    — temperatura inicial del gas [K]
    Ts_init   : scalar o array(N,)    — temperatura inicial del sólido [K]
    y_init    : array(nc,) o (nc, N)  — fracciones molares iniciales [-]
    n_comp    : int — número de componentes
    N         : int — número de celdas
    iso_fn    : callable — función de isoterma validada (de build_adsorbent_config)
    prop_gas  : dict — propiedades del gas (de build_gas_prop_config)
    epsi      : float — porosidad del lecho [-]
    gas_T_ref : float — temperatura de referencia de entalpías [K]
    q_init    : None o array(nc,) o array(nc, N) — carga adsorbida inicial [mol/kg].
                Si None se calcula el equilibrio isotérmico.

    Returns
    -------
    dict con claves:
        P_init, Tg_init, Ts_init, y_init, q_init  — arrays (N,) o (nc, N)
        C_init  : ndarray(nc, N) [mol/m³]
        Hg_init : ndarray(N,)    [J/m³]
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
    Ts0 = _expand_1d(Ts_init, "Ts_init")

    # y_init: (nc,) → tile, (nc, N) → copy
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

    # --- validaciones ---
    if not np.all(np.isfinite(P0)) or np.any(P0 <= 0.0):
        raise ValueError("P_init must contain finite values > 0 [bar]")
    if not np.all(np.isfinite(Tg0)) or np.any(Tg0 <= 0.0):
        raise ValueError("Tg_init must contain finite values > 0 [K]")
    if not np.all(np.isfinite(Ts0)) or np.any(Ts0 <= 0.0):
        raise ValueError("Ts_init must contain finite values > 0 [K]")
    if not np.all(np.isfinite(y0)) or np.any(y0 < 0.0):
        raise ValueError("y_init must be finite and >= 0")
    ysum = np.sum(y0, axis=0)
    if np.any(ysum <= 0.0):
        raise ValueError("y_init must have positive sum in every cell")
    if np.any(np.abs(ysum - 1.0) > 1.0e-6):
        raise ValueError("y_init must sum to 1 in every cell within tolerance 1e-6")

    # --- q_init: equilibrio o suministrado ---
    if q_init is None:
        P_part_list = [y0[i, :] * P0 for i in range(nc)]   # presiones parciales [bar]
        iso_result = iso_fn(P_part_list, Tg0)
        q0 = np.vstack([
            np.asarray(q_comp, dtype=float).reshape(-1) for q_comp in iso_result
        ])
    else:
        q_arr = np.asarray(q_init, dtype=float)
        if q_arr.ndim == 1:
            if q_arr.size != nc:
                raise ValueError(f"q_init must have length {nc} when given as 1D array")
            q0 = np.tile(q_arr.reshape(nc, 1), (1, nn))
        elif q_arr.ndim == 2:
            if q_arr.shape != (nc, nn):
                raise ValueError(f"q_init must have shape ({nc}, {nn}) when given as 2D array")
            q0 = q_arr.astype(float).copy()
        else:
            raise ValueError("q_init must be None, 1D or 2D array")

    if q0.shape != (nc, nn):
        raise ValueError(f"q_init must have shape ({nc}, {nn})")
    if not np.all(np.isfinite(q0)) or np.any(q0 < 0.0):
        raise ValueError("q_init must contain finite values >= 0")

    # --- C y Hg derivados ---
    P0_Pa = P0 * 1.0e5
    C0 = y0 * P0_Pa[None, :] / (R_GAS * Tg0[None, :])   # [mol/m³]

    if not np.all(np.isfinite(C0)) or np.any(C0 < 0.0):
        raise ValueError("Computed initial concentrations contain nan/inf or negative values")

    Hg0 = calc_volumetric_enthalpy(
        C=C0, Tg=Tg0, prop_gas=prop_gas,
        n_comp=nc, epsi=epsi, gas_T_ref=gas_T_ref,
    )

    if not np.all(np.isfinite(Hg0)):
        raise ValueError("Computed Hg_init contains nan or inf")

    return {
        "P_init":  P0,
        "Tg_init": Tg0,
        "Ts_init": Ts0,
        "y_init":  y0,
        "q_init":  q0,
        "C_init":  C0,
        "Hg_init": Hg0,
    }
