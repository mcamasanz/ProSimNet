"""
@module heater.config.transport
@brief Validación y construcción del bloque de configuración de transporte del heater.

@details
El heater es un tubo vacío: no hay lecho empaquetado, no hay adsorción,
no hay transferencia de masa. El único coeficiente de transporte relevante
es el coeficiente de transferencia de calor pared-gas (h_wall).

Dos modos:
    "constant"    — el usuario suministra h_wall_transfer [W/m²/K] directamente.
                    El valor se extiende uniformemente a todas las N celdas.
    "correlation" — h_wall se calcula en tiempo de ejecución mediante la
                    correlación de Dittus-Boelter / Nu laminar (nusselt.h_wall_tube).
                    En este modo h_wall_transfer es opcional: si se suministra,
                    sobreescribe la correlación (valor fijo).

Unidades: SI.
  h_wall_transfer  [W/m²/K]  — htc convectivo gas-pared interna
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.utils.profiling import profiled


@profiled
def build_transport_config(
    mode: str,
    N: int,
    h_wall_transfer: Optional[float] = None,
) -> dict:
    """
    @brief
    Valida los parámetros de transporte del heater y devuelve un dict de configuración.

    Parameters
    ----------
    mode             : str — "constant" o "correlation"
    N                : int — número de celdas
    h_wall_transfer  : float, opcional — htc gas-pared [W/m²/K]
                       Obligatorio en modo "constant".
                       En modo "correlation" sobreescribe la correlación si se suministra.

    Returns
    -------
    dict con claves:
        mode         : str
        h_wall_fixed : float o None
        h_wall       : ndarray(N,) o None  (solo en "constant")
    """
    nn = int(N)
    mode_str = str(mode).strip().lower()

    if mode_str not in ("constant", "correlation"):
        raise ValueError("mode must be 'constant' or 'correlation'")

    h_wall_val = None if h_wall_transfer is None else float(h_wall_transfer)

    if mode_str == "constant":
        if h_wall_val is None:
            raise ValueError("h_wall_transfer must be provided when mode='constant'")

    if h_wall_val is not None:
        if not np.isfinite(h_wall_val) or h_wall_val <= 0.0:
            raise ValueError("h_wall_transfer must be a finite scalar > 0 [W/m²/K]")

    h_wall_1d = (h_wall_val * np.ones(nn, dtype=float)
                 if mode_str == "constant" else None)

    return {
        "mode":          mode_str,
        "h_wall_fixed":  h_wall_val,
        "h_wall":        h_wall_1d,
    }
