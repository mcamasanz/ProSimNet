"""
@module config.transport
@brief Validación y construcción del bloque de configuración de transporte.

@details
Dos modos:
    "constant"    — el usuario suministra k_mass_transfer [1/s], D_dispersion [m²/s],
                    h_bed_transfer [W/m²/K] y h_wall_transfer [W/m²/K] directamente.
                    Todos son obligatorios.
    "correlation" — los coeficientes se calculan en tiempo de ejecución a partir
                    de correlaciones dimensionless (Re, Sc, Pr, Sh, Nu). En este
                    modo los parámetros fijos son opcionales (pueden suministrarse
                    para sobreescribir la correlación si se desea).

Unidades: SI.
  k_mass_transfer  [1/s]     — LDF overall mass transfer coefficient
  D_dispersion     [m²/s]    — dispersión axial por componente
  h_bed_transfer   [W/m²/K]  — htc gas-sólido
  h_wall_transfer  [W/m²/K]  — htc gas-pared interna
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.utils.profiling import profiled


@profiled
def build_transport_config(
    mode: str,
    n_comp: int,
    N: int,
    k_mass_transfer=None,
    D_dispersion=None,
    h_bed_transfer: Optional[float] = None,
    h_wall_transfer: Optional[float] = None,
    Pe_particle: float = 2.0,
    Pe_plugflow: float = 100.0,
    Pe_disponly: float = 0.1,
) -> dict:
    """
    @brief
    Valida los parámetros de transporte y devuelve un dict de configuración.

    Parameters
    ----------
    mode            : str — "constant" o "correlation"
    n_comp          : int — número de componentes
    N               : int — número de celdas
    k_mass_transfer : array-like shape (nc,) [1/s], obligatorio en "constant"
    D_dispersion    : array-like shape (nc,) [m²/s], obligatorio en "constant"
    h_bed_transfer  : float [W/m²/K], obligatorio en "constant"
    h_wall_transfer : float [W/m²/K], obligatorio en "constant"
    Pe_particle     : float  Bodenstein: D_ax = u·d_p/Pe_particle (default 2.0)
    Pe_plugflow     : float  Pe_bed threshold above which D_ax→0 (default 100.0)
    Pe_disponly     : float  Pe_bed threshold below which convection negligible (default 0.1)

    Returns
    -------
    dict con claves:
        mode            : str
        k_mtc_fixed     : ndarray(nc,) o None
        D_disp_fixed    : ndarray(nc,) o None
        h_bed_fixed     : float o None
        h_wall_fixed    : float o None
        k_mtc           : ndarray(nc, N) o None  (solo en "constant")
        D_disp          : ndarray(nc, N) o None  (solo en "constant")
        h_bed           : ndarray(N,) o None     (solo en "constant")
        h_wall          : ndarray(N,) o None     (solo en "constant")
        Pe_particle     : float
        Pe_plugflow     : float
        Pe_disponly     : float
    """
    nc, nn = int(n_comp), int(N)
    mode_str = str(mode).strip().lower()

    if mode_str not in ("constant", "correlation"):
        raise ValueError("mode must be 'constant' or 'correlation'")

    k_mtc_arr  = None if k_mass_transfer is None else np.asarray(k_mass_transfer, dtype=float).reshape(-1)
    D_disp_arr = None if D_dispersion    is None else np.asarray(D_dispersion,    dtype=float).reshape(-1)
    h_bed_val  = None if h_bed_transfer  is None else float(h_bed_transfer)
    h_wall_val = None if h_wall_transfer is None else float(h_wall_transfer)

    if mode_str == "constant":
        # Todos obligatorios en constant
        if k_mtc_arr is None:
            raise ValueError("k_mass_transfer must be provided when mode='constant'")
        if D_disp_arr is None:
            raise ValueError("D_dispersion must be provided when mode='constant'")
        if h_bed_val is None:
            raise ValueError("h_bed_transfer must be provided when mode='constant'")
        if h_wall_val is None:
            raise ValueError("h_wall_transfer must be provided when mode='constant'")

    # Validar k_mtc si está presente
    if k_mtc_arr is not None:
        if k_mtc_arr.shape != (nc,):
            raise ValueError(f"k_mass_transfer must have length {nc}")
        if not np.all(np.isfinite(k_mtc_arr)) or np.any(k_mtc_arr <= 0.0):
            raise ValueError("k_mass_transfer must contain finite values > 0")

    # Validar D_disp si está presente
    if D_disp_arr is not None:
        if D_disp_arr.shape != (nc,):
            raise ValueError(f"D_dispersion must have length {nc}")
        if not np.all(np.isfinite(D_disp_arr)) or np.any(D_disp_arr < 0.0):
            raise ValueError("D_dispersion must contain finite values >= 0")

    # Validar h_bed / h_wall si presentes
    if h_bed_val is not None:
        if not np.isfinite(h_bed_val) or h_bed_val <= 0.0:
            raise ValueError("h_bed_transfer must be a finite scalar > 0")
    if h_wall_val is not None:
        if not np.isfinite(h_wall_val) or h_wall_val <= 0.0:
            raise ValueError("h_wall_transfer must be a finite scalar > 0")

    # Construir fields extendidos al dominio (nc x N) en modo constant
    if mode_str == "constant":
        k_mtc_2d  = np.tile(k_mtc_arr.reshape(nc, 1),  (1, nn))
        D_disp_2d = np.tile(D_disp_arr.reshape(nc, 1), (1, nn))
        h_bed_1d  = h_bed_val  * np.ones(nn, dtype=float)
        h_wall_1d = h_wall_val * np.ones(nn, dtype=float)
    else:
        k_mtc_2d  = None
        D_disp_2d = None
        h_bed_1d  = None
        h_wall_1d = None

    return {
        "mode":          mode_str,
        "k_mtc_fixed":   None if k_mtc_arr  is None else k_mtc_arr.copy(),
        "D_disp_fixed":  None if D_disp_arr is None else D_disp_arr.copy(),
        "h_bed_fixed":   h_bed_val,
        "h_wall_fixed":  h_wall_val,
        "k_mtc":         k_mtc_2d,
        "D_disp":        D_disp_2d,
        "h_bed":         h_bed_1d,
        "h_wall":        h_wall_1d,
        "Pe_particle":   float(Pe_particle),
        "Pe_plugflow":   float(Pe_plugflow),
        "Pe_disponly":   float(Pe_disponly),
    }
