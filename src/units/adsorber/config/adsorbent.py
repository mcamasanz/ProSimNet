"""
@module config.adsorbent
@brief Validación y construcción del bloque de configuración del adsorbente.

@details
Centraliza la validación física y estructural de los parámetros del lecho
adsorbente: isoterma, geometría de partícula, propiedades térmicas del sólido,
entalpía de adsorción y parámetros de difusión intracristalina.

La función principal `build_adsorbent_config` devuelve un dict que agrupa
todos los parámetros validados listos para ser consumidos por el modelo.

Unidades: SI.
  D_particle [m]
  rho_s      [kg/m³]
  Cp_solid   [J/kg/K]
  k_solid    [W/m/K]
  dH_adsorption [J/mol] (negativo para exotérmica)
  tau        [-]
  r_pore     [m]
  epsi       [-]
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from src.utils.profiling import profiled


@profiled
def build_adsorbent_config(
    iso_fn: Callable,
    n_comp: int,
    N: int,
    epsi: float = 0.3,
    D_particle: float = 0.01,
    rho_s: float = 1000.0,
    Cp_solid: float = None,
    k_solid: float = None,
    dH_adsorption=None,
    tau: float = None,
    r_pore: float = None,
    P_test_range: Tuple[float, float] = (0.0, 10.0),
    T_test_range: Tuple[float, float] = (273.0, 373.0),
) -> dict:
    """
    @brief
    Valida todos los parámetros del adsorbente y devuelve un dict de configuración.

    @details
    Validaciones realizadas:
    - iso_fn debe ser callable y devolver una lista/tuple/ndarray de nc arrays
      con shape (N,) y valores no negativos finitos.
    - epsi en (0, 1)
    - D_particle, rho_s, Cp_solid, k_solid, tau, r_pore > 0
    - dH_adsorption: array de nc elementos finitos (puede ser negativo)
    - a_surf se deriva geométricamente: 6*(1-eps)/D_p

    Parameters
    ----------
    iso_fn       : callable(P_list, T) -> list[ndarray(N,)]
        Función de isoterma. P_list es una lista de nc arrays shape (N,) con
        las presiones parciales de cada componente [bar]; T es ndarray(N,) [K].
        Devuelve lista de nc arrays [mol/kg].
    n_comp       : int — número de componentes
    N            : int — número de celdas axiales
    epsi         : float — porosidad del lecho [-]
    D_particle   : float — diámetro de partícula [m]
    rho_s        : float — densidad del sólido [kg/m³]
    Cp_solid     : float — calor específico del sólido [J/kg/K]
    k_solid      : float — conductividad térmica del sólido [W/m/K]
    dH_adsorption: array-like, shape (nc,) — entalpía de adsorción [J/mol]
    tau          : float — tortuosidad del lecho [-]
    r_pore       : float — radio de poro [m]
    P_test_range : (float, float) — rango de presión parcial para test [bar]
    T_test_range : (float, float) — rango de temperatura para test [K]

    Returns
    -------
    dict con claves:
        iso       : callable — función de isoterma validada
        epsi      : float
        D_p       : float [m]
        rho_s     : float [kg/m³]
        Cp_s      : float [J/kg/K]
        k_s       : float [W/m/K]
        dH_comp   : ndarray, shape (nc,) [J/mol]
        tau       : float [-]
        r_pore    : float [m]
        a_surf    : float [m²/m³] — derivado: 6*(1-eps)/D_p
        prop_lecho: dict con arrays (N,): eps, D_p, tau, r_pore, a_surf
    """
    nc, nn = int(n_comp), int(N)

    # --- iso_fn ---
    if not callable(iso_fn):
        raise TypeError("iso_fn must be callable")

    # --- epsi ---
    if not np.isscalar(epsi) or not np.isfinite(float(epsi)):
        raise ValueError("epsi must be a finite scalar")
    epsi_val = float(epsi)
    if not (0.0 < epsi_val < 1.0):
        raise ValueError("epsi must satisfy 0 < epsi < 1")

    # --- D_particle ---
    if not np.isscalar(D_particle) or not np.isfinite(float(D_particle)):
        raise ValueError("D_particle must be a finite scalar")
    dp_val = float(D_particle)
    if dp_val <= 0.0:
        raise ValueError("D_particle must be > 0")

    # --- rho_s ---
    if not np.isscalar(rho_s) or not np.isfinite(float(rho_s)):
        raise ValueError("rho_s must be a finite scalar")
    rho_s_val = float(rho_s)
    if rho_s_val <= 0.0:
        raise ValueError("rho_s must be > 0")

    # --- Cp_solid ---
    if Cp_solid is None:
        raise ValueError("Cp_solid must be provided")
    if not np.isscalar(Cp_solid) or not np.isfinite(float(Cp_solid)):
        raise ValueError("Cp_solid must be a finite scalar")
    Cp_s_val = float(Cp_solid)
    if Cp_s_val <= 0.0:
        raise ValueError("Cp_solid must be > 0")

    # --- k_solid ---
    if k_solid is None:
        raise ValueError("k_solid must be provided")
    if not np.isscalar(k_solid) or not np.isfinite(float(k_solid)):
        raise ValueError("k_solid must be a finite scalar")
    k_s_val = float(k_solid)
    if k_s_val <= 0.0:
        raise ValueError("k_solid must be > 0")

    # --- dH_adsorption ---
    if dH_adsorption is None:
        raise ValueError("dH_adsorption must be provided")
    dH_arr = np.asarray(dH_adsorption, dtype=float).reshape(-1)
    if dH_arr.shape != (nc,):
        raise ValueError(f"dH_adsorption must have length {nc}")
    if not np.all(np.isfinite(dH_arr)):
        raise ValueError("dH_adsorption contains nan or inf")

    # --- tau ---
    if tau is None:
        raise ValueError("tau must be provided")
    if not np.isscalar(tau) or not np.isfinite(float(tau)):
        raise ValueError("tau must be a finite scalar")
    tau_val = float(tau)
    if tau_val <= 0.0:
        raise ValueError("tau must be > 0")

    # --- r_pore ---
    if r_pore is None:
        raise ValueError("r_pore must be provided")
    if not np.isscalar(r_pore) or not np.isfinite(float(r_pore)):
        raise ValueError("r_pore must be a finite scalar")
    r_pore_val = float(r_pore)
    if r_pore_val <= 0.0:
        raise ValueError("r_pore must be > 0")

    # --- rangos de test ---
    if len(P_test_range) != 2:
        raise ValueError("P_test_range must have length 2")
    if len(T_test_range) != 2:
        raise ValueError("T_test_range must have length 2")
    P_min, P_max = float(P_test_range[0]), float(P_test_range[1])
    T_min, T_max = float(T_test_range[0]), float(T_test_range[1])
    if not np.isfinite(P_min) or not np.isfinite(P_max):
        raise ValueError("P_test_range must contain finite values")
    if not np.isfinite(T_min) or not np.isfinite(T_max):
        raise ValueError("T_test_range must contain finite values")
    if P_min < 0.0 or P_max < P_min:
        raise ValueError("P_test_range must satisfy 0 <= P_min <= P_max")
    if T_min <= 0.0 or T_max < T_min:
        raise ValueError("T_test_range must satisfy 0 < T_min <= T_max")

    # --- test funcional de la isoterma ---
    T_test = np.linspace(T_min, T_max, nn, dtype=float)
    if nn == 1:
        P_base = np.array([0.5 * (P_min + P_max)], dtype=float)
    else:
        P_base = np.linspace(P_min, P_max, nn, dtype=float)

    P_test_list = []
    for comp_idx in range(nc):
        shift = comp_idx % nn
        P_test_list.append(np.roll(P_base, shift).astype(float))

    iso_test = iso_fn(P_test_list, T_test)

    if not isinstance(iso_test, (list, tuple, np.ndarray)):
        raise TypeError("iso_fn output must be list, tuple or ndarray")
    if len(iso_test) != nc:
        raise ValueError(f"iso_fn output must contain {nc} arrays")
    for i, q_comp in enumerate(iso_test):
        q_arr = np.asarray(q_comp, dtype=float).reshape(-1)
        if q_arr.shape != (nn,):
            raise ValueError(f"iso_fn output for component {i} must have shape ({nn},)")
        if not np.all(np.isfinite(q_arr)):
            raise ValueError(f"iso_fn output for component {i} contains nan or inf")
        if np.any(q_arr < 0.0):
            raise ValueError(f"iso_fn output for component {i} contains negative values")

    # --- a_surf derivado ---
    a_surf_val = 6.0 * (1.0 - epsi_val) / dp_val   # [m²/m³]

    prop_lecho = {
        "eps":    epsi_val  * np.ones(nn, dtype=float),
        "D_p":    dp_val    * np.ones(nn, dtype=float),
        "tau":    tau_val   * np.ones(nn, dtype=float),
        "r_pore": r_pore_val * np.ones(nn, dtype=float),
        "a_surf": a_surf_val * np.ones(nn, dtype=float),
    }

    return {
        "iso":       iso_fn,
        "epsi":      epsi_val,
        "D_p":       dp_val,
        "rho_s":     rho_s_val,
        "Cp_s":      Cp_s_val,
        "k_s":       k_s_val,
        "dH_comp":   np.tile(dH_arr.reshape(-1, 1), (1, nn)),   # (nc, N) — uniforme axialmente
        "tau":       tau_val,
        "r_pore":    r_pore_val,
        "a_surf":    a_surf_val,
        "prop_lecho": prop_lecho,
    }
