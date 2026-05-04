"""
@module state
@brief Gestión del vector de estado del modelo de calentador de gas 1D.

@details
Layout del vector de estado del integrador ODE:

    Sin shell-tube (wall_config ausente o None):
        sv = [C.reshape(-1), Hg]
        shape: (nc + 1) * N

    Con shell-tube (wall_config presente):
        sv = [C.reshape(-1), Hg, Tw]
        shape: (nc + 2) * N

donde:
    C   : concentraciones molares de las nc especies en N celdas  [mol/m³], shape (nc, N)
    Hg  : entalpía volumétrica del gas en N celdas                [J/m³],  shape (N,)
    Tw  : temperatura de la pared por celda (solo shell-tube)     [K],     shape (N,)

No existe q (sin adsorción) ni Ts (sin sólido). El heater es un tubo vacío donde
el gas se calienta mediante la pared. La porosidad efectiva es 1.0 (todo el volumen
de celda es espacio de gas).

Este módulo expone:
    pack_state_vector      — empaqueta C, Hg [y Tw] en el vector 1D del integrador
    unpack_state_vector    — desempaqueta y deriva las variables secundarias (Tg, P, y [, Tw])
    set_state              — construye el estado completo desde (P, Tg, y) del usuario
    init_heater_state      — prepara el estado inicial para el runner [con Tw opcional]
    build_sv0_from_results — extrae sv0 del último instante de un objeto heater (continuación)

Unidades: SI (P se almacena en bar para compatibilidad con la ley de gas ideal).
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from src.physics.thermodynamics.enthalpy import (
    calc_volumetric_enthalpy,
    recover_Tg_from_Hg,
)
from src.utils.profiling import profiled

R_GAS = 8.31446261815324   # [J/mol/K]

# Porosidad efectiva del tubo vacío: todo el volumen es gas
_EPSI_TUBE = 1.0


# =========================================================
# 1. Empaquetado / desempaquetado del vector de estado
# =========================================================

@profiled
def pack_state_vector(
    C: np.ndarray,
    Hg: np.ndarray,
    Tw: np.ndarray = None,
) -> np.ndarray:
    """
    @brief
    Empaqueta C, Hg (y opcionalmente Tw) en el vector 1D del integrador ODE.

    @details
    Layout sin shell-tube : [C.reshape(-1), Hg]          tamaño (nc + 1) * N
    Layout con shell-tube : [C.reshape(-1), Hg, Tw]       tamaño (nc + 2) * N

    Parameters
    ----------
    C  : np.ndarray, shape (nc, N) — concentraciones molares [mol/m³]
    Hg : np.ndarray, shape (N,)   — entalpía volumétrica del gas [J/m³]
    Tw : np.ndarray, shape (N,)   — temperatura de la pared [K]; None si no hay pared dinámica

    Returns
    -------
    sv : np.ndarray, shape ((nc + 1) * N,) o ((nc + 2) * N,)
    """
    parts = [
        np.asarray(C,  dtype=float).reshape(-1),
        np.asarray(Hg, dtype=float).reshape(-1),
    ]
    if Tw is not None:
        parts.append(np.asarray(Tw, dtype=float).reshape(-1))
    return np.concatenate(parts)


@profiled
def unpack_state_vector(
    sv: np.ndarray,
    n_comp: int,
    N: int,
    prop_gas: dict,
    Tg_guess: np.ndarray,
    gas_T_ref: float,
    newton_tol: float = 1.0e-8,
    newton_max_iter: int = 30,
    shell_tube: bool = False,
) -> Dict[str, Any]:
    """
    @brief
    Desempaqueta el vector de estado 1D y deriva las variables secundarias.

    @details
    Variables primarias extraídas directamente:
        C   : concentraciones molares [mol/m³], shape (nc, N)
        Hg  : entalpía volumétrica del gas [J/m³], shape (N,)
        Tw  : temperatura de pared [K], shape (N,) — solo si shell_tube=True

    Variables secundarias derivadas:
        Tg  : temperatura del gas recuperada de Hg mediante Newton [K]
        Ctot: concentración molar total = sum(C, axis=0) [mol/m³], shape (N,)
        P   : presión total por ley de gas ideal  P = Ctot * R * Tg / 1e5 [bar]
        y   : fracciones molares = C / Ctot, shape (nc, N)

    Parameters
    ----------
    sv            : np.ndarray — vector de estado; shape ((nc+1)*N,) o ((nc+2)*N,)
    n_comp        : int   — número de especies (nc)
    N             : int   — número de celdas
    prop_gas      : dict  — propiedades del gas (de build_gas_prop_config)
    Tg_guess      : np.ndarray, shape (N,) — temperatura inicial para Newton [K]
    gas_T_ref     : float — temperatura de referencia de las entalpías [K]
    newton_tol    : float — tolerancia de Newton [K]
    newton_max_iter: int  — máximo de iteraciones de Newton
    shell_tube    : bool  — True si hay pared dinámica (Tw es variable de estado)

    Returns
    -------
    dict con claves: C, Hg, Tg, Ctot, P, y [, Tw si shell_tube=True]
    """
    nc, nn = int(n_comp), int(N)
    sv_arr = np.asarray(sv, dtype=float).reshape(-1)

    C  = sv_arr[:nc * nn].reshape(nc, nn)
    Hg = sv_arr[nc * nn : (nc + 1) * nn]

    Ctot = np.sum(C, axis=0)                                  # [mol/m³], shape (N,)
    Tg = recover_Tg_from_Hg(
        C=C,
        Hg=Hg,
        prop_gas=prop_gas,
        n_comp=nc,
        epsi=_EPSI_TUBE,
        Tg_guess=np.asarray(Tg_guess, dtype=float).reshape(-1),
        gas_T_ref=gas_T_ref,
        tol_T=newton_tol,
        max_iter=newton_max_iter,
    )
    P = Ctot * R_GAS * Tg / 1.0e5                             # [bar]
    Ctot_safe = np.maximum(Ctot, 1.0e-300)
    y = C / Ctot_safe[None, :]                                 # [-], shape (nc, N)

    result = {
        "C":    C,
        "Hg":   Hg,
        "Tg":   Tg,
        "Ctot": Ctot,
        "P":    P,
        "y":    y,
    }
    if shell_tube:
        result["Tw"] = sv_arr[(nc + 1) * nn:]
    return result


# =========================================================
# 2. Interfaz de estado de alto nivel
# =========================================================

@profiled
def set_state(
    P_bar: np.ndarray,
    Tg: np.ndarray,
    y: np.ndarray,
    n_comp: int,
    N: int,
    prop_gas: dict,
    gas_T_ref: float,
) -> Dict[str, Any]:
    """
    @brief
    Construye el estado completo del heater a partir de (P, Tg, y) del usuario.

    @details
    Recibe las variables en unidades físicas naturales y calcula las variables
    primarias del integrador (C, Hg):

        C_i = y_i * P_Pa / (R * Tg)      [mol/m³]
        Hg  = Σ_i C_i * h_i(Tg)         [J/m³]  (epsi = 1 para tubo vacío)

    Parameters
    ----------
    P_bar   : array-like, shape (N,)    — presión total [bar]
    Tg      : array-like, shape (N,)    — temperatura del gas [K]
    y       : array-like, shape (nc, N) — fracciones molares [-]
    n_comp  : int   — número de especies
    N       : int   — número de celdas
    prop_gas: dict  — propiedades del gas
    gas_T_ref: float — temperatura de referencia de entalpías [K]

    Returns
    -------
    dict con claves: C [mol/m³], Hg [J/m³], Tg [K], P [bar], y [-]
    """
    nc, nn = int(n_comp), int(N)

    P_arr  = np.asarray(P_bar, dtype=float).reshape(-1)
    Tg_arr = np.asarray(Tg,    dtype=float).reshape(-1)
    y_arr  = np.asarray(y,     dtype=float).reshape(nc, nn)

    if P_arr.size != nn:
        raise ValueError(f"P_bar must have length {nn}")
    if Tg_arr.size != nn:
        raise ValueError(f"Tg must have length {nn}")
    if y_arr.shape != (nc, nn):
        raise ValueError(f"y must have shape ({nc}, {nn})")
    if not np.all(np.isfinite(P_arr)) or np.any(P_arr <= 0.0):
        raise ValueError("P_bar must be finite and > 0")
    if not np.all(np.isfinite(Tg_arr)) or np.any(Tg_arr <= 0.0):
        raise ValueError("Tg must be finite and > 0")
    if not np.all(np.isfinite(y_arr)) or np.any(y_arr < 0.0):
        raise ValueError("y must be finite and >= 0")
    ysum = np.sum(y_arr, axis=0)
    if np.any(ysum <= 0.0):
        raise ValueError("y must have positive sum in every cell")
    if np.any(np.abs(ysum - 1.0) > 1.0e-6):
        raise ValueError("y must sum to 1 in every cell within tolerance 1e-6")

    P_Pa = P_arr * 1.0e5
    C = y_arr * P_Pa[None, :] / (R_GAS * Tg_arr[None, :])   # [mol/m³]

    Hg = calc_volumetric_enthalpy(
        C=C, Tg=Tg_arr, prop_gas=prop_gas,
        n_comp=nc, epsi=_EPSI_TUBE, gas_T_ref=gas_T_ref,
    )

    return {
        "C":  C,
        "Hg": Hg,
        "Tg": Tg_arr.copy(),
        "P":  P_arr.copy(),
        "y":  y_arr.copy(),
    }


@profiled
def init_heater_state(
    C_init: np.ndarray,
    Hg_init: np.ndarray,
    Tg_init: np.ndarray,
    P_init: np.ndarray,
    y_init: np.ndarray,
    N: int,
    Tw_init: np.ndarray = None,
) -> Dict[str, Any]:
    """
    @brief
    Construye el estado vivo del heater a partir de los arrays iniciales.

    @details
    Copia todos los arrays de condición inicial a las variables vivas e
    inicializa los campos de velocidad a cero (se rellenan en el primer
    paso de integración por DW).

    Parameters
    ----------
    C_init   : shape (nc, N) — concentraciones iniciales [mol/m³]
    Hg_init  : shape (N,)   — entalpía volumétrica inicial [J/m³]
    Tg_init  : shape (N,)   — temperatura gas inicial [K]
    P_init   : shape (N,)   — presión inicial [bar]
    y_init   : shape (nc, N) — fracciones molares iniciales [-]
    N        : int  — número de celdas
    Tw_init  : shape (N,) — temperatura inicial de la pared [K]; None si no hay pared dinámica

    Returns
    -------
    dict con claves: C, Hg, Tg, P, y, v [m/s], v_face [m/s] [, Tw [K] si Tw_init dado]
    """
    nn = int(N)
    result = {
        "C":      np.array(C_init,  dtype=float),
        "Hg":     np.array(Hg_init, dtype=float),
        "Tg":     np.array(Tg_init, dtype=float),
        "P":      np.array(P_init,  dtype=float),
        "y":      np.array(y_init,  dtype=float),
        "v":      np.zeros(nn,      dtype=float),
        "v_face": np.zeros(nn + 1,  dtype=float),
    }
    if Tw_init is not None:
        result["Tw"] = np.array(Tw_init, dtype=float)
    return result


# =========================================================
# 3. Continuación entre pasos de simulación
# =========================================================

def build_sv0_from_results(heater) -> np.ndarray:
    """
    @brief
    Construye el vector de estado inicial sv0 para la siguiente simulación
    a partir del último instante de un objeto de resultados del heater.

    @details
    Extrae las variables primarias del instante final de `heater` y las
    empaqueta con `pack_state_vector`. Detecta automáticamente si el modo
    shell-tube está activo comprobando la presencia de `_Tw_results`.

    Uso típico (simulación multi-segmento):
        _, _, h1 = run_step(sv0, t1, params)
        sv0_next = build_sv0_from_results(h1)
        _, _, h2 = run_step(sv0_next, t2, params_new)

    Parameters
    ----------
    heater : SimpleNamespace devuelto por run_step.
             Requiere: _C_results, _Hg_results.
             Opcional: _Tw_results (presente solo con shell-tube).

    Returns
    -------
    sv0 : np.ndarray
        Vector de estado inicial listo para pasar a run_step.
        Shape: (nc+1)*N sin shell-tube, (nc+2)*N con shell-tube.
    """
    C  = np.asarray(heater._C_results[-1],  dtype=float)   # (nc, N)
    Hg = np.asarray(heater._Hg_results[-1], dtype=float)   # (N,)
    Tw = (np.asarray(heater._Tw_results[-1], dtype=float)
          if hasattr(heater, "_Tw_results") else None)      # (N,) o None
    return pack_state_vector(C, Hg, Tw=Tw)
