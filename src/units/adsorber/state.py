"""
@module state
@brief Gestión del vector de estado del modelo de columna PSA 1D.

@details
El vector de estado del integrador ODE tiene el layout:

    Sin shell-tube (wall_config ausente o None):
        sv = [C.reshape(-1), q.reshape(-1), Hg, Ts]
        Tamaño: 2*nc*N + 2*N

    Con shell-tube (wall_config presente):
        sv = [C.reshape(-1), q.reshape(-1), Hg, Ts, Tw]
        Tamaño: 2*nc*N + 3*N

donde:
    C   : concentraciones molares de las nc especies en N celdas  [mol/m³], shape (nc, N)
    q   : carga adsorbida de las nc especies en N celdas          [mol/kg], shape (nc, N)
    Hg  : entalpía volumétrica del gas en N celdas                [J/m³],  shape (N,)
    Ts  : temperatura del sólido en N celdas                      [K],     shape (N,)
    Tw  : temperatura de la pared por celda (solo shell-tube)     [K],     shape (N,)

Este módulo expone funciones puras sin estado:
    pack_state_vector   — empaqueta las arrays en el vector 1D del integrador
    unpack_state_vector — desempaqueta y deriva las variables secundarias
    set_state           — valida un dict de estado externo y calcula C y Hg
    get_state           — devuelve copias de las variables de estado actuales
    state_is_complete   — comprueba que todas las variables primarias son no-None
    init_column_state   — copia el estado inicial a las variables vivas y pone v=0
    build_sv0_from_results — genera sv0 de continuación desde el último instante

Unidades: SI (excepto P que se almacena en bar para compatibilidad con el modelo).
"""

from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np

from src.physics.thermodynamics.enthalpy import (
    calc_volumetric_enthalpy,
    recover_Tg_from_Hg,
)
from src.utils.profiling import profiled

R_GAS = 8.31446261815324   # constante de los gases ideales [J/mol/K]


# =========================================================
# 1. Empaquetado / desempaquetado del vector de estado
# =========================================================

@profiled
def pack_state_vector(
    C: np.ndarray,
    q: np.ndarray,
    Hg: np.ndarray,
    Ts: np.ndarray,
    Tw: np.ndarray = None,
) -> np.ndarray:
    """
    @brief
    Empaqueta las variables de estado en el vector 1D que consume el integrador ODE.

    @details
    Sin shell-tube: layout [C.reshape(-1), q.reshape(-1), Hg, Ts], tamaño 2*nc*N + 2*N.
    Con shell-tube: layout [C.reshape(-1), q.reshape(-1), Hg, Ts, Tw], tamaño 2*nc*N + 3*N.

    Parameters
    ----------
    C   : np.ndarray, shape (nc, N) — concentraciones molares [mol/m³]
    q   : np.ndarray, shape (nc, N) — carga adsorbida [mol/kg]
    Hg  : np.ndarray, shape (N,)   — entalpía volumétrica del gas [J/m³]
    Ts  : np.ndarray, shape (N,)   — temperatura del sólido [K]
    Tw  : np.ndarray, shape (N,) o None — temperatura de la pared [K] (shell-tube)

    Returns
    -------
    sv : np.ndarray, shape (2*nc*N + 2*N,) o (2*nc*N + 3*N,)
    """
    parts = [
        np.asarray(C,  dtype=float).reshape(-1),
        np.asarray(q,  dtype=float).reshape(-1),
        np.asarray(Hg, dtype=float).reshape(-1),
        np.asarray(Ts, dtype=float).reshape(-1),
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
    epsi: float,
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
        q   : carga adsorbida [mol/kg], shape (nc, N)
        Hg  : entalpía volumétrica del gas [J/m³], shape (N,)
        Ts  : temperatura del sólido [K], shape (N,)
        Tw  : temperatura de la pared [K], shape (N,) — solo si shell_tube=True

    Variables secundarias derivadas:
        Tg  : temperatura del gas recuperada de Hg (iteración Newton o analítica)
        Ctot: concentración molar total = sum(C, axis=0) [mol/m³], shape (N,)
        P   : presión total derivada de la ley de gas ideal  P = Ctot*R*Tg / 1e5 [bar]
        y   : fracciones molares = C / Ctot, shape (nc, N)

    Parameters
    ----------
    sv            : np.ndarray, shape (2*nc*N + 2*N,) o (2*nc*N + 3*N,) — vector de estado
    n_comp        : int  — número de especies (nc)
    N             : int  — número de celdas
    prop_gas      : dict — propiedades del gas (resultado de build_pure_gas_properties)
    epsi          : float — porosidad del lecho [-]
    Tg_guess      : np.ndarray, shape (N,) — temperatura inicial para Newton [K]
    gas_T_ref     : float — temperatura de referencia de las entalpías [K]
    newton_tol    : float — tolerancia de Newton [K]
    newton_max_iter: int  — máximo de iteraciones de Newton
    shell_tube    : bool — True si wall_config está activo (Tw en el sv)

    Returns
    -------
    dict con claves: C, q, Hg, Ts, Tg, Ctot, P, y [, Tw si shell_tube]
    """
    nc, nn = int(n_comp), int(N)
    sv_arr = np.asarray(sv, dtype=float).reshape(-1)

    # Límites de cada bloque
    i0 = 0
    i1 = nc * nn               # C
    i2 = 2 * nc * nn           # q
    i3 = 2 * nc * nn + nn      # Hg
    i4 = 2 * nc * nn + 2 * nn  # Ts  (Tw empieza aquí si shell_tube)

    C  = sv_arr[i0:i1].reshape(nc, nn)
    q  = sv_arr[i1:i2].reshape(nc, nn)
    Hg = sv_arr[i2:i3]
    Ts = sv_arr[i3:i4]
    Tw = sv_arr[i4:i4 + nn] if shell_tube else None

    # Variables secundarias
    Ctot = np.sum(C, axis=0)                             # [mol/m³], shape (N,)
    Tg = recover_Tg_from_Hg(
        C=C,
        Hg=Hg,
        prop_gas=prop_gas,
        n_comp=nc,
        epsi=epsi,
        Tg_guess=np.asarray(Tg_guess, dtype=float).reshape(-1),
        gas_T_ref=gas_T_ref,
        tol_T=newton_tol,
        max_iter=newton_max_iter,
    )
    P = Ctot * R_GAS * Tg / 1.0e5                        # [bar]
    Ctot_safe = np.maximum(Ctot, 1.0e-300)
    y = C / Ctot_safe[None, :]                            # [-], shape (nc, N)

    result = {
        "C":    C,
        "q":    q,
        "Hg":   Hg,
        "Ts":   Ts,
        "Tg":   Tg,
        "Ctot": Ctot,
        "P":    P,
        "y":    y,
    }
    if shell_tube:
        result["Tw"] = Tw
    return result


# =========================================================
# 2. Interfaz de estado de alto nivel
# =========================================================

@profiled
def set_state(
    P_bar: np.ndarray,
    Tg: np.ndarray,
    Ts: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    n_comp: int,
    N: int,
    prop_gas: dict,
    epsi: float,
    gas_T_ref: float,
) -> Dict[str, Any]:
    """
    @brief
    Valida un dict de estado externo (P, Tg, Ts, y, q) y calcula C y Hg.

    @details
    Recibe las variables en unidades de usuario (P en bar, Tg en K, y adimensional)
    y devuelve el estado completo incluyendo C [mol/m³] y Hg [J/m³].

    La concentración se calcula como ideal-gas:
        C_i = y_i * P_Pa / (R * Tg)

    Parameters
    ----------
    P_bar   : array-like, shape (N,)    — presión total [bar]
    Tg      : array-like, shape (N,)    — temperatura del gas [K]
    Ts      : array-like, shape (N,)    — temperatura del sólido [K]
    y       : array-like, shape (nc, N) — fracciones molares [-]
    q       : array-like, shape (nc, N) — carga adsorbida [mol/kg]
    n_comp  : int  — número de especies
    N       : int  — número de celdas
    prop_gas: dict — propiedades del gas
    epsi    : float — porosidad del lecho [-]
    gas_T_ref: float — temperatura de referencia de entalpías [K]

    Returns
    -------
    dict con claves: C, q, Hg, Ts, Tg, P, y
    """
    nc, nn = int(n_comp), int(N)

    P_arr  = np.asarray(P_bar, dtype=float).reshape(-1)
    Tg_arr = np.asarray(Tg,    dtype=float).reshape(-1)
    Ts_arr = np.asarray(Ts,    dtype=float).reshape(-1)
    y_arr  = np.asarray(y,     dtype=float).reshape(nc, nn)
    q_arr  = np.asarray(q,     dtype=float).reshape(nc, nn)

    # Validaciones mínimas
    if P_arr.size != nn:
        raise ValueError(f"P_bar must have length {nn}")
    if Tg_arr.size != nn:
        raise ValueError(f"Tg must have length {nn}")
    if Ts_arr.size != nn:
        raise ValueError(f"Ts must have length {nn}")
    if y_arr.shape != (nc, nn):
        raise ValueError(f"y must have shape ({nc}, {nn})")
    if q_arr.shape != (nc, nn):
        raise ValueError(f"q must have shape ({nc}, {nn})")
    if not np.all(np.isfinite(P_arr)) or np.any(P_arr <= 0.0):
        raise ValueError("P_bar must be finite and > 0")
    if not np.all(np.isfinite(Tg_arr)) or np.any(Tg_arr <= 0.0):
        raise ValueError("Tg must be finite and > 0")
    if not np.all(np.isfinite(Ts_arr)) or np.any(Ts_arr <= 0.0):
        raise ValueError("Ts must be finite and > 0")
    if not np.all(np.isfinite(y_arr)) or np.any(y_arr < 0.0):
        raise ValueError("y must be finite and >= 0")
    ysum = np.sum(y_arr, axis=0)
    if np.any(ysum <= 0.0):
        raise ValueError("y must have positive sum in every cell")
    if np.any(np.abs(ysum - 1.0) > 1.0e-6):
        raise ValueError("y must sum to 1 in every cell within tolerance 1e-6")
    if not np.all(np.isfinite(q_arr)) or np.any(q_arr < 0.0):
        raise ValueError("q must be finite and >= 0")

    P_Pa = P_arr * 1.0e5                                     # Pa
    C = y_arr * P_Pa[None, :] / (R_GAS * Tg_arr[None, :])   # [mol/m³]

    Hg = calc_volumetric_enthalpy(
        C=C, Tg=Tg_arr, prop_gas=prop_gas,
        n_comp=nc, epsi=epsi, gas_T_ref=gas_T_ref,
    )

    return {
        "C":  C,
        "q":  q_arr.copy(),
        "Hg": Hg,
        "Ts": Ts_arr.copy(),
        "Tg": Tg_arr.copy(),
        "P":  P_arr.copy(),
        "y":  y_arr.copy(),
    }


@profiled
def get_state(
    C: np.ndarray,
    Tg: np.ndarray,
    Ts: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    P: np.ndarray,
) -> Dict[str, Any]:
    """
    @brief
    Devuelve copias de las variables de estado actuales.

    Returns
    -------
    dict con claves: P [bar], Tg [K], Ts [K], y [-], q [mol/kg]
    (excluye C y Hg que son variables internas del integrador)
    """
    return {
        "P":  np.array(P,  dtype=float),
        "Tg": np.array(Tg, dtype=float),
        "Ts": np.array(Ts, dtype=float),
        "y":  np.array(y,  dtype=float),
        "q":  np.array(q,  dtype=float),
    }


@profiled
def state_is_complete(state: Dict[str, Any]) -> bool:
    """
    @brief
    Comprueba que todas las variables primarias del estado son no-None.

    Parameters
    ----------
    state : dict — debe contener las claves P, Tg, Hg, Ts, y, q

    Returns
    -------
    True si todas las claves existen y son distintas de None.
    """
    required_keys = ("P", "Tg", "Hg", "Ts", "y", "q")
    return all(state.get(k) is not None for k in required_keys)


@profiled
def init_column_state(
    C_init: np.ndarray,
    q_init: np.ndarray,
    Hg_init: np.ndarray,
    Ts_init: np.ndarray,
    Tg_init: np.ndarray,
    P_init: np.ndarray,
    y_init: np.ndarray,
    N: int,
) -> Dict[str, Any]:
    """
    @brief
    Construye el estado vivo de la columna a partir de los arrays iniciales.

    @details
    Copia todos los arrays de condición inicial a las variables vivas y
    pone a cero los campos de velocidad (v en centros de celda y v_face en caras).

    Parameters
    ----------
    C_init   : shape (nc, N) — concentraciones iniciales [mol/m³]
    q_init   : shape (nc, N) — carga adsorbida inicial [mol/kg]
    Hg_init  : shape (N,)   — entalpía volumétrica inicial [J/m³]
    Ts_init  : shape (N,)   — temperatura sólido inicial [K]
    Tg_init  : shape (N,)   — temperatura gas inicial [K]
    P_init   : shape (N,)   — presión inicial [bar]
    y_init   : shape (nc, N) — fracciones molares iniciales [-]
    N        : int  — número de celdas

    Returns
    -------
    dict con claves: C, q, Hg, Ts, Tg, P, y, v [m/s], v_face [m/s]
    """
    nn = int(N)
    return {
        "C":      np.array(C_init,  dtype=float),
        "q":      np.array(q_init,  dtype=float),
        "Hg":     np.array(Hg_init, dtype=float),
        "Ts":     np.array(Ts_init, dtype=float),
        "Tg":     np.array(Tg_init, dtype=float),
        "P":      np.array(P_init,  dtype=float),
        "y":      np.array(y_init,  dtype=float),
        "v":      np.zeros(nn,      dtype=float),   # velocidad en centros de celda [m/s]
        "v_face": np.zeros(nn + 1,  dtype=float),   # velocidad en caras [m/s]
    }


@profiled
def build_sv0_from_results(col) -> np.ndarray:
    """
    @brief
    Construye el vector de estado inicial para una continuación a partir del
    último instante del objeto de resultados devuelto por build_adsorber_results.

    @details
    Extrae C[-1], q[-1], Hg[-1], Ts[-1] y, si el objeto contiene _Tw_results,
    también Tw[-1]. Llama a pack_state_vector con los bloques correspondientes.

    Parameters
    ----------
    col : SimpleNamespace — objeto devuelto por build_adsorber_results

    Returns
    -------
    sv0 : np.ndarray — vector de estado inicial para el siguiente run_step
    """
    C  = np.asarray(col._C_results[-1],  dtype=float)
    q  = np.asarray(col._q_results[-1],  dtype=float)
    Hg = np.asarray(col._Hg_results[-1], dtype=float)
    Ts = np.asarray(col._Ts_results[-1], dtype=float)
    Tw = (np.asarray(col._Tw_results[-1], dtype=float)
          if hasattr(col, "_Tw_results") else None)
    return pack_state_vector(C=C, q=q, Hg=Hg, Ts=Ts, Tw=Tw)
