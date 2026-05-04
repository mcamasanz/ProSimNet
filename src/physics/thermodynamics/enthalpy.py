"""
@module enthalpy
@brief Cálculo de entalpías molares de especies gaseosas y recuperación de
       temperatura a partir de entalpía volumétrica del gas.

@details
Este módulo implementa tres operaciones termodinámicas relacionadas con
la energía del gas en la columna:

  1. calc_species_enthalpy : h_i(T) para cada especie y nodo.
  2. calc_volumetric_enthalpy : H_g = eps * Σ_i C_i h_i(T) [J/m³].
  3. recover_Tg_from_Hg : inversión T = f^{-1}(H_g) por Newton o analítica.

La separación de estas operaciones en funciones puras permite testearlas
de forma aislada y usarlas tanto en el RHS como en el postproceso.

Unidades:
  h_i     [J/mol]
  Hg      [J/m³]  (volumétrica, incluye porosidad eps)
  C       [mol/m³] (concentración molar, shape (nc, nn))
  T       [K]
"""

from __future__ import annotations

from typing import Union, List

import numpy as np

from src.physics.thermodynamics.pure_gas import eval_species_property
from src.utils.profiling import profiled


# =========================================================
# 1. Entalpía molar de cada especie: h_i(T)
# =========================================================

@profiled
def calc_species_enthalpy(
    Tg: np.ndarray,
    prop_gas: dict,
    n_comp: int,
    gas_T_ref: float = 298.15,
) -> np.ndarray:
    """
    @brief
    Calcula la entalpía molar de cada especie gaseosa a la temperatura Tg
    por nodo: h_i(T) [J/mol], shape (nc, nn).

    @details
    Dos casos según el modo de propiedades del gas:

    Caso 1 — modo polynomial (h_molar es lista de callables):
        h_i(T) se evalúa directamente desde la correlación polinómica.
        El resultado es la entalpía absoluta (referenciada a Tref del DB).

    Caso 2 — modo constant/fixed (h_molar es ndarray o None):
        h_i(T) = h_ref_i + Cp_i * (T - Tref_i)
        donde h_ref_i es la entalpía a Tref_i (constante o 0 por defecto).
        Esta aproximación es exacta cuando Cp no varía con T (modo constant).

    Parameters
    ----------
    Tg : np.ndarray, shape (nn,)
        Temperatura del gas por nodo [K].
    prop_gas : dict
        Diccionario de propiedades puras del gas (salida de build_pure_gas_properties).
        Claves relevantes: "Cp_molar", "h_molar" (opcional), "Tref" (opcional).
    n_comp : int
        Número de componentes (nc).
    gas_T_ref : float
        Temperatura de referencia fallback si "Tref" no está en prop_gas [K].

    Returns
    -------
    h_i : np.ndarray, shape (nc, nn)
        Entalpía molar por especie y nodo [J/mol].

    Notes
    -----
    - El shape de retorno es (nc, nn) — especie primero, nodo segundo —
      para ser consistente con la convención de concentraciones C (nc, nn)
      y facilitar la suma vectorizada Σ_i C_i h_i.
    - h_molar constante (ndarray) se interpreta como h_ref evaluada a T_ref,
      no como una entalpía constante independiente de T.
    """
    Tg_arr = np.asarray(Tg, dtype=float).reshape(-1)
    nn = Tg_arr.size

    Cp_entry = prop_gas["Cp_molar"]
    h_entry  = prop_gas.get("h_molar", None)
    Tref_entry = prop_gas.get("Tref", None)

    # ---- Caso 1: h_molar disponible como callable (modo polynomial)
    if h_entry is not None and not isinstance(h_entry, np.ndarray):
        h_eval = eval_species_property(h_entry, Tg_arr, n_comp)  # (nn, nc)
        return h_eval.T                                           # (nc, nn)

    # ---- Caso 2: h(T) = h_ref + Cp * (T - Tref)
    # Cp_molar
    if isinstance(Cp_entry, np.ndarray):
        Cp_const = np.asarray(Cp_entry, dtype=float).reshape(-1)
        Cp_i = np.tile(Cp_const.reshape(n_comp, 1), (1, nn))     # (nc, nn)
    else:
        Cp_eval = eval_species_property(Cp_entry, Tg_arr, n_comp) # (nn, nc)
        Cp_i = Cp_eval.T                                           # (nc, nn)

    # h_ref
    if h_entry is None:
        h_ref = np.zeros(n_comp, dtype=float)
    else:
        h_ref = np.asarray(h_entry, dtype=float).reshape(-1)

    # Tref
    if Tref_entry is not None:
        Tref = np.asarray(Tref_entry, dtype=float).reshape(-1)
    else:
        Tref = np.full(n_comp, float(gas_T_ref), dtype=float)

    # h_i(T) = h_ref_i + Cp_i * (T - Tref_i)
    h_i = h_ref[:, None] + Cp_i * (Tg_arr[None, :] - Tref[:, None])  # (nc, nn)
    return h_i


# =========================================================
# 2. Entalpía volumétrica del gas
# =========================================================

@profiled
def calc_volumetric_enthalpy(
    C: np.ndarray,
    Tg: np.ndarray,
    prop_gas: dict,
    n_comp: int,
    epsi: float,
    gas_T_ref: float = 298.15,
) -> np.ndarray:
    """
    @brief
    Calcula la entalpía volumétrica del gas en el lecho por nodo:

        H_g = eps * Σ_i C_i(z) * h_i(T_g(z))   [J/m³]

    @details
    H_g es la variable conservada del balance de energía del gas en formulación
    de volúmenes finitos. El factor eps convierte de espacio de poros a espacio
    total de celda.

    Parameters
    ----------
    C : np.ndarray, shape (nc, nn)
        Concentración molar de cada especie por nodo [mol/m³].
    Tg : np.ndarray, shape (nn,)
        Temperatura del gas por nodo [K].
    prop_gas : dict
        Diccionario de propiedades puras del gas.
    n_comp : int
        Número de componentes (nc).
    epsi : float
        Porosidad del lecho [-]. 0 < epsi < 1.
    gas_T_ref : float
        Temperatura de referencia fallback [K].

    Returns
    -------
    Hg : np.ndarray, shape (nn,)
        Entalpía volumétrica del gas por nodo [J/m³].

    Notes
    -----
    - La porosidad epsi multiplica la entalpía porque solo la fracción eps
      del volumen de celda contiene gas (el resto es sólido).
    - Esta función NO incluye la contribución del sólido; esa se trata por
      separado en el balance de energía del sólido.
    """
    C_mat  = np.asarray(C,  dtype=float)
    Tg_arr = np.asarray(Tg, dtype=float).reshape(-1)

    h_i = calc_species_enthalpy(Tg_arr, prop_gas, n_comp, gas_T_ref)  # (nc, nn)
    Hg  = float(epsi) * np.sum(C_mat * h_i, axis=0)                   # (nn,)
    return Hg


# =========================================================
# 3. Recuperación de T_g a partir de H_g
# =========================================================

@profiled
def recover_Tg_from_Hg(
    C: np.ndarray,
    Hg: np.ndarray,
    prop_gas: dict,
    n_comp: int,
    epsi: float,
    Tg_guess: np.ndarray,
    gas_T_ref: float = 298.15,
    T_clip_min: float = 200.0,
    T_clip_max: float = 5000.0,
    max_iter: int = 30,
    tol_T: float = 1.0e-8,
    tol_f: float = 1.0e-6,
) -> np.ndarray:
    """
    @brief
    Recupera el campo de temperatura T_g(z) a partir de la entalpía volumétrica
    del gas H_g(z) resolviendo la relación:

        H_g = eps * Σ_i C_i(z) * h_i(T_g(z))

    @details
    Esta inversión es necesaria porque el balance de energía del gas se
    resuelve en términos de H_g (variable conservada), pero las propiedades
    de transporte y la cinética necesitan T_g explícito.

    Dos estrategias según el modo de propiedades:

    Modo constant (Cp = constante por especie):
        Solución analítica directa:
            T = (H_g/eps - Σ C_i (h_ref_i - Cp_i Tref_i)) / Σ C_i Cp_i

    Modo polynomial (Cp = f(T)):
        Método de Newton nodo a nodo, con:
            f(T)  = eps * Σ C_i h_i(T) - H_g
            f'(T) = eps * Σ C_i Cp_i(T)
        Máximo max_iter iteraciones. El paso deltaT se limita a ±50 K.

    Parameters
    ----------
    C : np.ndarray, shape (nc, nn)
        Concentración molar por especie y nodo [mol/m³].
    Hg : np.ndarray, shape (nn,)
        Entalpía volumétrica del gas por nodo [J/m³].
    prop_gas : dict
        Diccionario de propiedades puras del gas.
    n_comp : int
        Número de componentes (nc).
    epsi : float
        Porosidad del lecho [-].
    Tg_guess : np.ndarray, shape (nn,)
        Predictor inicial de temperatura [K]. Se usa como punto de partida
        del Newton y también para nodos sin gas activo.
    gas_T_ref : float
        Temperatura de referencia fallback [K].
    T_clip_min, T_clip_max : float
        Límites de clipping para T_g durante la iteración [K].
    max_iter : int
        Número máximo de iteraciones de Newton (modo polynomial).
    tol_T : float
        Tolerancia en deltaT [K] para el criterio de convergencia.
    tol_f : float
        Tolerancia en el residuo f(T) [J/m³] para el criterio de convergencia.

    Returns
    -------
    Tg : np.ndarray, shape (nn,)
        Temperatura del gas recuperada por nodo [K].

    Notes
    -----
    - Los nodos sin gas activo (C_tot ≈ 0) mantienen Tg_guess sin iteración.
    - El Newton limita deltaT a ±50 K por iteración para robustez numérica
      en frentes de temperatura pronunciados.
    - Si Newton no converge en max_iter iteraciones, la función devuelve el
      mejor valor alcanzado sin lanzar excepción. El llamador puede activar
      debug externo si necesita detectar esta condición.
    """
    C_mat  = np.asarray(C,        dtype=float)
    Hg_arr = np.asarray(Hg,       dtype=float).reshape(-1)
    Tg0    = np.asarray(Tg_guess, dtype=float).reshape(-1)

    nn = Hg_arr.size

    Cp_entry = prop_gas["Cp_molar"]
    h_entry  = prop_gas.get("h_molar", None)
    Tref_entry = prop_gas.get("Tref", None)
    Tmax_entry = prop_gas.get("Tmax", None)

    if Tref_entry is not None:
        Tref_arr = np.asarray(Tref_entry, dtype=float).reshape(-1)
    else:
        Tref_arr = np.full(n_comp, float(gas_T_ref), dtype=float)

    if Tmax_entry is not None:
        T_clip_max = float(np.max(np.asarray(Tmax_entry, dtype=float)))

    Ctot = np.sum(C_mat, axis=0)                           # (nn,)
    active = Ctot > 1.0e-30                                 # nodos con gas

    Tg_arr = np.clip(Tg0.copy(), T_clip_min, T_clip_max)

    # ---- Modo constant: solución analítica
    if isinstance(Cp_entry, np.ndarray):
        Cp_const = np.asarray(Cp_entry, dtype=float).reshape(-1)

        if h_entry is None:
            h_ref = np.zeros(n_comp, dtype=float)
        else:
            h_ref = np.asarray(h_entry, dtype=float).reshape(-1)

        # A = Σ C_i (h_ref_i - Cp_i Tref_i)  shape (nn,)
        A = np.sum(C_mat * (h_ref[:, None] - Cp_const[:, None] * Tref_arr[:, None]),
                   axis=0)
        # B = Σ C_i Cp_i  shape (nn,)
        B = np.sum(C_mat * Cp_const[:, None], axis=0)

        denom = float(epsi) * np.maximum(B, 1.0e-30)
        Tg_arr[active] = (Hg_arr[active] - float(epsi) * A[active]) / denom[active]
        Tg_arr = np.clip(Tg_arr, T_clip_min, T_clip_max)
        return Tg_arr

    # ---- Modo polynomial: Newton nodo a nodo
    for _ in range(max_iter):
        h_eval  = eval_species_property(h_entry,  Tg_arr, n_comp).T   # (nc, nn)
        Cp_eval = eval_species_property(Cp_entry, Tg_arr, n_comp).T   # (nc, nn)

        f_arr  = float(epsi) * np.sum(C_mat * h_eval,  axis=0) - Hg_arr  # (nn,)
        df_arr = float(epsi) * np.sum(C_mat * Cp_eval, axis=0)            # (nn,)

        deltaT = np.zeros(nn, dtype=float)
        deltaT[active] = f_arr[active] / np.maximum(df_arr[active], 1.0e-20)
        deltaT = np.clip(deltaT, -50.0, 50.0)

        Tg_new = np.clip(Tg_arr - deltaT, T_clip_min, T_clip_max)

        converged = (~active) | ((np.abs(deltaT) < tol_T) & (np.abs(f_arr) < tol_f))
        Tg_arr = Tg_new

        if np.all(converged):
            break

    return Tg_arr
