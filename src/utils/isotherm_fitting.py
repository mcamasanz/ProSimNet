"""
@module isotherm_fitting
@brief Ajuste automático de isotermas de adsorción a datos experimentales.

@details
Flujo de uso estándar:

    1. Cargar datos experimentales (P en bar, q en mol/kg) para una o varias T.
    2. Llamar a `fit_single_T` o `fit_multi_T` según el caso.
    3. El resultado es un callable `iso_fn(P_bar, T_K) → q_mol_kg` listo para
       pasarse a `build_adsorbent_config`.

Funciones públicas:

    fit_objective    — función de coste MSE con penalización de parámetros negativos
    fit_single_model — ajuste de un modelo concreto con múltiples métodos de optimización
    fit_single_T     — selección automática del mejor modelo para una isoterma a T fija
    fit_multi_T      — ajuste multi-temperatura con estimación de dH por desplazamiento de Arrhenius

Unidades de entrada/salida:
    P   : bar
    q   : mol/kg
    T   : K
    dH  : J/mol  (devuelto por fit_multi_T)
"""

from __future__ import annotations

from typing import Callable, List, Tuple, Optional

import numpy as np
from scipy.optimize import minimize, differential_evolution, shgo

from src.utils.isotherm_models import MODEL_REGISTRY, arrh
from src.utils.profiling import profiled

_R_GAS = 8.31446261815324   # J/mol/K

# Métodos de optimización local usados secuencialmente
_LOCAL_METHODS = ["Nelder-Mead", "Powell", "COBYLA"]


# ═══════════════════════════════════════════════════════════════════════════════
# Función de coste
# ═══════════════════════════════════════════════════════════════════════════════

@profiled
def fit_objective(
    par: np.ndarray,
    P: np.ndarray,
    q: np.ndarray,
    model_fn: Callable,
) -> float:
    """
    MSE entre datos y predicción del modelo, con penalización por parámetros negativos.

    Parameters
    ----------
    par      : parámetros del modelo
    P        : presión [bar]
    q        : carga experimental [mol/kg]
    model_fn : callable(par, P) → q_pred

    Returns
    -------
    float — valor de la función de coste
    """
    par_arr = np.array(par, dtype=float)
    neg_mask = par_arr < 0.0
    penalty  = np.sum(par_arr[neg_mask] ** 2) * 50.0
    par_arr[neg_mask] = 0.0
    diff = model_fn(par_arr, np.asarray(P, dtype=float)) - np.asarray(q, dtype=float)
    return float(np.mean(diff ** 2)) + penalty


# ═══════════════════════════════════════════════════════════════════════════════
# Ajuste de un modelo concreto
# ═══════════════════════════════════════════════════════════════════════════════

@profiled
def fit_single_model(
    model_fn: Callable,
    n_par: int,
    P: np.ndarray,
    q: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Ajusta un modelo de isoterma a datos (P, q) probando varios optimizadores.

    Parameters
    ----------
    model_fn : función q = model_fn(par, P)
    n_par    : número de parámetros del modelo
    P        : presión [bar]
    q        : carga experimental [mol/kg]

    Returns
    -------
    par_best : ndarray — parámetros óptimos
    mse_best : float   — MSE mínimo alcanzado
    """
    P_arr = np.asarray(P, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    obj   = lambda par: fit_objective(par, P_arr, q_arr, model_fn)

    best_par = np.ones(n_par)
    best_mse = np.inf

    # Punto inicial: qs ~ q_max, resto ~ 1
    x0 = np.ones(n_par)
    x0[0] = float(q_arr[-1]) if len(q_arr) > 0 else 1.0

    # Métodos locales
    for method in _LOCAL_METHODS:
        try:
            res = minimize(obj, x0, method=method)
            if res.fun < best_mse:
                best_mse = res.fun
                best_par = res.x
        except Exception:
            pass

    # Búsqueda global con bounds conservadoras
    bounds = [(0.0, max(10.0 * float(q_arr.max()), 10.0))] * n_par
    try:
        res = shgo(obj, bounds)
        if res.fun < best_mse:
            best_mse = res.fun
            best_par = res.x
    except Exception:
        pass
    try:
        res = differential_evolution(obj, bounds)
        if res.fun < best_mse:
            best_mse = res.fun
            best_par = res.x
    except Exception:
        pass

    return best_par, float(best_mse)


# ═══════════════════════════════════════════════════════════════════════════════
# Selección automática del mejor modelo (T fija)
# ═══════════════════════════════════════════════════════════════════════════════

@profiled
def fit_single_T(
    P: np.ndarray,
    q: np.ndarray,
    model_names: Optional[List[str]] = None,
    mse_tol: float = 1.0e-5,
) -> Tuple[Callable, np.ndarray, str, float]:
    """
    Selecciona y ajusta el mejor modelo de isoterma para datos a temperatura fija.

    Prueba los modelos en orden creciente de complejidad (2 → 3 → 4 parámetros)
    y se detiene cuando el MSE cae por debajo de `mse_tol`.

    Parameters
    ----------
    P           : presión [bar]
    q           : carga experimental [mol/kg]
    model_names : lista de nombres del MODEL_REGISTRY a probar (None = todos)
    mse_tol     : umbral de MSE para detención temprana

    Returns
    -------
    iso_P       : callable(P_bar) → q [mol/kg]   — isoterma ajustada a T fija
    par_best    : ndarray — parámetros óptimos
    name_best   : str     — nombre del modelo seleccionado
    mse_best    : float   — MSE del ajuste
    """
    if model_names is None:
        model_names = list(MODEL_REGISTRY.keys())

    # Ordenar por número de parámetros (menor complejidad primero)
    candidates = sorted(
        [(name, MODEL_REGISTRY[name]) for name in model_names if name in MODEL_REGISTRY],
        key=lambda x: x[1][1],
    )

    overall_best_par  = None
    overall_best_mse  = np.inf
    overall_best_name = candidates[0][0]
    overall_best_fn   = candidates[0][1][0]

    for name, (model_fn, n_par) in candidates:
        par, mse = fit_single_model(model_fn, n_par, P, q)
        if mse < overall_best_mse:
            overall_best_mse  = mse
            overall_best_par  = par
            overall_best_name = name
            overall_best_fn   = model_fn
        if mse <= mse_tol:
            break

    par_final = overall_best_par
    fn_final  = overall_best_fn
    iso_P = lambda P_in: fn_final(par_final, np.asarray(P_in, dtype=float))

    return iso_P, par_final, overall_best_name, overall_best_mse


# ═══════════════════════════════════════════════════════════════════════════════
# Ajuste multi-temperatura con estimación de dH
# ═══════════════════════════════════════════════════════════════════════════════

@profiled
def fit_multi_T(
    P_list: List[np.ndarray],
    q_list: List[np.ndarray],
    T_list: List[float],
    i_ref: int = 0,
    model_names: Optional[List[str]] = None,
    mse_tol: float = 1.0e-5,
) -> Tuple[Callable, np.ndarray, str, float, float, float]:
    """
    Ajuste multi-temperatura: determina el modelo de isoterma y la entalpía
    de adsorción dH por minimización del error de desplazamiento de Arrhenius.

    Algoritmo
    ---------
    1. Ajusta el modelo a la curva de referencia (T_list[i_ref]).
    2. Para cada otra temperatura, encuentra el factor de escala θ tal que
       q(P·θ, T_ref) ≈ q(P, T_i)  → equivale a b(T_i) = θ · b(T_ref).
    3. Ajusta dH a los θ obtenidos vía Arrhenius.
    4. Devuelve iso_fn(P_bar, T_K) = iso_ref(P · arrh(T, dH, T_ref)).

    Parameters
    ----------
    P_list  : lista de nc arrays de presión [bar], uno por temperatura
    q_list  : lista de nc arrays de carga [mol/kg], uno por temperatura
    T_list  : lista de temperaturas [K]
    i_ref   : índice de la temperatura de referencia (defecto: 0)
    model_names : modelos a considerar (None = todos del registro)
    mse_tol : umbral de MSE para parada temprana en el ajuste del modelo

    Returns
    -------
    iso_fn   : callable(P_bar, T_K) → q [mol/kg]
    par_best : ndarray — parámetros del modelo ajustado
    name_best: str     — nombre del modelo seleccionado
    mse_best : float   — MSE del ajuste en T_ref
    dH       : float   — entalpía de adsorción estimada [J/mol]
    T_ref    : float   — temperatura de referencia [K]
    """
    T_arr = np.asarray(T_list, dtype=float)
    T_ref = float(T_arr[i_ref])

    # ── paso 1: ajuste en T_ref ──────────────────────────────────────────────
    iso_ref, par_best, name_best, mse_best = fit_single_T(
        P_list[i_ref], q_list[i_ref],
        model_names=model_names, mse_tol=mse_tol,
    )

    # ── paso 2: calcular factor θ para cada temperatura ──────────────────────
    theta_list = []
    for k, (P_k, q_k, T_k) in enumerate(zip(P_list, q_list, T_list)):
        if k == i_ref:
            theta_list.append(1.0)
            continue
        def _err_theta(theta, P_k=P_k, q_k=q_k):
            q_pred = iso_ref(np.asarray(P_k, dtype=float) * float(theta[0]))
            return float(np.mean((np.asarray(q_k, dtype=float) - q_pred) ** 2))
        res = minimize(_err_theta, [0.9], method="Nelder-Mead")
        theta_list.append(float(res.x[0]))

    # ── paso 3: estimar dH ───────────────────────────────────────────────────
    theta_arr = np.array(theta_list, dtype=float)
    T_ref_arr = np.full_like(T_arr, T_ref)

    def _err_dH(dH_val):
        theta_pred = np.exp(float(dH_val[0]) / _R_GAS * (1.0 / T_arr - 1.0 / T_ref_arr))
        return float(np.mean((theta_pred - theta_arr) ** 2))

    res_dH = minimize(_err_dH, [20000.0], method="Nelder-Mead")
    if res_dH.fun > 0.1:
        res_dH2 = minimize(_err_dH, [20000.0], method="Powell")
        if res_dH2.fun < res_dH.fun:
            res_dH = res_dH2
    dH = float(res_dH.x[0])

    # ── paso 4: construir iso_fn(P, T) ───────────────────────────────────────
    _iso_ref = iso_ref   # captura local
    _dH      = dH
    _T_ref   = T_ref

    def iso_fn(P_bar, T_K):
        P_in = np.asarray(P_bar, dtype=float)
        T_in = float(T_K)
        return _iso_ref(P_in * arrh(T_in, _dH, _T_ref))

    return iso_fn, par_best, name_best, mse_best, dH, T_ref
