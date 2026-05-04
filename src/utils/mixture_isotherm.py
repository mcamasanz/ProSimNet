"""
@module mixture_isotherm
@brief Isotermas de mezcla multicomponente: IAST y RAST.

@details
Implementa dos métodos para calcular la carga adsorbida de equilibrio
en mezclas multicomponente a partir de las isotermas puras individuales:

    IAST  — Ideal Adsorbed Solution Theory (Myers & Prausnitz, 1965)
            Hipótesis: solución adsorbida ideal (γ_i = 1).

    RAST  — Real Adsorbed Solution Theory
            Extiende IAST con coeficientes de actividad calculados mediante
            el modelo de Wilson (ln_gamma_i).

Ambas funciones reciben:
    - lista de callables iso_fn_i(P_bar, T_K) → q [mol/kg]  — isotermas puras
    - P_i  : presiones parciales de cada componente [bar]
    - T    : temperatura [K]

y devuelven:
    - q_mix : ndarray (nc,) — carga adsorbida de cada componente [mol/kg]

Unidades: bar, mol/kg, K.
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np

try:
    from scipy.integrate import trapezoid as _trapz
except ImportError:
    from scipy.integrate import trapz as _trapz   # scipy < 1.11

from scipy.optimize import minimize, differential_evolution

from src.utils.profiling import profiled


# ═══════════════════════════════════════════════════════════════════════════════
# Utilidad interna: presión de expansión (spreading pressure)
# ═══════════════════════════════════════════════════════════════════════════════

def _spreading_pressure(iso_fn: Callable, P_max: float, T: float, n_pts: int = 101) -> float:
    """
    Calcula la presión de expansión πA/RT = ∫₀^P* q(P)/P dP.

    Parameters
    ----------
    iso_fn : callable(P_bar, T_K) → q [mol/kg]
    P_max  : límite superior de integración [bar]
    T      : temperatura [K]
    n_pts  : puntos de cuadratura

    Returns
    -------
    float — πA/RT [-]
    """
    P_ran = np.linspace(1.0e-6, max(P_max, 1.0e-6), n_pts)
    q_over_P = iso_fn(P_ran, T) / P_ran
    return float(_trapz(q_over_P, P_ran))


# ═══════════════════════════════════════════════════════════════════════════════
# Coeficientes de actividad RAST (modelo de Wilson)
# ═══════════════════════════════════════════════════════════════════════════════

@profiled
def ln_gamma_wilson(
    x: np.ndarray,
    Lambda: np.ndarray,
    C: float,
    piA_RT: float,
) -> np.ndarray:
    """
    Logaritmo de los coeficientes de actividad según el modelo de Wilson.

    Parameters
    ----------
    x      : fracciones molares en la fase adsorbida (nc,)
    Lambda : matriz de parámetros de Wilson (nc, nc)
    C      : parámetro de heterogeneidad [-]
    piA_RT : presión de expansión adimensional πA/RT [-]

    Returns
    -------
    ln_gamma : ndarray (nc,) — ln de coeficientes de actividad
    """
    N = len(x)
    ln_gamma = np.zeros(N, dtype=float)
    exp_factor = 1.0 - np.exp(-C * piA_RT)

    sum_xLam = np.array([np.dot(x, Lambda[i, :]) for i in range(N)])

    for i in range(N):
        sum_ratio = sum(
            Lambda[k, i] * x[k] / max(sum_xLam[k], 1.0e-30)
            for k in range(N)
        )
        ln_gamma[i] = (1.0 - np.log(max(sum_xLam[i], 1.0e-30)) - sum_ratio) * exp_factor

    return ln_gamma


# ═══════════════════════════════════════════════════════════════════════════════
# IAST
# ═══════════════════════════════════════════════════════════════════════════════

@profiled
def iast(
    iso_fns: List[Callable],
    P_i: np.ndarray,
    T: float,
    n_pts: int = 101,
) -> np.ndarray:
    """
    Ideal Adsorbed Solution Theory (Myers & Prausnitz, 1965).

    Parameters
    ----------
    iso_fns : lista de nc callables iso_fn_i(P_bar, T_K) → q [mol/kg]
    P_i     : presiones parciales de cada componente [bar], shape (nc,)
    T       : temperatura [K]
    n_pts   : puntos de cuadratura para la presión de expansión

    Returns
    -------
    q_mix : ndarray (nc,) — carga adsorbida por componente [mol/kg]
    """
    nc   = len(iso_fns)
    P_i  = np.asarray(P_i, dtype=float)

    # ── función de error sobre (x_1..x_{nc-1}, πA/RT) ───────────────────────
    def _err(vars_):
        x_head = vars_[:nc - 1]
        x_last = 1.0 - np.sum(x_head)
        x = np.concatenate([x_head, [x_last]])

        penalty = 0.0
        for ii in range(nc):
            if x[ii] < 1.0e-4:
                penalty += 50.0 * x[ii] ** 2
                x[ii] = 1.0e-4
            elif x[ii] > 0.9995:
                penalty += 50.0 * (x[ii] - 0.9995) ** 2
                x[ii] = 0.9995

        piA_RT  = vars_[-1]
        P0_i    = P_i / x
        spr_new = np.array([_spreading_pressure(iso_fns[i], P0_i[i], T, n_pts) for i in range(nc)])
        return float(np.sum((spr_new - piA_RT) ** 2)) + penalty

    # ── estimación inicial ───────────────────────────────────────────────────
    y_i = P_i / np.sum(P_i)
    piA_RT_0 = [_spreading_pressure(iso_fns[i], P_i[i], T, n_pts) for i in range(nc)]
    x_init   = y_i[:-1]

    best_x, best_fn = None, np.inf
    for pi0 in piA_RT_0:
        x0 = np.concatenate([x_init, [pi0]])
        for method in ["Nelder-Mead", "Powell", "COBYLA"]:
            try:
                res = minimize(_err, x0, method=method)
                if res.fun < best_fn:
                    best_fn, best_x = res.fun, res.x
            except Exception:
                pass
        if best_fn < 1.0e-6:
            break
        # búsqueda global si los locales no convergen bien
        if best_fn > 1.0e-2:
            bounds = [[0.0, 1.0]] * (nc - 1) + [[0.0, 200.0]]
            try:
                res = differential_evolution(_err, bounds)
                if res.fun < best_fn:
                    best_fn, best_x = res.fun, res.x
            except Exception:
                pass

    # ── reconstruir q ────────────────────────────────────────────────────────
    x_sol = np.zeros(nc)
    x_sol[:-1] = best_x[:nc - 1]
    x_sol[-1]  = max(1.0 - np.sum(x_sol[:-1]), 0.0)

    arg_nz = x_sol > 0.0
    P0_sol = np.zeros(nc)
    P0_sol[arg_nz] = P_i[arg_nz] / x_sol[arg_nz]

    q_pure = np.array([iso_fns[i](P0_sol[i], T) for i in range(nc)])
    q_tot  = 1.0 / np.sum(x_sol / np.maximum(q_pure, 1.0e-30))
    return q_tot * x_sol


# ═══════════════════════════════════════════════════════════════════════════════
# RAST
# ═══════════════════════════════════════════════════════════════════════════════

@profiled
def rast(
    iso_fns: List[Callable],
    P_i: np.ndarray,
    T: float,
    Lambda: np.ndarray,
    C: float,
    n_pts: int = 101,
) -> np.ndarray:
    """
    Real Adsorbed Solution Theory con modelo de Wilson.

    Parameters
    ----------
    iso_fns : lista de nc callables iso_fn_i(P_bar, T_K) → q [mol/kg]
    P_i     : presiones parciales [bar], shape (nc,)
    T       : temperatura [K]
    Lambda  : matriz de parámetros de Wilson (nc, nc)
    C       : parámetro de heterogeneidad [-]
    n_pts   : puntos de cuadratura

    Returns
    -------
    q_mix : ndarray (nc,) — carga adsorbida por componente [mol/kg]
    """
    nc  = len(iso_fns)
    P_i = np.asarray(P_i, dtype=float)
    Lam = np.asarray(Lambda, dtype=float)

    def _err(vars_):
        x_head = vars_[:nc - 1]
        x_last = 1.0 - np.sum(x_head)
        x = np.concatenate([x_head, [x_last]])

        penalty = 0.0
        for ii in range(nc):
            if x[ii] < 1.0e-4:
                penalty += 50.0 * x[ii] ** 2
                x[ii] = 1.0e-4
            elif x[ii] > 0.9995:
                penalty += 50.0 * (x[ii] - 0.9995) ** 2
                x[ii] = 0.9995

        piA_RT   = vars_[-1]
        ln_gamma = ln_gamma_wilson(x, Lam, C, piA_RT)
        gamma    = np.exp(ln_gamma)
        P0_i     = P_i / (np.maximum(gamma, 1.0e-30) * np.maximum(x, 1.0e-30))
        spr_new  = np.array([_spreading_pressure(iso_fns[i], P0_i[i], T, n_pts) for i in range(nc)])
        return float(np.sum((spr_new - piA_RT) ** 2)) + penalty

    piA_RT_0 = [_spreading_pressure(iso_fns[i], P_i[i], T, n_pts) for i in range(nc)]
    y_i = P_i / np.sum(P_i)
    x_init = y_i[:-1]

    best_x, best_fn = None, np.inf
    for pi0 in piA_RT_0:
        x0 = np.concatenate([x_init, [pi0]])
        for method in ["Nelder-Mead", "Powell", "COBYLA"]:
            try:
                res = minimize(_err, x0, method=method)
                if res.fun < best_fn:
                    best_fn, best_x = res.fun, res.x
            except Exception:
                pass
        if best_fn < 1.0e-2:
            break

    x_sol = np.zeros(nc)
    x_sol[:-1] = best_x[:nc - 1]
    x_sol[-1]  = max(1.0 - np.sum(x_sol[:-1]), 0.0)
    piA_RT_sol = best_x[-1]

    ln_gamma_sol = ln_gamma_wilson(x_sol, Lam, C, piA_RT_sol)
    gamma_sol    = np.exp(ln_gamma_sol)

    arg_nz = x_sol > 0.0
    P0_sol = np.zeros(nc)
    P0_sol[arg_nz] = P_i[arg_nz] / (x_sol[arg_nz] * gamma_sol[arg_nz])

    q_pure = np.array([iso_fns[i](P0_sol[i], T) for i in range(nc)])
    q_tot  = 1.0 / np.sum(x_sol / np.maximum(q_pure, 1.0e-30))
    return q_tot * x_sol
