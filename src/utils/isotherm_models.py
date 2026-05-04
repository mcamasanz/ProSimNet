"""
@module isotherm_models
@brief Modelos de isoterma de adsorción pura y factor de Arrhenius.

@details
Todos los modelos reciben la presión en **bar** y devuelven la carga
adsorbida en **mol/kg**.

Modelos disponibles:

    Langmuir (2 parámetros)  : Lang(par, P)
    Freundlich (2 parámetros): Freu(par, P)
    Quadratic (3 parámetros) : Quad(par, P)
    Sips (3 parámetros)      : Sips(par, P)
    Dual-site Langmuir (4 p) : DSLa(par, P)

El factor de Arrhenius `arrh(T, dH, T_ref)` permite escalar la constante
de afinidad con la temperatura a partir de la entalpía de adsorción dH [J/mol].

El registro `MODEL_REGISTRY` mapea nombre → (función, n_parámetros) y es
la fuente de verdad para el ajuste automático de modelos.

Unidades:
    P    : bar
    q    : mol/kg
    T    : K
    dH   : J/mol
"""

from __future__ import annotations

import numpy as np

from src.utils.profiling import profiled

_R_GAS = 8.31446261815324   # J/mol/K


# ═══════════════════════════════════════════════════════════════════════════════
# Factor de Arrhenius
# ═══════════════════════════════════════════════════════════════════════════════

@profiled
def arrh(T: float | np.ndarray, dH: float, T_ref: float) -> float | np.ndarray:
    """
    Factor de Arrhenius para escalar la constante de afinidad con la temperatura.

    Parameters
    ----------
    T     : temperatura actual [K]
    dH    : entalpía de adsorción [J/mol] (valor absoluto; puede ser negativo)
    T_ref : temperatura de referencia [K]

    Returns
    -------
    float | ndarray — factor exp(|dH|/R · (1/T − 1/T_ref)) [-]
    """
    return np.exp(np.abs(dH) / _R_GAS * (1.0 / T - 1.0 / T_ref))


# ═══════════════════════════════════════════════════════════════════════════════
# Modelos de isoterma pura  q = f(par, P)
# ═══════════════════════════════════════════════════════════════════════════════

@profiled
def langmuir(par: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Isoterma de Langmuir.

    Parameters
    ----------
    par : [qs (mol/kg), b (1/bar)]
    P   : presión parcial [bar]

    Returns
    -------
    q [mol/kg]
    """
    qs, b = par[0], par[1]
    bP = b * np.asarray(P, dtype=float)
    return qs * bP / (1.0 + bP)


@profiled
def freundlich(par: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Isoterma de Freundlich.

    Parameters
    ----------
    par : [K (mol/kg/bar^n), n (-)]
    P   : presión parcial [bar]

    Returns
    -------
    q [mol/kg]
    """
    K, n = par[0], par[1]
    return K * np.asarray(P, dtype=float) ** n


@profiled
def quadratic(par: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Isoterma cuadrática (Quadratic).

    Parameters
    ----------
    par : [qs (mol/kg), b1 (1/bar), b2 (1/bar²)]
    P   : presión parcial [bar]

    Returns
    -------
    q [mol/kg]
    """
    qs, b1, b2 = par[0], par[1], par[2]
    P_arr = np.asarray(P, dtype=float)
    bP  = b1 * P_arr
    dPP = b2 * P_arr ** 2
    return qs * (bP + 2.0 * dPP) / (1.0 + bP + dPP)


@profiled
def sips(par: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Isoterma de Sips (Langmuir–Freundlich).

    Parameters
    ----------
    par : [qs (mol/kg), b (1/bar^n), n (-)]
    P   : presión parcial [bar]

    Returns
    -------
    q [mol/kg]
    """
    qs, b, n = par[0], par[1], par[2]
    P_arr = np.asarray(P, dtype=float)
    bPn = b * P_arr ** n
    return qs * bPn / (1.0 + bPn)


@profiled
def dual_site_langmuir(par: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Isoterma de Langmuir de dos sitios (DSL).

    Parameters
    ----------
    par : [qs1 (mol/kg), b1 (1/bar), qs2 (mol/kg), b2 (1/bar)]
    P   : presión parcial [bar]

    Returns
    -------
    q [mol/kg]
    """
    qs1, b1, qs2, b2 = par[0], par[1], par[2], par[3]
    P_arr = np.asarray(P, dtype=float)
    site1 = qs1 * b1 * P_arr / (1.0 + b1 * P_arr)
    site2 = qs2 * b2 * P_arr / (1.0 + b2 * P_arr)
    return site1 + site2


# ═══════════════════════════════════════════════════════════════════════════════
# Registro de modelos
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY: dict[str, tuple] = {
    #  nombre          función             n_par
    "Langmuir":        (langmuir,          2),
    "Freundlich":      (freundlich,        2),
    "Quadratic":       (quadratic,         3),
    "Sips":            (sips,              3),
    "Dual-site Langmuir": (dual_site_langmuir, 4),
}
