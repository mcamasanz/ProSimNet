"""
@module mixture
@brief Propiedades termofísicas de mezclas gaseosas: regla de Wilke y magnitudes derivadas.

@details
Este módulo implementa la regla de mezcla de Wilke para viscosidad y
conductividad térmica de mezclas multicomponente, y las funciones de
mezcla lineal (promedio molar) para Cp, h, u y s.

La regla de Wilke es la aproximación estándar para mezclas de gases ideales
con número de componentes pequeño (nc ≤ 10). Su coste es O(nc²) por nodo,
aceptable para las discretizaciones típicas de columnas PSA.

Convención dimensional:
  x        [-]       fracción molar, shape (nn, nc), suma 1 en cada nodo.
  prop_i   [SI]      propiedad pura por nodo y especie, shape (nn, nc).
  MW       [kg/mol]  masa molar por especie, shape (nc,).
  resultado [SI]     propiedad de mezcla por nodo, shape (nn,).

Referencia: Wilke, C.R. (1950). J. Chem. Phys. 18(4), 517-519.
"""

from __future__ import annotations

import numpy as np

from src.utils.profiling import profiled


# =========================================================
# 1. Factor phi_ij de Wilke (escalar)
# =========================================================

@profiled
def wilke_phi(MWi: float, MWj: float, propi: float, propj: float) -> float:
    """
    @brief
    Calcula el factor adimensional phi_ij de la regla de mezcla de Wilke
    para el par de especies (i, j).

    @details
    La expresión de Wilke es:

        phi_ij = [1 + (prop_i/prop_j)^0.5 * (MW_j/MW_i)^0.25]^2
                 / [sqrt(8) * sqrt(1 + MW_i/MW_j)]

    Es el peso que tiene la especie j sobre la contribución de la especie i
    a la propiedad de mezcla. phi_ii = 1 por definición.

    Parameters
    ----------
    MWi, MWj : float
        Masas molares de especies i y j [kg/mol].
    propi, propj : float
        Propiedad pura de las especies i y j (viscosidad o conductividad) [SI].

    Returns
    -------
    phi_ij : float
        Factor adimensional de Wilke [-].

    Notes
    -----
    - Los valores negativos o nulos en MW o prop se protegen con max(..., 1e-300)
      para evitar divisiones por cero. En un gas real estas condiciones no ocurren.
    - Esta función es escalar; la vectorización sobre nodos la realiza
      wilke_mix_property.
    """
    MWi_  = max(float(MWi),    1.0e-300)
    MWj_  = max(float(MWj),    1.0e-300)
    pi_   = max(float(propi),  1.0e-300)
    pj_   = max(float(propj),  1.0e-300)

    num = (1.0 + np.sqrt(pi_ / pj_) * (MWj_ / MWi_) ** 0.25) ** 2
    den = np.sqrt(8.0 * (1.0 + MWi_ / MWj_))
    return float(num / den)


# =========================================================
# 2. Propiedad de mezcla por la regla de Wilke
# =========================================================

@profiled
def wilke_mix_property(
    x: np.ndarray,
    prop_i: np.ndarray,
    MW: np.ndarray,
) -> np.ndarray:
    """
    @brief
    Calcula la propiedad de mezcla (viscosidad o conductividad térmica) usando
    la regla de Wilke para todos los nodos del dominio.

    @details
    La regla de Wilke para una propiedad phi_mix es:

        phi_mix = Σ_i [ x_i * phi_i / Σ_j (x_j * phi_ij) ]

    donde phi_ij es el factor de Wilke calculado por wilke_phi.

    Esta función itera sobre nodos (nn) y para cada nodo construye la matriz
    phi (nc × nc) y aplica la fórmula. El coste es O(nn * nc²).

    Parameters
    ----------
    x : np.ndarray, shape (nn, nc)
        Fracción molar por nodo y especie [-]. Se asume que suma 1 en cada fila.
    prop_i : np.ndarray, shape (nn, nc)
        Propiedad pura (mu o k) por nodo y especie [SI].
    MW : np.ndarray, shape (nc,)
        Masa molar por especie [kg/mol].

    Returns
    -------
    prop_mix : np.ndarray, shape (nn,)
        Propiedad de mezcla por nodo [SI].

    Notes
    -----
    - El denominador se protege con maximum(..., 1e-300) para evitar
      divisiones por cero en celdas con fracción molar nula.
    - La función no modifica los arrays de entrada.
    """
    x_mat     = np.asarray(x,      dtype=float)
    prop_mat  = np.asarray(prop_i, dtype=float)
    MW_arr    = np.asarray(MW,     dtype=float).reshape(-1)

    nn, nc = x_mat.shape
    prop_mix = np.zeros(nn, dtype=float)

    for n in range(nn):
        phi_mat = np.ones((nc, nc), dtype=float)
        for i in range(nc):
            for j in range(nc):
                if i == j:
                    continue
                phi_mat[i, j] = wilke_phi(MW_arr[i], MW_arr[j],
                                          prop_mat[n, i], prop_mat[n, j])

        denom = np.zeros(nc, dtype=float)
        for i in range(nc):
            denom[i] = np.sum(x_mat[n, :] * phi_mat[i, :])

        prop_mix[n] = np.sum(
            x_mat[n, :] * prop_mat[n, :] / np.maximum(denom, 1.0e-300)
        )

    return prop_mix


# =========================================================
# 3. Propiedades de mezcla lineales (promedio molar)
# =========================================================

@profiled
def molar_mix_property(
    x: np.ndarray,
    prop_i: np.ndarray,
) -> np.ndarray:
    """
    @brief
    Calcula una propiedad molar de mezcla como promedio molar ponderado:
    prop_mix = Σ_i x_i * prop_i.

    @details
    Válido para propiedades extensivas molares en mezcla ideal: Cp_molar,
    h_molar, u_molar, s_molar. No es correcto para propiedades de transporte
    (usar wilke_mix_property para mu y k).

    Parameters
    ----------
    x : np.ndarray, shape (nn, nc)
        Fracción molar por nodo y especie [-].
    prop_i : np.ndarray, shape (nn, nc)
        Propiedad molar pura por nodo y especie [SI].

    Returns
    -------
    prop_mix : np.ndarray, shape (nn,)
        Propiedad molar de mezcla por nodo [SI].

    Notes
    -----
    - La función no valida que x sume 1; esa responsabilidad es del llamador.
    - Se usa para Cp_molar, h_molar, u_molar y s_molar.
    """
    x_mat    = np.asarray(x,      dtype=float)
    prop_mat = np.asarray(prop_i, dtype=float)
    return np.sum(x_mat * prop_mat, axis=1)


@profiled
def mixture_molar_mass(
    x: np.ndarray,
    MW: np.ndarray,
) -> np.ndarray:
    """
    @brief
    Calcula la masa molar de mezcla por nodo: MWmix = Σ_i x_i * MW_i.

    Parameters
    ----------
    x : np.ndarray, shape (nn, nc)
        Fracción molar por nodo y especie [-].
    MW : np.ndarray, shape (nc,)
        Masa molar por especie [kg/mol].

    Returns
    -------
    MWmix : np.ndarray, shape (nn,)
        Masa molar de mezcla por nodo [kg/mol].
    """
    x_mat  = np.asarray(x,  dtype=float)
    MW_arr = np.asarray(MW, dtype=float).reshape(-1)
    return x_mat @ MW_arr


@profiled
def ideal_gas_density(
    P: np.ndarray,
    MWmix: np.ndarray,
    Tg: np.ndarray,
    R: float = 8.3145,
) -> np.ndarray:
    """
    @brief
    Calcula la densidad de la mezcla gaseosa asumiendo gas ideal:
    rho = P * MWmix / (R * Tg).

    @details
    Esta es la única ecuación de estado utilizada en el modelo de columna PSA.
    La hipótesis de gas ideal es explícita y documentada aquí.

    Parameters
    ----------
    P : np.ndarray, shape (nn,)
        Presión total por nodo [Pa].
    MWmix : np.ndarray, shape (nn,)
        Masa molar de mezcla por nodo [kg/mol].
    Tg : np.ndarray, shape (nn,)
        Temperatura del gas por nodo [K].
    R : float
        Constante universal de los gases [J/mol/K]. Por defecto 8.3145.

    Returns
    -------
    rho : np.ndarray, shape (nn,)
        Densidad de mezcla por nodo [kg/m³].

    Notes
    -----
    - Hipótesis: gas ideal. Válida a presiones moderadas (< 10 bar) y
      temperaturas > 200 K típicas de procesos PSA.
    - No se aplican correcciones de fugacidad ni factor de compresibilidad.
    """
    P_arr     = np.asarray(P,     dtype=float).reshape(-1)
    MWmix_arr = np.asarray(MWmix, dtype=float).reshape(-1)
    Tg_arr    = np.asarray(Tg,    dtype=float).reshape(-1)
    return (P_arr * MWmix_arr) / (R * np.maximum(Tg_arr, 1.0e-30))
