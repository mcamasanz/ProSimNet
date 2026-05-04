"""
@module diffusion
@brief Coeficientes de difusión multicomponente: binaria, mezcla, Knudsen,
       porosa, efectiva y dispersión axial.

@details
Jerarquía de difusión implementada en este módulo:

  Chapman-Enskog -> D_ij(T, P)          Difusión binaria en gas libre
  Stefan-Maxwell -> D_im(x, D_ij)       Difusión de la especie i en la mezcla
  Bosanquet      -> D_por(D_im, D_Kn)   Difusión combinada en poro (molecular + Knudsen)
  Medio poroso   -> D_eff(D_por, eps, tau)  Difusión efectiva en el lecho
  Dispersión     -> D_z(u, dp, Pe)      Dispersión axial de masa

Cada función es pura: acepta arrays numpy, devuelve arrays numpy, sin estado.

Unidades del sistema base: SI.
  T        [K]
  P        [Pa]
  MW       [kg/mol]
  sigma    [Å]   (Angstrom, parámetro LJ)
  epskB    [K]   (parámetro LJ)
  r_p      [m]   radio de poro
  D        [m²/s]
  eps      [-]   porosidad
  tau      [-]   tortuosidad
  u        [m/s] velocidad superficial
  dp       [m]   diámetro de partícula

Referencia Chapman-Enskog:
  Hirschfelder, J.O., Curtiss, C.F., Bird, R.B. (1964).
  Molecular Theory of Gases and Liquids. Wiley.
  Forma práctica: Bird, Stewart & Lightfoot, Transport Phenomena, 2002.
"""

from __future__ import annotations

import numpy as np

from src.utils.profiling import profiled

R_UNIVERSAL = 8.3145  # J/mol/K


# =========================================================
# 1. Integral de colisión para difusión (Neufeld)
# =========================================================

@profiled
def omega_D(Tstar: np.ndarray) -> np.ndarray:
    """
    @brief
    Integral de colisión adimensional Ω_D(T*) para difusión molecular,
    usando el ajuste clásico de Neufeld (1972).

    @details
    Ω_D(T*) controla la dependencia de D_ij con la temperatura. A T* alto
    (alta T o baja interacción ε/k) Ω_D → 1 y D crece como T^3/2.
    A T* bajo, la interacción molecular es más intensa y Ω_D > 1.

    La correlación de Neufeld es:
        Ω_D = 1.06036/T*^0.15610
            + 0.19300 * exp(-0.47635 * T*)
            + 1.03587 * exp(-1.52996 * T*)
            + 1.76474 * exp(-3.89411 * T*)

    Parameters
    ----------
    Tstar : np.ndarray (cualquier shape)
        Temperatura reducida T* = T / (ε/k) [-].

    Returns
    -------
    omega : np.ndarray, mismo shape que Tstar
        Integral de colisión [-].

    Notes
    -----
    - Válida para T* ∈ [0.3, 100]. Fuera de ese rango la precisión disminuye.
    - T* = 0 produciría división por cero; el llamador debe garantizar Tstar > 0.
    """
    T = np.asarray(Tstar, dtype=float)
    return (
        1.06036 / T ** 0.15610
        + 0.19300 * np.exp(-0.47635 * T)
        + 1.03587 * np.exp(-1.52996 * T)
        + 1.76474 * np.exp(-3.89411 * T)
    )


# =========================================================
# 2. Difusión binaria Chapman-Enskog: D_ij(T, P)
# =========================================================

@profiled
def binary_diffusivity(
    T: np.ndarray,
    P: np.ndarray,
    MW: np.ndarray,
    sigma_A: np.ndarray,
    epskB_K: np.ndarray,
) -> np.ndarray:
    """
    @brief
    Calcula los coeficientes de difusión binaria D_ij para todos los pares de
    especies y todos los nodos del dominio, usando la correlación de
    Chapman-Enskog.

    @details
    La correlación de Chapman-Enskog en unidades prácticas es:

        D_ij [cm²/s] = 0.001858 * T^1.5 / (P_atm * sigma_ij² * Ω_D)
                       * sqrt(1/Mi + 1/Mj)

    con Mi, Mj en g/mol, sigma_ij en Å, T en K, P en atm.
    Conversión: 1 cm²/s = 1e-4 m²/s.

    Los parámetros cruzados se calculan como:
        sigma_ij = 0.5 * (sigma_i + sigma_j)          [Å]
        eps_ij   = sqrt(eps_i * eps_j)                [K]

    La diagonal de D_ij se fija a np.inf para que la inversión en la
    aproximación de Stefan-Maxwell no contribuya al término i=j.

    Parameters
    ----------
    T : np.ndarray, shape (nn,)
        Temperatura por nodo [K].
    P : np.ndarray, shape (nn,)
        Presión total por nodo [Pa].
    MW : np.ndarray, shape (nc,)
        Masa molar por especie [kg/mol].
    sigma_A : np.ndarray, shape (nc,)
        Parámetro sigma de Lennard-Jones por especie [Å].
    epskB_K : np.ndarray, shape (nc,)
        Parámetro epsilon/k_B de Lennard-Jones por especie [K].

    Returns
    -------
    Dij : np.ndarray, shape (nn, nc, nc)
        Coeficientes de difusión binaria por nodo y par de especies [m²/s].
        La diagonal es np.inf.

    Notes
    -----
    - La correlación es válida para gases de bajo a medio peso molecular a
      presiones hasta ~10 atm y temperaturas 200-2000 K.
    - P_atm se protege con max(..., 1e-12) para evitar división por cero.
    - sqrt_invM_sum, sigma_ij, eps_ij son matrices (nc, nc) construidas una
      sola vez fuera del bucle sobre nodos para reducir coste.
    """
    T_arr       = np.asarray(T,       dtype=float).reshape(-1)
    P_arr       = np.asarray(P,       dtype=float).reshape(-1)
    MW_arr      = np.asarray(MW,      dtype=float).reshape(-1)
    sigma_arr   = np.asarray(sigma_A, dtype=float).reshape(-1)
    epskB_arr   = np.asarray(epskB_K, dtype=float).reshape(-1)

    nn = T_arr.size
    nc = MW_arr.size

    P_atm_arr   = P_arr / 1.01325e5           # Pa -> atm
    MW_gmol_arr = MW_arr * 1.0e3              # kg/mol -> g/mol

    # Matrices de parámetros cruzados (nc, nc), construidas una sola vez
    inv_M           = 1.0 / np.maximum(MW_gmol_arr, 1.0e-300)
    sqrt_invM_sum   = np.sqrt(inv_M[:, None] + inv_M[None, :])        # (nc, nc)
    sigma_ij        = 0.5 * (sigma_arr[:, None] + sigma_arr[None, :]) # (nc, nc) [Å]
    eps_ij          = np.sqrt(np.maximum(
        epskB_arr[:, None] * epskB_arr[None, :], 1.0e-300
    ))                                                                 # (nc, nc) [K]

    Dij = np.empty((nn, nc, nc), dtype=float)

    for k in range(nn):
        Tk   = float(T_arr[k])
        Patm = max(float(P_atm_arr[k]), 1.0e-12)

        Tstar = Tk / np.maximum(eps_ij, 1.0e-30)
        Omega = omega_D(Tstar)

        Dij_k = (
            1.0e-4                                       # cm²/s -> m²/s
            * (0.001858 * Tk ** 1.5)
            / (Patm
               * np.maximum(sigma_ij ** 2, 1.0e-300)
               * np.maximum(Omega,         1.0e-300))
            * sqrt_invM_sum
        )
        np.fill_diagonal(Dij_k, np.inf)
        Dij[k] = Dij_k

    return Dij


# =========================================================
# 3. Difusión de la especie i en la mezcla: Stefan-Maxwell
# =========================================================

@profiled
def mixture_diffusivity(
    x: np.ndarray,
    Dij: np.ndarray,
) -> np.ndarray:
    """
    @brief
    Calcula el coeficiente de difusión efectiva de cada especie en la mezcla
    usando la aproximación de Stefan-Maxwell (Wilke):

        1 / D_im = Σ_{j≠i} x_j / D_ij

    @details
    Esta aproximación es exacta para mezclas binarias y una aproximación
    razonable para multicomponente cuando una especie domina (dilución).
    Es la formulación estándar en modelos PSA de lecho fijo.

    Parameters
    ----------
    x : np.ndarray, shape (nn, nc)
        Fracción molar por nodo y especie [-].
    Dij : np.ndarray, shape (nn, nc, nc)
        Difusividades binarias por nodo [m²/s]. Diagonal = np.inf.

    Returns
    -------
    Dim : np.ndarray, shape (nn, nc)
        Difusividad de cada especie en la mezcla por nodo [m²/s].

    Notes
    -----
    - 1/inf = 0 por IEEE 754, por lo que la diagonal de Dij no contribuye
      al denominador de Stefan-Maxwell automáticamente.
    - El denominador se protege con maximum(..., 1e-30) para evitar
      explosión cuando todas las fracciones molares son nulas.
    """
    x_mat  = np.asarray(x,   dtype=float)
    Dij_mat = np.asarray(Dij, dtype=float)

    nn, nc = x_mat.shape
    Dim = np.empty((nn, nc), dtype=float)

    for k in range(nn):
        inv_D = 1.0 / Dij_mat[k]                            # diag -> 0 (1/inf=0)
        inv_D[np.eye(nc, dtype=bool)] = 0.0                 # garantizar 0 en diag
        # sum_j x_j / D_ij para cada i (fila de inv_D da los 1/D_ij por columna j)
        denom = (x_mat[k, None, :] @ inv_D.T).ravel()       # shape (nc,)
        Dim[k] = 1.0 / np.maximum(denom, 1.0e-30)

    return Dim


# =========================================================
# 4. Difusión de Knudsen intra-partícula
# =========================================================

@profiled
def knudsen_diffusivity(
    T: np.ndarray,
    MW: np.ndarray,
    r_pore: np.ndarray,
) -> np.ndarray:
    """
    @brief
    Calcula el coeficiente de difusión de Knudsen para cada especie en poros
    cilíndricos de radio r_pore:

        D_Kn,i = (2/3) * r_pore * sqrt(8 R T / (π M_i))

    @details
    La difusión de Knudsen domina cuando el recorrido libre medio del gas es
    mayor que el diámetro de poro (número de Knudsen Kn >> 1), típico en
    microporos de zeolitas (r_pore ~ 1-5 nm).

    Parameters
    ----------
    T : np.ndarray, shape (nn,)
        Temperatura del gas por nodo [K].
    MW : np.ndarray, shape (nc,)
        Masa molar por especie [kg/mol].
    r_pore : np.ndarray, shape (nn,) o escalar
        Radio de poro por nodo [m].

    Returns
    -------
    Dkn : np.ndarray, shape (nn, nc)
        Difusividad de Knudsen por nodo y especie [m²/s].

    Notes
    -----
    - La expresión es para poros cilíndricos con reflexión difusa en la pared.
    - r_pore puede ser escalar (poro uniforme) o array (poro variable axialmente).
    """
    T_arr   = np.asarray(T,  dtype=float).reshape(-1)
    MW_arr  = np.asarray(MW, dtype=float).reshape(-1)
    nn = T_arr.size
    nc = MW_arr.size

    rp = np.asarray(r_pore, dtype=float)
    if rp.ndim == 0:
        rp = np.full(nn, float(rp), dtype=float)
    else:
        rp = rp.reshape(-1)

    coef = (2.0 / 3.0) * rp[:, None]                    # (nn, 1)
    Dkn  = coef * np.sqrt(
        8.0 * R_UNIVERSAL * T_arr[:, None]
        / (np.pi * np.maximum(MW_arr[None, :], 1.0e-300))
    )                                                    # (nn, nc) [m²/s]
    return Dkn


# =========================================================
# 5. Difusión combinada en poro: Bosanquet
# =========================================================

@profiled
def pore_diffusivity(
    Dim: np.ndarray,
    Dkn: np.ndarray,
) -> np.ndarray:
    """
    @brief
    Combina difusión molecular (D_im) y de Knudsen (D_Kn,i) mediante la
    relación de Bosanquet para obtener la difusividad total en el poro:

        1 / D_por,i = 1 / D_im,i + 1 / D_Kn,i

    @details
    La relación de Bosanquet es la aproximación harmónica de las dos
    resistencias de difusión en serie. Dominada por la menor de las dos.
    Para poros grandes (meso/macro), D_Kn >> D_im y D_por ≈ D_im.
    Para microporos, D_Kn << D_im y D_por ≈ D_Kn.

    Parameters
    ----------
    Dim : np.ndarray, shape (nn, nc)
        Difusividad molecular en la mezcla [m²/s].
    Dkn : np.ndarray, shape (nn, nc)
        Difusividad de Knudsen [m²/s].

    Returns
    -------
    Dpor : np.ndarray, shape (nn, nc)
        Difusividad en el poro [m²/s].
    """
    Dim_ = np.asarray(Dim, dtype=float)
    Dkn_ = np.asarray(Dkn, dtype=float)
    return 1.0 / np.maximum(
        1.0 / np.maximum(Dim_, 1.0e-30) + 1.0 / np.maximum(Dkn_, 1.0e-30),
        1.0e-30
    )


# =========================================================
# 6. Difusividad efectiva en el lecho: corrección porosa
# =========================================================

@profiled
def effective_diffusivity(
    Dpor: np.ndarray,
    eps: np.ndarray,
    tau: np.ndarray,
) -> np.ndarray:
    """
    @brief
    Calcula la difusividad efectiva en el lecho poroso aplicando la corrección
    de porosidad y tortuosidad:

        D_eff,i = (eps / tau) * D_por,i

    @details
    Esta corrección transforma la difusividad en el espacio de los poros
    (D_por) en la difusividad efectiva a escala de celda del lecho (D_eff).
    El factor eps/tau < 1 refleja que solo una fracción eps del volumen de
    celda es accesible al gas, y que los caminos de difusión son más largos
    por un factor tau (tortuosidad).

    Parameters
    ----------
    Dpor : np.ndarray, shape (nn, nc)
        Difusividad en el poro [m²/s].
    eps : np.ndarray, shape (nn,) o escalar
        Porosidad del lecho por nodo [-]. 0 < eps < 1.
    tau : np.ndarray, shape (nn,) o escalar
        Tortuosidad por nodo [-]. tau >= 1.

    Returns
    -------
    Deff : np.ndarray, shape (nn, nc)
        Difusividad efectiva en el lecho [m²/s].

    Notes
    -----
    - En zonas de flujo libre (eps = 1), la corrección desaparece y Deff = Dpor.
    - tau se protege con maximum(..., 1e-30) para evitar división por cero.
    """
    Dpor_ = np.asarray(Dpor, dtype=float)
    nn    = Dpor_.shape[0]

    eps_ = np.asarray(eps, dtype=float)
    tau_ = np.asarray(tau, dtype=float)
    if eps_.ndim == 0:
        eps_ = np.full(nn, float(eps_), dtype=float)
    if tau_.ndim == 0:
        tau_ = np.full(nn, float(tau_), dtype=float)

    return (eps_[:, None] / np.maximum(tau_[:, None], 1.0e-30)) * Dpor_


# =========================================================
# 7. Dispersión axial de masa
# =========================================================

@profiled
def axial_dispersion(
    u: np.ndarray,
    dp: float,
    Pe: float,
) -> np.ndarray:
    """
    @brief
    Calcula el coeficiente de dispersión axial de masa mediante la correlación:

        D_z,i = u * dp / Pe

    @details
    La dispersión axial en lechos empacados se modela habitualmente como una
    difusión aumentada con un coeficiente D_z proporcional a la velocidad
    superficial u y al diámetro de partícula dp, e inversamente al número de
    Peclet de dispersión Pe.

    Esta expresión es uniforme para todas las especies (D_z no depende de i).
    Si se desea dispersión diferencial por especie, debe extenderse el modelo.

    Parameters
    ----------
    u : np.ndarray, shape (nn,)
        Velocidad superficial del gas por nodo [m/s].
    dp : float
        Diámetro de partícula [m].
    Pe : float
        Número de Peclet de dispersión axial [-]. Típicamente Pe ≈ 2.

    Returns
    -------
    Dz : np.ndarray, shape (nn, 1)
        Coeficiente de dispersión axial [m²/s]. Shape (nn, 1) para facilitar
        broadcasting con arrays de concentración (nn, nc).

    Notes
    -----
    - La formulación actual no depende de la especie. Si se necesita
      D_z,i = f(D_eff,i), extender esta función con un argumento Deff.
    - Pe se protege contra valores nulos con maximum(..., 1e-30).
    """
    u_arr = np.asarray(u, dtype=float).reshape(-1)
    Dz = (u_arr * float(dp)) / max(float(Pe), 1.0e-30)
    return Dz[:, None]   # (nn, 1) para broadcast con (nn, nc)
