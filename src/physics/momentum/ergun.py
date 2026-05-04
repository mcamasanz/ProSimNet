"""
@module ergun
@brief Velocidad superficial del gas en caras de celda mediante la ecuación de Ergun.

@details
Este módulo implementa el cálculo del campo de velocidades en caras interiores
del dominio discretizado aplicando la ecuación de Ergun, que relaciona el
gradiente de presión con la velocidad superficial en un lecho empacado.

La ecuación de Ergun es:

    -dP/dz = A * u² + B * u

donde:
    A = 1.75 * rho_g * (1 - eps) / (eps³ * dp)       [kg/m⁴]    término inercial
    B = 150 * mu_g  * (1 - eps)² / (eps³ * dp²)      [Pa·s/m²]  término viscoso

Nota: en la implementación se usa eps_ratio = (1-eps)/eps para simplificar,
lo que implica que eps³ aparece implícitamente al combinar con el término
de velocidad. Verificar consistencia con la formulación de referencia al
cambiar el modelo de presión.

La ecuación es cuadrática en u. Se resuelve analíticamente por nodo
para las caras interiores; las caras de contorno (inlet/outlet) reciben
las velocidades impuestas externamente.

Referencia: Ergun, S. (1952). Chem. Eng. Progress 48(2), 89-94.

Unidades: SI.
  P     [Pa]
  rho   [kg/m³]
  mu    [Pa·s]
  dp    [m]
  eps   [-]
  dz    [m]
  v     [m/s]
"""

from __future__ import annotations

import numpy as np

from src.utils.profiling import profiled


@profiled
def ergun_face_velocity(
    P: np.ndarray,
    rho_g: np.ndarray,
    mu_g: np.ndarray,
    epsi: float,
    dp: float,
    dz: float,
    v_in: float,
    v_out: float,
) -> np.ndarray:
    """
    @brief
    Calcula la velocidad superficial del gas en todas las caras del dominio
    (nn + 1 caras) usando la ecuación de Ergun para las caras internas y las
    velocidades de contorno para las caras extremas.

    @details
    Para cada cara interior k (k = 1, ..., nn-1) se calcula el gradiente
    de presión discretizado entre las celdas adyacentes:

        dP/dz ≈ (P[k] - P[k-1]) / dz

    y se resuelve la ecuación cuadrática de Ergun:

        A * v² + B * v + dP/dz = 0

    donde A y B se evalúan con propiedades interpoladas en la cara
    (media aritmética de los dos nodos adyacentes).

    Casos de la cuadrática:
    - A ≈ 0: régimen viscoso puro -> v = -dP/dz / B
    - A, B > 0, discriminante >= 0: tomar la raíz de menor módulo
    - discriminante < 0 (físicamente raro): se usa la raíz de la forma
      con signo seguro para mantener la dirección del flujo.

    Parameters
    ----------
    P : np.ndarray, shape (nn,)
        Presión total del gas en centros de celda [Pa].
    rho_g : np.ndarray, shape (nn,)
        Densidad del gas en centros de celda [kg/m³].
    mu_g : np.ndarray, shape (nn,)
        Viscosidad dinámica del gas en centros de celda [Pa·s].
    epsi : float
        Porosidad del lecho [-]. Debe satisfacer 0 < epsi < 1.
    dp : float
        Diámetro de partícula [m]. Debe ser > 0.
    dz : float
        Tamaño de celda axial [m]. Debe ser > 0.
    v_in : float
        Velocidad superficial impuesta en la cara de entrada (cara 0) [m/s].
        Positiva en la dirección del flujo.
    v_out : float
        Velocidad superficial impuesta en la cara de salida (cara nn) [m/s].

    Returns
    -------
    v_face : np.ndarray, shape (nn + 1,)
        Velocidad superficial del gas en todas las caras [m/s].
        v_face[0] = v_in, v_face[-1] = v_out.

    Notes
    -----
    - La convención de signo es: flujo positivo de cara 0 a cara nn (inlet
      a outlet en dirección forward). Un gradiente de presión negativo
      (P decreciente en z) produce velocidad positiva.
    - eps_ratio = (1 - eps) / eps. Los coeficientes de Ergun usan esta
      simplificación respecto a la forma canónica con eps³. Verificar si
      la formulación del resto del modelo (balance de momento) es coherente.
    - Para nc = 1 (gas puro) la función sigue siendo válida.

    Side Effects
    ------------
    Ninguno. Función pura.
    """
    P_arr   = np.asarray(P,     dtype=float).reshape(-1)
    rho_arr = np.asarray(rho_g, dtype=float).reshape(-1)
    mu_arr  = np.asarray(mu_g,  dtype=float).reshape(-1)

    nn = P_arr.size
    eps_ratio = (1.0 - float(epsi)) / float(epsi)   # (1-eps)/eps [-]
    dp_val = float(dp)
    dz_val = float(dz)

    v_face = np.zeros(nn + 1, dtype=float)
    v_face[0]  = float(v_in)
    v_face[-1] = float(v_out)

    for k in range(1, nn):
        # Propiedades interpoladas en la cara k (entre celdas k-1 y k)
        rho_face = 0.5 * (rho_arr[k - 1] + rho_arr[k])
        mu_face  = 0.5 * (mu_arr[k - 1]  + mu_arr[k])

        # Gradiente de presión (positivo -> P crece en z -> flujo negativo)
        dPdz = (P_arr[k] - P_arr[k - 1]) / dz_val

        # Coeficientes de Ergun en la cara
        A = 1.75 * rho_face * eps_ratio / dp_val         # [kg/m⁴]
        B = 150.0 * mu_face * eps_ratio ** 2 / dp_val ** 2  # [kg/m⁴/s]
        C = dPdz                                          # [Pa/m]

        # Resolver A v² + B v + C = 0
        if abs(A) < 1.0e-30:
            # Régimen viscoso: B v + C = 0
            v_face[k] = -C / B if abs(B) > 1.0e-30 else 0.0
        else:
            disc = B ** 2 - 4.0 * A * C
            if disc >= 0.0:
                # Dos raíces reales: tomar la de menor módulo (físicamente correcta)
                v_face[k] = (-B + np.sqrt(disc)) / (2.0 * A)
            else:
                # Discriminante negativo: usar forma alternativa numéricamente estable
                v_face[k] = -(B + np.sqrt(B ** 2 + 4.0 * A * C)) / (2.0 * A)

    return v_face
