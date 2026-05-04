"""
@module face_reconstruction
@brief Reconstrucción de propiedades escalares de centros de celda a caras en
       mallas 1D de volúmenes finitos.

@details
En una malla cell-centered 1D con nn celdas, existen nn+1 caras:
  - cara 0   : contorno de entrada (inlet)
  - caras 1..nn-1 : caras interiores
  - cara nn  : contorno de salida (outlet)

Este módulo implementa dos operaciones fundamentales de reconstrucción:

  1. face_property   : interpolación geométrica (aritmética o armónica) de una
                       propiedad escalar phi(celda) a phi(cara). Se usa para
                       propiedades difusivas como Gamma (conductividad, difusividad).

  2. upwind_reconstruction : reconstrucción upwind de primer orden de phi(celda)
                             a phi(cara), eligiendo el donor según el signo de la
                             velocidad en la cara. Se usa para propiedades
                             convectadas (concentración, temperatura, entalpía).

Convención de orientación:
  - El eje z crece de cara 0 (inlet) a cara nn (outlet).
  - Velocidad positiva: flujo de izquierda a derecha (inlet -> outlet).
  - Velocidad negativa: flujo de derecha a izquierda (outlet -> inlet).

Estas funciones son puras: no tienen estado, no modifican los arrays de entrada.

Unidades: las mismas que las del array de entrada (SI en el modelo de columna).
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from src.utils.profiling import profiled


# =========================================================
# 1. Interpolación geométrica de celda a cara
# =========================================================

@profiled
def face_property(
    phi_cell: np.ndarray,
    phi_in: Optional[float] = None,
    phi_out: Optional[float] = None,
    bc_in: Literal["copy", "dirichlet"] = "copy",
    bc_out: Literal["copy", "dirichlet"] = "copy",
    method: Literal["arithmetic", "harmonic"] = "arithmetic",
) -> np.ndarray:
    """
    @brief
    Interpola una propiedad escalar cell-centered phi(celda) a las caras de la
    malla, produciendo phi(cara) de shape (nn+1,).

    @details
    La operación distingue tres zonas:

    Cara de entrada (cara 0):
      - "copy"     : phi_face[0] = phi_cell[0]  (extrapolación con el primer nodo)
      - "dirichlet": phi_face[0] = phi_in        (valor impuesto)

    Caras interiores (caras 1..nn-1):
      - "arithmetic": phi_face[k] = 0.5 * (phi_cell[k-1] + phi_cell[k])
      - "harmonic"  : phi_face[k] = 2 phi_L phi_R / (phi_L + phi_R)
                      (media armónica, apropiada para propiedades como Γ cuando
                       hay gradientes fuertes o regiones heterogéneas)

    Cara de salida (cara nn):
      - "copy"     : phi_face[-1] = phi_cell[-1]
      - "dirichlet": phi_face[-1] = phi_out

    La interpolación aritmética es estándar y suficiente para propiedades
    suaves (mu, k, D). La armónica es más conservativa cuando Gamma cambia de
    orden de magnitud entre celdas adyacentes.

    Parameters
    ----------
    phi_cell : np.ndarray, shape (nn,)
        Propiedad escalar en centros de celda [SI].
    phi_in : float, opcional
        Valor en el contorno de entrada. Requerido si bc_in='dirichlet'.
    phi_out : float, opcional
        Valor en el contorno de salida. Requerido si bc_out='dirichlet'.
    bc_in : {"copy", "dirichlet"}
        Tipo de condición de contorno en la cara de entrada.
    bc_out : {"copy", "dirichlet"}
        Tipo de condición de contorno en la cara de salida.
    method : {"arithmetic", "harmonic"}
        Método de interpolación para caras interiores.

    Returns
    -------
    phi_face : np.ndarray, shape (nn+1,)
        Propiedad escalar interpolada en caras [SI].

    Notes
    -----
    - La media armónica tiene una singularidad si phi_L + phi_R = 0; se
      protege con maximum(..., 1e-300).
    - Esta función no aplica upwind: es solo geométrica. Para propiedades
      convectadas usar upwind_reconstruction.
    """
    phi_cell_arr = np.asarray(phi_cell, dtype=float).reshape(-1)
    nn = phi_cell_arr.size

    bc_in  = str(bc_in).strip().lower()
    bc_out = str(bc_out).strip().lower()
    method = str(method).strip().lower()

    phi_face = np.empty(nn + 1, dtype=float)

    # Cara de entrada
    phi_face[0] = float(phi_in) if bc_in == "dirichlet" else phi_cell_arr[0]

    # Caras interiores
    phi_L = phi_cell_arr[:-1]   # celdas izquierdas de cada cara interior
    phi_R = phi_cell_arr[1:]    # celdas derechas de cada cara interior

    if method == "harmonic":
        phi_face[1:nn] = (2.0 * phi_L * phi_R
                          / np.maximum(phi_L + phi_R, 1.0e-300))
    else:
        phi_face[1:nn] = 0.5 * (phi_L + phi_R)

    # Cara de salida
    phi_face[-1] = float(phi_out) if bc_out == "dirichlet" else phi_cell_arr[-1]

    return phi_face


# =========================================================
# 2. Reconstrucción upwind de primer orden
# =========================================================

@profiled
def upwind_reconstruction(
    phi_cell: np.ndarray,
    v_face: np.ndarray,
    phi_in: Optional[float] = None,
    phi_out: Optional[float] = None,
) -> np.ndarray:
    """
    @brief
    Reconstruye una propiedad escalar en caras usando el esquema upwind de
    primer orden: el valor en la cara es el del nodo donor (aguas arriba).

    @details
    La elección del donor depende del signo de la velocidad en la cara:

        v_face[k] >= 0  (flujo de izquierda a derecha):
            phi_face[k] = phi_cell[k-1]   (nodo izquierdo, donor)

        v_face[k] < 0  (flujo de derecha a izquierda):
            phi_face[k] = phi_cell[k]     (nodo derecho, donor)

    Para las caras de contorno:
        Cara 0 (entrada):
          v > 0: flujo entra -> phi_face[0] = phi_in  (valor exterior)
          v <= 0: flujo sale -> phi_face[0] = phi_cell[0]

        Cara nn (salida):
          v >= 0: flujo sale -> phi_face[-1] = phi_cell[-1]
          v < 0: flujo entra -> phi_face[-1] = phi_out (valor exterior)

    El upwind de primer orden es el esquema más robusto para convección
    dominante. Introduce difusión numérica proporcional a u * dz * d²phi/dz²,
    que es aceptable para las discretizaciones típicas de columnas PSA
    (frentes relativamente suaves).

    Parameters
    ----------
    phi_cell : np.ndarray, shape (nn,)
        Propiedad escalar en centros de celda [SI].
    v_face : np.ndarray, shape (nn+1,)
        Velocidad superficial en caras [m/s]. Positiva = izq -> der.
    phi_in : float, opcional
        Valor de la propiedad en el exterior de la cara de entrada.
        Requerido cuando v_face[0] > 0 (flujo que entra por la izquierda).
    phi_out : float, opcional
        Valor de la propiedad en el exterior de la cara de salida.
        Requerido cuando v_face[-1] < 0 (flujo que re-entra por la derecha).

    Returns
    -------
    phi_face : np.ndarray, shape (nn+1,)
        Propiedad escalar reconstruida en caras por upwind [SI].

    Notes
    -----
    - El esquema no interpola; selecciona. El valor en cada cara es
      exactamente el de uno de los dos nodos adyacentes.
    - Para v = 0 exacto, la convención adoptada es "derecha a izquierda"
      (v >= 0 -> izquierdo), que es la convención estándar en la clase Column.
    - phi_in y phi_out deben ser el valor de la propiedad en la condición de
      contorno, no el flujo. El flujo se calcula en convective_flux.
    """
    phi_cell_arr = np.asarray(phi_cell, dtype=float).reshape(-1)
    v_face_arr   = np.asarray(v_face,   dtype=float).reshape(-1)
    nn = phi_cell_arr.size

    phi_face = np.empty(nn + 1, dtype=float)

    # Cara de entrada (cara 0)
    if v_face_arr[0] > 0.0:
        # Flujo entra: el donor es el exterior (phi_in)
        phi_face[0] = float(phi_in) if phi_in is not None else phi_cell_arr[0]
    else:
        # Flujo sale: el donor es la primera celda
        phi_face[0] = phi_cell_arr[0]

    # Caras interiores vectorizadas
    v_int = v_face_arr[1:nn]                                 # (nn-1,)
    phi_face[1:nn] = np.where(
        v_int >= 0.0,
        phi_cell_arr[:-1],   # donor izquierdo
        phi_cell_arr[1:],    # donor derecho
    )

    # Cara de salida (cara nn)
    if v_face_arr[-1] >= 0.0:
        # Flujo sale: donor es la última celda
        phi_face[-1] = phi_cell_arr[-1]
    else:
        # Flujo re-entra: donor es el exterior (phi_out)
        phi_face[-1] = float(phi_out) if phi_out is not None else phi_cell_arr[-1]

    return phi_face
