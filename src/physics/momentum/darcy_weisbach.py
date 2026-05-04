"""
@module darcy_weisbach
@brief Velocidad superficial del gas en caras de celda mediante la ecuación de
       Darcy-Weisbach para flujo en tubo vacío.

@details
Análogo a `ergun_face_velocity` para lechos empaquetados, pero aplicado a
tubos de sección circular vacía. Relaciona el gradiente de presión local con
la velocidad superficial resolviendo la ecuación de Darcy-Weisbach en régimen
laminar o turbulento.

La ecuación de Darcy-Weisbach es:

    -dP/dz = f(Re) · ρ · v² / (2·D)

con el factor de fricción seleccionado por régimen:

    Laminar   (Re < 2300)  :  f = 64 / Re
                               → -dP/dz = 32 · µ · v / D²
                               → v = (-dP/dz) · D² / (32 · µ)        (lineal, exacta)

    Turbulento (Re ≥ 2300) :  f = 0.316 · Re^{-0.25}  [Blasius]
                               → -dP/dz = 0.158 · ρ^{0.75} · µ^{0.25} · D^{-1.25} · v^{1.75}
                               → v = ((-dP/dz) / C_turb)^{4/7}       (directa, sin iteración)

En ambos regímenes la solución es directa (sin iteración). El régimen se determina
calculando la velocidad laminar exacta y comprobando el Re resultante: si Re_lam
es autóconsistente con Re < 2300 se usa; en caso contrario se aplica Blasius.

La correlación de Blasius es válida para tubos hidráulicamente lisos con
4 000 < Re < 100 000. Para exploración fuera de este rango es conservadora pero
sigue siendo razonable como orden de magnitud.

Las caras de contorno (inlet y outlet) reciben las velocidades impuestas por
las condiciones de contorno del equipo; las caras interiores se resuelven con DW.

Referencia:
    Darcy, H. (1857). Recherches expérimentales relatives au mouvement de l'eau
    dans les tuyaux. Mallet-Bachelier, Paris.
    Blasius, H. (1913). Das Ähnlichkeitsgesetz bei Reibungsvorgängen in
    Flüssigkeiten. Forschungsarbeiten VDI 131.

Unidades: SI.
  P     [Pa]
  rho   [kg/m³]
  mu    [Pa·s]
  Di    [m]
  dz    [m]
  v     [m/s]
"""

from __future__ import annotations

import numpy as np

from src.utils.profiling import profiled


def continuity_face_velocity(
    C_tot_cell: np.ndarray,
    C_tot_in: float,
    v_in: float,
) -> np.ndarray:
    """
    @brief
    Propaga la velocidad de entrada a todas las caras usando conservación de masa.

    @details
    En flujo 1D compresible a baja velocidad, el gradiente de presión de
    Darcy-Weisbach es demasiado pequeño para ser resuelto por el integrador ODE
    (ΔP << RTOL·P). Usar DW para reconstruir v_face desde P conduce a velocidades
    interiores ≈ 0 mientras la BC impone v_in ≠ 0, creando un desequilibrio
    permanente en el RHS que fuerza pasos temporales diminutos.

    La solución correcta es usar la ecuación de continuidad en quasi-estado
    estacionario:

        ∂(v · C_tot) / ∂z = 0  →  v · C_tot = cte = v_in · C_tot_in

    de donde:

        v_face[k] = v_in · C_tot_in / C_tot_face[k]

    Esto es exacto para v_in = 0 (batch: todas las caras = 0) y robusto para
    cualquier velocidad. Captura la expansión del gas (C_tot decrece al calentarse
    → v aumenta aguas abajo) sin depender de la resolución de ΔP por el ODE.

    El cálculo de la caída de presión real (DW) queda como diagnóstico en
    `darcy_weisbach_face_velocity`, que recibe v_face y devuelve ΔP.

    Parameters
    ----------
    C_tot_cell : ndarray, shape (N,)
        Concentración molar total en centros de celda [mol/m³].
    C_tot_in : float
        Concentración molar total en la cara de entrada (de la BC) [mol/m³].
    v_in : float
        Velocidad superficial en la cara de entrada [m/s].

    Returns
    -------
    v_face : ndarray, shape (N + 1,)
        Velocidad superficial en todas las caras [m/s].
        v_face[0] = v_in. Las caras interiores y la de salida se derivan de
        la continuidad.
    """
    C_cell = np.asarray(C_tot_cell, dtype=float).reshape(-1)
    N      = C_cell.size

    v_face = np.zeros(N + 1, dtype=float)
    v_face[0] = float(v_in)

    if abs(v_in) < 1.0e-30:
        return v_face   # batch o flujo nulo: todas las caras = 0

    molar_flux = float(v_in) * max(float(C_tot_in), 1.0e-30)   # [mol/m²/s]

    # Caras interiores: C_face = media aritmética de celdas adyacentes
    if N > 1:
        C_face_int = 0.5 * (C_cell[:-1] + C_cell[1:])          # (N-1,)
        C_face_int = np.maximum(C_face_int, 1.0e-30)
        v_face[1:-1] = molar_flux / C_face_int

    # Cara de salida: extrapolación con la última celda
    v_face[-1] = molar_flux / max(float(C_cell[-1]), 1.0e-30)

    return v_face


@profiled
def darcy_weisbach_face_velocity(
    P: np.ndarray,
    rho_g: np.ndarray,
    mu_g: np.ndarray,
    Di: float,
    dz: float,
    v_in: float,
    v_out: float,
) -> np.ndarray:
    """
    @brief
    Calcula la velocidad superficial del gas en todas las caras del dominio
    (nn + 1 caras) usando la ecuación de Darcy-Weisbach para las caras internas
    y las velocidades de contorno para las caras extremas.

    @details
    Para cada cara interior k (k = 1, ..., nn-1) se calcula el gradiente de
    presión discretizado entre las celdas adyacentes:

        dP/dz ≈ (P[k] - P[k-1]) / dz

    y se resuelve Darcy-Weisbach con el factor de fricción del régimen local:

        Laminar   : v = (-dP/dz) · Di² / (32 · µ_face)
        Turbulento: v = sign(-dP/dz) · (|dP/dz| / C_turb)^{4/7}
                    con C_turb = 0.158 · ρ^{0.75} · µ^{0.25} · Di^{-1.25}

    El régimen se determina evaluando la velocidad laminar exacta y
    comprobando si su Reynolds es < 2300. La solución es siempre directa,
    sin iteración.

    Parameters
    ----------
    P : np.ndarray, shape (nn,)
        Presión total del gas en centros de celda [Pa].
    rho_g : np.ndarray, shape (nn,)
        Densidad del gas en centros de celda [kg/m³].
    mu_g : np.ndarray, shape (nn,)
        Viscosidad dinámica del gas en centros de celda [Pa·s].
    Di : float
        Diámetro interno del tubo [m]. Debe ser > 0.
    dz : float
        Tamaño de celda axial [m]. Debe ser > 0.
    v_in : float
        Velocidad superficial impuesta en la cara de entrada (cara 0) [m/s].
        Positiva en la dirección del flujo (inlet → outlet).
    v_out : float
        Velocidad superficial impuesta en la cara de salida (cara nn) [m/s].

    Returns
    -------
    v_face : np.ndarray, shape (nn + 1,)
        Velocidad superficial del gas en todas las caras [m/s].
        v_face[0] = v_in, v_face[-1] = v_out.

    Notes
    -----
    - Convención de signo idéntica a ergun_face_velocity: flujo positivo de
      cara 0 a cara nn. Un gradiente de presión negativo (P decreciente en z)
      produce velocidad positiva.
    - La transición laminar/turbulento se fija en Re_crit = 2300.
      En la zona de transición (2300–4000) Blasius sobreestima ligeramente
      la fricción; aceptable para aplicaciones industriales de calentamiento.
    - La correlación de Blasius asume tubo liso. Para tubos rugosos usar
      la correlación de Colebrook-White (no implementada aquí).
    - Función pura: no modifica los arrays de entrada ni tiene efectos laterales.
    """
    P_arr   = np.asarray(P,     dtype=float).reshape(-1)
    rho_arr = np.asarray(rho_g, dtype=float).reshape(-1)
    mu_arr  = np.asarray(mu_g,  dtype=float).reshape(-1)

    nn    = P_arr.size
    Di_v  = float(Di)
    dz_v  = float(dz)

    v_face = np.zeros(nn + 1, dtype=float)
    v_face[0]  = float(v_in)
    v_face[-1] = float(v_out)

    for k in range(1, nn):
        # ── Propiedades en la cara k (media aritmética de celdas adyacentes) ──
        rho_f = 0.5 * (rho_arr[k - 1] + rho_arr[k])
        mu_f  = max(0.5 * (mu_arr[k - 1] + mu_arr[k]), 1.0e-30)

        # ── Gradiente de presión local [Pa/m] ────────────────────────────────
        # neg_dPdz > 0 → P decrece en z → flujo en dirección positiva (+z)
        neg_dPdz = -(P_arr[k] - P_arr[k - 1]) / dz_v

        # ── Solución laminar exacta (Hagen-Poiseuille) ────────────────────────
        # -dP/dz = 32·µ·v / Di²  →  v = neg_dPdz · Di² / (32·µ)
        v_lam = neg_dPdz * Di_v ** 2 / (32.0 * mu_f)

        # ── Comprobación de régimen mediante Re de la solución laminar ────────
        Re_lam = rho_f * abs(v_lam) * Di_v / mu_f

        if Re_lam < 2300.0:
            # Régimen laminar: la solución es autóconsistente
            v_face[k] = v_lam
        else:
            # ── Solución turbulenta (Blasius) directa ─────────────────────────
            # -dP/dz = C_turb · v^{1.75}
            # C_turb = 0.158 · ρ^{0.75} · µ^{0.25} · Di^{-1.25}
            C_turb = 0.158 * rho_f ** 0.75 * mu_f ** 0.25 * Di_v ** (-1.25)
            C_turb = max(C_turb, 1.0e-30)

            sign_v  = 1.0 if neg_dPdz >= 0.0 else -1.0
            v_face[k] = sign_v * (abs(neg_dPdz) / C_turb) ** (4.0 / 7.0)

    return v_face
