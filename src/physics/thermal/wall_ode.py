"""
@module physics.thermal.wall_ode
@brief Funciones compartidas para el modelo de pared dinámica (shell-tube mode).

@details
Cuando `wall_config` está presente en `params`, la pared del equipo se modela
como un sólido con capacidad térmica propia: una ODE por celda para Tw(z, t).
Este módulo centraliza esa física de forma que sea reutilizable por cualquier
equipo (calentador, adsorbedor, reactor, etc.).

El ODE de pared por celda es:

    rho_w * cp_w * A_w * dz * dTw/dt = Q_gw + Q_ext + Q_ax

donde:
    Q_gw   — potencia gas↔pared:    h_wall * Pi * dz * (Tg − Tw)
    Q_ext  — potencia exterior:      función del modo térmico exterior
    Q_ax   — conducción axial:       Fourier con Neumann en ambos extremos

Funciones públicas:
    wall_exterior_q  — Q_ext por celda [W] según el modo de BC exterior
    wall_axial_q     — Q_ax por celda [W] por conducción axial en la pared
    wall_ode_rhs     — dTw/dt [K/s] sumando las tres contribuciones

Compatibilidad con thermal_bc_config:
    "adiabatic"    — exterior adiabático: Q_ext = 0
    "heatfluxwall" — potencia total impuesta, distribuida uniformemente
    "ambient_htc"  — conducción radial de la pared + convección exterior en serie
    "fixed_twall"  — NO compatible con shell_tube (Tw es variable del ODE)

Unidades: SI.
"""

from __future__ import annotations

import numpy as np


def wall_exterior_q(
    Tw_arr: np.ndarray,
    thermal_bc_cfg: dict,
    k_w_arr: np.ndarray,
    Pi: float,
    Po: float,
    dz: float,
    N: int,
) -> np.ndarray:
    """
    @brief
    Calcula la potencia [W] que entra a la pared desde el exterior, por celda.

    @details
    La resistencia de convección interna (gas↔pared) NO se incluye aquí: ya está
    representada en Q_gw = h_wall * Pi * dz * (Tg − Tw) en el RHS del equipo.

    Parameters
    ----------
    Tw_arr         : ndarray(N,) — temperatura de la pared [K]
    thermal_bc_cfg : dict — resultado de build_thermal_bc_config
    k_w_arr        : ndarray(N,) — conductividad térmica de la pared [W/m/K]
    Pi             : float — perímetro interior [m]
    Po             : float — perímetro exterior [m]
    dz             : float — longitud de celda [m]
    N              : int — número de celdas

    Returns
    -------
    Q_ext_cell : ndarray(N,) — potencia neta exterior→pared por celda [W]
                               signo positivo = calentamiento de la pared
    """
    mode = thermal_bc_cfg["mode"]
    nn   = int(N)

    if mode == "fixed_twall":
        raise ValueError(
            "thermal_bc_config mode='fixed_twall' es incompatible con shell_tube=True. "
            "La temperatura interior de la pared es ahora una variable dinámica del ODE. "
            "Usar 'adiabatic', 'heatfluxwall' o 'ambient_htc'."
        )

    if mode == "adiabatic":
        return np.zeros(nn, dtype=float)

    if mode == "heatfluxwall":
        Qwall = thermal_bc_cfg["Qwall"]
        if np.ndim(Qwall) == 0:
            # Escalar: potencia total distribuida uniformemente entre celdas
            return np.full(nn, float(Qwall) / nn, dtype=float)
        else:
            # Array 1D: potencia por celda [W] (suma = potencia total)
            Qwall_arr = np.asarray(Qwall, dtype=float)
            if Qwall_arr.shape != (nn,):
                raise ValueError(
                    f"Qwall array shape {Qwall_arr.shape} incompatible con N={nn}. "
                    f"Se espera ({nn},)."
                )
            return Qwall_arr.copy()

    if mode == "ambient_htc":
        h_ambi  = float(thermal_bc_cfg["h_ambi"])
        T_ambi  = float(thermal_bc_cfg["T_ambi"])
        Di_wall = Pi / np.pi
        Do_wall = Po / np.pi
        # Resistencia de conducción radial (variable con k_w) + convección exterior
        R_cond = np.log(Do_wall / Di_wall) / (2.0 * np.pi * np.maximum(k_w_arr, 1.0e-30) * dz)
        R_ext  = 1.0 / max(h_ambi * Po * dz, 1.0e-30)
        return (T_ambi - Tw_arr) / np.maximum(R_cond + R_ext, 1.0e-30)

    raise ValueError(f"thermal_bc_config mode desconocido: '{mode}'")


def wall_axial_q(
    Tw_arr: np.ndarray,
    k_w_arr: np.ndarray,
    A_w: float,
    dz: float,
) -> np.ndarray:
    """
    @brief
    Calcula la contribución neta de conducción axial a través de la pared [W/celda].

    @details
    Fourier: Q_face = −k * A_w * dT/dz, con condición de Neumann (flujo nulo)
    en ambos extremos axiales.
    La conducción axial de la pared siempre está activa; no existe opción de
    desactivarla.

    Parameters
    ----------
    Tw_arr  : ndarray(N,) — temperatura de la pared [K]
    k_w_arr : ndarray(N,) — conductividad de la pared [W/m/K]
    A_w     : float — área de sección transversal de la pared [m²]
    dz      : float — longitud de celda [m]

    Returns
    -------
    Q_ax_cell : ndarray(N,) — potencia neta axial por celda [W]
                              positivo = entrada neta de calor a la celda desde los vecinos
    """
    nn = len(Tw_arr)
    k_w_face = np.zeros(nn + 1, dtype=float)
    k_w_face[1:-1] = 0.5 * (k_w_arr[:-1] + k_w_arr[1:])  # interpolación aritmética en caras
    dT_face = np.zeros(nn + 1, dtype=float)
    dT_face[1:-1] = np.diff(Tw_arr)                        # Neumann: dT=0 en extremos
    Q_ax_face = -k_w_face * A_w * dT_face / dz
    return Q_ax_face[:-1] - Q_ax_face[1:]


def wall_ode_rhs(
    Q_gw_cell: np.ndarray,
    Q_ext_cell: np.ndarray,
    Q_ax_cell: np.ndarray,
    rho_w_arr: np.ndarray,
    cp_w_arr: np.ndarray,
    A_w: float,
    dz: float,
) -> np.ndarray:
    """
    @brief
    Calcula dTw/dt [K/s] para el ODE de temperatura de la pared.

    @details
    Ecuación:
        dTw/dt = (Q_gw + Q_ext + Q_ax) / (rho_w * cp_w * A_w * dz)

    Parameters
    ----------
    Q_gw_cell  : ndarray(N,) — potencia gas↔pared [W/celda],
                               positivo si Tg > Tw (el gas calienta la pared)
    Q_ext_cell : ndarray(N,) — potencia exterior→pared [W/celda]
    Q_ax_cell  : ndarray(N,) — conducción axial neta [W/celda]
    rho_w_arr  : ndarray(N,) — densidad de la pared [kg/m³]
    cp_w_arr   : ndarray(N,) — calor específico de la pared [J/kg/K]
    A_w        : float — área de sección transversal de la pared [m²]
    dz         : float — longitud de celda [m]

    Returns
    -------
    dTwdt : ndarray(N,) — derivada temporal de Tw [K/s]
    """
    thermal_mass = np.maximum(rho_w_arr * cp_w_arr, 1.0e-10) * A_w * dz   # [J/K/celda]
    return (Q_gw_cell + Q_ext_cell + Q_ax_cell) / thermal_mass
