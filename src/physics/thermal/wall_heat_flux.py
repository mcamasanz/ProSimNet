"""
@module wall_heat_flux
@brief Flujo térmico entre el gas y el exterior de la columna.

@details
Calcula la fuente térmica volumétrica que el intercambio con la pared/ambiente
aporta al balance de energía del gas, para los cuatro modos térmicos:

    "adiabatic"    → qwall_vol = 0
    "fixed_twall"  → qwall_vol = h_wall · (T_wall - Tg) · Pi·dz / (Ai·dz)
                                = h_wall · (T_wall - Tg) · Pi / Ai   [W/m³]
    "heatfluxwall" → qwall_vol = Qwall_total / (N · Ai · dz)         [W/m³]
    "ambient_htc"  → resistencias en serie: convección interna + conducción pared
                     + convección externa

Signo: positivo cuando el calor entra al gas (calentamiento).

Unidades: SI.
"""

from __future__ import annotations

import numpy as np

from src.utils.profiling import profiled


@profiled
def wall_heat_flux(
    Tg: np.ndarray,
    h_wall: np.ndarray,
    thermal_bc_config: dict,
    N: int,
    Ai: float,
    Pi: float,
    Po: float,
    dz: float,
) -> tuple:
    """
    @brief
    Calcula la fuente térmica volumétrica de pared y la potencia total intercambiada.

    Parameters
    ----------
    Tg              : ndarray(N,) — temperatura del gas por celda [K]
    h_wall          : ndarray(N,) — coeficiente htc gas-pared interno [W/m²/K]
    thermal_bc_config: dict       — resultado de build_thermal_bc_config
    N               : int         — número de celdas
    Ai              : float       — área interna de sección transversal [m²]
    Pi              : float       — perímetro interno de la sección [m]
    Po              : float       — perímetro externo de la sección [m]
    dz              : float       — longitud de celda [m]

    Returns
    -------
    (qwall_vol, Qwall_dot, qwall_area) donde:
        qwall_vol  : ndarray(N,) — fuente térmica volumétrica por celda [W/m³]
        Qwall_dot  : float       — potencia total intercambiada [W] (+ = entra al gas)
        qwall_area : ndarray(N,) — flujo térmico superficial equivalente [W/m²]
    """
    nn      = int(N)
    Ai_val  = float(Ai)
    Pi_val  = float(Pi)
    Po_val  = float(Po)
    dz_val  = float(dz)

    Tg_arr     = np.asarray(Tg,     dtype=float).reshape(-1)
    h_wall_arr = np.asarray(h_wall, dtype=float).reshape(-1)

    V_cell    = Ai_val * dz_val        # volumen interno de cada celda [m³]
    Aint_cell = Pi_val * dz_val        # área interior de pared por celda [m²]
    Aext_cell = Po_val * dz_val        # área exterior de pared por celda [m²]

    mode = thermal_bc_config["mode"]

    qwall_vol  = np.zeros(nn, dtype=float)
    qwall_area = np.zeros(nn, dtype=float)
    Qcell_arr  = np.zeros(nn, dtype=float)

    if mode == "adiabatic":
        pass  # todo cero

    elif mode == "fixed_twall":
        T_wall = float(thermal_bc_config["T_wall"])
        qwall_area = h_wall_arr * (T_wall - Tg_arr)    # [W/m²]
        Qcell_arr  = qwall_area * Aint_cell              # [W]
        qwall_vol  = Qcell_arr / V_cell                  # [W/m³]

    elif mode == "heatfluxwall":
        Qwall_total = float(thermal_bc_config["Qwall"])
        Qcell_arr[:] = Qwall_total / nn                  # uniforme por celda [W]
        qwall_vol  = Qcell_arr / V_cell
        qwall_area = Qcell_arr / Aint_cell

    elif mode == "ambient_htc":
        h_ambi  = float(thermal_bc_config["h_ambi"])
        T_ambi  = float(thermal_bc_config["T_ambi"])
        k_wall  = float(thermal_bc_config["k_wall"])
        Di_wall = Pi_val / np.pi    # diámetro interno [m]
        Do_wall = Po_val / np.pi    # diámetro externo [m]

        R_int  = 1.0 / np.maximum(h_wall_arr * Aint_cell, 1.0e-30)             # [K/W]
        R_cond = np.log(Do_wall / Di_wall) / (2.0 * np.pi * k_wall * dz_val)   # [K/W]
        R_ext  = 1.0 / max(h_ambi * Aext_cell, 1.0e-30)                        # [K/W] (escalar)
        R_tot  = R_int + R_cond + R_ext                                         # [K/W] (N,)

        Qcell_arr  = (T_ambi - Tg_arr) / np.maximum(R_tot, 1.0e-30)
        qwall_vol  = Qcell_arr / V_cell
        qwall_area = Qcell_arr / Aint_cell

    Qwall_dot = float(np.sum(Qcell_arr))

    return qwall_vol, Qwall_dot, qwall_area
