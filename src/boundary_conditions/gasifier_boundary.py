"""
@module boundary_conditions.gasifier_boundary
@brief Generic boundary condition evaluator for the gasifier model.

@details
El modo de operación se determina implícitamente a partir de los valores de bc_config,
no de un parámetro "mode" explícito. Las posibles configuraciones son:

    v_gas_in=None  + v_out=0.0   → batch sellado (sin flujo)
    v_gas_in=None  + v_out>0     → semibatch venteo proporcional (≈ isobaro)
    v_gas_in=None  + Cv>0        → semibatch venteo ISA-75.01 (≈ isobaro)
    v_gas_in=None  + v_out=None  → semibatch isobaro exacto (si hay reacciones)
    v_gas_in≠None  + v_out=None  → CSTR isobaro exacto
    v_gas_in≠None  + Cv>0        → CSTR / lecho con válvula ISA en la salida
    v_gas_in≠None  + v_solid>0   → lecho fijo (updraft/downdraft) o conveyor

Modos de salida (v_out y Cv son mutuamente excluyentes para venteo):
    v_out=None  → isobaro exacto:
                  v_out = (F_in + F_rxn) / Ctot_target
                  F_in   = v_in·Ctot_in               [mol/m²/s]  — flujo molar de entrada
                  F_rxn  = Σ_i source_i·ε·dz          [mol/m²/s]  — gas producido por reacciones
                  Ctot_target = P_out·1e5/(R·Tg_out)              — concentración a P_out
                  Con v_gas_in=None y sin reacciones: F_in=F_rxn=0 → v_out=0 (batch).
                  Requiere Tg_cell y source_total_flux en la llamada.
    v_out=0.0   → sellado (v_out = 0 siempre)
    v_out>0     → venteo proporcional: max(0,(P−P_out)/P_out)·v_out  [m/s]
    Cv>0        → válvula ISA: Q ∝ Cv·√(ΔP·P_up/(T_up·Sg))
                  Requiere Tg_cell, C_cell, MW_arr, epsi, Ai en la llamada.

El parámetro source_total_flux [mol/m²/s] debe provenir del caché del RHS:
    cache["source_total_flux_last"] = float(np.sum(source_gas)) * epsi_r * dz
Se usa el valor del paso RHS anterior (lag de un paso). Con BDF el error es O(dt).

Señales de BC (resolve):
    Todos los parámetros de inlet (v_gas_in, T_gas_in, y_gas_in) y outlet (v_out, Cv) pueden ser:
      - float / int                 → constante
      - callable(t) → value        → perfil temporal
      - callable(t, snap) → value  → retroalimentación de estado del equipo
    La resolución se hace vía src.utils.signals.resolve().  El snap proviene del RHS.

Este módulo es el ÚNICO lugar que evalúa valores de BC en el tiempo t.
El RHS recibe únicamente v_in, v_out, C_in, T_in, y el sólido; es agnóstico al modo.
"""

import numpy as np

from src.boundary_conditions.valve import valve_superficial_velocity
from src.utils.signals import resolve as _resolve

R_GAS = 8.31446261815324   # [J/mol/K]


def get_gasifier_boundary(
    t:          float,
    P_cell:     np.ndarray,    # (N,) [bar]  presión del gas en las celdas
    Ctot_cell:  np.ndarray,    # (N,) [mol/m³_gas] concentración total en celdas
    bc_config:  dict,           # output de build_bc_config()
    n_comp:     int,
    *,
    snap:                      dict | None  = None,  # equipment state snap (built in RHS paso 2)
    Tg_cell:           np.ndarray | None = None,  # (N,) [K]        — requerido para v_out=None y Cv
    C_cell:            np.ndarray | None = None,  # (nc,N) [mol/m³] — requerido para Cv
    MW_arr:            np.ndarray | None = None,  # (nc,) [kg/mol]  — requerido para Cv
    epsi:              float | None      = None,  # [-]             — requerido para Cv
    Ai:                float | None      = None,  # [m²]            — requerido para Cv
    source_total_flux:         float      = 0.0,   # [mol/m²/s]      — gas producido, caché del RHS
    thermal_expansion_flux:   float      = 0.0,   # [m/s]           — expansión térmica, caché del RHS
) -> dict:
    """
    Evaluate boundary conditions at time t.

    Parameters
    ----------
    t                  : float         current time [s]
    P_cell             : ndarray (N,)  gas pressure in cells [bar]
    Ctot_cell          : ndarray (N,)  total molar concentration in cells [mol/m³_gas]
    bc_config          : dict          output of build_bc_config()
    n_comp             : int           number of gas species
    snap               : dict or None  equipment state snap from RHS paso 2 (for state-feedback signals)
    Tg_cell            : ndarray (N,) or None  gas temperature [K]  — required for v_out=None and Cv
    C_cell             : ndarray (nc,N) or None  concentrations [mol/m³_gas] — required for Cv
    MW_arr             : ndarray (nc,) or None   molar masses [kg/mol]        — required for Cv
    epsi               : float or None           bed void fraction [-]         — required for Cv
    Ai                 : float or None           cross-sectional area [m²]     — required for Cv
    source_total_flux      : float               ① molar production by reactions [mol/m²/s]
                                                 = sum(source_gas) * epsi_r * dz  from RHS cache.
    thermal_expansion_flux : float               ② thermal expansion velocity [m/s]
                                                 = epsi_r * dz * max(0, dTg/dt) / Tg  from RHS cache.
                                                 Both used by v_out=None isobaric mode (no stiffness).

    Returns
    -------
    dict with keys:
        inlet : {
            T_K      : float or None      gas temperature at inlet [K]
            y        : ndarray(nc,) or None
            C_mol_m3 : ndarray(nc,) or None   [mol/m³_gas]
            v_m_s    : float              gas superficial velocity at inlet face [m/s]
        }
        outlet : {
            P_out_bar : float             outlet pressure [bar]
            v_m_s     : float             gas superficial velocity at outlet face [m/s]
        }
        solid_inlet : {
            rho_solid : ndarray(3,) or None   [kg/m³_bed]
            Ts_K      : float or None          [K]
            v_m_s     : float                  solid velocity magnitude [m/s]
            direction : str or None            "updraft" | "downdraft" | None
        }
    """
    _snap = snap if snap is not None else {}

    # ── Gas inlet ─────────────────────────────────────────────────────────────
    v_gas_in_raw = bc_config.get("v_gas_in")

    if v_gas_in_raw is None:
        # Sin entrada de gas (batch / semibatch)
        v_in = 0.0
        T_in = None
        y_in = None
        C_in = None
    else:
        v_in, T_in, y_in = _eval_gas_inlet(bc_config, t, n_comp, _snap)
        P_in_bar = float(P_cell[0])
        C_in     = y_in * (P_in_bar * 1.0e5) / (R_GAS * max(T_in, 1.0))

    # ── Gas outlet ────────────────────────────────────────────────────────────
    Cv_cfg    = bc_config.get("Cv")      # None, float>0, o callable
    v_out_cfg = bc_config.get("v_out")   # None, 0.0, float>0, o callable

    # Resolver Cv_cfg si es callable (para pasarlo a valve_superficial_velocity)
    Cv_resolved = float(_resolve(Cv_cfg, t, _snap)) if Cv_cfg is not None else None

    if Cv_resolved is not None:
        # Venteo ISA-75.01: Q ∝ Cv·√(ΔP·P_up / (T_up·Sg))
        # Requiere Tg_cell, C_cell, MW_arr, epsi, Ai del estado actual del RHS.
        if Tg_cell is None or C_cell is None or MW_arr is None or epsi is None or Ai is None:
            raise ValueError(
                "Cv mode requires Tg_cell, C_cell, MW_arr, epsi and Ai — "
                "pass them as keyword arguments to get_gasifier_boundary()"
            )
        P_out_bar  = float(bc_config["P_out_bar"])
        P_up_Pa    = float(P_cell[-1]) * 1.0e5
        P_down_Pa  = P_out_bar * 1.0e5
        T_up       = float(Tg_cell[-1])
        Ctot_out   = max(float(Ctot_cell[-1]), 1.0e-300)
        MW_mix_out = float(np.dot(C_cell[:, -1], MW_arr)) / Ctot_out   # [kg/mol]
        v_out = valve_superficial_velocity(
            Cv=Cv_resolved,
            P_up_Pa=P_up_Pa, P_down_Pa=P_down_Pa,
            T_up=T_up, MW_mix=MW_mix_out,
            epsi=float(epsi), Ai=float(Ai),
        )

    elif v_out_cfg is None:
        # Isobaro exacto — condición dP/dt=0 con gas ideal:
        #   v_out = (F_in + F_rxn)/Ctot_target  +  v_thermal              ①+②
        #
        # ① F_rxn = source_total_flux [mol/m²/s] — caché RHS anterior (lag 1 paso).
        # ② v_thermal = ε·dz·max(0, dTg/dt)/Tg [m/s] — caché RHS anterior.
        #
        # Ambos términos dependen solo de valores cacheados (no del estado actual):
        # → no introducen valores propios rápidos → no encarecen el solver.
        if Tg_cell is not None:
            P_out_Pa    = float(bc_config["P_out_bar"]) * 1.0e5
            Tg_out      = max(float(Tg_cell[-1]), 1.0)
            Ctot_target = P_out_Pa / (R_GAS * Tg_out)
            F_in_molar  = v_in * float(np.sum(C_in)) if C_in is not None else 0.0
            F_rxn_molar = float(source_total_flux)           # ① reacciones
            v_thermal   = float(thermal_expansion_flux)      # ② expansión térmica
            v_ff        = (F_in_molar + F_rxn_molar) / max(Ctot_target, 1.0e-300)
            v_out       = max(0.0, v_ff + v_thermal)
        else:
            # Fallback continuidad molar (Tg_cell no disponible — llamada externa sin estado)
            F_in     = v_in * float(np.sum(C_in)) if C_in is not None else 0.0
            Ctot_out = float(Ctot_cell[-1]) if len(Ctot_cell) > 0 else 1.0
            v_out    = F_in / max(Ctot_out, 1.0e-300)

    else:
        # v_out_cfg es float o callable — resolver y aplicar la lógica correspondiente
        v_out_resolved = float(_resolve(v_out_cfg, t, _snap))

        if v_out_resolved == 0.0:
            # Sistema sellado: sin salida de gas
            v_out = 0.0
        else:
            # Venteo proporcional al exceso de presión:
            #   v_out_actual = max(0, (P − P_out) / P_out) · v_vent_max
            # v_out_resolved es la velocidad máxima de venteo (válvula totalmente abierta)
            P_out_bar   = float(bc_config["P_out_bar"])
            P_current   = float(P_cell[-1])
            excess_frac = max(0.0, (P_current - P_out_bar) / P_out_bar)
            v_out = excess_frac * v_out_resolved

    # ── Solid inlet ───────────────────────────────────────────────────────────
    v_solid      = float(bc_config.get("v_solid", 0.0))
    direction    = bc_config.get("direction")
    rho_solid_in = bc_config.get("rho_solid_in")
    T_solid_in   = bc_config.get("T_solid_in")

    return {
        "inlet": {
            "T_K":      T_in,
            "y":        y_in,
            "C_mol_m3": C_in,
            "v_m_s":    float(v_in),
        },
        "outlet": {
            "P_out_bar": float(bc_config["P_out_bar"]),
            "v_m_s":     float(v_out),
        },
        "solid_inlet": {
            "rho_solid": (np.asarray(rho_solid_in, dtype=float)
                          if rho_solid_in is not None else None),
            "Ts_K":      T_solid_in,
            "v_m_s":     v_solid,
            "direction": direction,
        },
    }


# ─── Función auxiliar ─────────────────────────────────────────────────────────

def _eval_gas_inlet(bc_config: dict, t: float, n_comp: int, snap: dict):
    """
    Evalúa las condiciones de entrada del gas en el instante t.

    Soporta señales constantes, callable(t) y callable(t, snap) vía resolve().
    """
    v_in = float(_resolve(bc_config["v_gas_in"], t, snap))
    T_in = float(_resolve(bc_config["T_gas_in"], t, snap))
    y_raw = _resolve(bc_config["y_gas_in"], t, snap)
    y_in  = np.asarray(y_raw, dtype=float).reshape(-1)

    if len(y_in) != n_comp:
        raise ValueError(
            f"get_gasifier_boundary: y_gas_in has length {len(y_in)}, "
            f"expected n_comp={n_comp}"
        )
    return v_in, T_in, y_in
