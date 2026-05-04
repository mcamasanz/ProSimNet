"""
@module boundary_conditions.gasifier_boundary
@brief Generic boundary condition evaluator for the gasifier model.

@details
El modo de operación se determina implícitamente a partir de los valores de bc_config,
no de un parámetro "mode" explícito. Las posibles configuraciones son:

    v_gas_in=None  + outlet="open"  → batch (sin flujo, sistema cerrado)
    v_gas_in=None  + outlet="vent"  → semibatch (venteo controlado por presión)
    v_gas_in≠None  + v_solid=0      → CSTR (N=1) o paso continuo de gas (N>1)
    v_gas_in≠None  + v_solid>0      → lecho fijo (updraft/downdraft) o conveyor

Este módulo es el ÚNICO lugar que evalúa valores de BC en el tiempo t.
El RHS recibe únicamente v_in, v_out, C_in, T_in, y el sólido; es agnóstico al modo.
"""

import numpy as np

R_GAS = 8.31446261815324   # [J/mol/K]


def get_gasifier_boundary(
    t:          float,
    P_cell:     np.ndarray,    # (N,) [bar]  presión del gas en las celdas
    Ctot_cell:  np.ndarray,    # (N,) [mol/m³_gas] concentración total en celdas
    bc_config:  dict,           # output de build_bc_config()
    n_comp:     int,
) -> dict:
    """
    Evaluate boundary conditions at time t.

    Parameters
    ----------
    t          : float         current time [s]
    P_cell     : ndarray (N,)  gas pressure in cells [bar]
    Ctot_cell  : ndarray (N,)  total molar concentration in cells [mol/m³_gas]
    bc_config  : dict          output of build_bc_config()
    n_comp     : int           number of gas species

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
    # ── Gas inlet ─────────────────────────────────────────────────────────────
    v_gas_in_raw = bc_config.get("v_gas_in")

    if v_gas_in_raw is None:
        # Sin entrada de gas (batch / semibatch)
        v_in = 0.0
        T_in = None
        y_in = None
        C_in = None
    else:
        v_in, T_in, y_in = _eval_gas_inlet(bc_config, t, n_comp)
        P_in_bar = float(P_cell[0])
        C_in     = y_in * (P_in_bar * 1.0e5) / (R_GAS * max(T_in, 1.0))

    # ── Gas outlet ────────────────────────────────────────────────────────────
    outlet = str(bc_config.get("outlet", "open"))

    if outlet == "vent":
        # Venteo controlado por exceso de presión
        v_out = _compute_vent_velocity(bc_config, P_cell)
    else:
        # "open": v_out desde continuidad molar
        if C_in is None:
            v_out = 0.0
        else:
            Ctot_in  = float(np.sum(C_in))
            Ctot_out = float(Ctot_cell[-1]) if len(Ctot_cell) > 0 else Ctot_in
            v_out    = v_in * Ctot_in / max(Ctot_out, 1.0e-300)

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


# ─── Funciones auxiliares ──────────────────────────────────────────────────────

def _eval_gas_inlet(bc_config: dict, t: float, n_comp: int):
    """Evalúa las condiciones de entrada del gas en el instante t, resolviendo callables."""
    v_raw = bc_config["v_gas_in"]
    T_raw = bc_config["T_gas_in"]
    y_raw = bc_config["y_gas_in"]

    v_in = float(v_raw(t) if callable(v_raw) else v_raw)
    T_in = float(T_raw(t) if callable(T_raw) else T_raw)
    y_in = np.asarray(
        y_raw(t) if callable(y_raw) else y_raw,
        dtype=float,
    ).reshape(-1)

    if len(y_in) != n_comp:
        raise ValueError(
            f"get_gasifier_boundary: y_gas_in has length {len(y_in)}, "
            f"expected n_comp={n_comp}"
        )
    return v_in, T_in, y_in


def _compute_vent_velocity(bc_config: dict, P_cell: np.ndarray) -> float:
    """
    Velocidad de venteo controlada por presión.

        v_out = max(0, (P − P_out) / P_out) · v_vent_max

    - P = P_out  →  v_out = 0 (sin venteo)
    - P = 2·P_out →  v_out = v_vent_max (venteo máximo)
    - v_vent_max grande → equilibración rápida de presión ≈ proceso a presión constante
    """
    P_out      = float(bc_config["P_out_bar"])
    v_vent_max = float(bc_config.get("v_vent_max") or 0.10)
    P_current  = float(P_cell[-1])
    excess_frac = max(0.0, (P_current - P_out) / P_out)
    return excess_frac * v_vent_max
