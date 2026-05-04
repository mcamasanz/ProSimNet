"""
Boundary condition evaluator for the reactor model.

The operating regime is inferred from bc_config at runtime — there is no
explicit mode parameter:

    bc_config["v_in"] is None  →  batch (v_in = v_out = 0)
    bc_config["v_in"] is not None  →  continuous (prescribed inlet,
                                        outlet from molar continuity)

The distinction between cstr (N=1) and prf (N>1) is a spatial discretisation
choice made by the user through N and axial dispersion in trans_config,
not a boundary condition concept.

This is the only module that evaluates BC values at time t. The RHS receives
the returned dict and is agnostic to the operating regime.
"""

from __future__ import annotations

import numpy as np

R_GAS = 8.31446261815324   # [J/mol/K]


def get_reactor_boundary(
    t:          float,
    P_cell:     np.ndarray,   # (N,) [bar]
    Ctot_cell:  np.ndarray,   # (N,) [mol/m³_gas]
    bc_config:  dict,
    n_comp:     int,
) -> dict:
    """
    Evaluate boundary conditions at time t.

    Parameters
    ----------
    t         : float         current time [s]
    P_cell    : ndarray (N,)  gas pressure in cells [bar]
    Ctot_cell : ndarray (N,)  total molar concentration in cells [mol/m³_gas]
    bc_config : dict          output of build_boundary_c_config()
    n_comp    : int           number of gas species

    Returns
    -------
    dict with keys:
        inlet : {
            v_m_s    : float              superficial velocity at inlet face [m/s]
            T_K      : float or None
            y        : ndarray(nc,) or None
            C_mol_m3 : ndarray(nc,) or None  [mol/m³_gas]
        }
        outlet : {
            P_out_bar : float
            v_m_s     : float
        }
    """
    # Batch: v_in is None → no flow
    if bc_config["v_in"] is None:
        return {
            "inlet":  {"v_m_s": 0.0, "T_K": None, "y": None, "C_mol_m3": None},
            "outlet": {"P_out_bar": float(bc_config["P_out_bar"]), "v_m_s": 0.0},
        }

    # Continuo: inlet prescrito, outlet por continuidad molar
    v_in, T_in, y_in = _eval_inlet(bc_config, t, n_comp)

    P_in_Pa = float(P_cell[0]) * 1.0e5
    C_in    = y_in * P_in_Pa / (R_GAS * max(T_in, 1.0))   # (nc,) [mol/m³_gas]

    Ctot_in  = float(np.sum(C_in))
    Ctot_out = float(Ctot_cell[-1])
    v_out    = v_in * Ctot_in / max(Ctot_out, 1.0e-300)

    return {
        "inlet": {
            "v_m_s":    float(v_in),
            "T_K":      float(T_in),
            "y":        y_in,
            "C_mol_m3": C_in,
        },
        "outlet": {
            "P_out_bar": float(bc_config["P_out_bar"]),
            "v_m_s":     float(v_out),
        },
    }


# ─── Auxiliar ─────────────────────────────────────────────────────────────────

def _eval_inlet(bc_config: dict, t: float, n_comp: int):
    """Evalúa v_in, T_in, y_in en el instante t resolviendo callables."""
    v_raw = bc_config["v_in"]
    T_raw = bc_config["T_in"]
    y_raw = bc_config["y_in"]

    v_in = float(v_raw(t) if callable(v_raw) else v_raw)
    T_in = float(T_raw(t) if callable(T_raw) else T_raw)
    y_in = np.asarray(
        y_raw(t) if callable(y_raw) else y_raw, dtype=float
    ).reshape(-1)

    if len(y_in) != n_comp:
        raise ValueError(
            f"get_reactor_boundary: y_in has length {len(y_in)}, "
            f"expected n_comp={n_comp}"
        )
    return v_in, T_in, y_in
