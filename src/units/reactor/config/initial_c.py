"""
Build initial conditions for the reactor ODE.

Converts user-friendly physical variables into the packed state vector sv0.
Scalars are broadcast to uniform spatial profiles; arrays allow heterogeneous
initial conditions.
"""

from __future__ import annotations

import numpy as np

from src.units.reactor.state import set_state


def build_initial_conditions(
    P_bar:     float | np.ndarray,  # [bar]
    Tg:        float | np.ndarray,  # [K]
    y:         np.ndarray,          # (nc,) or (nc, N)
    n_comp:    int,
    N:         int,
    prop_gas:  dict,
    epsi:      float,
    gas_T_ref: float,
    Ts:        float | np.ndarray | None = None,  # [K]  catalyst temp
    Tw:        float | np.ndarray | None = None,  # [K]  wall temp
) -> dict:
    """
    Build and validate initial conditions for the reactor.

    Parameters
    ----------
    P_bar     : float or (N,)      initial gas pressure [bar]
    Tg        : float or (N,)      initial gas temperature [K]
    y         : (nc,) or (nc, N)   initial molar fractions [-]
    n_comp    : int                number of gas species
    N         : int                number of cells
    prop_gas  : dict               output of build_gas_prop_config
    epsi      : float              void fraction [-]  (1.0 for empty tube)
    gas_T_ref : float              enthalpy reference temperature [K]
    Ts        : float, (N,), or None  initial catalyst temperature [K];
                None if has_catalyst=False
    Tw        : float, (N,), or None  initial wall temperature [K];
                None if shell_tube=False

    Returns
    -------
    dict with keys:
        P_init   ndarray(N,)   [bar]
        Tg_init  ndarray(N,)   [K]
        Ts_init  ndarray(N,) or None  [K]
        Tw_init  ndarray(N,) or None  [K]
        y_init   ndarray(nc, N)  [-]
        C_init   ndarray(nc, N)  [mol/m³_gas]
        Hg_init  ndarray(N,)    [J/m³_bed]
        sv0      ndarray         packed state vector
    """
    if not (0.0 < float(epsi) <= 1.0):
        raise ValueError(f"epsi must be in (0, 1], got {epsi}")
    if float(gas_T_ref) <= 0.0:
        raise ValueError(f"gas_T_ref must be > 0, got {gas_T_ref}")

    state = set_state(
        P_bar=P_bar, Tg=Tg, y=y,
        n_comp=n_comp, N=N,
        prop_gas=prop_gas, epsi=epsi, gas_T_ref=gas_T_ref,
        Ts=Ts, Tw=Tw,
    )

    return {
        "P_init":  state["P"],
        "Tg_init": state["Tg"],
        "Ts_init": state["Ts"],
        "Tw_init": state["Tw"],
        "y_init":  state["y"],
        "C_init":  state["C"],
        "Hg_init": state["Hg"],
        "sv0":     state["sv0"],
    }
