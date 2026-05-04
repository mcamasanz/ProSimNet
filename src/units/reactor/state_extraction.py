"""
Extracción de resultados del reactor desde la historia del vector de estado.

Reconstruye el objeto de resultados (SimpleNamespace) con arrays de variables
físicas para cada instante guardado por el integrador ODE.
"""

from __future__ import annotations

import types

import numpy as np

from src.boundary_conditions.reactor_boundary import get_reactor_boundary
from src.units.reactor.state import unpack_state_vector

R_GAS = 8.31446261815324   # [J/mol/K]


def build_reactor_results(
    y_hist:  np.ndarray,   # (n_t, sv_size)
    t_arr:   np.ndarray,   # (n_t,)
    params:  dict,
) -> types.SimpleNamespace:
    """
    Reconstruct physical variables from the ODE state history.

    Parameters
    ----------
    y_hist : ndarray (n_t, sv_size)  state vector at each saved time step
    t_arr  : ndarray (n_t,)          time points [s]
    params : dict                    same params dict used during integration

    Returns
    -------
    SimpleNamespace with attributes:
        _t_results         (n_t,)         [s]
        _C_results         (n_t, nc, N)   [mol/m³_gas]
        _Hg_results        (n_t, N)       [J/m³_bed]
        _Tg_results        (n_t, N)       [K]
        _Ts_results        (n_t, N) or None
        _Tw_results        (n_t, N) or None
        _P_results         (n_t, N)       [bar]
        _y_results         (n_t, nc, N)   [-]
        _Ctot_results      (n_t, N)       [mol/m³_gas]
        _v_in_results      (n_t,)         [m/s]   (0 for batch)
        _v_out_results     (n_t,)         [m/s]   (0 for batch)
        _C_in_results      (n_t, nc)      [mol/m³_gas]  (NaN for batch)
        _T_in_results      (n_t,)         [K]           (NaN for batch)
        _species           list[str]
    """
    nc           = int(params["n_comp"])
    nn           = int(params["N"])
    dz           = float(params["dz"])
    epsi         = float(params["epsi"])
    prop_gas     = params["prop_gas"]
    gas_T_ref    = float(params["gas_T_ref"])
    has_catalyst = params.get("catalyst_config") is not None
    shell_tube   = params.get("wall_config") is not None
    bc_config    = params["bc_config"]
    species      = list(params["species"])

    z = (np.arange(nn) + 0.5) * dz   # centros de celda [m]

    n_t = len(t_arr)

    C_list    = []
    Hg_list   = []
    Tg_list   = []
    Ts_list   = []
    Tw_list   = []
    P_list    = []
    y_list    = []
    Ctot_list = []
    v_in_list  = []
    v_out_list = []
    C_in_list  = []
    T_in_list  = []

    Tg_guess = np.full(nn, 700.0)

    for k in range(n_t):
        sv    = y_hist[k]
        state = unpack_state_vector(
            sv=sv, n_comp=nc, N=nn,
            has_catalyst=has_catalyst, shell_tube=shell_tube,
            prop_gas=prop_gas, epsi=epsi,
            Tg_guess=Tg_guess, gas_T_ref=gas_T_ref,
        )
        C_k    = np.maximum(state["C"], 0.0)
        Tg_k   = state["Tg"]
        Ctot_k = np.sum(C_k, axis=0)
        safe   = np.maximum(Ctot_k, 1.0e-300)
        P_k    = np.maximum(Ctot_k * R_GAS * Tg_k / 1.0e5, 1.0e-6)

        C_list.append(C_k)
        Hg_list.append(state["Hg"].copy())
        Tg_list.append(Tg_k.copy())
        Ts_list.append(state["Ts"].copy() if state["Ts"] is not None else None)
        Tw_list.append(state["Tw"].copy() if state["Tw"] is not None else None)
        P_list.append(P_k)
        y_list.append(C_k / safe[None, :])
        Ctot_list.append(Ctot_k)

        # Condiciones de contorno evaluadas en t_k
        bc_k = get_reactor_boundary(t=float(t_arr[k]),
                                    P_cell=P_k, Ctot_cell=Ctot_k,
                                    bc_config=bc_config, n_comp=nc)
        v_in_list.append(float(bc_k["inlet"]["v_m_s"]))
        v_out_list.append(float(bc_k["outlet"]["v_m_s"]))
        C_in_k = bc_k["inlet"]["C_mol_m3"]
        T_in_k = bc_k["inlet"]["T_K"]
        C_in_list.append(C_in_k if C_in_k is not None else np.full(nc, np.nan))
        T_in_list.append(T_in_k if T_in_k is not None else np.nan)

        Tg_guess = Tg_k.copy()

    return types.SimpleNamespace(
        _t_results    = np.asarray(t_arr),
        _z            = z,
        _C_results    = np.stack(C_list,    axis=0),   # (n_t, nc, N)
        _Hg_results   = np.stack(Hg_list,   axis=0),   # (n_t, N)
        _Tg_results   = np.stack(Tg_list,   axis=0),   # (n_t, N)
        _Ts_results   = (np.stack(Ts_list, axis=0)
                         if has_catalyst else None),
        _Tw_results   = (np.stack(Tw_list, axis=0)
                         if shell_tube else None),
        _P_results    = np.stack(P_list,    axis=0),   # (n_t, N)
        _y_results    = np.stack(y_list,    axis=0),   # (n_t, nc, N)
        _Ctot_results = np.stack(Ctot_list, axis=0),   # (n_t, N)
        _v_in_results  = np.asarray(v_in_list),        # (n_t,)
        _v_out_results = np.asarray(v_out_list),        # (n_t,)
        _C_in_results  = np.stack(C_in_list, axis=0),  # (n_t, nc)
        _T_in_results  = np.asarray(T_in_list),         # (n_t,)
        _species       = species,
    )
