"""
@module gasifier.state_extraction
@brief Reconstruct the result object from the raw ODE history.

@details
Converts y_hist (n_t, 14*N) or (n_t, 15*N) into a SimpleNamespace with named
arrays, re-applying the same unpack logic as the RHS.

Shell-tube mode is detected from params["wall_config"] (not None → shell_tube=True).
When active, _Tw_results is an ndarray(n_t, N); otherwise it is None.
"""

import types
import numpy as np

from src.physics.thermodynamics.enthalpy import recover_Tg_from_Hg
from src.boundary_conditions.gasifier_boundary import get_gasifier_boundary

R_GAS = 8.31446261815324


def build_gasifier_results(
    y_hist: np.ndarray,   # (n_t, 14*N) or (n_t, 15*N)
    t_arr:  np.ndarray,   # (n_t,)
    params: dict,
) -> types.SimpleNamespace:
    """
    Build the gasifier result object from ODE history.

    Parameters
    ----------
    y_hist : ndarray (n_t, 14*N) or (n_t, 15*N)
    t_arr  : ndarray (n_t,) [s]
    params : dict

    Returns
    -------
    SimpleNamespace with attributes:
        _t_results         (n_t,)         [s]
        _z                 (N,)           [m]
        _species           list[str]      gas species names
        _P_results         (n_t, N)       [bar]
        _Tg_results        (n_t, N)       [K]
        _Ts_results        (n_t, N)       [K]
        _Tw_results        (n_t, N) or None  [K]  — None when shell_tube=False
        _Hg_results        (n_t, N)       [J/m³_bed]
        _y_results         (n_t, nc, N)   [-]
        _C_results         (n_t, nc, N)   [mol/m³_gas]
        _rho_solid_results (n_t, 3, N)    [kg/m³_bed]  [biomass, char, moisture]
        _v_results         (n_t, N)       [m/s]
        _v_in_results      (n_t,)         [m/s]
        _v_out_results     (n_t,)         [m/s]
        _C_in_results      (n_t, nc)      [mol/m³_gas]  NaN if batch
        _T_in_results      (n_t,)         [K]            NaN if batch
    """
    nc       = int(params["n_comp"])
    nn       = int(params["N"])
    dz       = float(params["dz"])
    epsi_r   = float(params["epsi_r"])
    gas_T_ref = float(params["gas_T_ref"])
    prop_gas  = params["prop_gas"]
    bc_config = params["bc_config"]
    species   = list(params["species"])

    shell_tube = params.get("wall_config") is not None

    n_t = len(t_arr)
    z   = (np.arange(nn) + 0.5) * dz

    # Pre-allocate output arrays
    P_hist           = np.zeros((n_t, nn),    dtype=float)
    Tg_hist          = np.zeros((n_t, nn),    dtype=float)
    Ts_hist          = np.zeros((n_t, nn),    dtype=float)
    Hg_hist          = np.zeros((n_t, nn),    dtype=float)
    C_hist           = np.zeros((n_t, nc, nn), dtype=float)
    y_hist_3d        = np.zeros((n_t, nc, nn), dtype=float)
    rho_s_hist       = np.zeros((n_t, 3, nn),  dtype=float)
    v_hist           = np.zeros((n_t, nn),    dtype=float)
    v_in_hist        = np.zeros(n_t,          dtype=float)
    v_out_hist       = np.zeros(n_t,          dtype=float)
    C_in_hist        = np.full((n_t, nc),     np.nan, dtype=float)
    T_in_hist        = np.full(n_t,           np.nan, dtype=float)
    Tw_hist          = np.zeros((n_t, nn),    dtype=float) if shell_tube else None
    Q_mt_acc_hist    = np.zeros((n_t, nn),    dtype=float)
    Q_rxn_acc_hist   = np.zeros((n_t, nn),    dtype=float)

    Tg_prev = np.full(nn, 700.0, dtype=float)   # Newton warm-start

    for k in range(n_t):
        sv = y_hist[k]

        # Extract primary blocks
        idx = 0
        C         = sv[idx: idx + nc * nn].reshape(nc, nn); idx += nc * nn
        rho_solid = sv[idx: idx + 3  * nn].reshape(3,  nn); idx += 3 * nn
        Hg        = sv[idx: idx + nn];                       idx += nn
        Ts        = sv[idx: idx + nn];                       idx += nn
        if shell_tube:
            Tw = sv[idx: idx + nn]; idx += nn
        # Acumuladores energéticos (siempre al final del sv)
        Q_mt_acc  = sv[idx: idx + nn]; idx += nn
        Q_rxn_acc = sv[idx: idx + nn]; idx += nn

        # Recover secondary variables
        Ctot     = np.sum(C, axis=0)
        Ctot_safe = np.maximum(Ctot, 1.0e-300)

        Tg = recover_Tg_from_Hg(
            C=C, Hg=Hg, prop_gas=prop_gas,
            n_comp=nc, epsi=epsi_r,
            Tg_guess=Tg_prev, gas_T_ref=gas_T_ref,
        )
        Tg_prev = Tg.copy()

        P    = Ctot * R_GAS * Tg / 1.0e5
        y    = C / Ctot_safe[None, :]

        # Boundary conditions at this time step
        bc = get_gasifier_boundary(
            t=float(t_arr[k]), P_cell=P, Ctot_cell=Ctot,
            bc_config=bc_config, n_comp=nc,
        )
        v_in  = float(bc["inlet"]["v_m_s"])
        v_out = float(bc["outlet"]["v_m_s"])
        C_in  = bc["inlet"]["C_mol_m3"]
        T_in  = bc["inlet"]["T_K"]

        # Store
        P_hist[k]        = P
        Tg_hist[k]       = Tg
        Ts_hist[k]       = Ts
        Hg_hist[k]       = Hg
        C_hist[k]        = C
        y_hist_3d[k]     = y
        rho_s_hist[k]    = rho_solid
        v_in_hist[k]     = v_in
        v_out_hist[k]    = v_out
        v_hist[k]        = 0.5 * (v_in + v_out)    # simple mean for 0D

        if shell_tube:
            Tw_hist[k] = Tw

        Q_mt_acc_hist[k]  = Q_mt_acc
        Q_rxn_acc_hist[k] = Q_rxn_acc

        if C_in is not None:
            C_in_hist[k] = C_in
        if T_in is not None:
            T_in_hist[k] = T_in

    result = types.SimpleNamespace(
        _t_results          = t_arr,
        _z                  = z,
        _species            = species,
        _P_results          = P_hist,
        _Tg_results         = Tg_hist,
        _Ts_results         = Ts_hist,
        _Tw_results         = Tw_hist,        # ndarray(n_t, N) or None
        _Hg_results         = Hg_hist,
        _y_results          = y_hist_3d,
        _C_results          = C_hist,
        _rho_solid_results  = rho_s_hist,
        _v_results          = v_hist,
        _v_in_results       = v_in_hist,
        _v_out_results      = v_out_hist,
        _C_in_results       = C_in_hist,
        _T_in_results       = T_in_hist,
        _Q_mt_acc_results   = Q_mt_acc_hist,  # (n_t, N) ∫q_mt dt  [J/m³_bed]
        _Q_rxn_acc_results  = Q_rxn_acc_hist, # (n_t, N) ∫Q_rxn dt [J/m³_bed]
    )
    return result
