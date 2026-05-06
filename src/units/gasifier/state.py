"""
@module gasifier.state
@brief Pack/unpack of the ODE state vector for the 1D gasifier model.

@details
Layout of the state vector:

    Without shell-tube  (wall_config absent or None):
        sv = [C(nc,N), rho_solid(3,N), Hg(N), Ts(N), Q_mt_acc(N), Q_rxn_acc(N), Q_gs_acc(N)]
        size = 17 * N

    With shell-tube  (wall_config present):
        sv = [C(nc,N), rho_solid(3,N), Hg(N), Ts(N), Tw(N), Q_mt_acc(N), Q_rxn_acc(N), Q_gs_acc(N)]
        size = 18 * N

Primary variables:
    C_i       — gas molar concentrations [mol/m³_gas] (per unit of void volume)
    rho_s_i   — solid bulk densities [kg/m³_bed] (per unit of bed volume)
    Hg        — volumetric enthalpy of gas [J/m³_bed] = epsi_r * Σ C_i * h_i(Tg)
    Ts        — solid temperature [K] (integrated directly)
    Tw        — wall temperature [K] (optional, only when shell_tube=True)

Accumulator variables (integrated, always present, start at 0):
    Q_mt_acc  — ∫q_mt dt  [J/m³_bed]  entalpía portada por masa sól→gas
    Q_rxn_acc — ∫Q_rxn dt [J/m³_bed]  calor de reacciones en el sólido
    Permiten cierres energéticos exactos en gasifier_balances sin re-evaluar el RHS.

Secondary variables (derived, never integrated):
    Tg    — gas temperature [K], recovered from Hg by Newton (recover_Tg_from_Hg)
    P     — total pressure [bar] = Ctot * R * Tg / 1e5
    y_i   — molar fractions [-] = C_i / Ctot
    Ctot  — total molar concentration [mol/m³_gas]

Why Ts is primary but Hg is used for the gas:
    The solid composition changes drastically during conversion (biomass → char),
    making the inversion Hs → Ts ill-conditioned when the solid thermal mass
    (Σ rho_s_i * Cp_s_i) changes by orders of magnitude. Integrating Ts directly
    avoids this. The gas phase uses Hg → Tg (Newton) as in the PSA/heater model,
    which is well-conditioned because the gas composition changes smoothly.

Species order (nc_gas = 9):
    0: CO   1: CO2   2: H2O   3: H2   4: O2
    5: CH4  6: C2H4  7: tar   8: N2

Solid species order (n_solid = 3):
    0: biomass   1: char   2: moisture
"""

import numpy as np

from src.physics.thermodynamics.enthalpy import (
    calc_volumetric_enthalpy,
    recover_Tg_from_Hg,
)
from src.utils.profiling import profiled

R_GAS    = 8.31446261815324   # [J/mol/K]
NC_GAS   = 9                  # number of gas species (fixed for gasifier)
N_SOLID  = 3                  # number of solid bulk-density variables


# ─── Pack / unpack ────────────────────────────────────────────────────────────

@profiled
def pack_state_vector(
    C:            np.ndarray,    # (nc_gas, N) [mol/m³_gas]
    rho_solid:    np.ndarray,    # (n_solid, N) [kg/m³_bed] — [biomass, char, moisture]
    Hg:           np.ndarray,    # (N,) [J/m³_bed]
    Ts:           np.ndarray,    # (N,) [K]
    Tw:           np.ndarray | None = None,   # (N,) [K] — only when shell_tube=True
    Q_mt_acc:     np.ndarray | None = None,   # (N,) [J/m³_bed] — accumulated ∫q_mt dt
    Q_rxn_acc:    np.ndarray | None = None,   # (N,) [J/m³_bed] — accumulated ∫Q_rxn dt
    Q_gs_acc:     np.ndarray | None = None,   # (N,) [J/m³_bed] — accumulated ∫q_gs dt
) -> np.ndarray:
    """
    Assemble the flat ODE state vector from primary and accumulator variables.

    Parameters
    ----------
    C         : ndarray (nc_gas, N)  gas concentrations [mol/m³_gas]
    rho_solid : ndarray (n_solid, N) solid bulk densities [kg/m³_bed]
    Hg        : ndarray (N,)         volumetric enthalpy of gas [J/m³_bed]
    Ts        : ndarray (N,)         solid temperature [K]
    Tw        : ndarray (N,) or None wall temperature [K]; None if no shell-tube
    Q_mt_acc  : ndarray (N,) or None accumulated ∫q_mt dt  [J/m³_bed]; zeros if None
    Q_rxn_acc : ndarray (N,) or None accumulated ∫Q_rxn dt [J/m³_bed]; zeros if None
    Q_gs_acc  : ndarray (N,) or None accumulated ∫q_gs dt  [J/m³_bed]; zeros if None

    Returns
    -------
    sv : ndarray (17*N,) or (18*N,)
    """
    nn = np.asarray(Hg).reshape(-1).shape[0]
    parts = [
        np.asarray(C,         dtype=float).reshape(-1),
        np.asarray(rho_solid, dtype=float).reshape(-1),
        np.asarray(Hg,        dtype=float).reshape(-1),
        np.asarray(Ts,        dtype=float).reshape(-1),
    ]
    if Tw is not None:
        parts.append(np.asarray(Tw, dtype=float).reshape(-1))
    # Acumuladores — siempre presentes, cero si no se pasan explícitamente
    parts.append(np.zeros(nn, dtype=float) if Q_mt_acc is None
                 else np.asarray(Q_mt_acc, dtype=float).reshape(-1))
    parts.append(np.zeros(nn, dtype=float) if Q_rxn_acc is None
                 else np.asarray(Q_rxn_acc, dtype=float).reshape(-1))
    parts.append(np.zeros(nn, dtype=float) if Q_gs_acc is None
                 else np.asarray(Q_gs_acc, dtype=float).reshape(-1))
    return np.concatenate(parts)


@profiled
def unpack_state_vector(
    sv:          np.ndarray,
    n_comp:      int,
    N:           int,
    prop_gas:    dict,
    epsi_r:      float,
    Tg_guess:    np.ndarray,    # (N,) [K] warm-start for Newton
    gas_T_ref:   float,
    shell_tube:  bool  = False,
    newton_tol:  float = 1.0e-8,
    newton_max_iter: int = 30,
) -> dict:
    """
    Extract primary variables from sv and compute secondary variables.

    Parameters
    ----------
    sv            : ndarray (17*N,) or (18*N,)
    n_comp        : int  — number of gas species (must be NC_GAS = 9)
    N             : int  — number of cells
    prop_gas      : dict — output of build_gas_prop_config (includes tar)
    epsi_r        : float — reactor bed void fraction [-]
    Tg_guess      : ndarray (N,) — Newton warm-start [K]
    gas_T_ref     : float — enthalpy reference temperature [K]
    shell_tube    : bool  — True if wall temperature Tw is part of the state
    newton_tol    : float — Newton convergence tolerance [K]
    newton_max_iter : int — Newton max iterations

    Returns
    -------
    dict with keys:
        C         (nc_gas, N)   [mol/m³_gas]
        rho_solid (n_solid, N)  [kg/m³_bed]   [biomass, char, moisture]
        Hg        (N,)          [J/m³_bed]
        Ts        (N,)          [K]
        Tg        (N,)          [K]
        Ctot      (N,)          [mol/m³_gas]
        P         (N,)          [bar]
        y         (nc_gas, N)   [-]
        Tw        (N,) or None  [K]
    """
    nc, nn = int(n_comp), int(N)

    # ── Extract primary blocks ───────────────────────────────────────────────
    idx = 0
    C         = sv[idx: idx + nc * nn].reshape(nc, nn); idx += nc * nn
    rho_solid = sv[idx: idx + N_SOLID * nn].reshape(N_SOLID, nn); idx += N_SOLID * nn
    Hg        = sv[idx: idx + nn].copy(); idx += nn
    Ts        = sv[idx: idx + nn].copy(); idx += nn
    Tw        = sv[idx: idx + nn].copy() if shell_tube else None

    # ── Secondary variables ──────────────────────────────────────────────────
    Ctot = np.sum(C, axis=0)                                    # (N,) [mol/m³_gas]
    Ctot_safe = np.maximum(Ctot, 1.0e-300)

    Tg = recover_Tg_from_Hg(
        C=C, Hg=Hg, prop_gas=prop_gas,
        n_comp=nc, epsi=epsi_r,
        Tg_guess=Tg_guess, gas_T_ref=gas_T_ref,
        max_iter=newton_max_iter, tol_T=newton_tol,
    )

    P = Ctot * R_GAS * Tg / 1.0e5                              # (N,) [bar]
    y = C / Ctot_safe[None, :]                                  # (nc_gas, N) [-]

    return {
        "C":         C,
        "rho_solid": rho_solid,
        "Hg":        Hg,
        "Ts":        Ts,
        "Tg":        Tg,
        "Ctot":      Ctot,
        "P":         P,
        "y":         y,
        "Tw":        Tw,
    }


# ─── Convenience constructors ─────────────────────────────────────────────────

def set_state(
    P_bar:        float | np.ndarray,    # [bar]
    Tg:           float | np.ndarray,    # [K]
    Ts:           float | np.ndarray,    # [K]
    y:            np.ndarray,            # (nc_gas,) or (nc_gas, N) [-]
    rho_biomass:  float | np.ndarray,    # [kg/m³_bed]
    rho_char:     float | np.ndarray,    # [kg/m³_bed]
    rho_moisture: float | np.ndarray,    # [kg/m³_bed]
    n_comp:       int,
    N:            int,
    prop_gas:     dict,
    epsi_r:       float,
    gas_T_ref:    float,
    Tw:           float | np.ndarray | None = None,  # [K] wall temp, optional
) -> dict:
    """
    Build the full initial state from physical variables.

    Computes C (ideal gas) and Hg (volumetric enthalpy) from (P, Tg, y).
    All scalar inputs are broadcast to shape (N,) or (nc_gas, N).

    Parameters
    ----------
    P_bar        : float or (N,) [bar]
    Tg           : float or (N,) [K]
    Ts           : float or (N,) [K]
    y            : (nc_gas,) or (nc_gas, N) molar fractions [-]
    rho_biomass  : float or (N,) [kg/m³_bed]
    rho_char     : float or (N,) [kg/m³_bed]
    rho_moisture : float or (N,) [kg/m³_bed]
    n_comp       : int — number of gas species
    N            : int — number of cells
    prop_gas     : dict — gas properties (includes tar)
    epsi_r       : float — bed void fraction [-]
    gas_T_ref    : float — enthalpy reference temperature [K]
    Tw           : float or (N,) or None [K] — initial wall temperature;
                   None if no shell-tube model

    Returns
    -------
    dict with keys: C, rho_solid, Hg, Ts, Tg, P, y, Tw, sv0
        sv0 : ndarray (17*N,) or (18*N,) — packed state vector
    """
    nc, nn = int(n_comp), int(N)

    P_Pa = np.broadcast_to(np.asarray(P_bar, float) * 1.0e5, (nn,)).copy()
    Tg_  = np.broadcast_to(np.asarray(Tg,    float),         (nn,)).copy()
    Ts_  = np.broadcast_to(np.asarray(Ts,    float),         (nn,)).copy()

    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim == 1:
        y_arr = np.tile(y_arr[:, None], (1, nn))   # (nc_gas, N)
    _validate_set_state(P_Pa, Tg_, Ts_, y_arr, nc, nn)

    # Gas concentrations from ideal gas law:  C_i = y_i * P / (R * Tg)
    C = y_arr * P_Pa[None, :] / (R_GAS * Tg_[None, :])        # (nc_gas, N) [mol/m³_gas]

    # Volumetric enthalpy of gas: Hg = epsi_r * Σ C_i * h_i(Tg)
    Hg = calc_volumetric_enthalpy(
        C=C, Tg=Tg_, prop_gas=prop_gas,
        n_comp=nc, epsi=epsi_r, gas_T_ref=gas_T_ref,
    )                                                            # (N,) [J/m³_bed]

    rho_s = np.stack([
        np.broadcast_to(np.asarray(rho_biomass,  float), (nn,)).copy(),
        np.broadcast_to(np.asarray(rho_char,     float), (nn,)).copy(),
        np.broadcast_to(np.asarray(rho_moisture, float), (nn,)).copy(),
    ], axis=0)                                                   # (3, N) [kg/m³_bed]

    Tw_ = (np.broadcast_to(np.asarray(Tw, float), (nn,)).copy()
           if Tw is not None else None)

    sv0 = pack_state_vector(C=C, rho_solid=rho_s, Hg=Hg, Ts=Ts_, Tw=Tw_)

    return {
        "C":         C,
        "rho_solid": rho_s,
        "Hg":        Hg,
        "Ts":        Ts_,
        "Tg":        Tg_,
        "P":         P_Pa / 1.0e5,
        "y":         y_arr,
        "Tw":        Tw_,
        "sv0":       sv0,
    }


def build_sv0_from_results(gasifier_results) -> np.ndarray:
    """
    Extract the state vector at the last time step of a gasifier result object.

    Parameters
    ----------
    gasifier_results : SimpleNamespace
        Output of build_gasifier_results (from state_extraction.py).

    Returns
    -------
    sv0 : ndarray (17*N,) or (18*N,) — packed state at t_final
    """
    r = gasifier_results

    C_last         = r._C_results[-1]             # (nc, N) [mol/m³_gas]
    rho_solid_last = r._rho_solid_results[-1]     # (3, N) [kg/m³_bed]
    Hg_last        = r._Hg_results[-1]            # (N,)
    Ts_last        = r._Ts_results[-1]            # (N,)
    Tw_last        = (r._Tw_results[-1]
                      if r._Tw_results is not None else None)   # (N,) or None

    return pack_state_vector(
        C=C_last, rho_solid=rho_solid_last, Hg=Hg_last, Ts=Ts_last, Tw=Tw_last,
    )


# ─── Internal validation ──────────────────────────────────────────────────────

def _validate_set_state(P_Pa, Tg, Ts, y, nc, N):
    if np.any(P_Pa <= 0.0):
        raise ValueError("set_state: P_bar must be > 0 everywhere")
    if np.any(Tg <= 0.0):
        raise ValueError("set_state: Tg must be > 0 everywhere")
    if np.any(Ts <= 0.0):
        raise ValueError("set_state: Ts must be > 0 everywhere")
    if y.shape != (nc, N):
        raise ValueError(
            f"set_state: y must have shape ({nc}, {N}), got {y.shape}"
        )
    y_sum = np.sum(y, axis=0)
    if np.any(np.abs(y_sum - 1.0) > 1.0e-4):
        raise ValueError(
            f"set_state: molar fractions must sum to 1; "
            f"max deviation = {np.max(np.abs(y_sum - 1.0)):.2e}"
        )
    if np.any(y < -1.0e-12):
        raise ValueError("set_state: molar fractions must be >= 0")
