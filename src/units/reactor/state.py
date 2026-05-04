"""
Pack/unpack of the ODE state vector for the 1D reactor model.

Layout depends on two flags (has_catalyst, shell_tube):

    No catalyst, no shell_tube:   [C(nc,N), Hg(N)]               → (nc+1)*N
    No catalyst, with shell_tube: [C(nc,N), Hg(N), Tw(N)]        → (nc+2)*N
    Catalyst,    no shell_tube:   [C(nc,N), Hg(N), Ts(N)]        → (nc+2)*N
    Catalyst,    with shell_tube: [C(nc,N), Hg(N), Ts(N), Tw(N)] → (nc+3)*N

Primary variables
-----------------
C_i  — gas molar concentrations [mol/m³_gas]
Hg   — volumetric gas enthalpy [J/m³_bed] = epsi * Σ C_i * h_i(Tg)
Ts   — catalyst/solid temperature [K]  (only when has_catalyst=True)
Tw   — wall temperature [K]            (only when shell_tube=True)

Secondary variables (derived, not integrated)
---------------------------------------------
Tg   — gas temperature [K], recovered from Hg by Newton
P    — total pressure [bar] = Ctot * R * Tg / 1e5
y    — molar fractions [-] = C_i / Ctot

Design notes
------------
- rho_catalyst is constant → not part of sv (stored in params).
- No accumulators: q_rxn is recoverable post-hoc by calling rate_fn on stored
  results (unlike the gasifier where q_masstransfer requires knowing solid mass
  at each internal ODE step).
- For empty tubes (no catalyst): epsi = 1.0 (all void); the user must pass
  epsi=1.0 explicitly in params.
- Ts is always dynamic when has_catalyst=True (no isothermal catalyst shortcut).
"""

from __future__ import annotations

import numpy as np

from src.physics.thermodynamics.enthalpy import (
    calc_volumetric_enthalpy,
    recover_Tg_from_Hg,
)
from src.utils.profiling import profiled

R_GAS = 8.31446261815324   # [J/mol/K]


# ─── Pack / Unpack ────────────────────────────────────────────────────────────

@profiled
def pack_state_vector(
    C:  np.ndarray,               # (nc, N)  [mol/m³_gas]
    Hg: np.ndarray,               # (N,)     [J/m³_bed]
    Ts: np.ndarray | None = None, # (N,)     [K]  — None if no catalyst
    Tw: np.ndarray | None = None, # (N,)     [K]  — None if no shell_tube
) -> np.ndarray:
    """
    Assemble the flat ODE state vector.

    Parameters
    ----------
    C  : ndarray (nc, N)  gas concentrations [mol/m³_gas]
    Hg : ndarray (N,)     volumetric gas enthalpy [J/m³_bed]
    Ts : ndarray (N,) or None  catalyst temperature [K]
    Tw : ndarray (N,) or None  wall temperature [K]

    Returns
    -------
    sv : ndarray  size (nc+1)*N, (nc+2)*N, or (nc+3)*N
    """
    parts = [
        np.asarray(C,  dtype=float).reshape(-1),
        np.asarray(Hg, dtype=float).reshape(-1),
    ]
    if Ts is not None:
        parts.append(np.asarray(Ts, dtype=float).reshape(-1))
    if Tw is not None:
        parts.append(np.asarray(Tw, dtype=float).reshape(-1))
    return np.concatenate(parts)


@profiled
def unpack_state_vector(
    sv:           np.ndarray,
    n_comp:       int,
    N:            int,
    has_catalyst: bool,
    shell_tube:   bool,
    prop_gas:     dict,
    epsi:         float,
    Tg_guess:     np.ndarray,   # (N,) [K]  warm-start for Newton
    gas_T_ref:    float,
    newton_tol:      float = 1.0e-8,
    newton_max_iter: int   = 30,
) -> dict:
    """
    Extract primary variables from sv and compute secondary variables.

    Parameters
    ----------
    sv           : ndarray — flat state vector
    n_comp       : int     — number of gas species
    N            : int     — number of cells
    has_catalyst : bool    — True → Ts is part of sv
    shell_tube   : bool    — True → Tw is part of sv
    prop_gas     : dict    — gas properties from build_gas_prop_config
    epsi         : float   — void fraction [-]
    Tg_guess     : ndarray (N,)  Newton warm-start [K]
    gas_T_ref    : float   — enthalpy reference temperature [K]

    Returns
    -------
    dict with keys:
        C    (nc, N)     [mol/m³_gas]
        Hg   (N,)        [J/m³_bed]
        Ts   (N,) or None [K]
        Tw   (N,) or None [K]
        Tg   (N,)        [K]
        Ctot (N,)        [mol/m³_gas]
        P    (N,)        [bar]
        y    (nc, N)     [-]
    """
    nc, nn = int(n_comp), int(N)
    idx = 0

    C  = sv[idx: idx + nc * nn].reshape(nc, nn); idx += nc * nn
    Hg = sv[idx: idx + nn].copy();               idx += nn
    Ts = sv[idx: idx + nn].copy() if has_catalyst else None
    if has_catalyst:
        idx += nn
    Tw = sv[idx: idx + nn].copy() if shell_tube else None

    Ctot      = np.sum(C, axis=0)                              # (N,) [mol/m³_gas]
    Ctot_safe = np.maximum(Ctot, 1.0e-300)

    Tg = recover_Tg_from_Hg(
        C=C, Hg=Hg, prop_gas=prop_gas,
        n_comp=nc, epsi=epsi,
        Tg_guess=Tg_guess, gas_T_ref=gas_T_ref,
        max_iter=newton_max_iter, tol_T=newton_tol,
    )

    P = Ctot * R_GAS * Tg / 1.0e5                             # (N,) [bar]
    y = C / Ctot_safe[None, :]                                 # (nc, N) [-]

    return {"C": C, "Hg": Hg, "Ts": Ts, "Tw": Tw,
            "Tg": Tg, "Ctot": Ctot, "P": P, "y": y}


# ─── Constructor de estado inicial ───────────────────────────────────────────

def set_state(
    P_bar:     float | np.ndarray,  # [bar]
    Tg:        float | np.ndarray,  # [K]
    y:         np.ndarray,          # (nc,) or (nc, N) [-]
    n_comp:    int,
    N:         int,
    prop_gas:  dict,
    epsi:      float,
    gas_T_ref: float,
    Ts:        float | np.ndarray | None = None,  # [K]  — required if has_catalyst
    Tw:        float | np.ndarray | None = None,  # [K]  — required if shell_tube
) -> dict:
    """
    Build the full initial state from physical variables.

    Computes C from ideal gas law and Hg from enthalpy sum.
    Scalar inputs are broadcast to (N,) or (nc, N).

    Parameters
    ----------
    P_bar     : float or (N,)      pressure [bar]
    Tg        : float or (N,)      gas temperature [K]
    y         : (nc,) or (nc, N)   molar fractions [-]
    n_comp    : int                number of gas species
    N         : int                number of cells
    prop_gas  : dict               gas properties
    epsi      : float              void fraction [-] (use 1.0 for empty tube)
    gas_T_ref : float              enthalpy reference temperature [K]
    Ts        : float, (N,), or None  catalyst temperature [K]
    Tw        : float, (N,), or None  wall temperature [K]

    Returns
    -------
    dict with keys: C, Hg, Ts, Tw, Tg, P, y, sv0
        sv0 : ndarray — packed state vector
    """
    nc, nn = int(n_comp), int(N)

    P_Pa = np.broadcast_to(np.asarray(P_bar, float) * 1.0e5, (nn,)).copy()
    Tg_  = np.broadcast_to(np.asarray(Tg,    float),         (nn,)).copy()

    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim == 1:
        y_arr = np.tile(y_arr[:, None], (1, nn))
    _validate_set_state(P_Pa, Tg_, y_arr, nc, nn)

    # C_i = y_i * P / (R * Tg)
    C  = y_arr * P_Pa[None, :] / (R_GAS * Tg_[None, :])      # (nc, N) [mol/m³_gas]

    # Hg = epsi * Σ C_i * h_i(Tg)
    Hg = calc_volumetric_enthalpy(
        C=C, Tg=Tg_, prop_gas=prop_gas,
        n_comp=nc, epsi=epsi, gas_T_ref=gas_T_ref,
    )                                                           # (N,) [J/m³_bed]

    Ts_ = (np.broadcast_to(np.asarray(Ts, float), (nn,)).copy()
           if Ts is not None else None)
    Tw_ = (np.broadcast_to(np.asarray(Tw, float), (nn,)).copy()
           if Tw is not None else None)

    sv0 = pack_state_vector(C=C, Hg=Hg, Ts=Ts_, Tw=Tw_)

    return {"C": C, "Hg": Hg, "Ts": Ts_, "Tw": Tw_,
            "Tg": Tg_, "P": P_Pa / 1.0e5, "y": y_arr, "sv0": sv0}


def build_sv0_from_results(reactor_results) -> np.ndarray:
    """
    Extract the packed state vector at the last time step of a result object.

    Parameters
    ----------
    reactor_results : SimpleNamespace  — output of build_reactor_results

    Returns
    -------
    sv0 : ndarray — state vector at t_final
    """
    r = reactor_results
    return pack_state_vector(
        C  = r._C_results[-1],
        Hg = r._Hg_results[-1],
        Ts = r._Ts_results[-1] if r._Ts_results is not None else None,
        Tw = r._Tw_results[-1] if r._Tw_results is not None else None,
    )


# ─── Validación interna ───────────────────────────────────────────────────────

def _validate_set_state(P_Pa, Tg, y, nc, N):
    if np.any(P_Pa <= 0.0):
        raise ValueError("set_state: P_bar must be > 0 everywhere")
    if np.any(Tg <= 0.0):
        raise ValueError("set_state: Tg must be > 0 everywhere")
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
