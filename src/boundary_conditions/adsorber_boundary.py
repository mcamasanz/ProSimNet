"""
@module boundary_conditions.adsorber_boundary
@brief Construcción del dict de condiciones de contorno para cada paso del ciclo PSA.

@details
En cada llamada al RHS el integrador necesita saber:
    - qué velocidad superficial imponer en la cara de entrada (cara 0)
    - qué velocidad superficial imponer en la cara de salida (cara N)
    - qué concentración / temperatura / fracción molar tiene el gas que entra

Estos valores dependen del paso PSA activo. Este módulo los encapsula en una
función pura `get_step_boundary` que devuelve un dict con sub-dicts "inlet" y
"outlet" homogéneos para todos los pasos.

Pasos soportados:
    "ads"       — adsorción: entrada a caudal fijo Q_in_feed, salida por válvula Cv_out_ads
    "purge"     — purga: entrada a caudal fijo Q_in_purge, salida por válvula Cv_out_purge
                  T_in_purge e y_in_purge pueden ser callables f(t)
    "pr_feed"   — presurización con feed: entrada por válvula Cv_in_prfeed, salida cerrada
    "blowdown"  — despresurización: entrada cerrada, salida por válvula Cv_out_blowdown
    "wait"      — espera: ambas caras cerradas (v_in = v_out = 0)

Convención de signo:
    v_in  > 0 — gas entra en la cara lógica 0 (dirección +z)
    v_out > 0 — gas sale por la cara lógica N (dirección +z)
    El llamador es responsable de aplicar la inversión de ejes si
    `flow_direction == "backward"` (multiplicar el campo de velocidad por -1
    o aplicar la matriz A_flip antes de llamar a esta función).

Las presiones en bc_config y P_cell están en bar. La función convierte a Pa
internamente para llamar a `valve_superficial_velocity`.

Unidades entrada/salida:
    P   [bar]
    T   [K]
    y   [-]
    C   [mol/m³]
    v   [m/s]
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from src.boundary_conditions.valve import valve_superficial_velocity

R_GAS = 8.31446261815324   # [J/mol/K]

# Pasos válidos de ciclo PSA single-column
_VALID_STEPS = frozenset({"ads", "purge", "pr_feed", "blowdown", "wait"})

from src.utils.profiling import profiled


@profiled
def get_step_boundary(
    t: float,
    P_cell: np.ndarray,
    Tg_cell: np.ndarray,
    y_cell: np.ndarray,
    step: str,
    bc_config: dict,
    n_comp: int,
    MW: np.ndarray,
    epsi: float,
    Ai: float,
) -> Dict[str, Any]:
    """
    @brief
    Construye el dict de contornos termodinámicos y cinemáticos para el paso PSA activo.

    @details
    Para cada paso PSA se determinan:
        - Las condiciones de la cara de entrada (P, T, y, C, v)
        - Las condiciones de la cara de salida (P_ext, P_cell, T, y, C, v)

    La velocidad de salida se calcula con la fórmula de válvula cuando la presión
    de la última celda supera la presión externa; en caso contrario v_out = 0.

    Parameters
    ----------
    t        : float — tiempo actual [s] (necesario para callables T/y en purge)
    P_cell   : np.ndarray, shape (N,) — presión por celda [bar]
    Tg_cell  : np.ndarray, shape (N,) — temperatura del gas por celda [K]
    y_cell   : np.ndarray, shape (nc, N) — fracciones molares por celda [-]
    step     : str — paso PSA activo: "ads", "purge", "pr_feed", "blowdown", "wait"
    bc_config: dict — resultado de `build_boundary_c_config`
    n_comp   : int  — número de componentes
    MW       : np.ndarray, shape (nc,) — masas molares [kg/mol]
    epsi     : float — porosidad del lecho [-]
    Ai       : float — área interna de la sección transversal [m²]

    Returns
    -------
    dict con claves:
        "step"   : str
        "inlet"  : dict con claves P_bar, T_K, y (nc,), C_mol_m3 (nc,), v_m_s
        "outlet" : dict con claves P_ext_bar, P_cell_bar, T_K, y (nc,), C_mol_m3 (nc,), v_m_s

    Notes
    -----
    - En pasos donde no hay entrada de gas (blowdown, wait): inlet.P_bar = None,
      inlet.T_K = None, inlet.y = None, inlet.C_mol_m3 = None.
    - En pasos donde no hay salida de gas (pr_feed, wait): outlet.P_ext_bar = None,
      outlet.v_m_s = 0.
    - Las fracciones molares y concentraciones de salida corresponden siempre
      al gas de la última celda (extrapolación tipo copy-BC).
    """
    step_str = str(step).strip().lower()
    if step_str not in _VALID_STEPS:
        raise ValueError(f"step must be one of {sorted(_VALID_STEPS)}, got '{step_str}'")

    P_arr  = np.asarray(P_cell,  dtype=float).reshape(-1)
    Tg_arr = np.asarray(Tg_cell, dtype=float).reshape(-1)
    y_mat  = np.asarray(y_cell,  dtype=float).reshape(int(n_comp), -1)
    MW_arr = np.asarray(MW,      dtype=float).reshape(-1)

    # --- propiedades de la cara de salida (última celda) ---
    y_out  = y_mat[:, -1].copy()                                      # [-], shape (nc,)
    P_out_cell_bar = float(P_arr[-1])                                 # [bar]
    T_out_cell_K   = float(Tg_arr[-1])                                # [K]
    MW_out = float(np.dot(y_out, MW_arr))                             # [kg/mol]
    C_out  = y_out * (P_out_cell_bar * 1.0e5) / (R_GAS * max(T_out_cell_K, 1.0e-12))  # [mol/m³]

    # --- helper: C a partir de (y, P_bar, T_K) ---
    def _make_C(y_arr, P_bar, T_K):
        return y_arr * (P_bar * 1.0e5) / (R_GAS * max(T_K, 1.0e-12))

    # =========================================================
    # Lógica por paso
    # =========================================================
    if step_str == "ads":
        P_in_bar  = float(bc_config["P_in_feed"])
        T_in_K    = float(bc_config["T_in_feed"])
        y_in      = np.asarray(bc_config["y_in_feed"], dtype=float).reshape(-1)
        C_in      = _make_C(y_in, P_in_bar, T_in_K)
        v_in      = float(bc_config["Q_in_feed"]) / (epsi * max(Ai, 1.0e-30))
        P_out_ext = float(bc_config["P_out_ads"])
        v_out     = valve_superficial_velocity(
            Cv=bc_config["Cv_out_ads"],
            P_up_Pa=P_out_cell_bar * 1.0e5,
            P_down_Pa=P_out_ext * 1.0e5,
            T_up=T_out_cell_K,
            MW_mix=MW_out,
            epsi=epsi, Ai=Ai,
        )

    elif step_str == "purge":
        # T_in_purge e y_in_purge pueden ser callables f(t)
        T_raw = bc_config["T_in_purge"]
        y_raw = bc_config["y_in_purge"]
        P_in_bar = float(bc_config["P_in_purge"])
        T_in_K   = float(T_raw(t)) if callable(T_raw) else float(T_raw)
        y_in_raw = y_raw(t)        if callable(y_raw) else y_raw
        y_in     = np.asarray(y_in_raw, dtype=float).reshape(-1)
        C_in     = _make_C(y_in, P_in_bar, T_in_K)
        v_in     = float(bc_config["Q_in_purge"]) / (epsi * max(Ai, 1.0e-30))
        P_out_ext = float(bc_config["P_out_purge"])
        v_out     = valve_superficial_velocity(
            Cv=bc_config["Cv_out_purge"],
            P_up_Pa=P_out_cell_bar * 1.0e5,
            P_down_Pa=P_out_ext * 1.0e5,
            T_up=T_out_cell_K,
            MW_mix=MW_out,
            epsi=epsi, Ai=Ai,
        )

    elif step_str == "pr_feed":
        P_in_bar  = float(bc_config["P_high"])
        T_in_K    = float(bc_config["T_in_feed"])
        y_in      = np.asarray(bc_config["y_in_feed"], dtype=float).reshape(-1)
        C_in      = _make_C(y_in, P_in_bar, T_in_K)
        MW_in     = float(np.dot(y_in, MW_arr))
        # Velocidad de entrada por válvula (entrada a la cara 0)
        v_in      = valve_superficial_velocity(
            Cv=bc_config["Cv_in_prfeed"],
            P_up_Pa=P_in_bar * 1.0e5,
            P_down_Pa=float(P_arr[0]) * 1.0e5,
            T_up=T_in_K,
            MW_mix=MW_in,
            epsi=epsi, Ai=Ai,
        )
        P_out_ext = None   # salida cerrada
        v_out     = 0.0

    elif step_str == "blowdown":
        P_in_bar  = None
        T_in_K    = None
        y_in      = None
        C_in      = None
        v_in      = 0.0
        P_out_ext = float(bc_config["P_low"])
        v_out     = valve_superficial_velocity(
            Cv=bc_config["Cv_out_blowdown"],
            P_up_Pa=P_out_cell_bar * 1.0e5,
            P_down_Pa=P_out_ext * 1.0e5,
            T_up=T_out_cell_K,
            MW_mix=MW_out,
            epsi=epsi, Ai=Ai,
        )

    else:  # "wait"
        P_in_bar  = None
        T_in_K    = None
        y_in      = None
        C_in      = None
        v_in      = 0.0
        P_out_ext = None
        v_out     = 0.0

    return {
        "step": step_str,
        "inlet": {
            "P_bar":      P_in_bar,
            "T_K":        T_in_K,
            "y":          None if y_in is None else np.asarray(y_in, dtype=float).reshape(-1),
            "C_mol_m3":   None if C_in is None else np.asarray(C_in, dtype=float).reshape(-1),
            "v_m_s":      float(v_in),
        },
        "outlet": {
            "P_ext_bar":  P_out_ext,
            "P_cell_bar": P_out_cell_bar,
            "T_K":        T_out_cell_K,
            "y":          y_out,
            "C_mol_m3":   C_out,
            "v_m_s":      float(v_out),
        },
    }
