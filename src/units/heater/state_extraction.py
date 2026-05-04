"""
@module heater.state_extraction
@brief Extracción y postproceso del vector de estado devuelto por runner_heater.

@details
Convierte la historia cruda `y_hist` (n_t, sv_size) en un objeto con los
atributos estandarizados que consumen las herramientas de visualización y
análisis del proyecto.

Layout del vector de estado (consistente con rhs_heater y pack_state_vector):
    sv = [C_0, C_1, …, C_{nc-1}, Hg]
    Cada bloque tiene N elementos (una celda por nodo).
    Tamaño total: (nc + 1) * N

El bucle temporal realiza en un único pase:
    1. Desempaquetado del estado: C, Hg, Tg (Newton warm-start), P, y.
    2. Propiedades de mezcla: rho, mu (para reconstrucción de velocidad).
    3. Contornos exactos: v_in, T_in, C_in, v_out via get_heater_boundary.
    4. Darcy-Weisbach → v_face (N+1,) → v_cell (N,) media de caras.

Función pública:
    build_heater_results(y_hist, t_arr, params) -> types.SimpleNamespace

El objeto devuelto expone:
    _t_results     : (n_t,)       — vector de tiempos [s]
    _z             : (N,)         — coordenadas axiales de centros de celda [m]
    _species       : list[str]    — nombres de las nc especies
    _prop_gas      : dict         — propiedades puras del gas
    _P_results     : (n_t, N)     — presión [bar]
    _Tg_results    : (n_t, N)     — temperatura del gas [K]
    _Hg_results    : (n_t, N)     — entalpía volumétrica del gas [J/m³]
    _y_results     : (n_t, nc, N) — fracción molar [-]
    _C_results     : (n_t, nc, N) — concentración molar [mol/m³]
    _v_results     : (n_t, N)     — velocidad superficial en centros de celda [m/s]
    _v_in_results  : (n_t,)       — velocidad en cara de entrada [m/s]
    _v_out_results : (n_t,)       — velocidad en cara de salida [m/s]
    _C_in_results  : (n_t, nc)    — concentración molar en inlet [mol/m³]
    _T_in_results  : (n_t,)       — temperatura en inlet [K]

Parámetros requeridos en params:
    n_comp, N, dz, gas_T_ref, prop_gas, Di, bc_config.
    Opcional: "species" — list[str] con nombres de componentes.

Unidades: SI.
"""

from __future__ import annotations

import types
from typing import Dict, Any

import numpy as np

from src.boundary_conditions.heater_boundary import get_heater_boundary
from src.physics.momentum.darcy_weisbach import continuity_face_velocity
from src.units.heater.state import unpack_state_vector
from src.utils.profiling import profiled


@profiled
def build_heater_results(
    y_hist: np.ndarray,
    t_arr:  np.ndarray,
    params: Dict[str, Any],
) -> types.SimpleNamespace:
    """
    @brief
    Convierte la historia del vector de estado del heater en un objeto con los
    atributos estandarizados de resultados.

    @details
    Realiza en un único bucle temporal:
        1. Desempaquetado del estado: C, Hg, Tg (Newton warm-start), P, y, Ctot.
        2. Propiedades de mezcla: rho, mu para reconstrucción de velocidad DW.
        3. Contornos exactos: v_in, T_in, C_in, v_out via get_heater_boundary.
        4. Darcy-Weisbach → v_face → v_cell como media de caras adyacentes.

    Parameters
    ----------
    y_hist : ndarray, shape (n_t, sv_size)
        Historia del vector de estado devuelta por run_step (orientación física).
    t_arr  : ndarray, shape (n_t,)
        Instantes de tiempo correspondientes [s].
    params : dict
        Mismo dict de parámetros que se pasó a run_step. Debe contener:
        n_comp, N, dz, gas_T_ref, prop_gas, Di, bc_config.
        Clave opcional: "species" — list[str] con nombres de componentes.

    Returns
    -------
    heater : types.SimpleNamespace
        Objeto con todos los atributos de resultados estandarizados.
    """
    # ── Lectura de parámetros ─────────────────────────────────────────────────
    nc        = int(params["n_comp"])
    nn        = int(params["N"])
    dz        = float(params["dz"])
    gas_T_ref = float(params["gas_T_ref"])
    prop_gas  = params["prop_gas"]
    bc_config = params["bc_config"]

    wall_config = params.get("wall_config")
    shell_tube  = wall_config is not None

    species = list(params.get("species", [f"comp_{i}" for i in range(nc)]))

    z_arr = np.linspace(dz / 2.0, nn * dz - dz / 2.0, nn)

    n_t = y_hist.shape[0]

    # ── Arrays de almacenamiento ──────────────────────────────────────────────
    C_hist      = np.zeros((n_t, nc, nn), dtype=float)
    Hg_hist     = np.zeros((n_t, nn),     dtype=float)
    Tg_hist     = np.zeros((n_t, nn),     dtype=float)
    P_hist      = np.zeros((n_t, nn),     dtype=float)
    y_hist_3d   = np.zeros((n_t, nc, nn), dtype=float)
    v_hist      = np.zeros((n_t, nn),     dtype=float)
    v_in_hist   = np.zeros(n_t,           dtype=float)
    v_out_hist  = np.zeros(n_t,           dtype=float)
    C_in_hist   = np.zeros((n_t, nc),     dtype=float)
    T_in_hist   = np.zeros(n_t,           dtype=float)

    if shell_tube:
        Tw_hist = np.zeros((n_t, nn), dtype=float)

    # Estimación inicial de Newton (se actualiza con el instante anterior)
    Tg_prev = np.full(nn, 300.0, dtype=float)

    # ── Bucle temporal ────────────────────────────────────────────────────────
    for k in range(n_t):
        sv_k = y_hist[k]

        # 1. Desempaquetar estado → C, Hg, Tg, P, y, Ctot [, Tw]
        state = unpack_state_vector(
            sv         = sv_k,
            n_comp     = nc,
            N          = nn,
            prop_gas   = prop_gas,
            Tg_guess   = Tg_prev,
            gas_T_ref  = gas_T_ref,
            shell_tube = shell_tube,
        )
        C_mat  = state["C"]    # (nc, N)
        Tg_arr = state["Tg"]  # (N,)
        P_bar  = state["P"]   # (N,) [bar]
        y_mat  = state["y"]   # (nc, N)
        Ctot   = state["Ctot"]# (N,)

        C_hist[k]    = C_mat
        Hg_hist[k]   = state["Hg"]
        Tg_hist[k]   = Tg_arr
        P_hist[k]    = P_bar
        y_hist_3d[k] = y_mat

        if shell_tube:
            Tw_hist[k] = state["Tw"]

        Tg_prev = Tg_arr.copy()   # warm-start Newton para el siguiente instante

        # 2. Contornos exactos (mismo cálculo que core_rhs)
        bc = get_heater_boundary(
            t         = float(t_arr[k]),
            P_cell    = P_bar,
            Ctot_cell = Ctot,
            bc_config = bc_config,
            n_comp    = nc,
        )
        v_in  = float(bc["inlet"]["v_m_s"])
        v_out = float(bc["outlet"]["v_m_s"])

        v_in_hist[k]  = v_in
        v_out_hist[k] = v_out
        C_in_hist[k]  = np.asarray(bc["inlet"]["C_mol_m3"], dtype=float).reshape(-1)
        T_in_hist[k]  = float(bc["inlet"]["T_K"])

        # 4. Continuidad → velocidad en caras → centros de celda
        C_tot_in_k = float(C_in_hist[k].sum())
        v_face_k = continuity_face_velocity(
            C_tot_cell = Ctot,
            C_tot_in   = C_tot_in_k,
            v_in       = v_in,
        )
        v_hist[k] = 0.5 * (v_face_k[:-1] + v_face_k[1:])

    # ── Construir objeto de resultados ────────────────────────────────────────
    result_dict = dict(
        _t_results     = t_arr,
        _z             = z_arr,
        _species       = species,
        _prop_gas      = prop_gas,
        # Campos principales (n_t, N) / (n_t, nc, N)
        _P_results     = P_hist,
        _Tg_results    = Tg_hist,
        _Hg_results    = Hg_hist,
        _y_results     = y_hist_3d,
        _C_results     = C_hist,
        _v_results     = v_hist,
        # Contornos exactos en caras
        _v_in_results  = v_in_hist,
        _v_out_results = v_out_hist,
        _C_in_results  = C_in_hist,
        _T_in_results  = T_in_hist,
    )
    if shell_tube:
        result_dict["_Tw_results"] = Tw_hist

    return types.SimpleNamespace(**result_dict)
