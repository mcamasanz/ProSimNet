"""
@module solvers.state_extraction
@brief Extracción y postproceso del vector de estado devuelto por run_step.

@details
Convierte la historia cruda `y_hist` (n_t, sv_size) en un objeto con los
atributos estandarizados que consumen las herramientas de visualización y
análisis del proyecto. El cálculo se realiza en un único pase temporal, por
lo que la recuperación de Tg (Newton) y la reconstrucción de velocidad
(Ergun) comparten el mismo bucle sin duplicar operaciones.

Layout del vector de estado (consistente con rhs_adsorption y pack_state_vector):
    Sin shell-tube:  sv = [C_0, …, C_{nc-1}, q_0, …, q_{nc-1}, Hg, Ts]
    Con shell-tube:  sv = [C_0, …, C_{nc-1}, q_0, …, q_{nc-1}, Hg, Ts, Tw]
    Cada bloque tiene N elementos (una celda por nodo).

Función pública:
    build_adsorber_results(y_hist, t_arr, params) -> types.SimpleNamespace

El objeto devuelto expone:
    _t_results  : (n_t,)       — vector de tiempos [s]
    _z          : (N,)         — coordenadas axiales de centros de celda [m]
    _species    : list[str]    — nombres de las nc especies
    _prop_gas   : dict         — propiedades puras del gas
    _prop_lecho : dict         — propiedades del lecho
    _P_results  : (n_t, N)     — presión [bar]
    _Tg_results : (n_t, N)     — temperatura del gas [K]
    _Ts_results : (n_t, N)     — temperatura del sólido [K]
    _Tw_results : (n_t, N)     — temperatura de la pared [K]  (solo si shell_tube)
    _v_results  : (n_t, N)     — velocidad superficial en centros de celda [m/s]
    _y_results  : (n_t, nc, N) — fracción molar [-]
    _C_results  : (n_t, nc, N) — concentración molar [mol/m³]
    _q_results  : (n_t, nc, N) — carga adsorbida [mol/kg]

Parámetros requeridos en params:
    n_comp, N, dz, epsi, gas_T_ref, prop_gas, MW, prop_lecho,
    bc_config, Ai, step.
    Opcionales: "species" — list[str] con nombres de componentes;
                "wall_config" — dict de pared dinámica (activa shell_tube).

Unidades: SI.
"""

from __future__ import annotations

import types
from typing import Dict, Any

import numpy as np

from src.boundary_conditions.adsorber_boundary import get_step_boundary
from src.physics.momentum.ergun import ergun_face_velocity
from src.physics.thermodynamics.mixture import (
    wilke_mix_property,
    mixture_molar_mass,
    ideal_gas_density,
)
from src.units.adsorber.state import unpack_state_vector
from src.utils.profiling import profiled


# ═══════════════════════════════════════════════════════════════════════════════
# Helper interno: viscosidad de cada especie pura en cada celda
# ═══════════════════════════════════════════════════════════════════════════════

def _species_viscosities(prop_gas: dict, Tg: np.ndarray, nc: int) -> np.ndarray:
    """
    Calcula la viscosidad pura de cada especie en cada celda.

    Parameters
    ----------
    prop_gas : dict — prop_gas["mu"] es ndarray(nc,) en modo constant/fixed,
                      o list[callable] en modo polynomial.
    Tg       : ndarray (N,) — temperatura del gas por celda [K].
    nc       : int — número de especies.

    Returns
    -------
    mu_sp : ndarray (nc, N) — viscosidad pura por especie y celda [Pa·s].
    """
    N_cells = Tg.shape[0]
    mu_sp   = np.zeros((nc, N_cells), dtype=float)
    mu_data = prop_gas["mu"]

    if isinstance(mu_data, np.ndarray):
        # Modos "constant" / "fixed": valor constante por especie
        for i in range(nc):
            mu_sp[i, :] = float(mu_data[i])
    else:
        # Modo "polynomial": lista de nc callables mu_i(T)
        for i, fn in enumerate(mu_data):
            mu_sp[i, :] = np.asarray(fn(Tg), dtype=float).reshape(-1)

    return mu_sp


# ═══════════════════════════════════════════════════════════════════════════════
# Función pública
# ═══════════════════════════════════════════════════════════════════════════════

@profiled
def build_adsorber_results(
    y_hist: np.ndarray,
    t_arr:  np.ndarray,
    params: Dict[str, Any],
) -> types.SimpleNamespace:
    """
    @brief
    Convierte la historia del vector de estado en un objeto con los atributos
    estandarizados de resultados.

    @details
    Realiza en un único bucle temporal:
        1. Desempaquetado del estado: C, q, Hg, Ts, Tg (Newton warm-start), P, y.
        2. Propiedades de mezcla: μ_mix, MW_mix, ρ_mix via Wilke y gas ideal.
        3. Contornos exactos del paso: v_in, v_out via get_step_boundary.
        4. Ergun → v_face (N+1,) → v_cell (N,) como media de caras adyacentes.

    El uso del warm-start de Newton (Tg del instante anterior como estimación
    inicial del siguiente) acelera significativamente la convergencia en cada paso.

    Parameters
    ----------
    y_hist : ndarray, shape (n_t, sv_size)
        Historia del vector de estado devuelta por run_step (orientación física).
    t_arr  : ndarray, shape (n_t,)
        Instantes de tiempo correspondientes [s].
    params : dict
        Mismo dict de parámetros que se pasó a run_step. Debe contener:
        n_comp, N, dz, epsi, gas_T_ref, prop_gas, MW, prop_lecho,
        bc_config, Ai, step.
        Clave opcional: "species" — list[str] con nombres de componentes.

    Returns
    -------
    col : types.SimpleNamespace
        Objeto con todos los atributos de resultados estandarizados rellenos,
        incluyendo _v_results (velocidad superficial reconstruida exactamente).

    Notes
    -----
    - Coste: O(n_t · N · nc²) por el Newton de Tg y la regla de Wilke.
    - params["step"] debe corresponder al paso que generó y_hist. Si se
      concatenan historiales de varios pasos, llamar separadamente por tramo.
    - Si "species" no está en params se generan nombres genéricos "comp_i".
    """
    # ── Lectura de parámetros ─────────────────────────────────────────────────
    nc         = int(params["n_comp"])
    nn         = int(params["N"])
    dz         = float(params["dz"])
    epsi       = float(params["epsi"])
    gas_T_ref  = float(params["gas_T_ref"])
    prop_gas   = params["prop_gas"]
    prop_lecho = params["prop_lecho"]
    MW         = np.asarray(params["MW"], dtype=float)
    bc_config  = params["bc_config"]
    Ai         = float(params["Ai"])
    step       = str(params["step"]).strip().lower()

    species    = list(params.get("species", [f"comp_{i}" for i in range(nc)]))
    shell_tube = params.get("wall_config") is not None

    # Diámetro de partícula: media si el campo es heterogéneo (N,)
    dp = float(np.mean(np.asarray(prop_lecho["D_p"], dtype=float).reshape(-1)))

    # Coordenada axial de centros de celda
    z_arr = np.linspace(dz / 2.0, nn * dz - dz / 2.0, nn)

    n_t = y_hist.shape[0]

    # ── Arrays de almacenamiento ──────────────────────────────────────────────
    C_hist      = np.zeros((n_t, nc, nn), dtype=float)
    q_hist      = np.zeros((n_t, nc, nn), dtype=float)
    Hg_hist     = np.zeros((n_t, nn),     dtype=float)
    Ts_hist     = np.zeros((n_t, nn),     dtype=float)
    Tg_hist     = np.zeros((n_t, nn),     dtype=float)
    P_hist      = np.zeros((n_t, nn),     dtype=float)
    y_hist_3d   = np.zeros((n_t, nc, nn), dtype=float)
    v_hist      = np.zeros((n_t, nn),     dtype=float)
    v_in_hist   = np.zeros(n_t,           dtype=float)
    v_out_hist  = np.zeros(n_t,           dtype=float)
    C_in_hist   = np.zeros((n_t, nc),     dtype=float)   # ghost cell inlet [mol/m³]
    T_in_hist   = np.full(n_t, np.nan,    dtype=float)   # NaN cuando no hay entrada
    P_in_hist   = np.full(n_t, np.nan,    dtype=float)   # [bar] presión upstream inlet (NaN si sin entrada)
    Tw_hist     = np.zeros((n_t, nn),     dtype=float) if shell_tube else None

    # Estimación inicial de Newton (se actualiza con el valor del instante anterior)
    Tg_prev = np.full(nn, 300.0, dtype=float)

    # ── Bucle temporal ────────────────────────────────────────────────────────
    for k in range(n_t):
        sv_k = y_hist[k]

        # 1. Desempaquetar estado completo → C, q, Hg, Ts, Tg, P, y [, Tw]
        state = unpack_state_vector(
            sv         = sv_k,
            n_comp     = nc,
            N          = nn,
            prop_gas   = prop_gas,
            epsi       = epsi,
            Tg_guess   = Tg_prev,
            gas_T_ref  = gas_T_ref,
            shell_tube = shell_tube,
        )

        C_hist[k]    = state["C"]    # (nc, N)
        q_hist[k]    = state["q"]    # (nc, N)
        Hg_hist[k]   = state["Hg"]   # (N,) [J/m³ column]
        Ts_hist[k]   = state["Ts"]   # (N,)
        Tg_hist[k]   = state["Tg"]   # (N,)
        P_hist[k]    = state["P"]    # (N,) [bar]
        y_hist_3d[k] = state["y"]    # (nc, N)
        if shell_tube:
            Tw_hist[k] = state["Tw"]  # (N,)

        Tg_prev = state["Tg"].copy()   # warm-start para el siguiente instante

        # 2. Propiedades de mezcla
        x_Nnc  = state["y"].T                                       # (N, nc)
        mu_sp  = _species_viscosities(prop_gas, state["Tg"], nc)    # (nc, N)
        mu_mix = wilke_mix_property(x_Nnc, mu_sp.T, MW)             # (N,) [Pa·s]
        MW_mix = mixture_molar_mass(x_Nnc, MW)                      # (N,) [kg/mol]
        rho_mix = ideal_gas_density(state["P"] * 1.0e5, MW_mix, state["Tg"])  # (N,) [kg/m³]

        # 3. Velocidades de contorno exactas (mismo cálculo que core_rhs)
        bc    = get_step_boundary(
            t        = float(t_arr[k]),
            P_cell   = state["P"],
            Tg_cell  = state["Tg"],
            y_cell   = state["y"],
            step     = step,
            bc_config= bc_config,
            n_comp   = nc,
            MW       = MW,
            epsi     = epsi,
            Ai       = Ai,
        )
        v_in  = float(bc["inlet"]["v_m_s"])
        v_out = float(bc["outlet"]["v_m_s"])

        v_in_hist[k]  = v_in
        v_out_hist[k] = v_out

        # Concentración, temperatura y presión de la celda fantasma de entrada
        bc_C_in = bc["inlet"]["C_mol_m3"]
        bc_T_in = bc["inlet"]["T_K"]
        bc_P_in = bc["inlet"]["P_bar"]
        if bc_C_in is not None:
            C_in_hist[k] = np.asarray(bc_C_in, dtype=float).reshape(-1)
        if bc_T_in is not None:
            T_in_hist[k] = float(bc_T_in)
        if bc_P_in is not None:
            P_in_hist[k] = float(bc_P_in)

        # 4. Ergun → velocidad en caras (N+1,) → centros de celda (N,)
        v_face_k = ergun_face_velocity(
            P     = state["P"] * 1.0e5,
            rho_g = rho_mix,
            mu_g  = mu_mix,
            epsi  = epsi,
            dp    = dp,
            dz    = dz,
            v_in  = v_in,
            v_out = v_out,
        )
        v_hist[k] = 0.5 * (v_face_k[:-1] + v_face_k[1:])

    # ── Construir objeto de resultados ────────────────────────────────────────
    ns = types.SimpleNamespace(
        _t_results   = t_arr,
        _z           = z_arr,
        _species     = species,
        _prop_gas    = prop_gas,
        _prop_lecho  = prop_lecho,
        # Campos principales (n_t, N) / (n_t, nc, N)
        _P_results   = P_hist,
        _Tg_results  = Tg_hist,
        _Ts_results  = Ts_hist,
        _Hg_results  = Hg_hist,
        _y_results   = y_hist_3d,
        _C_results   = C_hist,
        _q_results   = q_hist,
        _v_results   = v_hist,
        # Contornos exactos en caras (n_t,) / (n_t, nc)
        _v_in_results  = v_in_hist,
        _v_out_results = v_out_hist,
        _C_in_results  = C_in_hist,    # [mol/m³] ghost cell inlet (0 si sin entrada)
        _T_in_results  = T_in_hist,    # [K] temp. ghost cell inlet (NaN si sin entrada)
        _P_in_results  = P_in_hist,    # [bar] presión upstream inlet (NaN si sin entrada)
    )
    if shell_tube:
        ns._Tw_results = Tw_hist       # (n_t, N) [K] — temperatura de la pared
    return ns
