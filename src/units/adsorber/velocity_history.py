"""
@module units.adsorber.velocity_history
@brief Reconstrucción exacta del campo de velocidad superficial del gas a partir
       de la historia de estado devuelta por run_step.

@details
La velocidad superficial no es una variable de estado primaria del vector ODE.
Durante la integración se calcula en cada evaluación del RHS a partir del campo
de presión (Ergun) y las condiciones de contorno del paso activo, pero no queda
almacenada en y_hist.

Este módulo implementa un pase de postproceso que recorre los instantes de tiempo
guardados en y_hist y reproduce exactamente el mismo cálculo de velocidad que el
núcleo del RHS:

    1. Desempaqueta el estado (sv → C, q, Hg, Ts) y recupera Tg desde Hg
       mediante `unpack_state_vector`.
    2. Calcula μ_mix y ρ_mix en cada celda con Wilke y gas ideal.
    3. Obtiene v_in y v_out exactos llamando a `get_step_boundary` con el estado
       actual (P, Tg, y) del instante k — la misma función que usa core_rhs.
    4. Llama a `ergun_face_velocity` con esos v_in/v_out para obtener v_face (N+1,).
    5. Promedia caras adyacentes para obtener v en centros de celda (N,).

El resultado es idéntico al que el RHS habría producido si se hubiera guardado
v_cell en cada instante de t_eval.

Función pública:
    compute_velocity_history(y_hist, t_arr, params) → ndarray (n_t, N)

Parámetros de params requeridos:
    n_comp, N, dz, epsi, gas_T_ref
    prop_gas   — propiedades puras del gas (mu, MW, …)
    MW         — masas molares (nc,) [kg/mol]
    prop_lecho — propiedades del lecho (D_p, …)
    bc_config  — condiciones de contorno PSA
    Ai         — área interna de sección transversal [m²]
    step       — nombre del paso PSA activo

Unidades: SI.
"""

from __future__ import annotations

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
                      o list[callable] en modo polynomial
    Tg       : ndarray (N,) — temperatura del gas por celda [K]
    nc       : int — número de especies

    Returns
    -------
    mu_sp : ndarray (nc, N) — viscosidad pura por especie y celda [Pa·s]
    """
    N     = Tg.shape[0]
    mu_sp = np.zeros((nc, N), dtype=float)
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
def compute_velocity_history(
    y_hist: np.ndarray,
    t_arr:  np.ndarray,
    params: Dict[str, Any],
) -> np.ndarray:
    """
    @brief
    Reconstruye la velocidad superficial del gas en centros de celda para todos
    los instantes de tiempo guardados por run_step.

    @details
    El cálculo es exacto: usa las mismas funciones que core_rhs para obtener
    Tg, v_in, v_out y v_face, por lo que el resultado es idéntico al que
    se habría obtenido guardando v_cell en cada evaluación del RHS en t_eval.

    El estado se desempaqueta con `unpack_state_vector` (necesita Newton para
    recuperar Tg desde Hg). La temperatura del instante anterior se usa como
    estimación inicial de Newton para acelerar la convergencia.

    Parameters
    ----------
    y_hist : ndarray, shape (n_t, sv_size)
        Historia del vector de estado devuelta por run_step (orientación física).
    t_arr  : ndarray, shape (n_t,)
        Instantes de tiempo correspondientes [s].
    params : dict
        Mismo dict de parámetros que se pasó a run_step. Debe contener al menos:
        n_comp, N, dz, epsi, gas_T_ref, prop_gas, MW, prop_lecho, bc_config,
        Ai, step.

    Returns
    -------
    v_cell_hist : ndarray, shape (n_t, N) [m/s]
        Velocidad superficial del gas en centros de celda para cada instante.
        Valores positivos = flujo en dirección +z (inlet→outlet en forward).

    Notes
    -----
    - El coste es O(n_t · N · nc²) por las iteraciones de Newton y Wilke.
      Para grandes n_t se recomienda guardar v_face directamente en el runner.
    - `params["step"]` debe corresponder al paso que generó y_hist. Si se
      concatenan historiales de varios pasos, llamar a esta función por separado
      para cada tramo y concatenar los resultados.
    """
    # ── Leer parámetros del modelo ────────────────────────────────────────────
    nc        = int(params["n_comp"])
    nn        = int(params["N"])
    dz        = float(params["dz"])
    epsi      = float(params["epsi"])
    gas_T_ref = float(params["gas_T_ref"])
    prop_gas  = params["prop_gas"]
    MW        = np.asarray(params["MW"], dtype=float)
    bc_config = params["bc_config"]
    Ai        = float(params["Ai"])
    step      = str(params["step"]).strip().lower()

    # Diámetro de partícula: usar la media si el campo es heterogéneo (N,)
    D_p_raw = np.asarray(params["prop_lecho"]["D_p"], dtype=float).reshape(-1)
    dp      = float(np.mean(D_p_raw))

    # ── Iterar sobre instantes de tiempo ─────────────────────────────────────
    n_t          = y_hist.shape[0]
    v_cell_hist  = np.zeros((n_t, nn), dtype=float)
    Tg_prev      = np.full(nn, 300.0, dtype=float)   # estimación inicial Newton

    for k in range(n_t):
        sv_k = y_hist[k, :]

        # 1. Desempaquetar estado → C, q, Hg, Ts, Tg, P, y
        state = unpack_state_vector(
            sv         = sv_k,
            n_comp     = nc,
            N          = nn,
            prop_gas   = prop_gas,
            epsi       = epsi,
            Tg_guess   = Tg_prev,
            gas_T_ref  = gas_T_ref,
        )
        Tg_k = state["Tg"]   # (N,)  [K]
        P_k  = state["P"]    # (N,)  [bar]
        y_k  = state["y"]    # (nc, N)

        Tg_prev = Tg_k.copy()   # actualizar estimación para el siguiente instante

        # 2. Propiedades de mezcla en cada celda
        x_Nnc   = y_k.T                                             # (N, nc)
        mu_sp   = _species_viscosities(prop_gas, Tg_k, nc)          # (nc, N)
        mu_Nnc  = mu_sp.T                                           # (N, nc)

        mu_mix  = wilke_mix_property(x_Nnc, mu_Nnc, MW)             # (N,) [Pa·s]
        MW_mix  = mixture_molar_mass(x_Nnc, MW)                     # (N,) [kg/mol]
        rho_mix = ideal_gas_density(P_k * 1.0e5, MW_mix, Tg_k)      # (N,) [kg/m³]

        # 3. Velocidades de contorno exactas (mismo cálculo que core_rhs)
        bc    = get_step_boundary(
            t        = float(t_arr[k]),
            P_cell   = P_k,
            Tg_cell  = Tg_k,
            y_cell   = y_k,
            step     = step,
            bc_config= bc_config,
            n_comp   = nc,
            MW       = MW,
            epsi     = epsi,
            Ai       = Ai,
        )
        v_in  = float(bc["inlet"]["v_m_s"])
        v_out = float(bc["outlet"]["v_m_s"])

        # 4. Ergun → v en caras (N+1,)
        v_face_k = ergun_face_velocity(
            P     = P_k * 1.0e5,
            rho_g = rho_mix,
            mu_g  = mu_mix,
            epsi  = epsi,
            dp    = dp,
            dz    = dz,
            v_in  = v_in,
            v_out = v_out,
        )

        # 5. Media aritmética de caras adyacentes → v en centros de celda (N,)
        v_cell_hist[k, :] = 0.5 * (v_face_k[:-1] + v_face_k[1:])

    return v_cell_hist
