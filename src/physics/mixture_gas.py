"""
@module mixture_gas
@brief Propiedades termodinámicas y de transporte de la mezcla gaseosa.

@details
Calcula en un solo bloque todas las propiedades de mezcla necesarias para el RHS:
    MWmix  — masa molar de mezcla por celda      [kg/mol]
    rho    — densidad de mezcla por celda         [kg/m³]
    mu     — viscosidad dinámica de mezcla        [Pa·s]
    k      — conductividad térmica de mezcla      [W/m/K]
    Cp_molar — Cp molar de mezcla                [J/mol/K]
    Cp_mass  — Cp másico de mezcla               [J/kg/K]
    Cp_molar_i — Cp molar por especie y celda    [J/mol/K], shape (N, nc)
    h_i    — entalpía molar por especie y celda  [J/mol],   shape (N, nc)
    Dim    — difusión Stefan-Maxwell en mezcla   [m²/s],    shape (N, nc)

Reglas de dimensiones en este módulo:
    x     : fracción molar, shape (N, nc)  — node-first (convención de mezcla)
    y     : fracción molar, shape (nc, N)  — species-first (convención del estado)
    El llamador convierte: x_mat = y_mat.T

Hipótesis:
    - Gas ideal (ρ = P·MW / R·T)
    - Viscosidad de mezcla por regla de Wilke
    - Conductividad de mezcla por regla de Wilke
    - Cp de mezcla: media molar Σ xᵢ Cpᵢ
    - Difusión binaria: Chapman-Enskog con potencial LJ
    - Difusión en mezcla: Stefan-Maxwell (Wilke-Lee simplificado)

Unidades: SI.
"""

from __future__ import annotations

import numpy as np

from src.physics.thermodynamics.pure_gas import eval_species_property
from src.physics.thermodynamics.mixture import (
    wilke_mix_property,
    mixture_molar_mass,
    ideal_gas_density,
)
from src.physics.transport.diffusion import binary_diffusivity, mixture_diffusivity

R_GAS = 8.31446261815324   # [J/mol/K]

from src.utils.profiling import profiled


@profiled
def compute_gas_mixture_properties(
    P_Pa: np.ndarray,
    Tg: np.ndarray,
    x: np.ndarray,
    prop_gas: dict,
    n_comp: int,
    N: int,
) -> dict:
    """
    @brief
    Calcula todas las propiedades de mezcla del gas necesarias para el RHS.

    @details
    Las propiedades puras (mu_i, k_i, Cp_i, h_i) se evalúan usando
    `eval_species_property`, que soporta tanto el modo constant (arrays)
    como polynomial (callables f(T)).

    La entalpía molar de mezcla no se calcula porque no se usa directamente
    en los balances; en su lugar se devuelve h_i(T) por especie y celda.

    Parameters
    ----------
    P_Pa   : np.ndarray, shape (N,) — presión total por celda [Pa]
    Tg     : np.ndarray, shape (N,) — temperatura del gas por celda [K]
    x      : np.ndarray, shape (N, nc) — fracción molar por celda y componente [-]
    prop_gas: dict — propiedades puras del gas (de build_gas_prop_config)
    n_comp : int — número de componentes
    N      : int — número de celdas

    Returns
    -------
    dict con claves:
        MWmix     : ndarray(N,)    [kg/mol]
        rho       : ndarray(N,)    [kg/m³]
        mu        : ndarray(N,)    [Pa·s]
        k         : ndarray(N,)    [W/m/K]
        Cp_molar  : ndarray(N,)    [J/mol/K]
        Cp_mass   : ndarray(N,)    [J/kg/K]
        Cp_molar_i: ndarray(N, nc) [J/mol/K]
        h_i       : ndarray(N, nc) [J/mol]
        Dim       : ndarray(N, nc) [m²/s]
    """
    nc, nn = int(n_comp), int(N)

    P_arr  = np.asarray(P_Pa, dtype=float).reshape(-1)
    Tg_arr = np.asarray(Tg,   dtype=float).reshape(-1)
    x_mat  = np.asarray(x,    dtype=float).reshape(nn, nc)

    MW_arr     = np.asarray(prop_gas["MW"],     dtype=float).reshape(-1)  # (nc,)
    sigma_arr  = np.asarray(prop_gas["sigmaLJ"],dtype=float).reshape(-1)  # (nc,) [Å]
    epskB_arr  = np.asarray(prop_gas["epskB"],  dtype=float).reshape(-1)  # (nc,) [K]

    # --- propiedades puras por celda y especie ---
    # eval_species_property devuelve (nc,) para scalar T o (nn, nc) para array T
    mu_i_mat  = eval_species_property(prop_gas["mu"],       Tg_arr, nc)  # (nn, nc)
    k_i_mat   = eval_species_property(prop_gas["k"],        Tg_arr, nc)  # (nn, nc)
    Cp_i_mat  = eval_species_property(prop_gas["Cp_molar"], Tg_arr, nc)  # (nn, nc)

    # --- entalpía molar por especie ---
    h_entry    = prop_gas.get("h_molar", None)
    Tref_entry = prop_gas.get("Tref", None)

    if h_entry is not None and not isinstance(h_entry, np.ndarray):
        # modo polynomial: h_molar es lista de callables f(T)
        h_i_mat = eval_species_property(h_entry, Tg_arr, nc)             # (nn, nc)
    else:
        # modo constant/fixed: h = h_ref + Cp*(T - Tref)
        h_ref_arr  = np.zeros(nc, dtype=float) if h_entry is None else np.asarray(h_entry, dtype=float).reshape(-1)
        if Tref_entry is not None:
            Tref_arr = np.asarray(Tref_entry, dtype=float).reshape(-1)
        else:
            Tref_arr = np.asarray(prop_gas.get("Tref", np.full(nc, 298.15)), dtype=float).reshape(-1)
        h_i_mat = h_ref_arr[None, :] + Cp_i_mat * (Tg_arr[:, None] - Tref_arr[None, :])  # (nn, nc)

    # --- propiedades de mezcla ---
    MWmix_arr = x_mat @ MW_arr                                            # (nn,) [kg/mol]
    rho_arr   = (P_arr * MWmix_arr) / (R_GAS * Tg_arr)                   # (nn,) [kg/m³]

    mu_mix_arr = wilke_mix_property(x=x_mat, prop_i=mu_i_mat, MW=MW_arr)  # (nn,)
    k_mix_arr  = wilke_mix_property(x=x_mat, prop_i=k_i_mat,  MW=MW_arr)  # (nn,)
    Cp_mix_arr = np.sum(x_mat * Cp_i_mat, axis=1)                         # (nn,) [J/mol/K]
    Cp_mass_arr = Cp_mix_arr / np.maximum(MWmix_arr, 1.0e-30)             # (nn,) [J/kg/K]

    # --- difusión binaria y en mezcla ---
    Dij_mat  = binary_diffusivity(T=Tg_arr, P=P_arr, MW=MW_arr,
                                  sigma_A=sigma_arr, epskB_K=epskB_arr)   # (nn, nc, nc)
    Dim_mat  = mixture_diffusivity(x=x_mat, Dij=Dij_mat)                  # (nn, nc)

    return {
        "MWmix":      MWmix_arr,
        "rho":        rho_arr,
        "mu":         mu_mix_arr,
        "k":          k_mix_arr,
        "Cp_molar":   Cp_mix_arr,
        "Cp_mass":    Cp_mass_arr,
        "Cp_molar_i": Cp_i_mat,
        "h_i":        h_i_mat,
        "Dim":        Dim_mat,
    }
