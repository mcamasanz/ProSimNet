"""
@module gasifier.config.solid_props
@brief Build the solid thermal property config for the gasifier.

@details
Returns a dict consumed by the RHS to compute:
  - solid thermal mass: (1-epsi_r) * Σ_i rho_s_i * Cp_s_i(Ts)
  - solid thermal conductivity: λs(Ts) (weighted or effective)
  - emissivity (for radiative terms if needed in future)

Solid species order (matches state vector):
    0: biomass   1: char   2: moisture
"""

import numpy as np


def build_solid_prop_config(fuel_config: dict) -> dict:
    """
    Build solid thermal property config from a loaded fuel_config dict.

    Parameters
    ----------
    fuel_config : dict
        Output of fuels_reader.read_fueldb().

    Returns
    -------
    solid_config : dict with keys:
        Cp_fns : list[callable]  length 3 — Cp_s_i(Ts) [J/kg/K] per solid species
                                 order: [biomass, char, moisture]
        k_vals : ndarray(3,)    [W/m/K] thermal conductivity per solid species
        species : list[str]     ["biomass", "char", "moisture"]
        emissivity : float      solid emissivity [-] (Table 2 of reference article)
    """
    thermal = fuel_config["thermal"]

    Cp_fns = [
        thermal["biomass"]["Cp_fn"],
        thermal["char"]["Cp_fn"],
        thermal["moisture"]["Cp_fn"],
    ]
    h_fns = [
        thermal["biomass"]["h_fn"],
        thermal["char"]["h_fn"],
        thermal["moisture"]["h_fn"],
    ]
    k_vals = np.array([
        thermal["biomass"]["k"],
        thermal["char"]["k"],
        thermal["moisture"]["k"],
    ], dtype=float)

    return {
        "Cp_fns":     Cp_fns,
        "h_fns":      h_fns,    # H_j(T) = ∫_{273}^T Cp_j dT  (integral exacta de Cp)
        "k_vals":     k_vals,
        "species":    ["biomass", "char", "moisture"],
        "emissivity": 0.9,    # solid emissivity [-]  (Table 2, Anca-Couce 2021)
    }
