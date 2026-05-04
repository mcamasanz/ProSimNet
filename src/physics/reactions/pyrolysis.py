"""
@module physics.reactions.pyrolysis
@brief Primary pyrolysis kinetics and product distribution.

@details
Reaction:  Biomass → α_char · Char + Σ_i β_i · Gas_i + β_tar · Tar

Rate law (Arrhenius on the local bulk density of dry biomass):

    r_pyr [kg_biomass / m³_bed / s] = rho_biomass * A * exp(-E / (R * Ts))

Hypotheses:
  - Single-step first-order Arrhenius in solid temperature Ts.
  - No intra-particle gradient for heat or mass transfer during pyrolysis.
  - Particle size and solid velocity are not modified by pyrolysis
    (only bulk density of biomass and char change).
  - Product yields (mass fractions of dry biomass) are fixed and come
    from the fuel_config (RAC scheme outputs for the specific feedstock).
  - Enthalpy of pyrolysis is computed from the difference in heating values
    of reactants and products and transferred to the solid phase.

Source: [25] Gronli (1996). Kinetics: Table 1 of Anca-Couce et al. (2021).
Product composition: RAC scheme, Anca-Couce et al. (2017).
"""

import numpy as np

R_GAS = 8.31446261815324   # [J/mol/K]


def pyrolysis_rate(
    rho_biomass: np.ndarray,    # (N,) [kg/m³_bed]
    Ts:          np.ndarray,    # (N,) [K]
    kinetics:    dict,           # {"A": float [1/s], "E": float [J/mol]}
) -> np.ndarray:
    """
    Volumetric pyrolysis rate.

    Parameters
    ----------
    rho_biomass : ndarray (N,)  bulk density of dry biomass [kg/m³_bed]
    Ts          : ndarray (N,)  solid temperature [K]
    kinetics    : dict          {"A": float, "E": float}

    Returns
    -------
    r_pyr : ndarray (N,)  [kg_biomass / m³_bed / s]   ≥ 0

    Notes
    -----
    - Rate clipped at zero when rho_biomass ≤ 0.
    - In a fixed-bed gasifier Ts can reach 500-700°C in the pyrolysis zone.
      The single-step Arrhenius is adequate at the reactor scale; intra-particle
      gradients are neglected as stated in the model assumptions.
    """
    A = float(kinetics["A"])
    E = float(kinetics["E"])
    Ts_safe = np.maximum(np.asarray(Ts, dtype=float), 1.0)
    r = np.asarray(rho_biomass, dtype=float) * A * np.exp(-E / (R_GAS * Ts_safe))
    return np.maximum(r, 0.0)


def pyrolysis_sources(
    r_pyr:   np.ndarray,   # (N,) [kg_biomass/m³_bed/s]
    yields:  dict,          # {"char": f, "CO": f, ..., "tar": f}  mass fractions
    MW_gas:  np.ndarray,   # (nc_gas,) [kg/mol]  molar masses of gas species
    species: list,          # list[str] of nc_gas gas species names
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the volumetric sources for each gas species and for char.

    Parameters
    ----------
    r_pyr   : ndarray (N,)        pyrolysis rate [kg_biomass/m³_bed/s]
    yields  : dict                mass fraction of each product per unit biomass
    MW_gas  : ndarray (nc_gas,)   molar masses [kg/mol]
    species : list[str]           gas species names in state-vector order

    Returns
    -------
    source_gas  : ndarray (nc_gas, N)  [mol / m³_bed / s]
        Molar production rate of each gas species. Positive = production.
    source_char : ndarray (N,)         [kg / m³_bed / s]
        Mass production rate of char. Positive = char accumulation.

    Notes
    -----
    - tar is treated as a gas-phase species (it leaves the solid as a vapour).
    - Yields must sum to 1.0; validated upstream by fuels_reader.
    - Gas production rate for species i = yields[i] * r_pyr / MW_i [mol/m³/s].
    """
    nc     = len(species)
    N_cells = len(r_pyr)
    r      = np.asarray(r_pyr, dtype=float)   # (N,)

    source_gas = np.zeros((nc, N_cells), dtype=float)
    for j, sp in enumerate(species):
        if sp in yields and yields[sp] > 0.0:
            MW_j = float(MW_gas[j])
            source_gas[j] = yields[sp] * r / MW_j    # [mol/m³/s]

    source_char = yields.get("char", 0.0) * r        # [kg/m³/s]

    return source_gas, source_char


def pyrolysis_enthalpy_sink(
    r_pyr:           np.ndarray,  # (N,) [kg_biomass/m³_bed/s]
    dH_pyr_J_per_kg: float,       # [J/kg_biomass]  (positive = endothermic)
) -> np.ndarray:
    """
    Volumetric enthalpy sink in the solid energy balance due to pyrolysis.

    Parameters
    ----------
    r_pyr           : ndarray (N,)  [kg_biomass/m³_bed/s]
    dH_pyr_J_per_kg : float         [J/kg_biomass], positive = endothermic

    Returns
    -------
    Q_pyr : ndarray (N,)  [W/m³_bed]
        Heat consumed by the solid (positive = endothermic). Must be subtracted
        in the solid energy balance.

    Notes
    -----
    The pyrolysis enthalpy is computed externally from heating value differences:
        dH_pyr = LHV_biomass - (y_char * LHV_char + y_tar * LHV_tar + ...)
    and passed here as a scalar. The article (Anca-Couce 2021) computes it from
    the RAC scheme outputs and transfers the result entirely to the solid phase.
    """
    return np.asarray(r_pyr, dtype=float) * float(dH_pyr_J_per_kg)


def compute_pyrolysis_dH(
    heating_values: dict,   # {"biomass": J/kg, "char": J/kg, "tar": J/kg}
    yields:         dict,   # mass fractions per kg_biomass
) -> float:
    """
    Compute the specific enthalpy of pyrolysis from heating value differences.

    dH_pyr = LHV_biomass - Σ_products (yield_i * LHV_i)

    Only char and tar carry combustion enthalpy; permanent gases (CO, CO2, ...)
    are assumed to have their heating values implicitly accounted for in the
    gas-phase enthalpy (h_molar). Only the solid-retained value is relevant here.

    Parameters
    ----------
    heating_values : dict  {"biomass": float, "char": float, "tar": float} [J/kg]
    yields         : dict  mass fractions

    Returns
    -------
    dH_pyr : float  [J/kg_biomass]
        Positive = endothermic (heat consumed by the solid during pyrolysis).

    Notes
    -----
    This is a simplification. A rigorous approach would sum over all product
    enthalpies including gaseous species. Here we follow the article's convention
    of computing enthalpy from heating value differences for the condensed phases.
    """
    LHV_bio  = float(heating_values["biomass"])
    LHV_char = float(heating_values["char"])
    LHV_tar  = float(heating_values["tar"])

    y_char = float(yields.get("char", 0.0))
    y_tar  = float(yields.get("tar",  0.0))

    # Heat released by combustion of products - heat released by combustion of reactant
    # dH_pyr > 0 means pyrolysis is endothermic (absorbs heat from solid)
    dH_pyr = LHV_bio - (y_char * LHV_char + y_tar * LHV_tar)
    return dH_pyr
