"""
@module physics.reactions.drying
@brief Drying kinetics of biomass moisture content.

@details
Reaction:  H2O(l) → H2O(g)

Rate law (Arrhenius on the local bulk density of liquid moisture):

    r_dry [kg_moisture / m³_bed / s] = rho_moisture * A * exp(-E / (R * Ts))

Hypotheses:
  - Single-step first-order Arrhenius in solid temperature Ts.
  - No intra-particle gradient for moisture transport.
  - Particle size and solid velocity not affected by drying
    (only bulk density of moisture changes).
  - Enthalpy of drying transferred entirely to the solid phase
    as endothermic sink: dH_dry = h_vap_water [J/kg_moisture].

Source: [15] Mandl et al. (2010). Kinetic constants: Table 1 of
Anca-Couce et al., Fuel 296 (2021) 120687.
"""

import numpy as np

R_GAS = 8.31446261815324   # [J/mol/K]

# Latent heat of water vaporisation at ~100°C: 2260 kJ/kg
# Used as the enthalpy of drying (transferred to the solid as an endothermic sink).
_H_VAP_WATER = 2.260e6     # [J/kg_moisture]


def drying_rate(
    rho_moisture: np.ndarray,   # (N,) [kg/m³_bed]  bulk density of liquid moisture
    Ts:           np.ndarray,   # (N,) [K]           solid temperature
    kinetics:     dict,          # {"A": float [1/s], "E": float [J/mol]}
) -> np.ndarray:
    """
    Volumetric drying rate.

    Parameters
    ----------
    rho_moisture : ndarray (N,)  bulk density of liquid moisture [kg/m³_bed]
    Ts           : ndarray (N,)  solid temperature [K]
    kinetics     : dict          {"A": float, "E": float}

    Returns
    -------
    r_dry : ndarray (N,)  [kg_moisture / m³_bed / s]   ≥ 0
        Positive = moisture is consumed (drying proceeds).

    Notes
    -----
    - Rate is clipped at zero when rho_moisture ≤ 0 (no moisture left).
    - No rate limiting when Ts < 373 K; the Arrhenius term itself keeps
      the rate negligibly small below the boiling point.
    """
    A = float(kinetics["A"])
    E = float(kinetics["E"])
    Ts_safe = np.maximum(np.asarray(Ts, dtype=float), 1.0)    # avoid division by zero
    r = np.asarray(rho_moisture, dtype=float) * A * np.exp(-E / (R_GAS * Ts_safe))
    r = np.maximum(r, 0.0)
    return r


def drying_enthalpy_sink(r_dry: np.ndarray) -> np.ndarray:
    """
    Volumetric enthalpy sink in the solid energy balance due to drying.

    Parameters
    ----------
    r_dry : ndarray (N,)  drying rate [kg_moisture / m³_bed / s]

    Returns
    -------
    Q_dry : ndarray (N,)  [W/m³_bed]  (negative = endothermic sink in solid)

    Notes
    -----
    Sign convention: the solid energy balance is written as:
        (1-epsi_r)*rho_s*Cp_s * dTs/dt = ... - Q_dry - Q_gs - Q_wall
    so Q_dry > 0 represents heat consumed by the solid.
    """
    return np.asarray(r_dry, dtype=float) * _H_VAP_WATER    # [W/m³_bed]


def drying_gas_source(r_dry: np.ndarray, MW_H2O: float) -> np.ndarray:
    """
    Molar production rate of H2O(g) from drying [mol/m³_bed/s].

    Parameters
    ----------
    r_dry  : ndarray (N,)  [kg/m³_bed/s]
    MW_H2O : float         [kg/mol]  molar mass of water

    Returns
    -------
    source_H2O : ndarray (N,)  [mol / m³_bed / s]  ≥ 0
    """
    return np.asarray(r_dry, dtype=float) / MW_H2O
