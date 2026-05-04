"""
@module physics.reactions.char_conversion
@brief Heterogeneous char conversion reactions via the Shrinking Core Model (SCM).

@details
Three reactions are modelled:

    R_ox  : Char(CHαOβ) + (1 - η/2 + α/4 - β/2) O2  → η CO + (1-η) CO2 + α/2 H2O
    R_CO2 : Char(CHαOβ) + CO2  → 2 CO + (α/2 - β) H2 + β H2O   [Boudouard]
    R_H2O : Char(CHαOβ) + (1-β) H2O → CO + (1 + α/2 - β) H2    [Steam gasification]

Char model — Shrinking Core Model (SCM):
  - Bulk density of the char bed is constant: rho_char_bulk = const.
  - Particle size decreases with conversion: V_p = V_p0 * (1 - X_char)
    → d_p = d_p0 * (rho_char / rho_char0)^(1/3)
  - Specific surface area: A_p = 6 * (1 - epsi_r) / d_p  [m²_surface/m³_bed]

Rate law per reaction (surface kinetics + external mass transfer):
    r_char [kg_char/m³_bed/s] = A_p * MW_char * k_eff * C_gas
    k_eff  [m/s]              = 1 / (1/k_surf + 1/beta_mass_transfer)
    k_surf [m/s]              = A * exp(-E / (R * Ts))
    beta   [m/s]              = mass transfer coefficient (Ranz-Marshall)

CO/CO2 ratio in char oxidation (η):
    η = C1 * exp(-C2 / Ts) / (1 + C1 * exp(-C2 / Ts))
    Source: [30] Anca-Couce et al. (2017)

Stoichiometry:
    All stoichiometric coefficients (ν) are computed from α, β (char composition).
    Units: mol_gas_species / mol_C_reacted → converted to mol/kg_char.

Source: [13] Di Blasi (2004), Table 1 of Anca-Couce et al. (2021).
"""

import numpy as np

from src.physics.transport.diffusion import binary_diffusivity, mixture_diffusivity

R_GAS = 8.31446261815324   # [J/mol/K]

# ─── Stoichiometry (per mol C in char, accounting for CHαOβ composition) ─────

def char_stoichiometry(alpha: float, beta: float) -> dict:
    """
    Compute stoichiometric coefficients for the three char reactions.

    Parameters
    ----------
    alpha : float  mol H / mol C in char
    beta  : float  mol O / mol C in char

    Returns
    -------
    dict with keys "oxidation", "boudouard", "steam".
    Each entry is a dict of {species: ν_i} where ν_i > 0 means production
    and ν_i < 0 means consumption, in mol_i / mol_C.

    Notes
    -----
    Char formula is CH_alpha O_beta. One mole of char = one mol of C-unit.
    """
    return {
        "oxidation": {
            # η determined at runtime from Ts; placeholder η=0.5
            # O2 stoichiometry: -(1 - η/2 + α/4 - β/2), here η is TBD
            # Use η = 0.5 as nominal for bookkeeping; actual rate uses η(Ts)
            "O2":  None,         # computed from η(Ts) in reaction rate function
            "CO":  None,
            "CO2": None,
            "H2O": alpha / 2.0,  # [mol H2O / mol C]
        },
        "boudouard": {
            "CO2": -1.0,                    # consumed
            "CO":  2.0,                     # produced
            "H2":  max(alpha / 2.0 - beta, 0.0),
            "H2O": beta,
        },
        "steam": {
            "H2O": -(1.0 - beta),           # consumed
            "CO":  1.0,                     # produced
            "H2":  1.0 + alpha / 2.0 - beta,
        },
    }


# ─── Mass transfer coefficient (Ranz-Marshall) ───────────────────────────────

def _mass_transfer_coeff(
    rho_g:  np.ndarray,   # (N,) [kg/m³_gas]
    mu_g:   np.ndarray,   # (N,) [Pa·s]
    v_cell: np.ndarray,   # (N,) [m/s] superficial velocity at cell centres
    dp:     np.ndarray,   # (N,) [m]   particle diameter
    D_im:   np.ndarray,   # (N,) [m²/s] effective diffusivity of reactant in mixture
    epsi_r: float,
) -> np.ndarray:
    """
    External mass transfer coefficient β [m/s] via Ranz-Marshall.

    Sh = 2 + 1.1 * Re^0.6 * Sc^(1/3)
    β  = Sh * D_im / d_p

    Parameters
    ----------
    rho_g  : (N,) gas density [kg/m³]
    mu_g   : (N,) dynamic viscosity [Pa·s]
    v_cell : (N,) |superficial velocity| [m/s]
    dp     : (N,) particle diameter [m]
    D_im   : (N,) diffusivity of reactant in gas mixture [m²/s]
    epsi_r : float  bed void fraction

    Returns
    -------
    beta : ndarray (N,) [m/s]
    """
    # Interstitial velocity = v_sup / epsi_r
    v_int = np.abs(v_cell) / max(float(epsi_r), 1.0e-10)
    # Re y Sc clipados a ≥ 0: BDF puede perturbar rho_g o C a valores negativos
    # durante la estimación numérica del Jacobiano; potencias fraccionarias de
    # valores negativos producen NaN en NumPy real.
    Re    = np.maximum(rho_g * v_int * dp / np.maximum(mu_g, 1.0e-15), 0.0)
    Sc    = np.maximum(mu_g / np.maximum(rho_g * D_im, 1.0e-20), 0.0)
    Sh    = 2.0 + 1.1 * Re ** 0.6 * Sc ** (1.0 / 3.0)
    return Sh * D_im / np.maximum(dp, 1.0e-12)


# ─── Particle geometry from bulk density ──────────────────────────────────────

def particle_diameter(
    rho_char:  np.ndarray,   # (N,) [kg/m³_bed] current char bulk density
    rho_char0: float,        # [kg/m³_bed] initial char bulk density
    dp0:       float,        # [m] initial particle diameter
) -> np.ndarray:
    """
    Current particle diameter from SCM: V_p ∝ (1 - X_char) → d_p ∝ (ρ_char/ρ_char0)^(1/3).

    Parameters
    ----------
    rho_char  : ndarray (N,) current char bulk density [kg/m³_bed]
    rho_char0 : float        initial char bulk density  [kg/m³_bed]
    dp0       : float        initial particle diameter  [m]

    Returns
    -------
    dp : ndarray (N,)  [m]  current particle diameter (≥ 0)
    """
    ratio = np.maximum(np.asarray(rho_char, dtype=float) / max(rho_char0, 1.0e-12), 0.0)
    return dp0 * ratio ** (1.0 / 3.0)


def specific_surface_area(dp: np.ndarray, epsi_r: float) -> np.ndarray:
    """
    Specific surface area of spherical particles in a packed bed.

    A_p = 6 * (1 - epsi_r) / d_p  [m²_surface / m³_bed]

    Parameters
    ----------
    dp     : ndarray (N,)  particle diameter [m]
    epsi_r : float         bed void fraction [-]

    Returns
    -------
    Ap : ndarray (N,)  [m²/m³_bed]
    """
    return 6.0 * (1.0 - float(epsi_r)) / np.maximum(dp, 1.0e-12)


# ─── CO/CO2 product ratio in char oxidation ──────────────────────────────────

def eta_co_co2(Ts: np.ndarray, co_co2_ratio: dict) -> np.ndarray:
    """
    Molar fraction of CO in the {CO, CO2} product mix from char oxidation.

    η = C1 * exp(-C2/Ts) / (1 + C1 * exp(-C2/Ts))

    Parameters
    ----------
    Ts           : ndarray (N,)  solid temperature [K]
    co_co2_ratio : dict          {"C1": float, "C2": float}

    Returns
    -------
    eta : ndarray (N,)  [-]  0 ≤ η ≤ 1
    """
    C1 = float(co_co2_ratio["C1"])
    C2 = float(co_co2_ratio["C2"])
    Ts_safe = np.maximum(np.asarray(Ts, dtype=float), 1.0)
    x = C1 * np.exp(-C2 / Ts_safe)
    return x / (1.0 + x)


# ─── Char reaction rates ──────────────────────────────────────────────────────

def char_het_rates(
    rho_char:    np.ndarray,    # (N,) [kg/m³_bed]
    C_gas:       np.ndarray,    # (nc_gas, N) [mol/m³_gas]
    Ts:          np.ndarray,    # (N,) [K]
    v_cell:      np.ndarray,    # (N,) [m/s]
    rho_g:       np.ndarray,    # (N,) [kg/m³_gas]
    mu_g:        np.ndarray,    # (N,) [Pa·s]
    Tg:          np.ndarray,    # (N,) [K]
    P_Pa:        np.ndarray,    # (N,) [Pa]
    prop_gas:    dict,
    fuel_config: dict,
    params_rxn:  dict,          # from params: kinetics, char_composition, co_co2_ratio
    epsi_r:      float,
    species:     list,          # list[str] gas species names
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute volumetric rates for the three char heterogeneous reactions.

    Parameters
    ----------
    rho_char    : (N,)        current char bulk density [kg/m³_bed]
    C_gas       : (nc_gas, N) gas molar concentrations [mol/m³_gas]
    Ts          : (N,)        solid temperature [K]
    v_cell      : (N,)        superficial gas velocity at cell centres [m/s]
    rho_g       : (N,)        gas density [kg/m³_gas]
    mu_g        : (N,)        gas dynamic viscosity [Pa·s]
    Tg          : (N,)        gas temperature [K]
    P_Pa        : (N,)        total gas pressure [Pa]
    prop_gas    : dict        gas properties (MW, sigmaLJ, epskB, ...)
    fuel_config : dict        output of read_fueldb()
    params_rxn  : dict        {"kinetics": dict, "char_composition": dict,
                               "co_co2_ratio": dict, "dp0": float,
                               "rho_char0": float}
    epsi_r      : float       bed void fraction [-]
    species     : list[str]   gas species names in state-vector order

    Returns
    -------
    r_ox   : ndarray (N,)  char oxidation rate   [kg_char / m³_bed / s]  ≥ 0
    r_CO2  : ndarray (N,)  Boudouard rate         [kg_char / m³_bed / s]  ≥ 0
    r_H2O  : ndarray (N,)  steam gasification rate [kg_char / m³_bed / s]  ≥ 0

    Notes
    -----
    All three rates share the same particle geometry (Ap, dp).
    The external mass transfer coefficient β is reaction-specific because
    each reactant gas has a different diffusivity in the mixture.
    """
    kinetics     = params_rxn["kinetics"]
    char_comp    = params_rxn["char_composition"]
    co_co2       = params_rxn["co_co2_ratio"]
    dp0          = float(params_rxn["dp0"])
    rho_char0    = float(params_rxn["rho_char0"])
    MW_char      = float(char_comp["MW_per_C"])    # [kg/mol per C-atom]

    nc  = len(species)
    MW  = prop_gas["MW"]                            # (nc_gas,) [kg/mol]
    sig = prop_gas["sigmaLJ"]                       # (nc_gas,) [Å]
    eps = prop_gas["epskB"]                         # (nc_gas,) [K]

    # ── Particle geometry ─────────────────────────────────────────────────────
    dp = particle_diameter(rho_char, rho_char0, dp0)
    Ap = specific_surface_area(dp, epsi_r)          # (N,) [m²/m³_bed]

    # ── Binary diffusivity matrix for all species pairs ───────────────────────
    x_mat = (C_gas / np.maximum(np.sum(C_gas, axis=0)[None, :], 1.0e-300)).T  # (N, nc)
    Dij   = binary_diffusivity(T=Tg, P=P_Pa, MW=MW, sigma_A=sig, epskB_K=eps)  # (N,nc,nc)
    Dim   = mixture_diffusivity(x=x_mat, Dij=Dij)                              # (N, nc)

    sp_idx = {sp: i for i, sp in enumerate(species)}

    def _rate(kin_key: str, C_reac: np.ndarray, reac_idx: int) -> np.ndarray:
        """Single-reaction rate [kg_char/m³_bed/s]."""
        A = float(kinetics[kin_key]["A"])
        E = float(kinetics[kin_key]["E"])
        Ts_safe = np.maximum(np.asarray(Ts, dtype=float), 1.0)
        k_surf  = A * np.exp(-E / (R_GAS * Ts_safe))                # (N,) [m/s]
        D_im_r  = Dim[:, reac_idx]                                   # (N,) [m²/s]
        beta    = _mass_transfer_coeff(rho_g, mu_g, v_cell, dp, D_im_r, epsi_r)
        k_eff   = 1.0 / (1.0 / np.maximum(k_surf, 1.0e-20)
                          + 1.0 / np.maximum(beta,   1.0e-20))      # (N,) [m/s]
        C_r     = np.maximum(np.asarray(C_reac, dtype=float), 0.0)  # (N,)
        r       = Ap * MW_char * k_eff * C_r                        # (N,) [kg/m³/s]
        return np.maximum(r, 0.0)

    idx_O2  = sp_idx.get("O2",  4)
    idx_CO2 = sp_idx.get("CO2", 1)
    idx_H2O = sp_idx.get("H2O", 2)

    r_ox  = _rate("char_oxidation",         C_gas[idx_O2],  idx_O2)
    r_CO2 = _rate("char_CO2_gasification",  C_gas[idx_CO2], idx_CO2)
    r_H2O = _rate("char_H2O_gasification",  C_gas[idx_H2O], idx_H2O)

    # Suppress rates when there is effectively no char left
    has_char = np.asarray(rho_char, dtype=float) > 1.0e-10
    r_ox  = r_ox  * has_char
    r_CO2 = r_CO2 * has_char
    r_H2O = r_H2O * has_char

    return r_ox, r_CO2, r_H2O


def char_gas_sources(
    r_ox:    np.ndarray,    # (N,) [kg_char/m³_bed/s]
    r_CO2:   np.ndarray,
    r_H2O:   np.ndarray,
    Ts:      np.ndarray,    # (N,) [K]
    char_comp: dict,        # {"alpha", "beta", "MW_per_C"}
    co_co2_ratio: dict,     # {"C1", "C2"}
    MW_gas:  np.ndarray,    # (nc_gas,) [kg/mol]
    species: list,          # list[str]
) -> np.ndarray:
    """
    Compute the net molar production of each gas species from all char reactions.

    Parameters
    ----------
    r_ox, r_CO2, r_H2O : ndarray (N,)  char consumption rates [kg/m³_bed/s]
    Ts        : ndarray (N,)  solid temperature [K]
    char_comp : dict          char composition (alpha, beta, MW_per_C)
    co_co2_ratio : dict       η parameters
    MW_gas    : ndarray (nc,) molar masses [kg/mol]
    species   : list[str]     gas species names in state-vector order

    Returns
    -------
    source_gas : ndarray (nc_gas, N)  [mol / m³_bed / s]
        Net molar production rate of each gas species from char reactions.
        Positive = net production. Negative = net consumption.

    Notes
    -----
    r [kg_char/m³/s] / MW_char [kg/mol] = r_molar [mol_C/m³/s]
    Each stoichiometric coefficient gives mol_gas / mol_C.
    """
    alpha  = float(char_comp["alpha"])
    beta   = float(char_comp["beta"])
    MW_char = float(char_comp["MW_per_C"])          # [kg/mol per C-atom]

    # Molar rates [mol_C / m³_bed / s]
    r_ox_m  = r_ox  / max(MW_char, 1.0e-12)
    r_CO2_m = r_CO2 / max(MW_char, 1.0e-12)
    r_H2O_m = r_H2O / max(MW_char, 1.0e-12)

    eta = eta_co_co2(Ts, co_co2_ratio)              # (N,) CO/CO2 ratio

    # Stoichiometric coefficients [mol_i / mol_C]:
    #  Oxidation:
    nu_ox = {
        "O2":  -(1.0 - eta / 2.0 + alpha / 4.0 - beta / 2.0),
        "CO":  eta,
        "CO2": (1.0 - eta),
        "H2O": alpha / 2.0,
    }
    #  Boudouard:
    nu_co2 = {
        "CO2": -1.0,
        "CO":  2.0,
        "H2":  max(alpha / 2.0 - beta, 0.0),
        "H2O": beta,
    }
    #  Steam gasification:
    nu_h2o = {
        "H2O": -(1.0 - beta),
        "CO":  1.0,
        "H2":  1.0 + alpha / 2.0 - beta,
    }

    nc     = len(species)
    N_cells = len(r_ox)
    source  = np.zeros((nc, N_cells), dtype=float)

    sp_idx = {sp: j for j, sp in enumerate(species)}

    def _add(nu_dict, r_molar):
        for sp_name, coeff in nu_dict.items():
            if sp_name in sp_idx:
                j = sp_idx[sp_name]
                if callable(coeff) or isinstance(coeff, np.ndarray):
                    source[j] += coeff * r_molar
                else:
                    source[j] += float(coeff) * r_molar

    # Oxidation (η-dependent terms are arrays)
    j_O2  = sp_idx.get("O2",  4)
    j_CO  = sp_idx.get("CO",  0)
    j_CO2 = sp_idx.get("CO2", 1)
    j_H2O = sp_idx.get("H2O", 2)

    source[j_O2]  += nu_ox["O2"]  * r_ox_m        # (N,) — O2 consumed
    source[j_CO]  += eta           * r_ox_m        # (N,) — CO produced
    source[j_CO2] += (1.0 - eta)  * r_ox_m        # (N,)
    source[j_H2O] += alpha / 2.0  * r_ox_m        # (N,) — scalar coeff

    # Boudouard
    source[j_CO2]  += nu_co2["CO2"] * r_CO2_m
    source[j_CO]   += nu_co2["CO"]  * r_CO2_m
    if "H2" in sp_idx:
        source[sp_idx["H2"]] += nu_co2["H2"] * r_CO2_m
    source[j_H2O]  += nu_co2["H2O"] * r_CO2_m

    # Steam gasification
    source[j_H2O] += nu_h2o["H2O"] * r_H2O_m
    source[j_CO]  += nu_h2o["CO"]  * r_H2O_m
    if "H2" in sp_idx:
        source[sp_idx["H2"]] += nu_h2o["H2"] * r_H2O_m

    return source


def char_reaction_heat(
    r_ox:     np.ndarray,    # (N,) [kg_char/m³_bed/s]
    r_CO2:    np.ndarray,
    r_H2O:    np.ndarray,
    Ts:       np.ndarray,    # (N,) [K]
    heating_values: dict,    # {"biomass", "char", "tar"} in [J/kg]
    co_co2_ratio: dict,
    char_comp: dict,
) -> np.ndarray:
    """
    Volumetric heat release from char reactions [W/m³_bed].

    Computed from the difference in heating values of reactants and products,
    consistent with the article's approach. Sign convention: positive = exothermic
    (heat released to the solid).

    Parameters
    ----------
    r_ox, r_CO2, r_H2O : ndarray (N,) [kg_char/m³_bed/s]
    Ts     : ndarray (N,)
    heating_values : dict  [J/kg]
    co_co2_ratio   : dict
    char_comp      : dict

    Returns
    -------
    Q_char : ndarray (N,)  [W/m³_bed]
        Net heat released to the solid. Positive = exothermic (e.g., oxidation).
        Negative = endothermic (e.g., Boudouard, steam gasification).

    Notes
    -----
    ΔH [J/kg_char] for each reaction is approximated from heating value differences.
    Standard values (kJ/mol_C):
        Char oxidation (full): -393.5 kJ/mol  → exothermic
        Boudouard:             +172.5 kJ/mol  → endothermic
        Steam gasification:    +131.3 kJ/mol  → endothermic
    These are converted to J/kg_char using MW_char.
    """
    MW_char = float(char_comp["MW_per_C"])   # [kg/mol per C-atom]
    eta     = eta_co_co2(Ts, co_co2_ratio)   # (N,)

    # Approximate ΔH from heating values (article convention):
    # dH_ox = LHV_char (heat released per kg_char burnt)
    # For Boudouard and steam: taken from standard thermochemistry
    dH_ox_J_per_mol  = 393.5e3 * (1.0 - eta / 2.0)    # J/mol_C (approx, η-weighted)
    dH_CO2_J_per_mol = -172.5e3                          # J/mol_C (endothermic → negative)
    dH_H2O_J_per_mol = -131.3e3                          # J/mol_C (endothermic → negative)

    # Convert to J/kg_char
    dH_ox_kg  = dH_ox_J_per_mol  / MW_char     # [J/kg_char]
    dH_CO2_kg = dH_CO2_J_per_mol / MW_char
    dH_H2O_kg = dH_H2O_J_per_mol / MW_char

    Q_char = (r_ox  * dH_ox_kg
              + r_CO2 * dH_CO2_kg
              + r_H2O * dH_H2O_kg)             # (N,) [W/m³_bed]

    return Q_char
