"""
@module gasifier.config.initial_c
@brief Build initial conditions for the gasifier ODE.

@details
Computes the packed state vector sv0 from user-friendly physical variables:
  - gas state: (P_bar, Tg, y) → C (ideal gas), Hg (volumetric enthalpy)
  - solid state: rho_biomass, rho_char, rho_moisture, Ts (given directly)

The 0D case corresponds to N=1. The 1D case uses N>1 with spatial profiles
provided as scalars (uniform) or arrays (heterogeneous).
"""

from src.units.gasifier.state import set_state


def build_initial_c_config(
    P_init,           # float or ndarray(N,) [bar]
    Tg_init,          # float or ndarray(N,) [K]
    Ts_init,          # float or ndarray(N,) [K]
    y_init,           # ndarray(nc,) or ndarray(nc, N) [-]
    rho_biomass_init, # float or ndarray(N,) [kg/m³_bed]
    rho_char_init,    # float or ndarray(N,) [kg/m³_bed]
    rho_moisture_init,# float or ndarray(N,) [kg/m³_bed]
    n_comp: int,
    N: int,
    prop_gas: dict,
    epsi_r: float,
    gas_T_ref: float,
    Tw_init=None,     # float or ndarray(N,) [K] — initial wall temp; None if no shell-tube
) -> dict:
    """
    Build and validate the initial condition configuration dict.

    Parameters
    ----------
    P_init           : float or (N,) [bar]  initial gas pressure
    Tg_init          : float or (N,) [K]   initial gas temperature
    Ts_init          : float or (N,) [K]   initial solid temperature
    y_init           : (nc,) or (nc, N)    initial molar fractions
    rho_biomass_init : float or (N,) [kg/m³_bed]  initial biomass bulk density
    rho_char_init    : float or (N,) [kg/m³_bed]  initial char bulk density
    rho_moisture_init: float or (N,) [kg/m³_bed]  initial moisture bulk density
    n_comp           : int — number of gas species
    N                : int — number of cells
    prop_gas         : dict — output of build_gas_prop_config
    epsi_r           : float — bed void fraction [-]
    gas_T_ref        : float — enthalpy reference temperature [K]
    Tw_init          : float or (N,) or None [K] — initial wall temperature;
                       None if no shell-tube model (default)

    Returns
    -------
    dict with keys:
        P_init            ndarray(N,) [bar]
        Tg_init           ndarray(N,) [K]
        Ts_init           ndarray(N,) [K]
        y_init            ndarray(nc, N) [-]
        rho_biomass_init  ndarray(N,) [kg/m³_bed]
        rho_char_init     ndarray(N,) [kg/m³_bed]
        rho_moisture_init ndarray(N,) [kg/m³_bed]
        C_init            ndarray(nc, N) [mol/m³_gas]
        Hg_init           ndarray(N,) [J/m³_bed]
        Tw_init           ndarray(N,) or None [K]
        sv0               ndarray(14*N,) or (15*N,)
    """
    _validate_positive_scalar("epsi_r",    epsi_r,    lo=0.0, hi=1.0)
    _validate_positive_scalar("gas_T_ref", gas_T_ref, lo=0.0)

    state = set_state(
        P_bar        = P_init,
        Tg           = Tg_init,
        Ts           = Ts_init,
        y            = y_init,
        rho_biomass  = rho_biomass_init,
        rho_char     = rho_char_init,
        rho_moisture = rho_moisture_init,
        n_comp       = n_comp,
        N            = N,
        prop_gas     = prop_gas,
        epsi_r       = epsi_r,
        gas_T_ref    = gas_T_ref,
        Tw           = Tw_init,
    )

    return {
        "P_init":            state["P"],
        "Tg_init":           state["Tg"],
        "Ts_init":           state["Ts"],
        "y_init":            state["y"],
        "rho_biomass_init":  state["rho_solid"][0],
        "rho_char_init":     state["rho_solid"][1],
        "rho_moisture_init": state["rho_solid"][2],
        "C_init":            state["C"],
        "Hg_init":           state["Hg"],
        "Tw_init":           state["Tw"],
        "sv0":               state["sv0"],
    }


def _validate_positive_scalar(name: str, val: float,
                               lo: float = 0.0, hi: float = None):
    v = float(val)
    if v <= lo:
        raise ValueError(f"{name} must be > {lo}, got {v}")
    if hi is not None and v >= hi:
        raise ValueError(f"{name} must be < {hi}, got {v}")
