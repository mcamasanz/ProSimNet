"""
Build gas property configuration for the reactor.

Re-export of build_pure_gas_properties from src/physics/thermodynamics/pure_gas.py.
Unlike the gasifier, the reactor has no hardcoded species list — the user defines
which species to include (e.g. ["N2", "H2", "NH3"] for ammonia synthesis).
"""

from src.physics.thermodynamics.pure_gas import build_pure_gas_properties


def build_gas_prop_config(
    species: list[str],
    mode:    str   = "polynomial",
    Temp:    float = 298.15,
    db_path: str   = "materials/fluids/gasdb.txt",
) -> dict:
    """
    Build the prop_gas dict for the reactor model.

    Parameters
    ----------
    species : list[str]
        Gas species in the desired order (e.g. ["N2", "H2", "NH3"]).
        Must all exist in gasdb.txt.
    mode    : {"polynomial", "constant"}
        "polynomial" — properties as callables f(T)  [recommended]
        "constant"   — properties evaluated at Temp [K]
    Temp    : float
        Reference temperature [K] for "constant" mode.
    db_path : str
        Path to gasdb.txt.

    Returns
    -------
    dict — output of build_pure_gas_properties (species, MW, mu, k, Cp_molar, h_molar, …)
    """
    return build_pure_gas_properties(
        species=species,
        mode=mode,
        Temp=Temp,
        db_path=db_path,
    )
