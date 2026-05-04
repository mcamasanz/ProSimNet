"""
@module gasifier.config.gas_props
@brief Build the prop_gas dict for the gasifier, including the tar pseudo-component.

@details
The gasifier uses 9 gas species in the following fixed order:

    Index  Species
    ─────  ───────
      0    CO
      1    CO2
      2    H2O
      3    H2
      4    O2
      5    CH4
      6    C2H4
      7    tar      ← defined in the fuel YAML, NOT in gasdb.txt
      8    N2

All 8 permanent species (non-tar) are loaded from gasdb via
build_pure_gas_properties. The tar properties are extracted from
fuel_config (output of fuels_reader.read_fueldb) and inserted at
position 7 to produce the final 9-species prop_gas dict.
"""

import numpy as np

from src.io.fuels_reader import build_tar_gas_props
from src.physics.thermodynamics.pure_gas import build_pure_gas_properties

# Fixed species order for the gasifier (tar always at index 7)
GASIFIER_GAS_SPECIES: list[str] = [
    "CO", "CO2", "H2O", "H2", "O2", "CH4", "C2H4", "tar", "N2",
]
_NON_TAR_SPECIES: list[str] = [
    "CO", "CO2", "H2O", "H2", "O2", "CH4", "C2H4", "N2",
]
_TAR_IDX: int = 7   # insertion position in the 9-species list


def build_gas_prop_config(
    fuel_config: dict,
    mode: str = "polynomial",
    Temp: float = 298.15,
    db_path: str = "materials/fluids/gasdb.txt",
) -> dict:
    """
    Build the prop_gas dict for the gasifier model.

    Parameters
    ----------
    fuel_config : dict
        Output of fuels_reader.read_fueldb().
    mode : {"polynomial", "constant"}
        "polynomial" — all properties as callables f(T) [recommended for gasifier].
        "constant"   — all properties evaluated at Temp [K] (fast, for exploration).
    Temp : float
        Reference temperature [K] for "constant" mode. Ignored for "polynomial".
    db_path : str
        Path to gasdb.txt.

    Returns
    -------
    prop_gas : dict with keys species(9), MW(9,), sigmaLJ(9,), epskB(9,),
               Tref(9,), Tmax(9,), mu, k, Cp_molar, h_molar
               where mu/k/Cp_molar/h_molar are ndarray(9,) or list[callable](9)
               depending on mode.

    Notes
    -----
    - The species list is always GASIFIER_GAS_SPECIES (fixed order).
    - tar properties come from fuel_config, NOT from gasdb.
    - In "polynomial" mode the tar callables clip T to [300, 1000] K
      (tar decomposes above ~800 K in practice, but the cap prevents divergence).
    """
    # 1. Build prop_gas for the 8 permanent species from gasdb
    pg8 = build_pure_gas_properties(
        species=_NON_TAR_SPECIES,
        mode=mode,
        Temp=Temp,
        db_path=db_path,
    )

    # 2. Build tar properties in compatible format
    tar_props = build_tar_gas_props(fuel_config, mode=mode)

    # 3. Assemble the 9-species prop_gas by inserting tar at _TAR_IDX
    species_9 = _NON_TAR_SPECIES[:_TAR_IDX] + ["tar"] + _NON_TAR_SPECIES[_TAR_IDX:]

    prop_gas = {
        "species": species_9,
        "MW":      np.insert(pg8["MW"],      _TAR_IDX, tar_props["MW"]),
        "sigmaLJ": np.insert(pg8["sigmaLJ"], _TAR_IDX, tar_props["sigmaLJ"]),
        "epskB":   np.insert(pg8["epskB"],   _TAR_IDX, tar_props["epskB"]),
        "Tref":    np.insert(pg8["Tref"],    _TAR_IDX, tar_props["Tref"]),
        "Tmax":    np.insert(pg8["Tmax"],    _TAR_IDX, tar_props["Tmax"]),
    }

    if mode == "constant":
        prop_gas["mu"]       = np.insert(pg8["mu"],       _TAR_IDX, tar_props["mu"])
        prop_gas["k"]        = np.insert(pg8["k"],        _TAR_IDX, tar_props["k"])
        prop_gas["Cp_molar"] = np.insert(pg8["Cp_molar"], _TAR_IDX, tar_props["Cp_molar"])
        prop_gas["h_molar"]  = np.insert(pg8["h_molar"],  _TAR_IDX, tar_props["h_molar"])
    else:
        prop_gas["mu"]       = pg8["mu"][:_TAR_IDX]       + [tar_props["mu"]]       + pg8["mu"][_TAR_IDX:]
        prop_gas["k"]        = pg8["k"][:_TAR_IDX]        + [tar_props["k"]]        + pg8["k"][_TAR_IDX:]
        prop_gas["Cp_molar"] = pg8["Cp_molar"][:_TAR_IDX] + [tar_props["Cp_molar"]] + pg8["Cp_molar"][_TAR_IDX:]
        prop_gas["h_molar"]  = pg8["h_molar"][:_TAR_IDX]  + [tar_props["h_molar"]]  + pg8["h_molar"][_TAR_IDX:]

    return prop_gas
