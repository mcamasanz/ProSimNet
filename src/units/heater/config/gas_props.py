"""
@module heater.config.gas_props
@brief Validación y carga del bloque de propiedades puras del gas para el heater.

@details
Copia autónoma del builder de propiedades puras del gas, independiente de
cualquier otro equipo. Misma implementación que adsorber.config.gas_props,
mantenida por separado para evitar dependencias cruzadas entre equipos.

Tres modos de operación:
    "fixed"      — el usuario suministra directamente el dict prop_gas con arrays
                   constantes (numpy) para mu, k, Cp_molar, h_molar.
    "constant"   — se lee la base de datos gasdb a T_ref y se construyen arrays
                   constantes evaluados en esa temperatura.
    "polynomial" — se lee la base de datos gasdb y se construyen listas de
                   callables f(T) para mu, k, Cp_molar, h_molar.

La función devuelve un dict prop_gas validado, listo para ser consumido
por los módulos de física (thermodynamics, transport).

Claves mínimas del dict devuelto:
    species, MW, sigmaLJ, epskB, Tref, Tmax,
    mu, k, Cp_molar, h_molar
"""

from __future__ import annotations

import copy
from typing import List, Optional

import numpy as np

from src.physics.thermodynamics.pure_gas import build_pure_gas_properties
from src.utils.profiling import profiled


@profiled
def build_gas_prop_config(
    species: List[str],
    mode: str,
    n_comp: int,
    N: int,
    T_ref: Optional[float] = None,
    db_path: str = "materials/fluids/gasdb.txt",
    prop_gas_fixed: Optional[dict] = None,
) -> dict:
    """
    @brief
    Valida los parámetros de propiedades puras del gas y devuelve un dict validado.

    Parameters
    ----------
    species        : list[str] — fórmulas de las nc especies (ej. ["N2", "CO2"])
    mode           : str — "fixed", "constant" o "polynomial"
    n_comp         : int — número de componentes (debe coincidir con len(species))
    N              : int — número de celdas (para validación de callables en polynomial)
    T_ref          : float, opcional — temperatura de referencia [K] (obligatorio en "constant")
    db_path        : str — ruta a la base de datos gasdb.txt
    prop_gas_fixed : dict, opcional — dict de propiedades en modo "fixed"

    Returns
    -------
    dict prop_gas validado con claves:
        species, MW [kg/mol], sigmaLJ [Å], epskB [K], Tref [K], Tmax [K],
        mu, k, Cp_molar, h_molar  (arrays nc o listas de nc callables)
    """
    nc, nn = int(n_comp), int(N)

    # --- validar species ---
    if not isinstance(species, (list, tuple)):
        raise TypeError("species must be a list or tuple")
    species_list = list(species)
    if len(species_list) != nc:
        raise ValueError(f"species must have length {nc}")
    for i, s in enumerate(species_list):
        if not isinstance(s, str):
            raise TypeError(f"species[{i}] must be a string")
        if len(s.strip()) == 0:
            raise ValueError(f"species[{i}] cannot be empty")

    # --- validar mode ---
    mode_str = str(mode).strip().lower()
    if mode_str not in ("fixed", "constant", "polynomial"):
        raise ValueError("mode must be 'fixed', 'constant' or 'polynomial'")

    # --- validar T_ref ---
    T_ref_val = None if T_ref is None else float(T_ref)
    if mode_str == "constant":
        if T_ref_val is None:
            raise ValueError("T_ref must be provided when mode='constant'")
    if T_ref_val is not None:
        if not np.isfinite(T_ref_val) or T_ref_val <= 0.0:
            raise ValueError("T_ref must be a finite scalar > 0 [K]")

    # --- cargar prop_gas ---
    if mode_str == "fixed":
        if prop_gas_fixed is None:
            raise ValueError("prop_gas_fixed must be provided when mode='fixed'")
        if not isinstance(prop_gas_fixed, dict):
            raise TypeError("prop_gas_fixed must be a dict")
        prop_gas_loaded = copy.deepcopy(prop_gas_fixed)
        # Claves mínimas requeridas en fixed
        for key in ("species", "MW", "mu", "k", "Cp_molar", "sigmaLJ", "epskB"):
            if key not in prop_gas_loaded:
                raise ValueError(f"prop_gas_fixed is missing required key '{key}'")
        # Completar opcionales con defaults
        if "h_molar" not in prop_gas_loaded:
            prop_gas_loaded["h_molar"] = np.zeros(nc, dtype=float)
        if "Tref" not in prop_gas_loaded:
            default_tref = 298.15 if T_ref_val is None else T_ref_val
            prop_gas_loaded["Tref"] = np.full(nc, default_tref, dtype=float)
        if "Tmax" not in prop_gas_loaded:
            prop_gas_loaded["Tmax"] = np.full(nc, 5000.0, dtype=float)

    elif mode_str == "constant":
        prop_gas_loaded = build_pure_gas_properties(
            species=species_list,
            mode="constant",
            Temp=T_ref_val,
            db_path=str(db_path),
        )
    else:  # polynomial
        prop_gas_loaded = build_pure_gas_properties(
            species=species_list,
            mode="polynomial",
            Temp=300.0,   # Temp irrelevante en polynomial — solo se usa para construir callables
            db_path=str(db_path),
        )

    # --- validación estructural del dict cargado ---
    required_keys = ("species", "MW", "mu", "k", "Cp_molar", "h_molar",
                     "sigmaLJ", "epskB", "Tref", "Tmax")
    for key in required_keys:
        if key not in prop_gas_loaded:
            raise ValueError(f"prop_gas is missing required key '{key}'")

    # species debe coincidir
    species_loaded = list(prop_gas_loaded["species"])
    if len(species_loaded) != nc:
        raise ValueError(f"prop_gas['species'] must have length {nc}")
    if species_loaded != species_list:
        raise ValueError("prop_gas['species'] does not match input species list")

    # Arrays LJ y MW
    for key in ("MW", "sigmaLJ", "epskB", "Tref", "Tmax"):
        arr = np.asarray(prop_gas_loaded[key], dtype=float).reshape(-1)
        if arr.shape != (nc,):
            raise ValueError(f"prop_gas['{key}'] must have length {nc}")
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
            raise ValueError(f"prop_gas['{key}'] contains invalid values (must be finite > 0)")
    Tref_arr = np.asarray(prop_gas_loaded["Tref"], dtype=float).reshape(-1)
    Tmax_arr = np.asarray(prop_gas_loaded["Tmax"], dtype=float).reshape(-1)
    if np.any(Tmax_arr < Tref_arr):
        raise ValueError("Each component must satisfy Tmax >= Tref")

    # Propiedades termofísicas: arrays en constant/fixed, callables en polynomial
    if mode_str in ("fixed", "constant"):
        for key in ("mu", "k", "Cp_molar", "h_molar"):
            arr = np.asarray(prop_gas_loaded[key], dtype=float).reshape(-1)
            if arr.shape != (nc,):
                raise ValueError(f"prop_gas['{key}'] must have length {nc}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"prop_gas['{key}'] contains nan or inf")
    else:  # polynomial
        T_test = np.linspace(float(np.min(Tref_arr)), float(np.max(Tmax_arr)), nn, dtype=float)
        for key in ("mu", "k", "Cp_molar", "h_molar"):
            entry = prop_gas_loaded[key]
            if not isinstance(entry, (list, tuple)):
                raise TypeError(f"prop_gas['{key}'] must be a list in polynomial mode")
            if len(entry) != nc:
                raise ValueError(f"prop_gas['{key}'] must have length {nc}")
            for i, fn in enumerate(entry):
                if not callable(fn):
                    raise TypeError(f"prop_gas['{key}'][{i}] must be callable")
                vals = np.asarray(fn(T_test), dtype=float).reshape(-1)
                if vals.shape != (nn,):
                    raise ValueError(f"prop_gas['{key}'][{i}] must return shape ({nn},)")
                if not np.all(np.isfinite(vals)):
                    raise ValueError(f"prop_gas['{key}'][{i}] returns nan or inf")

    return prop_gas_loaded
