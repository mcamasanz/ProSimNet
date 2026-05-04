"""
@module io.fuels_reader
@brief Loader and validator for solid fuel definition files (YAML).

@details
Reads a fuel YAML file from materials/fuels/ and returns a validated
fuel_config dict consumed by the gasifier config modules.

Responsibilities:
  - Load and parse the YAML.
  - Validate required fields, types and physical ranges.
  - Convert heating values from MJ/kg to J/kg.
  - Build Cp callables for solid species (polynomial in T [K]).
  - Expose tar properties in a format compatible with build_gas_prop_config.

What this module does NOT do:
  - Build prop_gas dicts (that is the job of units/gasifier/config/gas_props.py).
  - Compute reaction enthalpies (computed dynamically in the reaction modules).
  - Perform any numerical simulation.
"""

import numpy as np
import yaml

# ─── Required top-level sections ─────────────────────────────────────────────
_REQUIRED_SECTIONS = (
    "fuel_id", "physical", "heating_values", "thermal",
    "char_composition", "tar", "pyrolysis_yields", "kinetics", "co_co2_ratio",
)

_REQUIRED_PHYSICAL = ("rho_particle", "dp_initial")

_REQUIRED_HEATING_VALUES = ("biomass", "char", "tar")

_REQUIRED_THERMAL_SOLID = ("Cp_poly_T", "k")

_REQUIRED_CHAR = ("alpha", "beta", "MW_per_C")

_REQUIRED_TAR = ("a", "b", "c", "MW", "Cp_poly_T",
                  "mu_ref", "k_ref", "Tref_transport", "sigmaLJ", "epskB")

_REQUIRED_KINETICS = (
    "drying", "pyrolysis",
    "char_oxidation", "char_CO2_gasification", "char_H2O_gasification",
)

_PYROLYSIS_SPECIES = ("char", "CO", "H2O", "CO2", "H2", "CH4", "C2H4", "tar")


# ─── Public API ───────────────────────────────────────────────────────────────

def read_fueldb(fuel_path: str) -> dict:
    """
    Load a fuel YAML file and return a validated fuel_config dict.

    Parameters
    ----------
    fuel_path : str
        Path to the YAML file (e.g. "materials/fuels/softwood_spruce.yaml").

    Returns
    -------
    dict with keys:
        fuel_id          : str
        description      : str
        source           : str
        physical         : dict  — rho_particle [kg/m³], dp_initial [m]
        heating_values   : dict  — biomass/char/tar [J/kg]
        thermal          : dict  — per species: {Cp_fn: callable(T)->float, k: float}
        char_composition : dict  — alpha [-], beta [-], MW_per_C [kg/mol]
        tar              : dict  — a,b,c,MW,Cp_fn,mu_ref,k_ref,Tref,sigmaLJ,epskB
        pyrolysis_yields : dict  — species → mass fraction [-] (sum = 1)
        kinetics         : dict  — reaction → {A, E}
        co_co2_ratio     : dict  — {model, C1, C2}
    """
    with open(fuel_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    _validate(raw, fuel_path)

    fuel_config = {
        "fuel_id":     str(raw["fuel_id"]),
        "description": str(raw.get("description", "")),
        "source":      str(raw.get("source", "")),
        "physical":    _build_physical(raw["physical"]),
        "heating_values": _build_heating_values(raw["heating_values"]),
        "thermal":     _build_thermal(raw["thermal"]),
        "char_composition": _build_char_composition(raw["char_composition"]),
        "tar":         _build_tar(raw["tar"]),
        "pyrolysis_yields": _build_pyrolysis_yields(raw["pyrolysis_yields"]),
        "kinetics":    _build_kinetics(raw["kinetics"]),
        "co_co2_ratio": _build_co_co2_ratio(raw["co_co2_ratio"]),
    }
    return fuel_config


def build_tar_gas_props(fuel_config: dict, mode: str = "polynomial") -> dict:
    """
    Build a gasdb-compatible property dict for the tar pseudo-component.

    The returned dict matches the format produced by build_pure_gas_properties
    for a single species and can be merged into prop_gas by the gasifier
    config module.

    Parameters
    ----------
    fuel_config : dict
        Output of read_fueldb().
    mode : str
        "polynomial" — Cp, h, mu, k as callables f(T).
        "constant"   — evaluated at Tref=298 K.

    Returns
    -------
    dict with keys matching prop_gas single-species sub-dicts:
        MW        : float [kg/mol]
        sigmaLJ   : float [Å]
        epskB     : float [K]
        Tref      : float [K]
        Tmax      : float [K]
        mu        : callable(T) or float  [Pa·s]
        k         : callable(T) or float  [W/m/K]
        Cp_molar  : callable(T) or float  [J/mol/K]
        h_molar   : callable(T) or float  [J/mol]
    """
    tar   = fuel_config["tar"]
    MW    = tar["MW"]          # [kg/mol]
    T_ref = 298.0              # [K]  standard enthalpy reference
    T_max = 1000.0             # [K]  tar decomposes above this

    Cp_fn = tar["Cp_fn"]       # callable(T) [J/kg/K]

    def Cp_molar_fn(T: np.ndarray) -> np.ndarray:
        return MW * np.asarray(Cp_fn(T), dtype=float)

    def h_molar_fn(T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        h_ref = float(fuel_config["tar"].get("h_ref_J_mol", 0.0))
        # Integrate Cp_molar from T_ref to T (trapezoidal approximation replaced
        # by analytical integration of the polynomial in T).
        coeffs_mass = tar["Cp_poly_T_raw"]     # polynomial coeffs in T basis
        h = np.full_like(T, h_ref, dtype=float)
        for i, a in enumerate(coeffs_mass):
            # integral of a*T^i from T_ref to T = a/(i+1)*(T^(i+1) - Tref^(i+1))
            h += MW * a / (i + 1) * (T ** (i + 1) - T_ref ** (i + 1))
        return h

    T_tr  = float(tar["Tref_transport"])
    mu_r  = float(tar["mu_ref"])
    k_r   = float(tar["k_ref"])

    # Power-law transport: prop ∝ (T/T_tr)^0.65 (organic vapour approximation)
    def mu_fn(T: np.ndarray) -> np.ndarray:
        return mu_r * (np.asarray(T, dtype=float) / T_tr) ** 0.65

    def k_fn(T: np.ndarray) -> np.ndarray:
        return k_r * (np.asarray(T, dtype=float) / T_tr) ** 0.65

    if mode == "constant":
        return {
            "MW":       MW,
            "sigmaLJ":  float(tar["sigmaLJ"]),
            "epskB":    float(tar["epskB"]),
            "Tref":     T_ref,
            "Tmax":     T_max,
            "mu":       float(mu_fn(T_ref)),
            "k":        float(k_fn(T_ref)),
            "Cp_molar": float(Cp_molar_fn(T_ref)),
            "h_molar":  float(h_molar_fn(T_ref)),
        }

    return {
        "MW":       MW,
        "sigmaLJ":  float(tar["sigmaLJ"]),
        "epskB":    float(tar["epskB"]),
        "Tref":     T_ref,
        "Tmax":     T_max,
        "mu":       mu_fn,
        "k":        k_fn,
        "Cp_molar": Cp_molar_fn,
        "h_molar":  h_molar_fn,
    }


# ─── Internal builders ────────────────────────────────────────────────────────

def _make_Cp_fn(coeffs: list, T_clip_min: float = 273.0,
                T_clip_max: float = 1700.0) -> "callable":
    """
    Build a vectorised Cp(T) callable from polynomial coefficients in T basis.

    Parameters
    ----------
    coeffs : list[float]
        [a0, a1, a2, ...] such that Cp(T) = sum(a_i * T^i).
    T_clip_min, T_clip_max : float
        Clipping bounds [K] to avoid polynomial divergence.
    """
    arr = np.asarray(coeffs, dtype=float)

    def Cp_fn(T):
        T_c = np.clip(np.asarray(T, dtype=float), T_clip_min, T_clip_max)
        result = np.zeros_like(T_c)
        for i, a in enumerate(arr):
            result += a * T_c ** i
        return result

    return Cp_fn


def _make_h_fn(coeffs: list, T_clip_min: float = 273.0,
               T_clip_max: float = 1700.0) -> "callable":
    """
    Build H(T) = ∫_{T_clip_min}^T Cp(T') dT' consistent with _make_Cp_fn clipping.

    H(T_clip_min) = 0.  For T in [T_clip_min, T_clip_max]: analytical polynomial
    integral.  Outside range: linear extrapolation with boundary Cp value.

    Parameters
    ----------
    coeffs : list[float]
        Same polynomial coefficients as Cp_fn.
    T_clip_min, T_clip_max : float
        Same clip bounds as Cp_fn.
    """
    arr = np.asarray(coeffs, dtype=float)

    # Polynomial integral: P(T) = Σ a_i/(i+1) · T^(i+1)
    def _poly_int(T_arr):
        res = np.zeros_like(np.asarray(T_arr, dtype=float))
        for i, a in enumerate(arr):
            res += a / (i + 1) * T_arr ** (i + 1)
        return res

    # Cp at clip boundaries (precomputed scalars)
    _Cp_min = float(sum(arr[i] * T_clip_min ** i for i in range(len(arr))))
    _Cp_max = float(sum(arr[i] * T_clip_max ** i for i in range(len(arr))))
    _P_min  = float(_poly_int(np.array([T_clip_min]))[0])
    _P_max  = float(_poly_int(np.array([T_clip_max]))[0])
    _H_max  = _P_max - _P_min   # H(T_clip_max)

    def h_fn(T):
        T_in  = np.asarray(T, dtype=float)
        scalar = T_in.ndim == 0
        T_arr  = np.atleast_1d(T_in)
        result = np.empty_like(T_arr, dtype=float)

        in_rng = (T_arr >= T_clip_min) & (T_arr <= T_clip_max)
        below  = T_arr < T_clip_min
        above  = T_arr > T_clip_max

        if np.any(in_rng):
            result[in_rng] = _poly_int(T_arr[in_rng]) - _P_min
        if np.any(below):
            result[below]  = _Cp_min * (T_arr[below] - T_clip_min)
        if np.any(above):
            result[above]  = _H_max + _Cp_max * (T_arr[above] - T_clip_max)

        return float(result[0]) if scalar else result

    return h_fn


def _build_physical(raw: dict) -> dict:
    rho = float(raw["rho_particle"])
    dp  = float(raw["dp_initial"])
    if rho <= 0.0:
        raise ValueError(f"physical.rho_particle must be > 0, got {rho}")
    if dp <= 0.0:
        raise ValueError(f"physical.dp_initial must be > 0, got {dp}")
    return {"rho_particle": rho, "dp_initial": dp}


def _build_heating_values(raw: dict) -> dict:
    out = {}
    for key in _REQUIRED_HEATING_VALUES:
        val = float(raw[key])
        if val <= 0.0:
            raise ValueError(f"heating_values.{key} must be > 0, got {val}")
        out[key] = val * 1.0e6    # MJ/kg → J/kg
    return out


def _build_thermal(raw: dict) -> dict:
    out = {}
    for solid in ("biomass", "char", "moisture"):
        if solid not in raw:
            raise KeyError(f"thermal.{solid} section missing in fuel file")
        entry = raw[solid]
        for req in _REQUIRED_THERMAL_SOLID:
            if req not in entry:
                raise KeyError(f"thermal.{solid}.{req} missing in fuel file")
        coeffs = [float(c) for c in entry["Cp_poly_T"]]
        k_val  = float(entry["k"])
        if k_val <= 0.0:
            raise ValueError(f"thermal.{solid}.k must be > 0, got {k_val}")
        out[solid] = {
            "Cp_fn": _make_Cp_fn(coeffs),
            "h_fn":  _make_h_fn(coeffs),   # H_j(T) = ∫_{273}^T Cp_j dT
            "k":     k_val,
        }
    return out


def _build_char_composition(raw: dict) -> dict:
    alpha    = float(raw["alpha"])
    beta     = float(raw["beta"])
    MW_per_C = float(raw["MW_per_C"])
    if alpha < 0:
        raise ValueError(f"char_composition.alpha must be >= 0, got {alpha}")
    if beta < 0:
        raise ValueError(f"char_composition.beta must be >= 0, got {beta}")
    if MW_per_C <= 0:
        raise ValueError(f"char_composition.MW_per_C must be > 0, got {MW_per_C}")
    return {
        "alpha":    alpha,
        "beta":     beta,
        "MW_per_C": MW_per_C * 1.0e-3,    # g/mol → kg/mol
    }


def _build_tar(raw: dict) -> dict:
    for req in _REQUIRED_TAR:
        if req not in raw:
            raise KeyError(f"tar.{req} missing in fuel file")
    a  = float(raw["a"])
    b  = float(raw["b"])
    c  = float(raw["c"])
    MW = float(raw["MW"]) * 1.0e-3    # g/mol → kg/mol

    coeffs = [float(x) for x in raw["Cp_poly_T"]]
    return {
        "a":               a,
        "b":               b,
        "c":               c,
        "MW":              MW,
        "Cp_fn":           _make_Cp_fn(coeffs, T_clip_min=300.0, T_clip_max=1200.0),
        "Cp_poly_T_raw":   coeffs,    # kept for analytical integration in h_molar
        "mu_ref":          float(raw["mu_ref"]),
        "k_ref":           float(raw["k_ref"]),
        "Tref_transport":  float(raw["Tref_transport"]),
        "sigmaLJ":         float(raw["sigmaLJ"]),
        "epskB":           float(raw["epskB"]),
        "h_ref_J_mol":     float(raw.get("h_ref_J_mol", 0.0)),
    }


def _build_pyrolysis_yields(raw: dict) -> dict:
    out = {}
    for sp in _PYROLYSIS_SPECIES:
        if sp not in raw:
            raise KeyError(f"pyrolysis_yields.{sp} missing in fuel file")
        val = float(raw[sp])
        if val < 0.0:
            raise ValueError(f"pyrolysis_yields.{sp} must be >= 0, got {val}")
        out[sp] = val
    total = sum(out.values())
    if abs(total - 1.0) > 1.0e-3:
        raise ValueError(
            f"pyrolysis_yields must sum to 1.0, got {total:.6f} — "
            "check the YAML values."
        )
    return out


def _build_kinetics(raw: dict) -> dict:
    out = {}
    for rxn in _REQUIRED_KINETICS:
        if rxn not in raw:
            raise KeyError(f"kinetics.{rxn} section missing in fuel file")
        entry = raw[rxn]
        A = float(entry["A"])
        E = float(entry["E"])
        if A <= 0.0:
            raise ValueError(f"kinetics.{rxn}.A must be > 0, got {A}")
        if E <= 0.0:
            raise ValueError(f"kinetics.{rxn}.E must be > 0, got {E}")
        out[rxn] = {"A": A, "E": E}
    return out


def _build_co_co2_ratio(raw: dict) -> dict:
    model = str(raw.get("model", "anca_couce_2017"))
    C1    = float(raw["C1"])
    C2    = float(raw["C2"])
    if C1 <= 0.0:
        raise ValueError(f"co_co2_ratio.C1 must be > 0, got {C1}")
    if C2 <= 0.0:
        raise ValueError(f"co_co2_ratio.C2 must be > 0, got {C2}")
    return {"model": model, "C1": C1, "C2": C2}


# ─── Validation ──────────────────────────────────────────────────────────────

def _validate(data: dict, path: str) -> None:
    missing = [k for k in _REQUIRED_SECTIONS if k not in data]
    if missing:
        raise KeyError(
            f"fuels_reader: missing top-level sections in '{path}': {missing}"
        )
    for solid in ("biomass", "char", "moisture"):
        if solid not in data["thermal"]:
            raise KeyError(
                f"fuels_reader: thermal.{solid} section missing in '{path}'"
            )
        for req in _REQUIRED_THERMAL_SOLID:
            if req not in data["thermal"][solid]:
                raise KeyError(
                    f"fuels_reader: thermal.{solid}.{req} missing in '{path}'"
                )
    for req in _REQUIRED_PHYSICAL:
        if req not in data["physical"]:
            raise KeyError(
                f"fuels_reader: physical.{req} missing in '{path}'"
            )
    for req in _REQUIRED_TAR:
        if req not in data["tar"]:
            raise KeyError(
                f"fuels_reader: tar.{req} missing in '{path}'"
            )
    for rxn in _REQUIRED_KINETICS:
        if rxn not in data["kinetics"]:
            raise KeyError(
                f"fuels_reader: kinetics.{rxn} missing in '{path}'"
            )
