"""
@module solid_props
@brief Propiedades termofísicas de materiales sólidos: densidad, calor específico
       y conductividad térmica.

@details
Módulo análogo a pure_gas.py pero para la fase sólida (paredes de tubo, rellenos
cerámicos, soportes estructurales, etc.).

Tres modos de operación:
  - "fixed"      : el usuario suministra directamente los valores de rho, cp y k
                   como escalares. No se accede a la BD. Útil para prototipos
                   rápidos o parámetros de ajuste manual.
  - "constant"   : lee la BD y devuelve el valor de cada propiedad en Tref (a0),
                   es decir, constante con la temperatura.
  - "polynomial" : lee la BD y devuelve callables f(T) construidos con el
                   polinomio deltaT (mismo esquema que pure_gas.py).

Propiedades disponibles:
    rho  [kg/m³]   — densidad
    cp   [J/kg/K]  — calor específico másico
    k    [W/m/K]   — conductividad térmica

La función pública principal es build_solid_prop_config().

El diccionario devuelto es el contrato de interfaz con el RHS del heater y
cualquier otro módulo que necesite propiedades de la pared.

Unidades: SI.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional, Union

import numpy as np

from src.io.soliddb_reader import read_soliddb
from src.utils.profiling import profiled

# Reutiliza la factory de polinomios de pure_gas para evitar duplicación.
from src.physics.thermodynamics.pure_gas import poly_deltaT_factory


# =========================================================
# 1. Evaluación de una propiedad sólida escalar / callable
# =========================================================

def eval_solid_property(
    prop: Union[float, Callable],
    T: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    @brief
    Evalúa una propiedad sólida (escalar constante o callable) a la temperatura T.

    Parameters
    ----------
    prop : float o callable
        En modo "fixed"/"constant": escalar float.
        En modo "polynomial": callable f(T) -> float o ndarray.
    T : float o ndarray
        Temperatura de evaluación [K].

    Returns
    -------
    float o np.ndarray
        Valor de la propiedad. Si prop es escalar y T es array, devuelve un
        array de igual shape con el valor constante.
    """
    if callable(prop):
        return prop(T)

    # Escalar constante
    val = float(prop)
    T_arr = np.asarray(T, dtype=float)
    if T_arr.ndim == 0:
        return val
    return np.full(T_arr.shape, val, dtype=float)


# =========================================================
# 2. Construcción del diccionario de propiedades del sólido
# =========================================================

@profiled
def build_solid_prop_config(
    material_id: str,
    mode: Literal["fixed", "constant", "polynomial"] = "constant",
    db_path: str = "materials/solids/soliddb.txt",
    rho_fixed: Optional[float] = None,
    cp_fixed: Optional[float] = None,
    k_fixed: Optional[float] = None,
) -> dict:
    """
    @brief
    Construye el diccionario de propiedades termofísicas de un material sólido.

    @details
    Tres modos de operación:

    "fixed"
        Los valores de rho, cp y k se toman directamente de los parámetros
        rho_fixed, cp_fixed y k_fixed. No se accede a la BD. Útil para
        materiales no catalogados o pruebas rápidas.

    "constant"
        Se lee la BD y se devuelven los valores de a0 (valor en Tref) para
        cada propiedad, como escalares float. Las propiedades son constantes
        con la temperatura.

    "polynomial"
        Se lee la BD y se devuelven callables f(T) construidos a partir de
        los coeficientes del polinomio deltaT. Las propiedades varían con T.
        La temperatura se recorta a [Tref, Tmax] antes de evaluar.

    Parameters
    ----------
    material_id : str
        Identificador del material en soliddb (ej. "SS316L", "Al2O3").
        Se ignora en modo "fixed".
    mode : {"fixed", "constant", "polynomial"}
        Modo de representación de las propiedades.
    db_path : str
        Ruta al archivo soliddb.txt. Se ignora en modo "fixed".
    rho_fixed : float, opcional
        Densidad [kg/m³]. Requerido en modo "fixed".
    cp_fixed : float, opcional
        Calor específico [J/kg/K]. Requerido en modo "fixed".
    k_fixed : float, opcional
        Conductividad térmica [W/m/K]. Requerido en modo "fixed".

    Returns
    -------
    config : dict con las claves:
        "material" : str         — id del material (o "fixed")
        "mode"     : str         — modo de operación
        "Tref"     : float | None — temperatura de referencia [K] (None en fixed)
        "Tmax"     : float | None — temperatura máxima de validez [K] (None en fixed)
        "rho"      : float o callable — densidad [kg/m³]
        "cp"       : float o callable — calor específico [J/kg/K]
        "k"        : float o callable — conductividad térmica [W/m/K]

    Raises
    ------
    ValueError
        Si mode no es válido, faltan parámetros en modo "fixed", o el material
        no existe en la BD.
    KeyError
        Si el material no está en soliddb.

    Notes
    -----
    - En modo "constant", rho, cp y k son el valor del polinomio en Tref, es
      decir simplemente a0 de cada propiedad.
    - En modo "polynomial", las propiedades devueltas son callables f(T) que
      aceptan T escalar o np.ndarray y devuelven el mismo tipo.
    - Para evaluar una propiedad en tiempo de simulación, usar eval_solid_property().
    """
    mode_str = str(mode).strip().lower()
    if mode_str not in {"fixed", "constant", "polynomial"}:
        raise ValueError(
            f"mode must be 'fixed', 'constant', or 'polynomial'. Got: {mode!r}"
        )

    # ── Modo "fixed": el usuario provee los valores directamente ────────────
    if mode_str == "fixed":
        if rho_fixed is None or cp_fixed is None or k_fixed is None:
            raise ValueError(
                "mode='fixed' requires rho_fixed, cp_fixed and k_fixed to be provided."
            )
        rho_val = float(rho_fixed)
        cp_val  = float(cp_fixed)
        k_val   = float(k_fixed)
        if not (np.isfinite(rho_val) and rho_val > 0.0):
            raise ValueError(f"rho_fixed must be finite and > 0. Got: {rho_val}")
        if not (np.isfinite(cp_val) and cp_val > 0.0):
            raise ValueError(f"cp_fixed must be finite and > 0. Got: {cp_val}")
        if not (np.isfinite(k_val) and k_val > 0.0):
            raise ValueError(f"k_fixed must be finite and > 0. Got: {k_val}")
        return {
            "material": "fixed",
            "mode":     "fixed",
            "Tref":     None,
            "Tmax":     None,
            "rho":      rho_val,
            "cp":       cp_val,
            "k":        k_val,
        }

    # ── Modos "constant" y "polynomial": leer la BD ─────────────────────────
    db = read_soliddb(db_path)
    mat_id = str(material_id).strip()
    if mat_id not in db:
        raise KeyError(
            f"Material '{mat_id}' not found in soliddb at '{db_path}'. "
            f"Available: {list(db.keys())}"
        )

    rec      = db[mat_id]
    Tref_val = float(rec["reference"]["Tref_K"])
    Tmax_val = float(rec["limits"]["Tmax_K"])
    polys    = rec["polynomials"]["properties"]

    a_rho = polys["rho"]["a0_to_a7"]
    a_cp  = polys["cp"]["a0_to_a7"]
    a_k   = polys["k"]["a0_to_a7"]

    f_rho = poly_deltaT_factory(a_rho, Tref_val, Tmax_val)
    f_cp  = poly_deltaT_factory(a_cp,  Tref_val, Tmax_val)
    f_k   = poly_deltaT_factory(a_k,   Tref_val, Tmax_val)

    if mode_str == "constant":
        return {
            "material": mat_id,
            "mode":     "constant",
            "Tref":     Tref_val,
            "Tmax":     Tmax_val,
            "rho":      float(f_rho(Tref_val)),
            "cp":       float(f_cp(Tref_val)),
            "k":        float(f_k(Tref_val)),
        }

    # mode == "polynomial"
    return {
        "material": mat_id,
        "mode":     "polynomial",
        "Tref":     Tref_val,
        "Tmax":     Tmax_val,
        "rho":      f_rho,
        "cp":       f_cp,
        "k":        f_k,
    }
