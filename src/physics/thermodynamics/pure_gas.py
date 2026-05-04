"""
@module pure_gas
@brief Propiedades termofísicas puras de especies gaseosas mediante correlaciones polinómicas.

@details
Este módulo expone dos funciones públicas:

  - poly_deltaT_factory  : construye una función f(T) a partir de coeficientes
                           polinómicos en (T - Tref), con clipping bilateral.
  - build_pure_gas_properties : carga y evalúa (o construye callables) para
                                todas las propiedades puras de una lista de
                                especies a partir de gasdb.

El diccionario devuelto por build_pure_gas_properties es el contrato de
interfaz entre la capa de I/O+termodinámica pura y el resto del modelo:
clases Column, mezcla, difusión, etc.

Unidades del sistema de base: SI.
  MW       [kg/mol]
  mu       [Pa·s]
  k        [W/m/K]
  Cp_molar [J/mol/K]
  h_molar  [J/mol]
  sigmaLJ  [Å]  (parámetro de transporte Lennard-Jones)
  epskB    [K]  (parámetro de transporte Lennard-Jones)
  Tref     [K]
  Tmax     [K]
"""

from __future__ import annotations

from typing import Callable, List, Literal, Union

import numpy as np

from src.io.gasdb_reader import read_gasdb
from src.utils.profiling import profiled

# Tipo alias para propiedades evaluadas (constante o callable)
_PropEntry = Union[np.ndarray, List[Callable]]


# =========================================================
# 1. Factory de correlaciones polinómicas en (T - Tref)
# =========================================================

@profiled
def poly_deltaT_factory(
    a: List[float],
    Tref: float,
    Tmax: float,
) -> Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]:
    """
    @brief
    Construye una correlación polinómica f(T) = Σ aᵢ (T − Tref)^i con
    clipping bilateral al intervalo [Tref, Tmax].

    @details
    La evaluación usa el esquema de Horner para reducir el número de
    multiplicaciones. La función producida acepta T escalar o array y
    devuelve el mismo tipo.

    El clipping garantiza que la extrapolación fuera del rango de validez
    no produce valores físicamente absurdos (ej. viscosidad negativa a
    temperaturas muy bajas). Es una salvaguarda numérica, no un modelo
    físico.

    Parameters
    ----------
    a : list[float]
        Coeficientes [a0, a1, ..., an] del polinomio en (T − Tref).
        a0 es el valor de la propiedad en T = Tref.
    Tref : float
        Temperatura de referencia [K]. Debe ser > 0.
    Tmax : float
        Temperatura máxima de validez [K]. Debe satisfacer Tmax >= Tref.

    Returns
    -------
    f : callable
        Función f(T) -> float o np.ndarray según el tipo de T de entrada.
        Evalúa la propiedad en SI.

    Notes
    -----
    - El clipping se aplica antes de calcular dT = T_eff - Tref, por lo que
      la extrapolación produce el valor en el límite del rango, no un
      valor polinómico divergente.
    - La función producida cierra sobre a_arr, Tref, Tmax (sin estado mutable).
    """
    a_arr = np.asarray(a, dtype=float).reshape(-1)
    Tref_val = float(Tref)
    Tmax_val = float(Tmax)

    def f(T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        T_arr = np.asarray(T, dtype=float)
        T_eff = np.clip(T_arr, Tref_val, Tmax_val)
        dT = T_eff - Tref_val

        # Horner: recorre a[::-1] de mayor a menor grado
        result = np.zeros_like(dT, dtype=float)
        for coeff in a_arr[::-1]:
            result = result * dT + coeff

        if np.isscalar(T):
            return float(result)
        return result

    return f


# =========================================================
# 2. Evaluación vectorizada de una propiedad pura
# =========================================================

@profiled
def eval_species_property(
    prop_entry: _PropEntry,
    T: Union[float, np.ndarray],
    n_comp: int,
) -> np.ndarray:
    """
    @brief
    Evalúa una propiedad pura por especie a la temperatura T dada,
    soportando tanto el modo constante (ndarray) como el polinómico
    (lista de callables).

    @details
    Esta función es el punto de entrada único para evaluar mu, k, Cp_molar
    y h_molar en cualquier nodo del dominio, independientemente de si las
    propiedades fueron cargadas como constantes (modo "constant") o como
    funciones (modo "polynomial").

    - Modo constant: prop_entry es np.ndarray shape (nc,). Devuelve un
      array repetido si T es array.
    - Modo polynomial: prop_entry es list[callable] len nc. Evalúa cada
      función en T.

    Parameters
    ----------
    prop_entry : np.ndarray shape (nc,) o list[callable] len nc
        Propiedad pura por especie en modo constante o polinómico.
    T : float o np.ndarray shape (nn,)
        Temperatura de evaluación [K].
    n_comp : int
        Número de componentes (nc). Se usa para validar la longitud.

    Returns
    -------
    result : np.ndarray
        - Si T es escalar: shape (nc,)
        - Si T es array de nn elementos: shape (nn, nc)

    Notes
    -----
    - En modo constant, el array devuelto es una copia; no aliasa el original.
    - En modo polynomial, la evaluación es component-by-component sin
      vectorización cruzada entre especies.
    """
    T_arr = np.asarray(T, dtype=float)
    scalar_T = T_arr.ndim == 0

    if isinstance(prop_entry, np.ndarray):
        # Modo constante
        prop_const = np.asarray(prop_entry, dtype=float).reshape(-1)
        if scalar_T:
            return prop_const.copy()                           # shape (nc,)
        nn = T_arr.size
        return np.tile(prop_const.reshape(1, n_comp), (nn, 1))  # shape (nn, nc)

    # Modo polinómico
    funcs = prop_entry
    if scalar_T:
        result = np.zeros(n_comp, dtype=float)
        for j, fn in enumerate(funcs):
            result[j] = float(fn(float(T_arr)))
        return result                                          # shape (nc,)

    nn = T_arr.size
    result = np.zeros((nn, n_comp), dtype=float)
    for j, fn in enumerate(funcs):
        result[:, j] = np.asarray(fn(T_arr), dtype=float).reshape(-1)
    return result                                              # shape (nn, nc)


# =========================================================
# 3. Construcción del diccionario de propiedades puras
# =========================================================

@profiled
def build_pure_gas_properties(
    species: List[str],
    mode: Literal["constant", "polynomial"] = "constant",
    Temp: float = 298.15,
    db_path: str = "materials/fluids/gasdb.txt",
) -> dict:
    """
    @brief
    Carga las propiedades termofísicas puras de una lista de especies gaseosas
    desde gasdb y las devuelve en un diccionario normalizado.

    @details
    Este es el punto de entrada estándar para obtener las propiedades puras del
    gas que se usarán a lo largo de toda la simulación. El diccionario devuelto
    es el contrato de interfaz entre la capa de datos y los módulos de mezcla,
    difusión, entalpía y la clase Column.

    Dos modos de operación:
    - "constant": evalúa todas las correlaciones a Temp [K] y devuelve arrays
      numpy shape (nc,). Apropiado cuando se asume que las propiedades no varían
      con T durante la simulación (modo gas-fixed).
    - "polynomial": devuelve listas de callables f(T). Cada callable acepta T
      escalar o array y devuelve la propiedad en SI. Apropiado cuando las
      propiedades se reevalúan en cada paso de tiempo (modo gas-polynomial).

    Parameters
    ----------
    species : list[str]
        Lista de fórmulas químicas, ej. ["N2", "CO2"]. Deben existir en gasdb.
    mode : {"constant", "polynomial"}
        Modo de representación de las propiedades dependientes de T.
    Temp : float
        Temperatura de evaluación [K] para modo "constant". Ignorado en
        modo "polynomial".
    db_path : str
        Ruta al archivo gasdb.txt.

    Returns
    -------
    props : dict con las siguientes claves:
        "species"   : list[str]              - fórmulas en el mismo orden que la entrada
        "MW"        : np.ndarray (nc,)       - masa molar [kg/mol]
        "sigmaLJ"   : np.ndarray (nc,)       - sigma Lennard-Jones [Å]
        "epskB"     : np.ndarray (nc,)       - epsilon/kB Lennard-Jones [K]
        "Tref"      : np.ndarray (nc,)       - temperatura de referencia [K]
        "Tmax"      : np.ndarray (nc,)       - temperatura máxima de validez [K]
        "mu"        : ndarray (nc,) o list[callable]  - viscosidad dinámica [Pa·s]
        "k"         : ndarray (nc,) o list[callable]  - conductividad térmica [W/m/K]
        "Cp_molar"  : ndarray (nc,) o list[callable]  - Cp molar [J/mol/K]
        "h_molar"   : ndarray (nc,) o list[callable]  - entalpía molar [J/mol]

    Raises
    ------
    ValueError
        Si species está vacía, mode no es válido, o Temp <= 0 en modo constant.
    KeyError
        Si alguna especie no existe en gasdb.

    Notes
    -----
    - Las propiedades Cp_molar se derivan de cp_mass (J/kg/K) multiplicando
      por MW [kg/mol]: Cp_molar = cp_mass * MW.
    - sigmaLJ y epskB son parámetros de transporte Lennard-Jones usados por
      el módulo de difusión (diffusion.py) para calcular D_ij mediante
      Chapman-Enskog.
    - Tref y Tmax se conservan en el diccionario para que recover_Tg_from_Hg
      (enthalpy.py) pueda definir los límites de la inversión térmica.
    """
    species_list = list(species)
    if not species_list:
        raise ValueError("species must be a non-empty list.")

    mode = str(mode).strip().lower()
    if mode not in {"constant", "polynomial"}:
        raise ValueError("mode must be 'constant' or 'polynomial'.")

    if mode == "constant":
        Temp_val = float(Temp)
        if not np.isfinite(Temp_val) or Temp_val <= 0.0:
            raise ValueError("Temp must be a finite scalar > 0 [K].")

    db = read_gasdb(db_path)
    for sp in species_list:
        if sp not in db:
            raise KeyError(f"Species '{sp}' not found in gasdb at '{db_path}'.")

    nc = len(species_list)
    props: dict = {
        "species": list(species_list),
        "MW":      np.zeros(nc, dtype=float),
        "sigmaLJ": np.zeros(nc, dtype=float),
        "epskB":   np.zeros(nc, dtype=float),
        "Tref":    np.zeros(nc, dtype=float),
        "Tmax":    np.zeros(nc, dtype=float),
        "mu":      None,
        "k":       None,
        "Cp_molar": None,
        "h_molar": None,
    }

    mu_list   = []
    k_list    = []
    Cp_list   = []
    h_list    = []

    for i, sp in enumerate(species_list):
        rec = db[sp]

        MW_val    = float(rec["molar_mass"]["value"])
        Tref_val  = float(rec["reference"]["Tref_K"])
        Tmax_val  = float(rec["limits"]["Tmax_K"])
        sigma_val = float(rec["transport"]["lj"]["sigma_A"])
        epskB_val = float(rec["transport"]["lj"]["epsilon_over_k_K"])

        polys = rec["polynomials"]["properties"]
        a_mu  = polys["mu"]["a0_to_a7"]    # Pa·s
        a_k   = polys["k"]["a0_to_a7"]     # W/m/K
        a_cp  = polys["cp"]["a0_to_a7"]    # J/kg/K  (másico; se convierte a molar)
        a_h   = polys["h"]["a0_to_a7"]     # J/mol

        props["MW"][i]      = MW_val
        props["sigmaLJ"][i] = sigma_val
        props["epskB"][i]   = epskB_val
        props["Tref"][i]    = Tref_val
        props["Tmax"][i]    = Tmax_val

        f_mu      = poly_deltaT_factory(a_mu, Tref_val, Tmax_val)
        f_k       = poly_deltaT_factory(a_k,  Tref_val, Tmax_val)
        f_cp_mass = poly_deltaT_factory(a_cp, Tref_val, Tmax_val)
        f_h       = poly_deltaT_factory(a_h,  Tref_val, Tmax_val)

        # Cp molar = Cp_mass [J/kg/K] * MW [kg/mol]  -> [J/mol/K]
        # Se construye como closure sobre f_cp_mass y MW_val para evitar
        # captura tardía en el bucle.
        def _make_f_cp_molar(f_cp, mw):
            return lambda T: f_cp(T) * mw

        f_cp_molar = _make_f_cp_molar(f_cp_mass, MW_val)

        if mode == "polynomial":
            mu_list.append(f_mu)
            k_list.append(f_k)
            Cp_list.append(f_cp_molar)
            h_list.append(f_h)
        else:
            mu_list.append(float(f_mu(Temp_val)))
            k_list.append(float(f_k(Temp_val)))
            Cp_list.append(float(f_cp_molar(Temp_val)))
            h_list.append(float(f_h(Temp_val)))

    if mode == "polynomial":
        props["mu"]       = mu_list
        props["k"]        = k_list
        props["Cp_molar"] = Cp_list
        props["h_molar"]  = h_list
    else:
        props["mu"]       = np.array(mu_list,  dtype=float)
        props["k"]        = np.array(k_list,   dtype=float)
        props["Cp_molar"] = np.array(Cp_list,  dtype=float)
        props["h_molar"]  = np.array(h_list,   dtype=float)

    return props
