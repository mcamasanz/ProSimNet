"""
@module solvers.runner_gasifier
@brief Runner de simulación para el gasificador de lecho fijo 1D.

@details
Este runner es el punto de entrada para cualquier simulación del gasificador,
independientemente del modo de operación (batch, CSTR, updraft, conveyor).
Sus responsabilidades son:

1. **Validación de params**: comprobar que el dict contiene todas las claves
   necesarias para la física del gasificador antes de lanzar el integrador.
   Separa claves comunes (malla, gas, geometría) de claves específicas del
   gasificador (combustible, sólido, cinéticas).

2. **Cálculo de dH_pyr**: si no está pre-calculado en params, se inyecta
   la entalpía de pirólisis a partir de los poderes caloríficos y los yields.

3. **Integración ODE**: invocar `solve_ivp` (BDF) u `odeint` (LSODA) con el
   RHS del gasificador (`core_rhs` de `rhs_gasifier`).

4. **Post-proceso automático**: llamar a `build_gasifier_results()` para
   producir el objeto `gasifier` con todos los atributos estandarizados.

Separación respecto a otros runners:
    runner_adsorption.py  → rhs_adsorption.py   (LDF + Ergun + energía gas/sólido)
    runner_heater.py      → rhs_heater.py        (tubo vacío, energía del gas)
    runner_gasifier.py    → rhs_gasifier.py      (reacciones + sólido convectivo)

Claves requeridas en params:
    Comunes (_REQUIRED_COMMON):
        n_comp, N, dz, Ai, Di, Pi, Po
        prop_gas, MW, gas_T_ref
        bc_config, trans_config, thermal_bc_config, energy

    Específicas del gasificador (_REQUIRED_GASIFIER):
        epsi_r          — porosidad del reactor [-]
        dp0             — diámetro inicial de partícula [m]
        rho_char0       — densidad de referencia del char para el SCM [kg/m³_bed]
        fuel_config     — output de read_fueldb()
        solid_config    — output de build_solid_prop_config()
        species         — list[str] de las nc=9 especies gaseosas

    Clave opcional:
        wall_config     — output de build_wall_config(); activa el modelo de pared
                          dinámica Tw(z,t). El vector de estado crece de 14·N a 15·N.

Unidades: SI.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from scipy.integrate import solve_ivp

try:
    from scipy.integrate import odeint as _odeint
    _HAS_ODEINT = True
except ImportError:
    _HAS_ODEINT = False

from src.physics.reactions.pyrolysis import compute_pyrolysis_dH
from src.solvers.rhs.rhs_gasifier import core_rhs
from src.units.gasifier.state_extraction import build_gasifier_results
from src.utils.profiling import profiled


# ═══════════════════════════════════════════════════════════════════════════════
# Claves requeridas en params
# ═══════════════════════════════════════════════════════════════════════════════

_REQUIRED_COMMON = (
    "n_comp",            # int   — número de especies gaseosas
    "N",                 # int   — número de celdas
    "dz",                # float — tamaño de celda [m]
    "Ai",                # float — área interna de sección transversal [m²]
    "Di",                # float — diámetro interno [m]
    "Pi",                # float — perímetro interno [m]
    "Po",                # float — perímetro externo [m]
    "prop_gas",          # dict  — propiedades puras del gas
    "MW",                # ndarray (nc,) — masas molares [kg/mol]
    "gas_T_ref",         # float — temperatura de referencia entálpica [K]
    "bc_config",         # dict  — contornos (output de build_bc_config)
    "trans_config",      # dict  — coeficientes de transporte
    "thermal_bc_config", # dict  — condición térmica de pared
    "energy",            # bool  — activar balance de energía
)

_REQUIRED_GASIFIER = (
    "epsi_r",            # float — porosidad del lecho [-]
    "dp0",               # float — diámetro inicial de partícula [m]
    "rho_char0",         # float — densidad de referencia char (SCM) [kg/m³_bed]
    "fuel_config",       # dict  — output de read_fueldb()
    "solid_config",      # dict  — output de build_solid_prop_config()
    "species",           # list[str] — nc=9 especies gaseosas en orden fijo
)


# ═══════════════════════════════════════════════════════════════════════════════
# Validación de params
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_gasifier_params(params: dict) -> None:
    """
    @brief
    Comprueba que params contiene todas las claves necesarias antes de integrar.

    @details
    Valida por separado las claves comunes y las específicas del gasificador
    para que el mensaje de error indique exactamente qué parte falta.
    También valida rangos físicos de las claves críticas y, si wall_config
    está presente, verifica su estructura e incompatibilidades con thermal_bc.

    Parameters
    ----------
    params : dict — dict de parámetros del modelo

    Raises
    ------
    KeyError   Si alguna clave obligatoria no está presente.
    TypeError  Si fuel_config no es un dict.
    ValueError Si algún parámetro tiene un valor físicamente inválido.
    """
    # ── Claves comunes ────────────────────────────────────────────────────────
    missing_common = [k for k in _REQUIRED_COMMON if k not in params]
    if missing_common:
        raise KeyError(
            f"runner_gasifier: faltan claves comunes en params: {missing_common}\n"
            f"  Requeridas para cualquier equipo 1D de flujo compresible."
        )

    # ── Claves específicas del gasificador ────────────────────────────────────
    missing_gas = [k for k in _REQUIRED_GASIFIER if k not in params]
    if missing_gas:
        raise KeyError(
            f"runner_gasifier: faltan claves del gasificador en params: {missing_gas}\n"
            f"  Construir con read_fueldb() + build_solid_prop_config()."
        )

    # ── Validaciones de tipos y rangos críticos ───────────────────────────────
    if not isinstance(params["fuel_config"], dict):
        raise TypeError(
            "params['fuel_config'] debe ser un dict (output de read_fueldb)"
        )
    nc = int(params["n_comp"])
    sp = params["species"]
    if len(sp) != nc:
        raise ValueError(
            f"len(params['species'])={len(sp)} != n_comp={nc}"
        )
    if not (0.0 < float(params["epsi_r"]) < 1.0):
        raise ValueError(f"epsi_r debe estar en (0, 1), got {params['epsi_r']}")
    if float(params["dp0"]) <= 0.0:
        raise ValueError(f"dp0 debe ser > 0, got {params['dp0']}")
    if float(params["rho_char0"]) <= 0.0:
        raise ValueError(f"rho_char0 debe ser > 0, got {params['rho_char0']}")

    # ── Validación de wall_config (si está presente) ──────────────────────────
    wall_config = params.get("wall_config")
    if wall_config is not None:
        for wkey in ("material", "A_w", "Di", "Do", "T_w_init"):
            if wkey not in wall_config:
                raise KeyError(
                    f"params['wall_config'] debe contener '{wkey}'. "
                    f"Construir con build_wall_config del gasificador."
                )
        # Todos los modos de thermal_bc son compatibles con wall_config.
        # Con shell_tube=True, T_wall en 'fixed_twall' prescribe la temperatura
        # de la pared EXTERIOR (To); la pared interior Tw_i sigue siendo dinámica.


# ═══════════════════════════════════════════════════════════════════════════════
# Función pública
# ═══════════════════════════════════════════════════════════════════════════════

@profiled
def run_step(
    sv0:           np.ndarray,
    t_max:         float,
    params:        Dict[str, Any],
    solver:        str   = "solve_ivp",
    rtol:          float = 1.0e-6,
    atol:          float = 1.0e-8,
    n_sec:         int   = 20,
    max_step:      float = np.inf,
    show_progress: bool  = False,
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    @brief
    Integra un intervalo temporal del modelo del gasificador.

    @details
    Antes de integrar, valida params, inyecta dH_pyr si es necesario y
    reinicia el caché de propiedades. Devuelve la historia de estados y el
    objeto de resultados estandarizado.

    Parameters
    ----------
    sv0           : ndarray (14*N,) o (15*N,) — vector de estado inicial
    t_max         : float                       duración [s]
    params        : dict                        parámetros completos (validados aquí)
    solver        : {"solve_ivp", "odeint"}     integrador temporal
    rtol          : float                       tolerancia relativa
    atol          : float                       tolerancia absoluta
    n_sec         : int                         puntos de salida por segundo
    show_progress : bool                        mostrar barra de progreso (requiere tqdm)

    Returns
    -------
    t_arr    : ndarray (n_t,)          instantes de tiempo [s]
    y_hist   : ndarray (n_t, sv_size)  historia del vector de estado
    gasifier : SimpleNamespace         objeto de resultados estandarizado
    """
    # ── Validación de params ──────────────────────────────────────────────────
    _validate_gasifier_params(params)

    # ── Inyección de dH_pyr ───────────────────────────────────────────────────
    if "dH_pyr" not in params:
        fc = params["fuel_config"]
        params["dH_pyr"] = compute_pyrolysis_dH(
            heating_values=fc["heating_values"],
            yields=fc["pyrolysis_yields"],
        )

    # ── Reinicio del caché al inicio de cada paso ─────────────────────────────
    params.setdefault("_cache", {})
    params["_cache"].pop("gas_props",             None)
    params["_cache"].pop("trans_props",           None)
    params["_cache"].pop("h_wall",                None)   # clave legacy, segura limpiar
    params["_cache"].pop("rho_solid_in_conveyor", None)   # conveyor: recalcular desde t=0
    # "Tg_last" se conserva: warm-start de Newton entre pasos consecutivos

    # ── Dominio temporal ──────────────────────────────────────────────────────
    t_max_val = float(t_max)
    if t_max_val <= 0.0:
        raise ValueError("t_max debe ser > 0")
    n_sec_val = max(int(n_sec), 1)
    t_max_int = int(np.floor(t_max_val))
    nt_int    = t_max_int * n_sec_val + 1
    t_arr     = np.linspace(0.0, t_max_int, nt_int, dtype=float)
    if t_max_int < t_max_val:
        t_arr = np.append(t_arr, t_max_val)

    sv0_arr = np.asarray(sv0, dtype=float)

    # ── Barra de progreso (opcional, requiere tqdm) ───────────────────────────
    _pbar    = None
    _t_pbar  = [0.0]

    if show_progress:
        try:
            from tqdm.auto import tqdm as _tqdm
            _pbar = _tqdm(
                total=t_max_val,
                desc="gasifier",
                unit="s",
                bar_format=(
                    "{l_bar}{bar}| {n:.1f}/{total:.0f} s"
                    " [{elapsed}<{remaining}, {rate_fmt}]"
                ),
                dynamic_ncols=True,
            )
        except ImportError:
            pass

    def _update_pbar(t_val: float) -> None:
        if _pbar is None:
            return
        delta = t_val - _t_pbar[0]
        if delta > 0.0:
            _pbar.update(min(delta, t_max_val - _t_pbar[0]))
            _t_pbar[0] = t_val

    def _close_pbar() -> None:
        if _pbar is not None:
            remaining = t_max_val - _t_pbar[0]
            if remaining > 0.0:
                _pbar.update(remaining)
            _pbar.close()

    # ── Integración ───────────────────────────────────────────────────────────
    def _rhs_ivp(t_val: float, sv_val: np.ndarray) -> np.ndarray:
        _update_pbar(t_val)
        return core_rhs(t_val, sv_val, params)

    solver_str = str(solver).strip().lower()

    if solver_str == "solve_ivp":
        sol = solve_ivp(
            fun=_rhs_ivp,
            t_span=(t_arr[0], t_arr[-1]),
            y0=sv0_arr,
            t_eval=t_arr,
            method="BDF",
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        _close_pbar()
        if not sol.success:
            raise RuntimeError(
                f"runner_gasifier: solve_ivp (BDF) falló — {sol.message}"
            )
        y_hist = sol.y.T    # (n_t, sv_size)

    elif solver_str == "odeint":
        if not _HAS_ODEINT:
            raise ImportError("scipy.integrate.odeint no está disponible")

        def _rhs_odeint(sv_val: np.ndarray, t_val: float) -> np.ndarray:
            _update_pbar(t_val)
            return core_rhs(t_val, sv_val, params)

        from scipy.integrate import odeint
        y_hist = odeint(
            _rhs_odeint, sv0_arr, t_arr,
            rtol=rtol, atol=atol, mxstep=5000,
        )
        _close_pbar()

    else:
        _close_pbar()
        raise ValueError(
            f"runner_gasifier: solver debe ser 'solve_ivp' o 'odeint', "
            f"got '{solver}'"
        )

    # ── Post-proceso automático ───────────────────────────────────────────────
    gasifier = build_gasifier_results(y_hist=y_hist, t_arr=t_arr, params=params)
    return t_arr, y_hist, gasifier
