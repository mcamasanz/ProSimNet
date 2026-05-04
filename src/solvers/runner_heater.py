"""
@module runner_heater
@brief Runner de simulación para el calentador de gas 1D.

@details
Este runner es el punto de entrada para la simulación del heater. Sus responsabilidades:

1. **Validación de params**: comprobar que el dict contiene todas las claves
   necesarias para la física del heater antes de lanzar el integrador.

2. **Integración ODE**: invocar `solve_ivp` (BDF) u `odeint` (LSODA) con el RHS
   del heater (`core_rhs` de `rhs_heater`).

3. **Post-proceso automático**: llamar a `build_heater_results()` para producir
   el objeto de resultados con todos los atributos estandarizados.

El heater no tiene ciclo de pasos (ni ads, purge, blowdown…): opera en un único
régimen continuo. La función `run_step` mantiene el mismo nombre que el runner
del adsorbedor para homogeneidad de interfaz entre equipos.

Separación respecto a otros runners:
    runner_adsorption.py → rhs_adsorption.py  (LDF + Ergun + energía gas/sólido)
    runner_heater.py     → rhs_heater.py      (DW + convección + energía gas)

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

from src.solvers.rhs.rhs_heater import core_rhs
from src.units.heater.state_extraction import build_heater_results
from src.utils.profiling import profiled


# ═══════════════════════════════════════════════════════════════════════════════
# Claves requeridas en params
# ═══════════════════════════════════════════════════════════════════════════════

_REQUIRED_HEATER = (
    "n_comp",            # int — número de especies
    "N",                 # int — número de celdas
    "dz",                # float — longitud de celda [m]
    "Ai",                # float — área interna sección transversal [m²]
    "Di",                # float — diámetro interno [m]
    "Pi",                # float — perímetro interno [m]
    "Po",                # float — perímetro externo [m]
    "prop_gas",          # dict — propiedades puras del gas
    "gas_T_ref",         # float — temperatura de referencia entálpica [K]
    "bc_config",         # dict — contornos (de build_boundary_c_config del heater)
    "trans_config",      # dict — transporte (de build_transport_config del heater)
    "thermal_bc_config", # dict — condición térmica de pared
    "energy",            # bool — activar balance de energía
)


def _validate_heater_params(params: dict) -> None:
    """
    @brief
    Comprueba que params contiene todas las claves necesarias para el heater.

    Raises
    ------
    KeyError   si alguna clave obligatoria falta.
    ValueError si n_comp, N o dz tienen valores inválidos.
    """
    missing = [k for k in _REQUIRED_HEATER if k not in params]
    if missing:
        raise KeyError(
            f"runner_heater: faltan claves en params: {missing}\n"
            f"  Construir con build_gas_prop_config, build_boundary_c_config, "
            f"build_transport_config y build_thermal_bc_config del heater."
        )

    nc  = int(params["n_comp"])
    nn  = int(params["N"])
    dz  = float(params["dz"])

    if nc < 1:
        raise ValueError(f"params['n_comp'] debe ser >= 1, got {nc}")
    if nn < 1:
        raise ValueError(f"params['N'] debe ser >= 1, got {nn}")
    if dz <= 0.0:
        raise ValueError(f"params['dz'] debe ser > 0, got {dz}")

    for key in ("Di", "Ai", "Pi", "Po"):
        val = float(params[key])
        if val <= 0.0:
            raise ValueError(f"params['{key}'] debe ser > 0, got {val}")

    bc = params["bc_config"]
    for key in ("v_in", "T_in", "y_in", "P_out"):
        if key not in bc:
            raise KeyError(
                f"params['bc_config'] debe contener '{key}'. "
                f"Construir con build_boundary_c_config del heater."
            )

    tc = params["trans_config"]
    if "mode" not in tc or "h_wall_fixed" not in tc:
        raise KeyError(
            "params['trans_config'] incompleto. "
            "Construir con build_transport_config del heater."
        )

    if "mode" not in params["thermal_bc_config"]:
        raise KeyError(
            "params['thermal_bc_config'] incompleto. "
            "Construir con build_thermal_bc_config."
        )

    # Validación de wall_config cuando shell-tube está activo
    wall_config = params.get("wall_config")
    if wall_config is not None:
        for wkey in ("material", "A_w", "Di", "Do", "T_w_init"):
            if wkey not in wall_config:
                raise KeyError(
                    f"params['wall_config'] debe contener '{wkey}'. "
                    f"Construir con build_wall_config del heater."
                )
        tbc_mode = params["thermal_bc_config"]["mode"]
        if tbc_mode == "fixed_twall":
            raise ValueError(
                "thermal_bc_config mode='fixed_twall' es incompatible con wall_config "
                "(shell_tube=True). Usar 'adiabatic', 'heatfluxwall' o 'ambient_htc'."
            )

    # Advertencia: prop_update_mode="frozen" con gas properties polinómicas es
    # físicamente inconsistente en un heater (grandes variaciones de temperatura).
    prop_mode   = str(params.get("prop_update_mode", "always")).strip().lower()
    gas_mu_mode = params["prop_gas"].get("mu")
    if prop_mode == "frozen" and not isinstance(gas_mu_mode, np.ndarray):
        import warnings
        warnings.warn(
            "runner_heater: prop_update_mode='frozen' con prop_gas modo 'polynomial' "
            "congela las propiedades del gas en la primera evaluación. "
            "En un heater la temperatura varía significativamente; "
            "considerar prop_update_mode='always' para mayor exactitud.",
            UserWarning,
            stacklevel=3,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Función pública
# ═══════════════════════════════════════════════════════════════════════════════

@profiled
def run_step(
    sv0: np.ndarray,
    t_max: float,
    params: Dict[str, Any],
    solver: str = "solve_ivp",
    rtol: float = 1.0e-8,
    atol: float = 1.0e-10,
    n_sec: int = 20,
    show_progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    @brief
    Integra un intervalo temporal del heater y devuelve la historia de estados.

    @details
    Valida params, prepara el dominio temporal, reinicia las cachés de propiedades
    congeladas y lanza el integrador ODE. Al finalizar llama a `build_heater_results`
    para producir el objeto de resultados.

    Parameters
    ----------
    sv0          : ndarray — vector de estado inicial, shape ((nc+1)*N,)
    t_max        : float — duración de la integración [s]
    params       : dict  — parámetros del modelo (validados antes de integrar)
    solver       : str   — "solve_ivp" (BDF) o "odeint" (LSODA)
    rtol         : float — tolerancia relativa del integrador
    atol         : float — tolerancia absoluta del integrador
    n_sec        : int   — número de puntos de salida por segundo de simulación
    show_progress: bool  — mostrar barra de progreso (requiere tqdm)

    Returns
    -------
    t_arr  : ndarray, shape (nt,) — instantes de tiempo [s]
    y_hist : ndarray, shape (nt, sv_size) — historia del vector de estado
    heater : SimpleNamespace — objeto con atributos de resultado estandarizados
    """
    # ── Validación ────────────────────────────────────────────────────────────
    _validate_heater_params(params)

    # ── Dominio temporal ──────────────────────────────────────────────────────
    t_max_val = float(t_max)
    if t_max_val <= 0.0:
        raise ValueError("t_max must be > 0")

    n_sec_val = max(int(n_sec), 1)
    t_max_int = int(np.floor(t_max_val))
    nt_int    = t_max_int * n_sec_val + 1
    t_arr     = np.linspace(0.0, t_max_int, nt_int, dtype=float)
    if t_max_int < t_max_val:
        t_arr = np.append(t_arr, t_max_val)

    # ── Caché de propiedades ──────────────────────────────────────────────────
    params.setdefault("_cache", {})
    params["_cache"].pop("gas_props", None)
    params["_cache"].pop("h_wall",    None)
    # "Tg_last" se conserva: es el warm-start de Newton entre pasos consecutivos

    # ── Barra de progreso (opcional) ──────────────────────────────────────────
    _pbar = None
    _t_pbar = [0.0]

    if show_progress:
        try:
            from tqdm.auto import tqdm as _tqdm
            _pbar = _tqdm(
                total=t_max_val,
                desc="heater",
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
    sv0_arr = np.asarray(sv0, dtype=float).copy()

    solver_str = str(solver).strip().lower()

    if solver_str == "solve_ivp":
        def _rhs_ivp(t_val, sv_val):
            _update_pbar(t_val)
            return core_rhs(t_val, sv_val, params)

        sol = solve_ivp(
            fun=_rhs_ivp,
            t_span=(t_arr[0], t_arr[-1]),
            y0=sv0_arr,
            t_eval=t_arr,
            method="BDF",
            rtol=rtol,
            atol=atol,
        )
        _close_pbar()
        if not sol.success:
            raise RuntimeError(
                f"solve_ivp (BDF) falló en heater: {sol.message}"
            )
        y_hist = sol.y.T   # (nt, sv_size)

    elif solver_str == "odeint":
        if not _HAS_ODEINT:
            raise ImportError("scipy.integrate.odeint not available")

        def _rhs_odeint(sv_val, t_val):
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
        raise ValueError(f"solver must be 'solve_ivp' o 'odeint', got '{solver}'")

    # ── Post-proceso ──────────────────────────────────────────────────────────
    heater = build_heater_results(y_hist, t_arr, params)

    return t_arr, y_hist, heater
