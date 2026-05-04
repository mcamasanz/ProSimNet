"""
Runner de simulación para el reactor tubular 1D.

Responsabilidades:
    1. Validar params antes de integrar (claves, rangos, coherencia física).
    2. Calcular a_p del catalizador si no está pre-calculado.
    3. Integrar el ODE con solve_ivp (BDF).
    4. Construir y devolver el objeto de resultados estandarizado.

Claves requeridas en params
----------------------------
Comunes (_REQUIRED_COMMON):
    n_comp, N, dz, Ai, Di, Pi, Po
    prop_gas, MW, gas_T_ref, species
    bc_config, trans_config, thermal_bc_config, energy

Específicas del reactor (_REQUIRED_REACTOR):
    epsi            — fracción de vacíos [-]  (1.0 para tubo vacío)

Opcionales:
    reactions_config — list[dict]; cada dict debe tener:
                       rate_fn  : callable(C, Tg, Ts, P_Pa, params) → (N,)
                       stoich   : dict {species_name: coeff}
                       dH_rxn   : float o callable(T) → float  [J/mol_rxn]
                       type     : "homogeneous" | "heterogeneous"
    catalyst_config  — dict con:
                       dp       : float  [m]  diámetro de partícula
                       rho_bulk : float  [kg/m³_bed]  densidad bulk del catalizador
                       Cp_fn    : callable(T) → float  [J/kg/K]
                       a_p      : float (opcional; se calcula de dp y epsi si falta)
    induction_config — dict con:
                       mode     : "uniform" | "profile"
                       q_vol    : float  [W/m³_bed]  (si mode="uniform")
                       q_fn     : callable(z_cells, t) → (N,)  (si mode="profile")
    wall_config      — dict con: material, A_w, Di, Do, T_w_init
                       (activa ODE de pared dinámica Tw)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from src.solvers.rhs.rhs_reactor import core_rhs
from src.units.reactor.state_extraction import build_reactor_results
from src.utils.profiling import profiled

# ─── Claves obligatorias ──────────────────────────────────────────────────────

_REQUIRED_COMMON = (
    "n_comp", "N", "dz", "Ai", "Di", "Pi", "Po",
    "prop_gas", "MW", "gas_T_ref", "species",
    "bc_config", "trans_config", "thermal_bc_config", "energy",
)

_REQUIRED_REACTOR = (
    "epsi",
)


# ─── Validación ───────────────────────────────────────────────────────────────

def _validate_reactor_params(params: dict) -> None:
    """
    Validate params before integration.

    Raises
    ------
    KeyError   Missing required key.
    TypeError  Wrong type for a key.
    ValueError Physically invalid value.
    """
    missing_common = [k for k in _REQUIRED_COMMON if k not in params]
    if missing_common:
        raise KeyError(
            f"runner_reactor: faltan claves comunes en params: {missing_common}"
        )

    missing_reactor = [k for k in _REQUIRED_REACTOR if k not in params]
    if missing_reactor:
        raise KeyError(
            f"runner_reactor: faltan claves del reactor en params: {missing_reactor}"
        )

    nc = int(params["n_comp"])
    sp = list(params["species"])
    if len(sp) != nc:
        raise ValueError(
            f"len(params['species'])={len(sp)} != n_comp={nc}"
        )

    epsi = float(params["epsi"])
    if not (0.0 < epsi <= 1.0):
        raise ValueError(f"epsi debe estar en (0, 1], got {epsi}")

    # ── catalyst_config (opcional) ────────────────────────────────────────────
    cat = params.get("catalyst_config")
    if cat is not None:
        for key in ("dp", "rho_bulk", "Cp_fn"):
            if key not in cat:
                raise KeyError(
                    f"catalyst_config debe contener '{key}'"
                )
        if float(cat["dp"]) <= 0.0:
            raise ValueError(f"catalyst_config['dp'] debe ser > 0, got {cat['dp']}")
        if float(cat["rho_bulk"]) <= 0.0:
            raise ValueError(
                f"catalyst_config['rho_bulk'] debe ser > 0, got {cat['rho_bulk']}"
            )
        if not callable(cat["Cp_fn"]):
            raise TypeError("catalyst_config['Cp_fn'] debe ser callable")

    # ── reactions_config (opcional) ───────────────────────────────────────────
    rxns = params.get("reactions_config", [])
    has_catalyst = cat is not None
    for i, rxn in enumerate(rxns):
        for key in ("rate_fn", "stoich", "dH_rxn", "type"):
            if key not in rxn:
                raise KeyError(
                    f"reactions_config[{i}] debe contener '{key}'"
                )
        if not callable(rxn["rate_fn"]):
            raise TypeError(f"reactions_config[{i}]['rate_fn'] debe ser callable")
        rxn_type = str(rxn["type"]).lower()
        if rxn_type not in {"homogeneous", "heterogeneous"}:
            raise ValueError(
                f"reactions_config[{i}]['type'] debe ser 'homogeneous' o "
                f"'heterogeneous', got '{rxn['type']}'"
            )
        if rxn_type == "heterogeneous" and not has_catalyst:
            raise ValueError(
                f"reactions_config[{i}] es 'heterogeneous' pero catalyst_config "
                f"es None. Las reacciones heterogéneas requieren catalizador."
            )
        # Verificar que las especies del stoich existen en la lista de especies
        unknown = [s for s in rxn["stoich"] if s not in sp]
        if unknown:
            raise ValueError(
                f"reactions_config[{i}]['stoich'] contiene especies desconocidas: "
                f"{unknown}. Especies disponibles: {sp}"
            )

    # ── induction_config (opcional) ───────────────────────────────────────────
    ind = params.get("induction_config")
    if ind is not None:
        if not has_catalyst:
            raise ValueError(
                "induction_config requiere catalyst_config (q_induction se aplica "
                "sobre el sólido catalítico)"
            )
        mode = str(ind.get("mode", "uniform")).lower()
        if mode == "uniform" and "q_vol" not in ind:
            raise KeyError("induction_config con mode='uniform' requiere 'q_vol'")
        if mode == "profile" and "q_fn" not in ind:
            raise KeyError("induction_config con mode='profile' requiere 'q_fn'")
        if mode == "profile" and not callable(ind["q_fn"]):
            raise TypeError("induction_config['q_fn'] debe ser callable")

    # ── wall_config (opcional) ────────────────────────────────────────────────
    wall = params.get("wall_config")
    if wall is not None:
        for key in ("material", "A_w", "Di", "Do", "T_w_init"):
            if key not in wall:
                raise KeyError(
                    f"wall_config debe contener '{key}'. "
                    f"Construir con build_wall_config()."
                )


def _prepare_catalyst_params(params: dict) -> None:
    """
    Compute derived catalyst params (a_p) if not already in catalyst_config.

    a_p = 6 * (1 - epsi) / dp   [m²/m³_bed]  — partículas esféricas
    """
    cat = params.get("catalyst_config")
    if cat is None:
        return
    if "a_p" not in cat:
        dp   = float(cat["dp"])
        epsi = float(params["epsi"])
        cat["a_p"] = 6.0 * (1.0 - epsi) / dp


# ─── Runner principal ─────────────────────────────────────────────────────────

@profiled
def run_step(
    sv0:           np.ndarray,
    t_max:         float,
    params:        dict[str, Any],
    rtol:          float = 1.0e-6,
    atol:          float = 1.0e-8,
    max_step:      float = np.inf,
    n_sec:         int   = 20,
    show_progress: bool  = False,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """
    Integrate one time interval of the reactor model.

    Parameters
    ----------
    sv0           : ndarray   initial state vector (from set_state or build_sv0_from_results)
    t_max         : float     simulation duration [s]
    params        : dict      complete params dict (validated here)
    rtol          : float     relative tolerance for BDF
    atol          : float     absolute tolerance for BDF
    max_step      : float     maximum time step [s]
    n_sec         : int       output points per second of simulation
    show_progress : bool      show tqdm progress bar (requires tqdm)

    Returns
    -------
    t_arr   : ndarray (n_t,)         time points [s]
    y_hist  : ndarray (n_t, sv_size) state history
    reactor : SimpleNamespace        result object from build_reactor_results
    """
    # ── Validar y preparar ────────────────────────────────────────────────────
    _validate_reactor_params(params)
    _prepare_catalyst_params(params)

    if float(t_max) <= 0.0:
        raise ValueError(f"t_max debe ser > 0, got {t_max}")

    # ── Reinicio del caché al inicio de cada paso ─────────────────────────────
    # Se limpian propiedades cacheadas para que se recalculen con el nuevo sv0.
    # "Tg_last" se conserva: warm-start de Newton entre pasos consecutivos.
    # "_sp_idx" se conserva: es estático, no cambia entre pasos.
    cache = params.setdefault("_cache", {})
    cache.pop("gas_props",  None)
    cache.pop("trans_props", None)

    # ── Dominio temporal ──────────────────────────────────────────────────────
    t_max_f   = float(t_max)
    n_sec_val = max(int(n_sec), 1)
    t_max_int = int(math.floor(t_max_f))
    nt_int    = t_max_int * n_sec_val + 1
    t_arr     = np.linspace(0.0, float(t_max_int), nt_int)
    if t_max_int < t_max_f:
        t_arr = np.append(t_arr, t_max_f)

    sv0_arr = np.asarray(sv0, dtype=float)

    # ── Barra de progreso ─────────────────────────────────────────────────────
    _pbar   = None
    _t_prev = [0.0]

    if show_progress:
        try:
            from tqdm.auto import tqdm as _tqdm
            _pbar = _tqdm(
                total=t_max_f, desc="reactor", unit="s",
                bar_format=(
                    "{l_bar}{bar}| {n:.1f}/{total:.0f} s"
                    " [{elapsed}<{remaining}, {rate_fmt}]"
                ),
                dynamic_ncols=True,
            )
        except ImportError:
            pass

    def _tick(t_val: float) -> None:
        if _pbar is None:
            return
        delta = t_val - _t_prev[0]
        if delta > 0.0:
            _pbar.update(min(delta, t_max_f - _t_prev[0]))
            _t_prev[0] = t_val

    def _close() -> None:
        if _pbar is not None:
            rem = t_max_f - _t_prev[0]
            if rem > 0.0:
                _pbar.update(rem)
            _pbar.close()

    # ── Integración BDF ───────────────────────────────────────────────────────
    def _rhs(t_val: float, sv_val: np.ndarray) -> np.ndarray:
        _tick(t_val)
        return core_rhs(t_val, sv_val, params)

    sol = solve_ivp(
        fun=_rhs,
        t_span=(t_arr[0], t_arr[-1]),
        y0=sv0_arr,
        t_eval=t_arr,
        method="BDF",
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    _close()

    if not sol.success:
        raise RuntimeError(
            f"runner_reactor: solve_ivp (BDF) falló — {sol.message}"
        )

    y_hist = sol.y.T    # (n_t, sv_size)

    # ── Post-proceso ──────────────────────────────────────────────────────────
    reactor = build_reactor_results(y_hist=y_hist, t_arr=t_arr, params=params)
    return t_arr, y_hist, reactor
