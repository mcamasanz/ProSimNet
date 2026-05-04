"""
@module runner_adsorption
@brief Runner de simulación para columnas de adsorción 1D (PSA, TSA, VSA).

@details
Este runner es el punto de entrada para cualquier proceso basado en adsorción
en lecho empaquetado. Sus responsabilidades son:

1. **Validación de params**: comprobar que el dict de parámetros contiene todas
   las claves necesarias para la física de adsorción antes de lanzar el integrador.
   Esto incluye claves comunes (malla, propiedades del gas, geometría) y claves
   específicas de adsorción (isoterma, entalpía de adsorción, prop. del lecho).

2. **Orientación del flujo**: manejar la inversión del dominio si
   `bc_config["flow_direction"] == "backward"`.

3. **Integración ODE**: invocar `solve_ivp` (BDF) u `odeint` (LSODA) con el RHS
   de adsorción (`core_rhs` de `rhs_adsorption`).

4. **Post-proceso automático**: llamar a `build_adsorber_results()` para producir
   el objeto `col` con todos los atributos de resultado estandarizados.

Separación respecto a otros runners:
    runner_adsorption.py  → rhs_adsorption.py   (LDF + Ergun + energía gas/sólido)
    runner_reactor.py     → rhs_reactor.py       (cinética química + balance de pared)
    runner_<proceso>.py   → rhs_<proceso>.py     (física específica del proceso)

Cada runner valida el subconjunto de params que su RHS necesita, de modo que
los errores se detectan antes de integrar, con mensajes claros sobre qué falta.

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

from src.solvers.rhs.rhs_adsorption import core_rhs
from src.units.adsorber.state_extraction import build_adsorber_results
from src.utils.profiling import profiled


# ═══════════════════════════════════════════════════════════════════════════════
# Claves requeridas en params
# ═══════════════════════════════════════════════════════════════════════════════

# Claves comunes a cualquier simulación de columna 1D de flujo compresible
_REQUIRED_COMMON = (
    "n_comp",            # int — número de especies
    "N",                 # int — número de celdas
    "dz",                # float — longitud de celda [m]
    "Ai",                # float — área interna sección transversal [m²]
    "Di",                # float — diámetro interno [m]
    "Pi",                # float — perímetro interno [m]
    "Po",                # float — perímetro externo [m]
    "prop_gas",          # dict — propiedades puras del gas
    "MW",                # ndarray (nc,) — masas molares [kg/mol]
    "gas_T_ref",         # float — temperatura de referencia entálpica [K]
    "bc_config",         # dict — contornos por paso
    "trans_config",      # dict — configuración de coeficientes de transporte
    "thermal_bc_config", # dict — condición térmica de pared
    "energy",            # bool — activar balance de energía
)

# Claves específicas de la física de adsorción
_REQUIRED_ADSORPTION = (
    "iso_fn",            # callable: iso_fn(P_part_list, Ts) → q_eq (nc, N)
    "dH",                # ndarray (nc, N) [J/mol] — entalpía de adsorción
    "epsi",              # float — porosidad del lecho [-]
    "rho_s",             # float — densidad del sólido [kg/m³]
    "Cp_s",              # float — calor específico del sólido [J/kg/K]
    "k_s",               # float — conductividad térmica del sólido [W/m/K]
    "prop_lecho",        # dict — propiedades del lecho: D_p, a_surf, tau, r_pore, eps
)


def _validate_adsorption_params(params: dict) -> None:
    """
    @brief
    Comprueba que el dict de parámetros contiene todas las claves necesarias
    para la simulación de adsorción antes de lanzar el integrador.

    @details
    Valida por separado las claves comunes y las específicas de adsorción,
    para que el mensaje de error indique exactamente qué parte del params falta
    (geometría, propiedades del gas, datos del adsorbente, isoterma, etc.).

    Parameters
    ----------
    params : dict — dict de parámetros del modelo

    Raises
    ------
    KeyError
        Si alguna clave obligatoria no está presente en params.
    TypeError
        Si `iso_fn` no es callable o `energy` no es bool-compatible.
    ValueError
        Si n_comp, N o dz tienen valores inválidos.
    """
    # ── Claves comunes ────────────────────────────────────────────────────────
    missing_common = [k for k in _REQUIRED_COMMON if k not in params]
    if missing_common:
        raise KeyError(
            f"runner_adsorption: faltan claves comunes en params: {missing_common}\n"
            f"  Estas claves son requeridas para cualquier columna 1D de flujo compresible."
        )

    # ── Claves específicas de adsorción ───────────────────────────────────────
    missing_ads = [k for k in _REQUIRED_ADSORPTION if k not in params]
    if missing_ads:
        raise KeyError(
            f"runner_adsorption: faltan claves de adsorción en params: {missing_ads}\n"
            f"  Construir con build_adsorbent_config() y build_mixture_isotherm()."
        )

    # ── Validaciones de tipos críticos ────────────────────────────────────────
    if not callable(params["iso_fn"]):
        raise TypeError(
            "params['iso_fn'] debe ser un callable: iso_fn(P_part_list, Ts) → q_eq (nc, N)"
        )

    nc = int(params["n_comp"])
    nn = int(params["N"])
    dz = float(params["dz"])

    if nc < 1:
        raise ValueError(f"params['n_comp'] debe ser >= 1, got {nc}")
    # N=1 es válido: CSTR 0D (celda única bien mezclada). Ergun no resuelve
    # caras interiores; gradientes axiales nulos. Útil para exploración rápida.
    if nn < 1:
        raise ValueError(f"params['N'] debe ser >= 1, got {nn}")
    if dz <= 0.0:
        raise ValueError(f"params['dz'] debe ser > 0, got {dz}")

    dH = np.asarray(params["dH"], dtype=float)
    if dH.shape != (nc, nn):
        raise ValueError(
            f"params['dH'] debe tener shape ({nc}, {nn}), got {dH.shape}"
        )

    prop_lecho = params["prop_lecho"]
    for key in ("D_p", "a_surf"):
        if key not in prop_lecho:
            raise KeyError(
                f"params['prop_lecho'] debe contener la clave '{key}'. "
                f"Construir con build_adsorbent_config()."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers de inversión de dominio (flujo backward)
# ═══════════════════════════════════════════════════════════════════════════════

def _flip_state(sv: np.ndarray, n_comp: int, N: int, shell_tube: bool = False) -> np.ndarray:
    """Invierte el orden axial de cada bloque del vector de estado."""
    nc, nn = n_comp, N
    n_blocks = 2 * nc + 2 + (1 if shell_tube else 0)
    blocks = []
    for ib in range(n_blocks):
        block = sv[ib * nn:(ib + 1) * nn]
        blocks.append(block[::-1].copy())
    return np.concatenate(blocks)


def _unflip_history(y_hist: np.ndarray, n_comp: int, N: int, shell_tube: bool = False) -> np.ndarray:
    """Restaura la orientación física en la historia almacenada (nt, sv_size)."""
    nc, nn = n_comp, N
    n_blocks = 2 * nc + 2 + (1 if shell_tube else 0)
    y_out = y_hist.copy()
    for ib in range(n_blocks):
        col_start = ib * nn
        col_end   = (ib + 1) * nn
        y_out[:, col_start:col_end] = y_hist[:, col_start:col_end][:, ::-1]
    return y_out


# ═══════════════════════════════════════════════════════════════════════════════
# Función pública
# ═══════════════════════════════════════════════════════════════════════════════

@profiled
def run_step(
    step: str,
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
    Integra un paso temporal de adsorción y devuelve la historia de estados.

    @details
    Antes de integrar, valida que params contiene todas las claves requeridas
    por la física de adsorción (claves comunes + específicas de adsorción).
    Si falta alguna clave, lanza KeyError con mensaje informativo antes de
    llamar al integrador.

    Inyecta el nombre de paso en `params["step"]` antes de llamar al integrador,
    de modo que el core_rhs accede al step correcto en cada evaluación.

    Si `params["bc_config"]["flow_direction"] == "backward"`, el dominio se
    invierte antes de integrar y se restaura al finalizar.

    Parameters
    ----------
    step          : str   — nombre del paso ("ads", "purge", "pr_feed", "blowdown", "wait")
    sv0           : ndarray — vector de estado inicial
    t_max         : float — duración del paso [s]
    params        : dict  — parámetros del modelo (validados aquí antes de integrar)
    solver        : str   — "solve_ivp" (BDF) o "odeint" (LSODA)
    rtol          : float — tolerancia relativa del integrador
    atol          : float — tolerancia absoluta del integrador
    n_sec         : int   — número de puntos de salida por segundo de simulación
    show_progress : bool  — mostrar barra de progreso (requiere tqdm)

    Returns
    -------
    t_arr  : ndarray, shape (nt,) — instantes de tiempo [s]
    y_hist : ndarray, shape (nt, sv_size) — historia del vector de estado (orientación física)
    col    : SimpleNamespace — objeto con atributos de resultado estandarizados
    """
    # ── Validación de params ──────────────────────────────────────────────────
    _validate_adsorption_params(params)

    # ── Inyectar el step en params ────────────────────────────────────────────
    params["step"] = str(step).strip().lower()

    nc         = int(params["n_comp"])
    nn         = int(params["N"])
    shell_tube = params.get("wall_config") is not None

    # Validación adicional de wall_config (si está presente)
    if shell_tube:
        wall_config = params["wall_config"]
        for wkey in ("material", "A_w", "Di", "Do", "T_w_init"):
            if wkey not in wall_config:
                raise KeyError(
                    f"params['wall_config'] debe contener '{wkey}'. "
                    f"Construir con build_wall_config del adsorbedor."
                )
        tbc_mode = params["thermal_bc_config"]["mode"]
        if tbc_mode == "fixed_twall":
            raise ValueError(
                "thermal_bc_config mode='fixed_twall' es incompatible con wall_config "
                "(shell_tube=True). Usar 'adiabatic', 'heatfluxwall' o 'ambient_htc'."
            )

    # ── Orientación del flujo ─────────────────────────────────────────────────
    backward = (params["bc_config"].get("flow_direction", "forward") == "backward")

    sv0_arr = np.asarray(sv0, dtype=float).copy()
    if backward:
        sv0_solver = _flip_state(sv0_arr, nc, nn, shell_tube=shell_tube)
    else:
        sv0_solver = sv0_arr

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

    # ── Reiniciar caché de propiedades al inicio de cada paso ─────────────────
    params.setdefault("_cache", {})
    params["_cache"].pop("gas_props",   None)
    params["_cache"].pop("trans_props", None)
    # "Tg_last" se conserva: es el warm-start de Newton entre pasos consecutivos

    # ── Barra de progreso (opcional, requiere tqdm) ───────────────────────────
    _pbar = None
    _t_pbar = [0.0]

    if show_progress:
        try:
            from tqdm.auto import tqdm as _tqdm
            _pbar = _tqdm(
                total=t_max_val,
                desc=f"step={params['step']}",
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
    def _rhs_ivp(t_val, sv_val):
        _update_pbar(t_val)
        return core_rhs(t_val, sv_val, params)

    solver_str = str(solver).strip().lower()

    if solver_str == "solve_ivp":
        sol = solve_ivp(
            fun=_rhs_ivp,
            t_span=(t_arr[0], t_arr[-1]),
            y0=sv0_solver,
            t_eval=t_arr,
            method="BDF",
            rtol=rtol,
            atol=atol,
        )
        _close_pbar()
        if not sol.success:
            raise RuntimeError(
                f"solve_ivp (BDF) failed at step '{step}': {sol.message}"
            )
        y_hist_solver = sol.y.T    # (nt, sv_size)

    elif solver_str == "odeint":
        if not _HAS_ODEINT:
            raise ImportError("scipy.integrate.odeint not available")

        def _rhs_odeint(sv_val, t_val):
            _update_pbar(t_val)
            return core_rhs(t_val, sv_val, params)

        from scipy.integrate import odeint
        y_hist_solver = odeint(
            _rhs_odeint, sv0_solver, t_arr,
            rtol=rtol, atol=atol, mxstep=5000,
        )
        _close_pbar()

    else:
        _close_pbar()
        raise ValueError(f"solver must be 'solve_ivp' or 'odeint', got '{solver}'")

    # ── Restaurar orientación física si backward ──────────────────────────────
    if backward:
        y_hist = _unflip_history(y_hist_solver, nc, nn, shell_tube=shell_tube)
    else:
        y_hist = y_hist_solver

    # ── Post-proceso automático ───────────────────────────────────────────────
    col = build_adsorber_results(y_hist, t_arr, params)

    return t_arr, y_hist, col
