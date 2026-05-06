"""
Equipment-agnostic optimization and parametric analysis utilities.

All three functions follow the same contract:

    run_fn(params: dict) -> result
        Runs one simulation and returns a result object.
        Must be a closure capturing sv0, t_max, rtol, etc.
        Example (gasifier)::

            def run_case(params):
                t, _, g = run_step(sv0=sv0, t_max=T_END, params=params,
                                   rtol=1e-5, atol=1e-7, n_sec=3)
                return g

    objective_fn(result) -> dict[str, float]  (parametric_sweep)
    objective_fn(result) -> float             (optimize_bc, sensitivity_analysis)
        Extracts scalar metrics from the result object.

    param_patcher(params, name, value) -> dict
        Applies a named parameter change and returns a modified copy.
        If None, defaults to params[name] = value (top-level keys only).
        Required for nested params::

            def patcher(params, name, value):
                p = {**params, "_cache": {}}
                if name == "T_wall":
                    p["thermal_bc_config"] = {**p["thermal_bc_config"], "T_wall": value}
                elif name == "Cv":
                    p["bc_config"] = {**p["bc_config"], "Cv": value}
                else:
                    p[name] = value
                return p

The _cache is always reset after patching, regardless of the patcher implementation,
to prevent warm-start contamination between independent cases.
"""
from __future__ import annotations

import copy
import itertools
from typing import Any, Callable

import numpy as np


# ── Default patcher (top-level keys) ──────────────────────────────────────────

def _patch(params: dict, patcher: Callable | None,
           name: str, value: Any) -> dict:
    """Apply one parameter change, then always reset _cache."""
    if patcher is not None:
        p = patcher(params, name, value)
    else:
        p = {**params}
        p[name] = value
    p["_cache"] = {}
    return p


# ── parametric_sweep ───────────────────────────────────────────────────────────

def parametric_sweep(
    base_params: dict,
    sweep_vars: dict[str, list],
    run_fn: Callable[[dict], Any],
    objective_fn: Callable[[Any], dict[str, float]],
    param_patcher: Callable[[dict, str, Any], dict] | None = None,
    verbose: bool = True,
) -> "pd.DataFrame":
    """
    Cartesian product sweep over sweep_vars.

    Parameters
    ----------
    base_params   : dict
        Base parameter dict for the equipment.
    sweep_vars    : dict
        {param_name: [value1, value2, ...]}. Cartesian product of all lists.
    run_fn        : callable
        run_fn(params) -> result. Closure capturing sv0, t_max, rtol, etc.
    objective_fn  : callable
        objective_fn(result) -> dict[str, float]. Extracts scalar metrics.
    param_patcher : callable or None
        (params, name, value) -> patched_params. Default: params[name] = value.
    verbose       : bool
        Print case index, param values and metrics.

    Returns
    -------
    pd.DataFrame
        One row per parameter combination. Columns: sweep var names + metric names
        + "_status" ("OK" or "ERROR: ...").

    Example
    -------
    >>> def run_case(params):
    ...     t, _, g = run_step(sv0=sv0, t_max=T_END, params=params,
    ...                        rtol=1e-5, atol=1e-7, n_sec=3)
    ...     return g
    >>>
    >>> def metrics(g):
    ...     bio0 = g._rho_solid_results[0, 0, 0]
    ...     biof = g._rho_solid_results[-1, 0, 0]
    ...     return {"conv_bio": 1 - biof / bio0,
    ...             "P_max":    float(g._P_results[:, 0].max()),
    ...             "Ts_fin":   float(g._Ts_results[-1, 0]) - 273.15}
    >>>
    >>> df = parametric_sweep(base_params,
    ...                       sweep_vars={"T_wall": [700, 900, 1073],
    ...                                   "mc_wb":  [0.10, 0.165]},
    ...                       run_fn=run_case, objective_fn=metrics,
    ...                       param_patcher=patcher)
    """
    import pandas as pd

    names  = list(sweep_vars.keys())
    combos = list(itertools.product(*[sweep_vars[n] for n in names]))
    n_total = len(combos)

    rows = []
    for i, combo in enumerate(combos):
        # Build patched params for this combination
        p = copy.copy(base_params)
        p["_cache"] = {}
        for name, val in zip(names, combo):
            p = _patch(p, param_patcher, name, val)

        label = ", ".join(f"{n}={v}" for n, v in zip(names, combo))
        if verbose:
            print(f"  [{i+1:>{len(str(n_total))}}/{n_total}]  {label}", end="  ", flush=True)

        try:
            result  = run_fn(p)
            metrics = objective_fn(result)
            status  = "OK"
        except Exception as exc:
            metrics = {}
            status  = f"ERROR: {exc}"
            if verbose:
                print(f"⚠ {exc}", end="")

        row = {n: v for n, v in zip(names, combo)}
        row.update(metrics)
        row["_status"] = status
        rows.append(row)

        if verbose:
            if metrics:
                print("  ".join(f"{k}={v:.4g}" for k, v in metrics.items()))
            else:
                print()

    df = pd.DataFrame(rows)
    if verbose:
        n_ok = (df["_status"] == "OK").sum()
        print(f"\n✓ {n_ok}/{n_total} casos completados.")
    return df


# ── optimize_bc ────────────────────────────────────────────────────────────────

def optimize_bc(
    base_params: dict,
    decision_vars: dict[str, tuple[float, float]],
    run_fn: Callable[[dict], Any],
    objective_fn: Callable[[Any], float],
    param_patcher: Callable[[dict, str, Any], dict] | None = None,
    method: str = "L-BFGS-B",
    options: dict | None = None,
    verbose: bool = True,
) -> dict:
    """
    Minimize a scalar objective over continuous decision variables using scipy.

    Parameters
    ----------
    base_params    : dict
        Base parameter dict.
    decision_vars  : dict
        {param_name: (lower_bound, upper_bound)}. All vars are varied simultaneously.
    run_fn         : callable
        run_fn(params) -> result.
    objective_fn   : callable
        objective_fn(result) -> float. To maximize, return -metric.
    param_patcher  : callable or None
        (params, name, value) -> patched_params.
    method         : str
        scipy.optimize.minimize method. "L-BFGS-B" (bounded, gradient-free via
        finite differences) is recommended for smooth objectives.
    options        : dict or None
        scipy options dict (e.g. {"maxiter": 50, "ftol": 1e-4}).
    verbose        : bool
        Print each function evaluation.

    Returns
    -------
    dict with keys:
        "x_opt"   : dict {param_name: optimal_value}
        "f_opt"   : float   objective value at optimum
        "n_eval"  : int     number of run_fn evaluations
        "success" : bool
        "message" : str
        "raw"     : scipy OptimizeResult

    Example
    -------
    >>> result = optimize_bc(
    ...     base_params,
    ...     decision_vars={"T_wall": (700.0, 1200.0)},
    ...     run_fn=run_case,
    ...     objective_fn=lambda g: -(1 - g._rho_solid_results[-1,0,0]
    ...                                / g._rho_solid_results[0,0,0]),
    ...     param_patcher=patcher,
    ... )
    >>> print(f"T_wall óptima: {result['x_opt']['T_wall']:.1f} K  "
    ...       f"conv={-result['f_opt']:.3f}")
    """
    from scipy.optimize import minimize

    names  = list(decision_vars.keys())
    bounds = [decision_vars[n] for n in names]
    x0     = np.array([0.5 * (lo + hi) for lo, hi in bounds])
    n_eval = [0]

    def _objective(x: np.ndarray) -> float:
        n_eval[0] += 1
        p = copy.copy(base_params)
        p["_cache"] = {}
        for name, val in zip(names, x):
            p = _patch(p, param_patcher, name, float(val))
        result = run_fn(p)
        f = float(objective_fn(result))
        if verbose:
            vals = "  ".join(f"{n}={v:.4g}" for n, v in zip(names, x))
            print(f"  eval {n_eval[0]:3d}: {vals}  →  f={f:.6g}")
        return f

    opts = {"maxiter": 100, "ftol": 1.0e-5}
    if options:
        opts.update(options)

    raw = minimize(_objective, x0, method=method, bounds=bounds, options=opts)

    x_opt = {n: float(v) for n, v in zip(names, raw.x)}
    if verbose:
        marker = "✓" if raw.success else "⚠"
        print(f"\n{marker}  {raw.message}")
        print(f"   x_opt = {x_opt}")
        print(f"   f_opt = {raw.fun:.6g}   ({n_eval[0]} evaluaciones)")

    return {
        "x_opt":   x_opt,
        "f_opt":   float(raw.fun),
        "n_eval":  n_eval[0],
        "success": bool(raw.success),
        "message": str(raw.message),
        "raw":     raw,
    }


# ── sensitivity_analysis ───────────────────────────────────────────────────────

def sensitivity_analysis(
    base_params: dict,
    param_specs: dict[str, float],
    run_fn: Callable[[dict], Any],
    objective_fn: Callable[[Any], float],
    param_patcher: Callable[[dict, str, Any], dict] | None = None,
    delta_pct: float = 0.10,
    verbose: bool = True,
) -> "pd.DataFrame":
    """
    One-at-a-time (OAT) sensitivity analysis: perturb each parameter by
    ±delta_pct and measure the normalized change in the objective.

    Parameters
    ----------
    base_params  : dict
        Base parameter dict.
    param_specs  : dict
        {param_name: base_value}. Base values are provided explicitly to
        support nested parameters not accessible from base_params top-level.
    run_fn       : callable
        run_fn(params) -> result.
    objective_fn : callable
        objective_fn(result) -> float.
    param_patcher: callable or None
        (params, name, value) -> patched_params.
    delta_pct    : float
        Relative perturbation (default 0.10 = ±10 %).
    verbose      : bool
        Print progress.

    Returns
    -------
    pd.DataFrame indexed by param_name with columns:
        v_base, v_plus, v_minus   — parameter values
        f_base, f_plus, f_minus   — objective values
        S_plus, S_minus           — normalized sensitivity [(Δf/f) / (Δp/p)]
        S_mean                    — mean of |S_plus| and |S_minus|

    Notes
    -----
    Normalized sensitivity S = (Δf/f_base) / (Δp/p_base).
    |S| > 1 means the objective is more sensitive than proportional to the parameter.
    |S| < 1 means the objective is less sensitive.

    Example
    -------
    >>> df_sens = sensitivity_analysis(
    ...     base_params,
    ...     param_specs={"T_wall": 1073.15, "epsi_r": 0.60, "dp0": 0.008},
    ...     run_fn=run_case,
    ...     objective_fn=lambda g: 1 - g._rho_solid_results[-1,0,0]
    ...                                / g._rho_solid_results[0,0,0],
    ...     param_patcher=patcher,
    ...     delta_pct=0.10,
    ... )
    >>> display(df_sens.sort_values("S_mean", ascending=False))
    """
    import pandas as pd

    # Base case
    p_base = copy.copy(base_params)
    p_base["_cache"] = {}
    result_base = run_fn(p_base)
    f_base = float(objective_fn(result_base))
    if verbose:
        print(f"Caso base: f={f_base:.6g}")

    rows = []
    for param_name, v_base in param_specs.items():
        v_plus  = v_base * (1.0 + delta_pct)
        v_minus = v_base * (1.0 - delta_pct)

        if verbose:
            print(f"  {param_name}: {v_base:.4g}  "
                  f"+{delta_pct*100:.0f}%→{v_plus:.4g}  "
                  f"-{delta_pct*100:.0f}%→{v_minus:.4g}", end="  ", flush=True)

        p_plus  = _patch(copy.copy(base_params), param_patcher, param_name, v_plus)
        p_minus = _patch(copy.copy(base_params), param_patcher, param_name, v_minus)

        f_plus  = float(objective_fn(run_fn(p_plus)))
        f_minus = float(objective_fn(run_fn(p_minus)))

        # Normalized sensitivity: (Δf/f_base) / (Δp/p_base)
        _f_ref  = max(abs(f_base), 1.0e-30)
        S_plus  = ((f_plus  - f_base) / _f_ref) / delta_pct
        S_minus = ((f_minus - f_base) / _f_ref) / delta_pct
        S_mean  = 0.5 * (abs(S_plus) + abs(S_minus))

        if verbose:
            print(f"f+={f_plus:.4g}  f-={f_minus:.4g}  S_mean={S_mean:.3f}")

        rows.append({
            "param":   param_name,
            "v_base":  v_base,
            "v_plus":  v_plus,
            "v_minus": v_minus,
            "f_base":  f_base,
            "f_plus":  f_plus,
            "f_minus": f_minus,
            "S_plus":  S_plus,
            "S_minus": S_minus,
            "S_mean":  S_mean,
        })

    df = pd.DataFrame(rows).set_index("param")
    if verbose:
        print(f"\n✓ Análisis de sensibilidad completo ({len(rows)} parámetros).")
    return df
