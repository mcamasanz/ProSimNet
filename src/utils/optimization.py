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
import os
from typing import Any, Callable

import numpy as np


# ── Parallel execution helpers ─────────────────────────────────────────────────

def _n_workers(n_jobs: int) -> int:
    """Resolve n_jobs=-1 to cpu_count."""
    if n_jobs == -1:
        return max(1, os.cpu_count() or 1)
    return max(1, int(n_jobs))


def _run_one(args: tuple) -> tuple:
    """Module-level worker (must be picklable for joblib).

    Reads two special keys from params (injected by parametric_sweep):
      _case_desc     : str  — label printed at job start in parallel mode
      _show_progress : bool — controls inner run_fn simulation bar
    Both are consumed here; _show_progress is passed through for run_fn to read.
    """
    import os
    p, run_fn, objective_fn, return_result = args
    case_desc = p.pop("_case_desc", "")   # run_fn must not see this key
    # _show_progress stays in p so run_fn can read params.get("_show_progress")
    if case_desc:
        try:
            from tqdm.auto import tqdm as _tqdm
            _tqdm.write(f"  [PID {os.getpid()}] inicio: {case_desc}")
        except Exception:
            print(f"  [PID {os.getpid()}] inicio: {case_desc}", flush=True)
    try:
        result  = run_fn(p)
        metrics = objective_fn(result)
        return "OK", result if return_result else None, metrics
    except Exception as exc:
        return f"ERROR: {exc}", None, {}


def _parallel_map(fn_args: list[tuple], n_jobs: int, verbose: bool) -> list[tuple]:
    """
    Execute a list of (params, run_fn, objective_fn, return_result) tuples in parallel.

    Uses joblib (pip install joblib) which handles notebook closures via cloudpickle.
    Falls back to serial execution with a RuntimeWarning if joblib is not installed.
    Shows a tqdm progress bar (one tick per completed case).
    Preserves input order in the output.

    Returns list of (status, result_or_None, metrics_dict).
    """
    nw = _n_workers(n_jobs)
    try:
        from joblib import Parallel, delayed
    except ImportError:
        import warnings
        warnings.warn(
            "joblib is required for parallel execution "
            "(pip install joblib). Falling back to serial.",
            RuntimeWarning, stacklevel=4,
        )
        return [_run_one(a) for a in fn_args]

    try:
        from tqdm.auto import tqdm as _tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    n_total = len(fn_args)
    if verbose:
        print(f"  [paralelo] {n_total} casos / {nw} workers (joblib) ...", flush=True)

    runner = Parallel(n_jobs=nw, verbose=0, prefer="processes")

    if has_tqdm:
        try:
            # joblib >= 1.3: generator interface lets tqdm tick per completed case
            outcomes = []
            with _tqdm(total=n_total, desc="Paralelo", unit="caso", leave=True) as pbar:
                for result in Parallel(n_jobs=nw, verbose=0, prefer="processes",
                                       return_as="generator")(
                    delayed(_run_one)(a) for a in fn_args
                ):
                    outcomes.append(result)
                    status = result[0]
                    pbar.update(1)
                    if verbose:
                        marker = "✓" if status == "OK" else "⚠"
                        pbar.write(f"    {marker} caso {len(outcomes)}/{n_total}")
        except TypeError:
            # joblib < 1.3 fallback: run all at once, then show bar at 100%
            with _tqdm(total=n_total, desc="Paralelo", unit="caso", leave=True) as pbar:
                outcomes = runner(delayed(_run_one)(a) for a in fn_args)
                pbar.update(n_total)
    else:
        outcomes = runner(delayed(_run_one)(a) for a in fn_args)

    return outcomes


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
    return_results: bool = False,
    n_jobs: int = 1,
    show_sim_progress: bool = True,
    verbose: bool = True,
) -> "pd.DataFrame | tuple[pd.DataFrame, list]":
    """
    Cartesian product sweep over sweep_vars.

    Parameters
    ----------
    base_params       : dict
        Base parameter dict for the equipment.
    sweep_vars        : dict
        {param_name: [value1, value2, ...]}. Cartesian product of all lists.
    run_fn            : callable
        run_fn(params) -> result. Closure capturing sv0, t_max, rtol, etc.
        Read ``params.get("_show_progress", False)`` to control the inner
        simulation bar: it is True in serial+show_sim_progress, False otherwise.
    objective_fn      : callable
        objective_fn(result) -> dict[str, float]. Extracts scalar metrics.
    param_patcher     : callable or None
        (params, name, value) -> patched_params. Default: params[name] = value.
    return_results    : bool
        If True, also return the list of raw result objects (same order as df rows)
        so that full time-series can be plotted across cases. Errors stored as None.
    n_jobs            : int
        Number of parallel workers.
        1   → serial execution (default, always works).
        -1  → use all available CPUs (os.cpu_count()).
        N   → use N parallel workers.
        Requires joblib (pip install joblib) — handles notebook closures via cloudpickle.
        Falls back to serial with a warning if joblib is not installed.
    show_sim_progress : bool
        Serial only: inject params["_show_progress"]=True so that run_fn can
        show its inner simulation progress bar. In parallel mode this is always
        False (multiple concurrent bars would interleave).
    verbose           : bool
        Print case index, param values and metrics (serial) or summary (parallel).

    Returns
    -------
    df : pd.DataFrame
        One row per combination. Columns: sweep var names + metric names + "_status".
    results : list (only when return_results=True)
        Raw result objects in the same order as df rows. Errors stored as None.
        Access: ``df, results = parametric_sweep(..., return_results=True)``

    Example
    -------
    >>> # Serial with inner simulation bar:
    >>> df, results = parametric_sweep(
    ...     base_params,
    ...     sweep_vars={"T_wall": [700, 900, 1073], "mc_wb": [0.10, 0.165]},
    ...     run_fn=run_case, objective_fn=metrics,
    ...     param_patcher=patcher, return_results=True, show_sim_progress=True,
    ... )
    >>> # Parallel (4 cores) — inner bar suppressed automatically:
    >>> df, results = parametric_sweep(..., n_jobs=4, return_results=True)
    >>> # In run_fn, read show_progress:
    >>> def run_case(params):
    ...     show_p = bool(params.get("_show_progress", False))
    ...     t, _, g = run_step(..., show_progress=show_p)
    ...     return g
    """
    import pandas as pd

    try:
        from tqdm.auto import tqdm as _tqdm
        _has_tqdm = True
    except ImportError:
        _has_tqdm = False

    names   = list(sweep_vars.keys())
    combos  = list(itertools.product(*[sweep_vars[n] for n in names]))
    n_total = len(combos)

    # serial: show inner sim bar if requested; parallel: always suppress
    _show_p = show_sim_progress and (n_jobs == 1)

    # Construir lista de params parchados con metadatos inyectados
    all_params = []
    for combo in combos:
        p = copy.copy(base_params)
        p["_cache"] = {}
        for name, val in zip(names, combo):
            p = _patch(p, param_patcher, name, val)
        label = ", ".join(f"{n}={v}" for n, v in zip(names, combo))
        p["_show_progress"] = _show_p   # leído por run_fn
        p["_case_desc"]     = label     # leído por _run_one en modo paralelo
        all_params.append(p)

    # ── Ejecución ─────────────────────────────────────────────────────────────
    if n_jobs == 1:
        outcomes = []
        pbar = _tqdm(total=n_total, desc="Barrido", unit="caso", leave=True) if _has_tqdm else None
        for i, (p, combo) in enumerate(zip(all_params, combos)):
            label = ", ".join(f"{n}={v}" for n, v in zip(names, combo))
            if pbar is not None:
                pbar.set_postfix_str(label[:60])
            elif verbose:
                print(f"  [{i+1:>{len(str(n_total))}}/{n_total}]  {label}",
                      end="  ", flush=True)
            try:
                result  = run_fn(p)
                metrics = objective_fn(result)
                status  = "OK"
            except Exception as exc:
                result, metrics, status = None, {}, f"ERROR: {exc}"
            outcomes.append((status, result, metrics))
            if pbar is not None:
                pbar.update(1)
                if verbose:
                    marker = "✓" if status == "OK" else "⚠"
                    summary = ("  ".join(f"{k}={v:.4g}" for k, v in metrics.items())
                               if metrics else status)
                    pbar.write(f"  {marker} [{i+1}/{n_total}] {label}  |  {summary}")
            elif verbose:
                if metrics:
                    print("  ".join(f"{k}={v:.4g}" for k, v in metrics.items()))
                else:
                    print()
        if pbar is not None:
            pbar.close()
    else:
        # Paralelo: _parallel_map preserva el orden de entrada
        fn_args  = [(p, run_fn, objective_fn, return_results) for p in all_params]
        outcomes = _parallel_map(fn_args, n_jobs, verbose)

    # ── Construir DataFrame y lista de resultados ──────────────────────────────
    rows        = []
    raw_results = []
    for combo, (status, result, metrics) in zip(combos, outcomes):
        row = {n: v for n, v in zip(names, combo)}
        row.update(metrics)
        row["_status"] = status
        rows.append(row)
        raw_results.append(result)

    df = pd.DataFrame(rows)
    if verbose:
        n_ok = (df["_status"] == "OK").sum()
        print(f"\n✓ {n_ok}/{n_total} casos completados.")

    return (df, raw_results) if return_results else df


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
    return_results: bool = False,
    n_jobs: int = 1,
    show_sim_progress: bool = True,
    verbose: bool = True,
) -> "pd.DataFrame | tuple[pd.DataFrame, dict]":
    """
    One-at-a-time (OAT) sensitivity: perturb each parameter by ±delta_pct
    and measure the normalized change in the objective.

    Parameters
    ----------
    base_params    : dict
        Base parameter dict.
    param_specs    : dict
        {param_name: base_value}. Base values provided explicitly to support
        nested parameters not accessible from base_params top-level.
    run_fn         : callable
        run_fn(params) -> result.
    objective_fn   : callable
        objective_fn(result) -> float.
    param_patcher  : callable or None
        (params, name, value) -> patched_params.
    delta_pct      : float
        Relative perturbation (default 0.10 = ±10 %).
    return_results : bool
        If True, also return a dict of raw result objects keyed by case label.
        Keys: "base", "{param}_plus", "{param}_minus" for each param.
        Errors stored as None.
    n_jobs         : int
        Number of parallel workers.
        1   → serial (default). -1 → all CPUs. N → N workers.
        All cases (base + perturbations) are dispatched as a single batch,
        so the base case also runs in parallel when n_jobs > 1.
    verbose        : bool
        Print progress.

    Returns
    -------
    df : pd.DataFrame indexed by param_name
        Columns: v_base, v_plus, v_minus, f_base, f_plus, f_minus,
                 S_plus, S_minus, S_mean.
    results : dict (only when return_results=True)
        {"base": result_base,
         "{param}_plus":  result_plus,   # result at +delta_pct
         "{param}_minus": result_minus}  # result at -delta_pct
        Access: ``df, results = sensitivity_analysis(..., return_results=True)``

    Notes
    -----
    Normalized sensitivity S = (Δf/f_base) / (Δp/p_base).
    |S| > 1: objective varies more than proportionally with the parameter.
    |S| < 1: objective is less sensitive than proportional.

    Example
    -------
    >>> df_s, res = sensitivity_analysis(
    ...     base_params,
    ...     param_specs={"T_wall": 1073.15, "epsi_r": 0.60, "dp0": 0.008},
    ...     run_fn=run_case,
    ...     objective_fn=lambda g: 1 - g._rho_solid_results[-1,0,0]
    ...                                / g._rho_solid_results[0,0,0],
    ...     param_patcher=patcher, return_results=True,
    ... )
    >>> display(df_s.sort_values("S_mean", ascending=False))
    >>> # Compare Ts profiles for T_wall +10% vs base vs -10%:
    >>> ax.plot(t, res["base"]._Ts_results[:,0], label="base")
    >>> ax.plot(t, res["T_wall_plus"]._Ts_results[:,0],  label="+10%")
    >>> ax.plot(t, res["T_wall_minus"]._Ts_results[:,0], label="-10%")
    """
    import pandas as pd

    try:
        from tqdm.auto import tqdm as _tqdm
        _has_tqdm = True
    except ImportError:
        _has_tqdm = False

    _show_p = show_sim_progress and (n_jobs == 1)

    # ── Construir TODOS los casos: base primero, luego perturbaciones ──────────
    # El caso base es una simulación más — no hay razón para ejecutarlo en serie.
    # Orden: [base, param0_plus, param0_minus, param1_plus, param1_minus, ...]
    p_base_entry = copy.copy(base_params)
    p_base_entry["_cache"]         = {}
    p_base_entry["_show_progress"] = _show_p
    p_base_entry["_case_desc"]     = "base"

    all_labels = ["base"]
    all_params = [p_base_entry]

    for param_name, v_base in param_specs.items():
        v_plus  = v_base * (1.0 + delta_pct)
        v_minus = v_base * (1.0 - delta_pct)
        for sign, val in (("+", v_plus), ("-", v_minus)):
            lbl = f"{param_name}_plus" if sign == "+" else f"{param_name}_minus"
            p   = _patch(copy.copy(base_params), param_patcher, param_name, val)
            p["_show_progress"] = _show_p
            p["_case_desc"]     = f"{param_name}={val:.4g} ({sign}{delta_pct*100:.0f}%)"
            all_labels.append(lbl)
            all_params.append(p)

    # ── Ejecución (serie o paralelo) — base incluido en el lote ───────────────
    n_cases = len(all_params)
    if n_jobs == 1:
        all_outcomes = []
        pbar = _tqdm(total=n_cases, desc="Sensibilidad", unit="caso",
                     leave=True) if _has_tqdm else None
        for p, lbl in zip(all_params, all_labels):
            if pbar is not None:
                pbar.set_postfix_str(lbl[:60])
            try:
                r = run_fn(p)
                f = float(objective_fn(r))
                all_outcomes.append(("OK", r, f))
                if pbar is not None:
                    pbar.update(1)
                    if verbose:
                        pbar.write(f"  ✓ {lbl}: f={f:.4g}")
            except Exception as exc:
                all_outcomes.append((f"ERROR: {exc}", None, float("nan")))
                if pbar is not None:
                    pbar.update(1)
        if pbar is not None:
            pbar.close()
    else:
        fn_args  = [(p, run_fn, objective_fn, return_results) for p in all_params]
        raw_out  = _parallel_map(fn_args, n_jobs, verbose)
        all_outcomes = [(st, r, float(m) if isinstance(m, (int, float)) else float("nan"))
                        for st, r, m in raw_out]

    # ── Extraer caso base y perturbaciones ─────────────────────────────────────
    base_st, result_base, f_base = all_outcomes[0]
    if base_st != "OK":
        raise RuntimeError(f"El caso base falló: {base_st}")
    outcomes    = all_outcomes[1:]
    case_labels = all_labels[1:]
    if verbose:
        print(f"Caso base: f={f_base:.6g}")

    # ── Recopilar resultados y calcular S ──────────────────────────────────────
    perturb_results: dict = {"base": result_base}
    for label, (_, r, _) in zip(case_labels, outcomes):
        perturb_results[label] = r

    rows = []
    _f_ref = max(abs(f_base), 1.0e-30)

    for param_name, v_base in param_specs.items():
        st_p, _, f_plus  = outcomes[case_labels.index(f"{param_name}_plus")]
        st_m, _, f_minus = outcomes[case_labels.index(f"{param_name}_minus")]
        v_plus  = v_base * (1.0 + delta_pct)
        v_minus = v_base * (1.0 - delta_pct)

        S_plus  = ((f_plus  - f_base) / _f_ref) / delta_pct if not np.isnan(f_plus)  else float("nan")
        S_minus = ((f_minus - f_base) / _f_ref) / delta_pct if not np.isnan(f_minus) else float("nan")
        S_vals  = [abs(v) for v in (S_plus, S_minus) if not np.isnan(v)]
        S_mean  = float(np.mean(S_vals)) if S_vals else float("nan")

        if verbose:
            print(f"  {param_name}: f+={f_plus:.4g}  f-={f_minus:.4g}  S_mean={S_mean:.3f}")

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

    return (df, perturb_results) if return_results else df
