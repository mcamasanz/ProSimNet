"""
@module postprocessing.gasifier_plots
@brief Visualization functions for gasifier simulation results.

All functions accept the `col` SimpleNamespace returned by runner_gasifier.run_step()
and return (fig, axes) so the caller can further customize the figure.

Convention:
    0D (N=1) → temporal evolution plots (variable vs time)
    1D (N>1) → spatial profile plots at selected time indices, plus temporal at cell 0
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace


# ── helpers ───────────────────────────────────────────────────────────────────

def _pick_t_indices(col: SimpleNamespace, n: int = 5) -> list[int]:
    """Return n evenly-spaced time indices from the result array."""
    nt = len(col._t_results)
    return list(np.linspace(0, nt - 1, min(n, nt), dtype=int))


def _label_K_C(val: float) -> str:
    return f"{val:.0f} K ({val - 273.15:.0f} °C)"


# ── temperature ───────────────────────────────────────────────────────────────

def plot_temperatures(
    col: SimpleNamespace,
    t_indices: list[int] | None = None,
    T_ref: float | None = None,
    T_ref_label: str | None = None,
    figsize: tuple = (9, 4),
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot gas and solid temperature.

    Parameters
    ----------
    col         : gasifier result object
    t_indices   : time indices for spatial profiles (1D only); None → auto-select
    T_ref       : optional reference temperature [K] drawn as a horizontal dashed line
                  (e.g. wall temperature); ignored in 1D mode
    T_ref_label : label for the reference line; None → auto-generated from T_ref value
    figsize     : figure size [inches]
    ax          : existing axes to draw on; None → create new figure

    Returns
    -------
    (fig, ax)
    """
    N  = col._Tg_results.shape[1]
    t  = col._t_results

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if N == 1:
        # 0D: temporal evolution
        ax.plot(t, col._Tg_results[:, 0] - 273.15, lw=2, label="$T_g$ (gas)")
        ax.plot(t, col._Ts_results[:, 0] - 273.15, lw=2, label="$T_s$ (sólido)")
        if col._Tw_results is not None:
            ax.plot(t, col._Tw_results[:, 0] - 273.15, lw=1.5, ls="--",
                    label="$T_w$ (pared)")
        if T_ref is not None:
            lbl = T_ref_label or f"$T_{{ref}}$ = {T_ref - 273.15:.0f} °C"
            ax.axhline(T_ref - 273.15, ls="--", color="gray", alpha=0.7, label=lbl)
        ax.set_xlabel("Tiempo [s]")
        ax.set_ylabel("Temperatura [°C]")
        ax.set_title("Evolución de temperaturas (0D)")
    else:
        # 1D: spatial profiles at selected times
        z = col._z
        if t_indices is None:
            t_indices = _pick_t_indices(col)
        cmap = plt.cm.plasma
        for k, idx in enumerate(t_indices):
            color = cmap(k / max(len(t_indices) - 1, 1))
            lbl = f"t = {t[idx]:.0f} s"
            ax.plot(z, col._Tg_results[idx] - 273.15, color=color, lw=2, label=lbl)
            ax.plot(z, col._Ts_results[idx] - 273.15, color=color, lw=2, ls="--")
        ax.plot([], [], "k-",  lw=2, label="$T_g$ (gas)")
        ax.plot([], [], "k--", lw=2, label="$T_s$ (sólido)")
        ax.set_xlabel("Posición axial z [m]")
        ax.set_ylabel("Temperatura [°C]")
        ax.set_title("Perfiles axiales de temperatura (1D)")

    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ── solid phase ───────────────────────────────────────────────────────────────

def plot_solid_evolution(
    col: SimpleNamespace,
    t_indices: list[int] | None = None,
    figsize: tuple = (9, 4),
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot solid bulk densities (biomass, char, moisture).

    Parameters
    ----------
    col       : gasifier result object
    t_indices : time indices for spatial profiles (1D only)
    figsize   : figure size
    ax        : existing axes; None → create new

    Returns
    -------
    (fig, ax)
    """
    N   = col._rho_solid_results.shape[2]
    t   = col._t_results
    rho = col._rho_solid_results   # (n_t, 3, N)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    labels = ["Biomasa", "Char", "Humedad"]
    colors = ["saddlebrown", "dimgray", "steelblue"]

    if N == 1:
        for j, (lbl, col_) in enumerate(zip(labels, colors)):
            ax.plot(t, rho[:, j, 0], lw=2, color=col_, label=f"{lbl} [kg/m³$_{{bed}}$]")
        ax.set_xlabel("Tiempo [s]")
        ax.set_ylabel("Densidad bulk [kg/m³$_{bed}$]")
        ax.set_title("Evolución de la fase sólida (0D)")
    else:
        z = col._z
        if t_indices is None:
            t_indices = _pick_t_indices(col)
        cmap = plt.cm.viridis
        for k, idx in enumerate(t_indices):
            color = cmap(k / max(len(t_indices) - 1, 1))
            lbl = f"t = {t[idx]:.0f} s"
            for j, ls in enumerate(["-", "--", ":"]):
                ax.plot(z, rho[idx, j], color=color, lw=2, ls=ls)
        for j, (lbl, ls) in enumerate(zip(labels, ["-", "--", ":"])):
            ax.plot([], [], "k" + ls, lw=2, label=lbl)
        ax.set_xlabel("Posición axial z [m]")
        ax.set_ylabel("Densidad bulk [kg/m³$_{bed}$]")
        ax.set_title("Perfiles axiales de fase sólida (1D)")

    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ── gas composition ───────────────────────────────────────────────────────────

def plot_gas_composition(
    col: SimpleNamespace,
    t_indices: list[int] | None = None,
    threshold: float = 1e-3,
    figsize: tuple = (9, 4),
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot molar fractions of gas species.

    Parameters
    ----------
    col       : gasifier result object
    t_indices : time indices for 1D spatial profiles; None → auto-select
    threshold : minimum final mole fraction to include a species in the plot
    figsize   : figure size
    ax        : existing axes; None → create new

    Returns
    -------
    (fig, ax)
    """
    N       = col._y_results.shape[2]
    t       = col._t_results
    y       = col._y_results          # (n_t, nc, N)
    species = col._species

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if N == 1:
        for i, sp in enumerate(species):
            y_final = float(y[-1, i, 0])
            if y_final > threshold or float(y[:, i, 0].max()) > threshold:
                ax.plot(t, y[:, i, 0], lw=2,
                        label=f"{sp}  ({y_final*100:.1f} %)")
        ax.set_xlabel("Tiempo [s]")
        ax.set_ylabel("Fracción molar [-]")
        ax.set_title("Composición del gas (0D)")
    else:
        z = col._z
        if t_indices is None:
            t_indices = _pick_t_indices(col, n=3)
        # Solo especies con presencia significativa
        active = [i for i, sp in enumerate(species)
                  if y[-1, i, :].max() > threshold or y[:, i, :].max() > threshold]
        cmap = plt.cm.tab10
        for k, idx in enumerate(t_indices):
            for j, i in enumerate(active):
                color = cmap(j / max(len(active) - 1, 1))
                lbl = f"{species[i]} t={t[idx]:.0f}s" if k == 0 else None
                ax.plot(z, y[idx, i], color=color, lw=2 if k == 0 else 1,
                        ls=["-", "--", ":"][k], label=lbl)
        ax.set_xlabel("Posición axial z [m]")
        ax.set_ylabel("Fracción molar [-]")
        ax.set_title("Perfiles axiales de composición (1D)")

    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ── pressure ──────────────────────────────────────────────────────────────────

def plot_pressure(
    col: SimpleNamespace,
    figsize: tuple = (9, 3),
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot pressure evolution (0D: vs time; 1D: temporal at all cells).

    Returns
    -------
    (fig, ax)
    """
    N = col._P_results.shape[1]
    t = col._t_results

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if N == 1:
        ax.plot(t, col._P_results[:, 0], lw=2, color="steelblue")
        ax.set_xlabel("Tiempo [s]")
        ax.set_ylabel("Presión [bar]")
        ax.set_title("Evolución de la presión (0D)")
    else:
        z = col._z
        t_indices = _pick_t_indices(col)
        cmap = plt.cm.plasma
        for k, idx in enumerate(t_indices):
            color = cmap(k / max(len(t_indices) - 1, 1))
            ax.plot(z, col._P_results[idx], color=color, lw=2,
                    label=f"t = {t[idx]:.0f} s")
        ax.set_xlabel("Posición axial z [m]")
        ax.set_ylabel("Presión [bar]")
        ax.set_title("Perfiles axiales de presión (1D)")
        ax.legend(fontsize=9)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ── velocities ────────────────────────────────────────────────────────────────

def plot_velocities(
    col: SimpleNamespace,
    figsize: tuple = (9, 3),
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot gas velocity (inlet/outlet for 0D; spatial profiles for 1D).

    Returns
    -------
    (fig, ax)
    """
    t = col._t_results

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.plot(t, col._v_out_results, lw=2, color="darkorange", label="$v_{out}$")
    if col._v_in_results is not None:
        v_in = np.asarray(col._v_in_results, float)
        if not np.all(np.isnan(v_in)):
            ax.plot(t, v_in, lw=2, ls="--", color="navy", label="$v_{in}$")

    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Velocidad superficial [m/s]")
    ax.set_title("Velocidad del gas")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ── summary dashboard ─────────────────────────────────────────────────────────

def plot_summary(
    col: SimpleNamespace,
    t_indices: list[int] | None = None,
    figsize: tuple = (14, 8),
) -> tuple[plt.Figure, np.ndarray]:
    """
    2×3 dashboard with temperatures, solid evolution, gas composition,
    pressure, velocity and a blank axes for custom use.

    Returns
    -------
    (fig, axes)  where axes.shape = (2, 3)
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    plot_temperatures(col,     t_indices=t_indices, ax=axes[0, 0])
    plot_solid_evolution(col,  t_indices=t_indices, ax=axes[0, 1])
    plot_gas_composition(col,  t_indices=t_indices, ax=axes[0, 2])
    plot_pressure(col,                              ax=axes[1, 0])
    plot_velocities(col,                            ax=axes[1, 1])

    # axes[1, 2] reservado para uso externo (balance plot, etc.)
    axes[1, 2].set_visible(False)

    fig.tight_layout()
    return fig, axes


# ── sweep postprocessing ──────────────────────────────────────────────────────
# Functions that operate on the output of parametric_sweep:
#   df      : pd.DataFrame — one row per case, columns = sweep vars + metrics
#   results : list of col SimpleNamespace objects (same order as df rows)
#
# Public API:
#   plot_sweep_profiles     — time-series comparison of one attribute
#   plot_sweep_composition  — final gas composition as grouped bars
#   plot_sweep_solid        — solid components evolution (rho_bio, rho_char, rho_moi)
#   plot_sweep_pressure     — P(t) and v_out(t) side by side
#   plot_sweep_metrics      — scalar metrics as bar subplots


def _sweep_colors(df, sweep_col, use_tab10=False):
    """
    One color per row.  Viridis gradient for all-numeric sweep_col;
    tab10 discrete palette when any value is None/NaN or use_tab10=True.
    """
    import pandas as pd
    vals = df[sweep_col].tolist()
    has_missing = any(pd.isna(v) for v in vals)
    if use_tab10 or has_missing:
        return list(plt.cm.tab10.colors[: len(vals)])
    floats = [float(v) for v in vals]
    lo, hi = min(floats), max(floats)
    if lo == hi:
        return [plt.cm.viridis(0.5)] * len(floats)
    return [plt.cm.viridis((v - lo) / (hi - lo) * 0.70 + 0.15) for v in floats]


def _sweep_labels(df, sweep_col, label_fn=None):
    """
    One label string per row.

    Parameters
    ----------
    label_fn : callable(row) -> str, optional.
        If None, auto-generates ``"{sweep_col}={val:.4g}"``.
    """
    import pandas as pd
    labels = []
    for _, row in df.iterrows():
        if label_fn is not None:
            labels.append(str(label_fn(row)))
            continue
        val = row[sweep_col]
        if pd.isna(val):
            labels.append(f"{sweep_col}=None")
        elif isinstance(val, (int, np.integer)):
            labels.append(f"{sweep_col}={int(val)}")
        else:
            labels.append(f"{sweep_col}={float(val):.4g}")
    return labels


def plot_sweep_profiles(
    df,
    results,
    attr_fn,
    ylabel,
    title,
    sweep_col,
    y_transform=None,
    label_fn=None,
    use_tab10=False,
    h_ref=None,
    h_ref_label=None,
    figsize=(9, 4),
    ax=None,
):
    """
    Time-series comparison of one physical attribute across sweep cases.

    Parameters
    ----------
    df          : DataFrame from parametric_sweep
    results     : list of col objects (same order as df rows)
    attr_fn     : callable(col) -> ndarray(n_t,) — extracts the time profile
    ylabel      : y-axis label
    title       : subplot title
    sweep_col   : column in df used for labels and colormap
    y_transform : optional transform, e.g. ``lambda x: x - 273.15``
    label_fn    : callable(row) -> str for custom legend labels
    use_tab10   : force discrete tab10 palette
    h_ref       : optional horizontal reference line value
    h_ref_label : label for h_ref; auto-generated if None

    Returns
    -------
    (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    colors = _sweep_colors(df, sweep_col, use_tab10)
    labels = _sweep_labels(df, sweep_col, label_fn)

    for col_res, color, label in zip(results, colors, labels):
        if col_res is None:
            continue
        t   = col_res._t_results
        arr = np.atleast_1d(attr_fn(col_res)).copy()
        if y_transform is not None:
            arr = y_transform(arr)
        ax.plot(t, arr, color=color, lw=2, label=label)

    if h_ref is not None:
        lbl = h_ref_label or f"ref = {h_ref:.4g}"
        ax.axhline(h_ref, ls="--", color="gray", alpha=0.7, label=lbl)

    ax.set(xlabel="Tiempo [s]", ylabel=ylabel, title=title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_sweep_composition(
    df,
    results,
    sweep_col,
    species=None,
    species_show=None,
    threshold=1e-3,
    label_fn=None,
    figsize=(10, 4),
    ax=None,
):
    """
    Grouped bar chart of final molar fractions across sweep cases.

    Parameters
    ----------
    df           : DataFrame from parametric_sweep
    results      : list of col objects
    sweep_col    : column used for labels/colormap
    species      : full ordered species list; if None, read from results[0]._species
    species_show : subset to display; if None, auto-detect species with y_final > threshold
    threshold    : min mole fraction for auto-detection

    Returns
    -------
    (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if species is None:
        for r in results:
            if r is not None:
                species = list(r._species)
                break
    if species is None:
        raise ValueError("species list could not be inferred — pass species explicitly")

    if species_show is None:
        active = []
        for sp in species:
            idx = species.index(sp)
            for r in results:
                if r is not None and float(r._y_results[-1, idx, 0]) > threshold:
                    active.append(sp)
                    break
        species_show = active or species

    n_sp   = len(species_show)
    n_case = len(results)
    colors = _sweep_colors(df, sweep_col, use_tab10=(n_case <= 10))
    labels = _sweep_labels(df, sweep_col, label_fn)
    width  = 0.8 / n_case
    x_base = np.arange(n_sp)

    for k, (r, color, label) in enumerate(zip(results, colors, labels)):
        if r is None:
            continue
        yf = [float(r._y_results[-1, species.index(sp), 0]) for sp in species_show]
        ax.bar(x_base + k * width, yf, width, color=color, label=label, alpha=0.85)

    ax.set_xticks(x_base + width * (n_case - 1) / 2)
    ax.set_xticklabels(species_show)
    ax.set(ylabel="Fracción molar final [-]",
           title=f"Composición gas final — barrido de {sweep_col}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig, ax


def plot_sweep_solid(
    df,
    results,
    sweep_col,
    label_fn=None,
    figsize=(15, 4),
    axes=None,
):
    """
    Solid component bulk densities (biomasa, char, humedad) across sweep cases.

    Parameters
    ----------
    df        : DataFrame from parametric_sweep
    results   : list of col objects
    sweep_col : column used for labels/colormap

    Returns
    -------
    (fig, axes)  axes.shape = (3,)
    """
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    else:
        fig = axes.flat[0].get_figure()

    colors  = _sweep_colors(df, sweep_col)
    labels  = _sweep_labels(df, sweep_col, label_fn)
    titles  = ["Biomasa", "Char", "Humedad"]
    ylabels = ["ρ_biomasa [kg/m³_bed]", "ρ_char [kg/m³_bed]", "ρ_humedad [kg/m³_bed]"]

    for r, color, label in zip(results, colors, labels):
        if r is None:
            continue
        t = r._t_results
        for j, ax in enumerate(axes):
            ax.plot(t, r._rho_solid_results[:, j, 0], color=color, lw=2, label=label)

    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set(xlabel="Tiempo [s]", ylabel=ylabel, title=title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, axes


def plot_sweep_pressure(
    df,
    results,
    sweep_col,
    P_ref=None,
    label_fn=None,
    use_tab10=False,
    figsize=(13, 4),
    axes=None,
):
    """
    P(t) and v_out(t) comparison across sweep cases.

    Parameters
    ----------
    P_ref : reference pressure [bar] drawn as a horizontal dashed line

    Returns
    -------
    (fig, axes)  axes.shape = (2,)
    """
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        fig = axes.flat[0].get_figure()

    colors = _sweep_colors(df, sweep_col, use_tab10)
    labels = _sweep_labels(df, sweep_col, label_fn)

    for r, color, label in zip(results, colors, labels):
        if r is None:
            continue
        t = r._t_results
        axes[0].plot(t, r._P_results[:, 0], color=color, lw=2, label=label)
        axes[1].plot(t, r._v_out_results,   color=color, lw=2, label=label)

    if P_ref is not None:
        axes[0].axhline(P_ref, ls=":", color="gray", alpha=0.7,
                        label=f"P_ref = {P_ref:.3g} bar")

    axes[0].set(xlabel="Tiempo [s]", ylabel="P [bar]",
                title=f"Presión — barrido de {sweep_col}")
    axes[1].set(xlabel="Tiempo [s]", ylabel="v_out [m/s]",
                title=f"Velocidad de salida — barrido de {sweep_col}")
    for ax in axes:
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, axes


def plot_sweep_metrics(
    df,
    metric_cols,
    sweep_col,
    label_fn=None,
    figsize=None,
):
    """
    Scalar metrics from a parametric_sweep DataFrame as bar subplots.
    One subplot per metric; x-axis = sweep cases.

    Parameters
    ----------
    df          : DataFrame from parametric_sweep
    metric_cols : list of column names to plot
    sweep_col   : column used for x-tick labels

    Returns
    -------
    (fig, axes)  axes.shape = (n_metrics,)
    """
    n_m = len(metric_cols)
    if figsize is None:
        figsize = (max(4.0, n_m * 3.5), 4.0)

    fig, raw_axes = plt.subplots(1, n_m, figsize=figsize)
    axes = np.atleast_1d(raw_axes)

    colors = _sweep_colors(df, sweep_col)
    labels = _sweep_labels(df, sweep_col, label_fn)
    x      = np.arange(len(labels))

    for ax, metric in zip(axes, metric_cols):
        vals = df[metric].values.astype(float)
        ax.bar(x, vals, color=colors, alpha=0.85, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set(ylabel=metric, title=metric)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Métricas del barrido de {sweep_col}", fontweight="bold")
    fig.tight_layout()
    return fig, axes
