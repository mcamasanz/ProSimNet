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
