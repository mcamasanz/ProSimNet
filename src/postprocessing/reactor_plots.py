"""
Visualization functions for reactor simulation results.

All functions accept the `reactor` SimpleNamespace returned by
runner_reactor.run_step() and return (fig, axes) so the caller can
further customize the figure.

Convention
----------
0D (N=1) → temporal evolution plots (variable vs time)
1D (N>1) → spatial profile plots at selected time indices
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace


# ── helpers ───────────────────────────────────────────────────────────────────

def _pick_t_indices(reactor: SimpleNamespace, n: int = 5) -> list[int]:
    """Return n evenly-spaced time indices from the result array."""
    nt = len(reactor._t_results)
    return list(np.linspace(0, nt - 1, min(n, nt), dtype=int))


# ── temperatures ──────────────────────────────────────────────────────────────

def plot_temperatures(
    reactor: SimpleNamespace,
    t_indices: list[int] | None = None,
    T_ref: float | None = None,
    T_ref_label: str | None = None,
    figsize: tuple = (9, 4),
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot gas temperature (and catalyst/wall if present).

    Parameters
    ----------
    reactor     : reactor result object
    t_indices   : time indices for spatial profiles (1D only); None → auto-select
    T_ref       : optional reference temperature [K] drawn as horizontal dashed line
    T_ref_label : label for the reference line; None → auto from T_ref value
    figsize     : figure size [inches]
    ax          : existing axes; None → create new figure

    Returns
    -------
    (fig, ax)
    """
    N = reactor._Tg_results.shape[1]
    t = reactor._t_results

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if N == 1:
        ax.plot(t, reactor._Tg_results[:, 0] - 273.15, lw=2, label="$T_g$ (gas)")
        if reactor._Ts_results is not None:
            ax.plot(t, reactor._Ts_results[:, 0] - 273.15, lw=2, ls="--",
                    label="$T_s$ (catalizador)")
        if reactor._Tw_results is not None:
            ax.plot(t, reactor._Tw_results[:, 0] - 273.15, lw=1.5, ls=":",
                    label="$T_w$ (pared)")
        if T_ref is not None:
            lbl = T_ref_label or f"$T_{{ref}}$ = {T_ref - 273.15:.0f} °C"
            ax.axhline(T_ref - 273.15, ls="--", color="gray", alpha=0.7, label=lbl)
        ax.set_xlabel("Tiempo [s]")
        ax.set_ylabel("Temperatura [°C]")
        ax.set_title("Evolución de temperaturas (0D)")
    else:
        z = reactor._z
        if t_indices is None:
            t_indices = _pick_t_indices(reactor)
        cmap = plt.cm.plasma
        for k, idx in enumerate(t_indices):
            color = cmap(k / max(len(t_indices) - 1, 1))
            lbl = f"t = {t[idx]:.0f} s"
            ax.plot(z, reactor._Tg_results[idx] - 273.15, color=color, lw=2, label=lbl)
            if reactor._Ts_results is not None:
                ax.plot(z, reactor._Ts_results[idx] - 273.15, color=color, lw=2, ls="--")
        ax.plot([], [], "k-",  lw=2, label="$T_g$ (gas)")
        if reactor._Ts_results is not None:
            ax.plot([], [], "k--", lw=2, label="$T_s$ (catalizador)")
        ax.set_xlabel("Posición axial z [m]")
        ax.set_ylabel("Temperatura [°C]")
        ax.set_title("Perfiles axiales de temperatura (1D)")

    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ── gas composition ───────────────────────────────────────────────────────────

def plot_gas_composition(
    reactor: SimpleNamespace,
    t_indices: list[int] | None = None,
    threshold: float = 1e-3,
    figsize: tuple = (9, 4),
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot molar fractions of gas species.

    Parameters
    ----------
    reactor   : reactor result object
    t_indices : time indices for 1D spatial profiles; None → auto-select
    threshold : minimum mole fraction to include a species in the plot
    figsize   : figure size
    ax        : existing axes; None → create new

    Returns
    -------
    (fig, ax)
    """
    N       = reactor._y_results.shape[2]
    t       = reactor._t_results
    y       = reactor._y_results   # (n_t, nc, N)
    species = reactor._species

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if N == 1:
        for i, sp in enumerate(species):
            y_max = float(y[:, i, 0].max())
            if y_max > threshold:
                y_fin = float(y[-1, i, 0])
                ax.plot(t, y[:, i, 0], lw=2,
                        label=f"{sp}  ({y_fin*100:.1f} %)")
        ax.set_xlabel("Tiempo [s]")
        ax.set_ylabel("Fracción molar [-]")
        ax.set_title("Composición del gas (0D)")
    else:
        z = reactor._z
        if t_indices is None:
            t_indices = _pick_t_indices(reactor, n=3)
        active = [i for i, sp in enumerate(species)
                  if y[:, i, :].max() > threshold]
        cmap = plt.cm.tab10
        for k, idx in enumerate(t_indices):
            for j, i in enumerate(active):
                color = cmap(j / max(len(active) - 1, 1))
                lbl = f"{species[i]} t={t[idx]:.0f}s" if k == 0 else None
                ax.plot(z, y[idx, i], color=color,
                        lw=2 if k == 0 else 1,
                        ls=["-", "--", ":"][k],
                        label=lbl)
        ax.set_xlabel("Posición axial z [m]")
        ax.set_ylabel("Fracción molar [-]")
        ax.set_title("Perfiles axiales de composición (1D)")

    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ── pressure ──────────────────────────────────────────────────────────────────

def plot_pressure(
    reactor: SimpleNamespace,
    figsize: tuple = (9, 3),
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot pressure (0D: vs time; 1D: spatial profiles at selected instants).

    Returns
    -------
    (fig, ax)
    """
    N = reactor._P_results.shape[1]
    t = reactor._t_results

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if N == 1:
        ax.plot(t, reactor._P_results[:, 0], lw=2, color="steelblue")
        ax.set_xlabel("Tiempo [s]")
        ax.set_ylabel("Presión [bar]")
        ax.set_title("Evolución de la presión (0D)")
    else:
        z = reactor._z
        t_indices = _pick_t_indices(reactor)
        cmap = plt.cm.plasma
        for k, idx in enumerate(t_indices):
            color = cmap(k / max(len(t_indices) - 1, 1))
            ax.plot(z, reactor._P_results[idx], color=color, lw=2,
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
    reactor: SimpleNamespace,
    figsize: tuple = (9, 3),
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot gas superficial velocities (inlet and outlet) vs time.

    Returns
    -------
    (fig, ax)
    """
    t   = reactor._t_results
    v_in  = reactor._v_in_results
    v_out = reactor._v_out_results

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if not np.all(v_in == 0.0):
        ax.plot(t, v_in,  lw=2, ls="--", color="navy",       label="$v_{in}$")
    ax.plot(    t, v_out, lw=2,           color="darkorange", label="$v_{out}$")

    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Velocidad superficial [m/s]")
    ax.set_title("Velocidad del gas")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ── conversion ────────────────────────────────────────────────────────────────

def plot_conversion(
    reactor: SimpleNamespace,
    reactant: str,
    figsize: tuple = (9, 3),
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot conversion of a reactant species over time (0D) or axial position (1D).

    Conversion defined as X = 1 - y_i(t) / y_i(0)  for 0D,
    or X(z) = 1 - y_i(z,t) / y_i(z=0, t)  for 1D (last stored time).

    Parameters
    ----------
    reactor  : reactor result object
    reactant : species name (must be in reactor._species)
    figsize  : figure size
    ax       : existing axes; None → create new

    Returns
    -------
    (fig, ax)
    """
    species = reactor._species
    if reactant not in species:
        raise ValueError(
            f"plot_conversion: '{reactant}' not in species {species}"
        )
    idx_sp = species.index(reactant)
    y      = reactor._y_results    # (n_t, nc, N)
    N      = y.shape[2]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if N == 1:
        y0 = float(y[0, idx_sp, 0])
        if y0 < 1e-12:
            ax.set_title(f"Conversión {reactant} — fracción inicial ≈ 0, no calculable")
        else:
            X = 1.0 - y[:, idx_sp, 0] / y0
            ax.plot(reactor._t_results, X * 100.0, lw=2, color="darkgreen")
            ax.set_xlabel("Tiempo [s]")
            ax.set_ylabel(f"Conversión {reactant} [%]")
            ax.set_title(f"Conversión de {reactant} (0D)")
    else:
        z  = reactor._z
        t  = reactor._t_results
        t_indices = _pick_t_indices(reactor)
        cmap = plt.cm.plasma
        for k, tidx in enumerate(t_indices):
            y_inlet = float(y[tidx, idx_sp, 0])
            if y_inlet < 1e-12:
                continue
            X = 1.0 - y[tidx, idx_sp, :] / y_inlet
            color = cmap(k / max(len(t_indices) - 1, 1))
            ax.plot(z, X * 100.0, color=color, lw=2, label=f"t = {t[tidx]:.0f} s")
        ax.set_xlabel("Posición axial z [m]")
        ax.set_ylabel(f"Conversión {reactant} [%]")
        ax.set_title(f"Perfil axial de conversión de {reactant} (1D)")
        ax.legend(fontsize=9)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ── summary dashboard ─────────────────────────────────────────────────────────

def plot_summary(
    reactor: SimpleNamespace,
    t_indices: list[int] | None = None,
    figsize: tuple = (14, 8),
) -> tuple[plt.Figure, np.ndarray]:
    """
    2×3 dashboard: temperatures, gas composition, pressure, velocities,
    and (if catalyst present) temperature gap Tg-Ts.

    Returns
    -------
    (fig, axes)  — axes.shape = (2, 3)
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    plot_temperatures(  reactor, t_indices=t_indices, ax=axes[0, 0])
    plot_gas_composition(reactor, t_indices=t_indices, ax=axes[0, 1])
    plot_pressure(      reactor,                       ax=axes[0, 2])
    plot_velocities(    reactor,                       ax=axes[1, 0])

    # Diferencia Tg-Ts si hay catalizador
    if reactor._Ts_results is not None:
        ax = axes[1, 1]
        t  = reactor._t_results
        N  = reactor._Tg_results.shape[1]
        dT = reactor._Tg_results - reactor._Ts_results   # (n_t, N)
        if N == 1:
            ax.plot(t, dT[:, 0], lw=2, color="crimson")
            ax.set_xlabel("Tiempo [s]")
        else:
            if t_indices is None:
                t_indices = _pick_t_indices(reactor)
            cmap = plt.cm.plasma
            for k, idx in enumerate(t_indices):
                color = cmap(k / max(len(t_indices) - 1, 1))
                ax.plot(reactor._z, dT[idx], color=color, lw=2,
                        label=f"t = {t[idx]:.0f} s")
            ax.set_xlabel("z [m]")
            ax.legend(fontsize=8)
        ax.axhline(0, ls="--", color="gray", alpha=0.5)
        ax.set_ylabel("$T_g - T_s$ [K]")
        ax.set_title("Diferencia gas-catalizador")
        ax.grid(True, alpha=0.3)
    else:
        axes[1, 1].set_visible(False)

    # axes[1, 2] reservado para uso externo
    axes[1, 2].set_visible(False)

    fig.tight_layout()
    return fig, axes
