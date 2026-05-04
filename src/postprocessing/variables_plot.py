"""
@module postprocessing.column_plots
@brief Visualización de resultados de simulación de columna 1D.

@details
Funciones standalone que reciben un objeto de unidad (columna, reactor, …)
y acceden a los atributos estandarizados de resultados:

    col._t_results   : (n_t,)       — vector de tiempos [s]
    col._z           : (N,)         — coordenadas axiales de centros de celda [m]
    col._species     : list[str]    — nombres de las nc especies
    col._P_results   : (n_t, N)     — presión [bar]
    col._Tg_results  : (n_t, N)     — temperatura del gas [K]
    col._Ts_results  : (n_t, N)     — temperatura del sólido [K]
    col._v_results   : (n_t, N)     — velocidad superficial [m/s]
    col._y_results   : (n_t, nc, N) — fracción molar [-]
    col._C_results   : (n_t, nc, N) — concentración molar [mol/m³]
    col._q_results   : (n_t, nc, N) — carga adsorbida [mol/kg]

Layout de cada figura:
    ┌────────┬──────────────────────┬────────┐
    │ Inlet  │   Perfil axial       │ Outlet │
    │ vs t   │   (varios tiempos)   │ vs t   │
    └────────┴──────────────────────┴────────┘
    Cols 1 y 4: series temporales en contorno real (inlet/outlet).
    Cols 2-3 fusionadas: perfiles axiales a los tiempos seleccionados,
        coloreados de azul (t=0) a rojo (t=final) usando el mapa jet.
    Leyenda única centrada en la parte inferior.

Selección de tiempos (plot_every):
    int        → cada N snapshots guardados
    list[int]  → índices exactos de snapshot
    list[float]→ valores de tiempo en segundos (se busca el más cercano)

Unidades: SI (ver tabla de atributos).
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers internos
# ═══════════════════════════════════════════════════════════════════════════════

_LINESTYLES: List[str] = ["-", "--", "-.", ":"]


def _parse_plot_every(
    plot_every: Union[int, List[int], List[float]],
    t_results: np.ndarray,
) -> List[int]:
    """
    Convierte plot_every a una lista ordenada de índices de snapshot.

    Parameters
    ----------
    plot_every  : int, list[int] o list[float]
        - int        → cada N snapshots (siempre incluye el último)
        - list[int]  → índices directos (0-based, dentro de rango)
        - list[float]→ tiempos en segundos; se selecciona el índice más cercano
    t_results   : ndarray (n_t,) — vector de tiempos del resultado

    Returns
    -------
    list[int] ordenado sin duplicados

    Raises
    ------
    TypeError  si el tipo de plot_every o de sus elementos no es el esperado
    ValueError si los valores están fuera de rango o la lista está vacía
    """
    n_t = len(t_results)

    # ── modo int: cada N snapshots ────────────────────────────────────────────
    if isinstance(plot_every, int) and not isinstance(plot_every, bool):
        if plot_every <= 0:
            raise ValueError("plot_every como int debe ser > 0")
        indices = list(range(0, n_t, plot_every))
        if (n_t - 1) not in indices:
            indices.append(n_t - 1)
        return indices

    # ── modo lista ────────────────────────────────────────────────────────────
    if isinstance(plot_every, (list, tuple)):
        if len(plot_every) == 0:
            raise ValueError("plot_every como lista no puede estar vacía")

        # Detectar tipo de los elementos
        all_strict_int  = all(
            isinstance(v, int) and not isinstance(v, bool)
            for v in plot_every
        )
        all_numeric = all(
            isinstance(v, (int, float)) and not isinstance(v, bool)
            for v in plot_every
        )

        if not all_numeric:
            raise TypeError(
                "Los elementos de plot_every deben ser int o float"
            )

        if all_strict_int:
            # list[int] → índices directos
            for idx in plot_every:
                if idx < 0 or idx >= n_t:
                    raise ValueError(
                        f"Índice {idx} fuera de rango [0, {n_t - 1}]"
                    )
            return sorted(set(int(v) for v in plot_every))

        else:
            # list[float] → tiempos en segundos
            indices = []
            for t_target in plot_every:
                idx = int(np.argmin(np.abs(t_results - float(t_target))))
                indices.append(idx)
            return sorted(set(indices))

    raise TypeError(
        "plot_every debe ser int, list[int] o list[float]; "
        f"se recibió {type(plot_every).__name__}"
    )


def _make_colors(n: int) -> List:
    """
    Genera n colores de frío (azul) a caliente (rojo) con el mapa jet.

    Con n == 1 devuelve azul (t=0).
    """
    if n <= 0:
        return []
    if n == 1:
        return [cm.jet(0.0)]
    return [cm.jet(v) for v in np.linspace(0.0, 1.0, n)]


def _build_figure(
    suptitle: str,
    figsize: Optional[Tuple[float, float]],
) -> Tuple[Figure, plt.Axes, plt.Axes, plt.Axes]:
    """
    Crea la figura con layout 4 columnas (col 1, cols 2-3 fusionadas, col 4).

    Returns
    -------
    fig, ax_inlet, ax_profile, ax_outlet
    """
    if figsize is None:
        figsize = (14, 4.0)

    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(
        1, 4,
        figure=fig,
        width_ratios=[1, 1.4, 1.4, 1],
        wspace=0.38,
    )
    ax_in  = fig.add_subplot(gs[0, 0])
    ax_pro = fig.add_subplot(gs[0, 1:3])
    ax_out = fig.add_subplot(gs[0, 3])

    fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.02)
    return fig, ax_in, ax_pro, ax_out


def _style_axes(
    ax_in:  plt.Axes,
    ax_pro: plt.Axes,
    ax_out: plt.Axes,
    ylabel: str,
    xlabel_z: str = "z [m]",
    xlabel_t: str = "t [s]",
) -> None:
    """Aplica etiquetas, títulos y rejilla a los tres ejes."""
    ax_in.set_title("Entrada",  fontsize=9)
    ax_pro.set_title("Perfil axial", fontsize=9)
    ax_out.set_title("Salida",  fontsize=9)

    ax_in.set_xlabel(xlabel_t, fontsize=8)
    ax_in.set_ylabel(ylabel,   fontsize=8)
    ax_pro.set_xlabel(xlabel_z, fontsize=8)
    ax_pro.set_ylabel(ylabel,   fontsize=8)
    ax_out.set_xlabel(xlabel_t, fontsize=8)
    ax_out.set_ylabel(ylabel,   fontsize=8)

    for ax in (ax_in, ax_pro, ax_out):
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.tick_params(labelsize=7)


def _finalize(
    fig:       Figure,
    handles:   list,
    labels:    list,
    save_path: Optional[str],
) -> Figure:
    """
    Añade leyenda compartida centrada en la parte inferior y guarda o muestra.

    Returns
    -------
    fig
    """
    n_col_legend = min(len(labels), 7)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=n_col_legend,
        frameon=True,
        fontsize=7.5,
        handlelength=1.8,
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.show()

    return fig


# ── Validación del objeto de unidad ──────────────────────────────────────────

def _require(col, attr: str) -> np.ndarray:
    """Comprueba que col tiene el atributo y lo devuelve como ndarray."""
    if not hasattr(col, attr):
        raise AttributeError(
            f"El objeto {type(col).__name__} no tiene el atributo '{attr}'. "
            f"Asegúrese de que la simulación ha terminado y de que el atributo "
            f"está definido como resultado estandarizado."
        )
    val = getattr(col, attr)
    if val is None:
        raise ValueError(
            f"El atributo '{attr}' es None. "
            f"Compruebe que la simulación se ha ejecutado correctamente."
        )
    return np.asarray(val, dtype=float)


# ═══════════════════════════════════════════════════════════════════════════════
# Funciones públicas — variables escalares
# ═══════════════════════════════════════════════════════════════════════════════

def Graph_P(
    col,
    plot_every: Union[int, List[int], List[float]] = 10,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """
    Gráfico de presión: serie temporal en entrada/salida + perfil axial.

    Parameters
    ----------
    col        : objeto de unidad con atributos estandarizados de resultados
    plot_every : int, list[int] o list[float] — tiempos a representar en el perfil
    save_path  : str opcional — ruta de guardado; si None, muestra la figura
    figsize    : tuple opcional — (ancho, alto) en pulgadas

    Returns
    -------
    (fig, (ax_inlet, ax_profile, ax_outlet))
    """
    t = _require(col, "_t_results")
    P = _require(col, "_P_results")   # (n_t, N)
    z = _require(col, "_z")           # (N,)

    idx    = _parse_plot_every(plot_every, t)
    colors = _make_colors(len(idx))

    fig, ax_in, ax_pro, ax_out = _build_figure("Presión", figsize)

    handles, labels = [], []
    for ii, color in zip(idx, colors):
        lbl = f"t = {t[ii]:.1f} s"
        ax_pro.plot(z, P[ii, :], color=color, linewidth=1.4)
        handles.append(mlines.Line2D([], [], color=color, linewidth=1.4))
        labels.append(lbl)

    # Inlet: presión upstream de la celda fantasma si está disponible (NaN en pasos sin entrada)
    if hasattr(col, "_P_in_results"):
        ax_in.plot(t, np.asarray(col._P_in_results, dtype=float), color="k", linewidth=1.2)
    else:
        ax_in.plot(t, P[:, 0], color="k", linewidth=1.2)
    ax_out.plot(t, P[:, -1], color="k", linewidth=1.2)

    # Marcadores de los tiempos seleccionados en las series temporales
    for ii, color in zip(idx, colors):
        ax_in.axvline(t[ii],  color=color, linewidth=0.7, alpha=0.7)
        ax_out.axvline(t[ii], color=color, linewidth=0.7, alpha=0.7)

    _style_axes(ax_in, ax_pro, ax_out, ylabel="P [bar]")
    fig = _finalize(fig, handles, labels, save_path)
    return fig, (ax_in, ax_pro, ax_out)


def Graph_Tg(
    col,
    plot_every: Union[int, List[int], List[float]] = 10,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """Gráfico de temperatura del gas [K]."""
    t  = _require(col, "_t_results")
    Tg = _require(col, "_Tg_results")  # (n_t, N)
    z  = _require(col, "_z")

    idx    = _parse_plot_every(plot_every, t)
    colors = _make_colors(len(idx))

    fig, ax_in, ax_pro, ax_out = _build_figure("Temperatura del gas", figsize)

    handles, labels = [], []
    for ii, color in zip(idx, colors):
        lbl = f"t = {t[ii]:.1f} s"
        ax_pro.plot(z, Tg[ii, :], color=color, linewidth=1.4)
        handles.append(mlines.Line2D([], [], color=color, linewidth=1.4))
        labels.append(lbl)

    # Inlet: temperatura real de la celda fantasma (NaN en pasos sin entrada)
    if hasattr(col, "_T_in_results"):
        ax_in.plot(t, np.asarray(col._T_in_results, dtype=float), color="k", linewidth=1.2)
    else:
        ax_in.plot(t, Tg[:, 0], color="k", linewidth=1.2)
    ax_out.plot(t, Tg[:, -1], color="k", linewidth=1.2)

    for ii, color in zip(idx, colors):
        ax_in.axvline(t[ii],  color=color, linewidth=0.7, alpha=0.7)
        ax_out.axvline(t[ii], color=color, linewidth=0.7, alpha=0.7)

    _style_axes(ax_in, ax_pro, ax_out, ylabel="Tg [K]")
    fig = _finalize(fig, handles, labels, save_path)
    return fig, (ax_in, ax_pro, ax_out)


def Graph_Ts(
    col,
    plot_every: Union[int, List[int], List[float]] = 10,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """Gráfico de temperatura del sólido [K]."""
    t  = _require(col, "_t_results")
    Ts = _require(col, "_Ts_results")  # (n_t, N)
    z  = _require(col, "_z")

    idx    = _parse_plot_every(plot_every, t)
    colors = _make_colors(len(idx))

    fig, ax_in, ax_pro, ax_out = _build_figure("Temperatura del sólido", figsize)

    handles, labels = [], []
    for ii, color in zip(idx, colors):
        lbl = f"t = {t[ii]:.1f} s"
        ax_pro.plot(z, Ts[ii, :], color=color, linewidth=1.4)
        handles.append(mlines.Line2D([], [], color=color, linewidth=1.4))
        labels.append(lbl)

    ax_in.plot(t, Ts[:, 0],  color="k", linewidth=1.2)
    ax_out.plot(t, Ts[:, -1], color="k", linewidth=1.2)

    for ii, color in zip(idx, colors):
        ax_in.axvline(t[ii],  color=color, linewidth=0.7, alpha=0.7)
        ax_out.axvline(t[ii], color=color, linewidth=0.7, alpha=0.7)

    _style_axes(ax_in, ax_pro, ax_out, ylabel="Ts [K]")
    fig = _finalize(fig, handles, labels, save_path)
    return fig, (ax_in, ax_pro, ax_out)


def Graph_Tw(
    col,
    plot_every: Union[int, List[int], List[float]] = 10,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """Gráfico de temperatura de la pared [K] — solo disponible con shell_tube activo."""
    t  = _require(col, "_t_results")
    Tw = _require(col, "_Tw_results")  # (n_t, N)
    z  = _require(col, "_z")

    idx    = _parse_plot_every(plot_every, t)
    colors = _make_colors(len(idx))

    fig, ax_in, ax_pro, ax_out = _build_figure("Temperatura de la pared", figsize)

    handles, labels = [], []
    for ii, color in zip(idx, colors):
        lbl = f"t = {t[ii]:.1f} s"
        ax_pro.plot(z, Tw[ii, :], color=color, linewidth=1.4)
        handles.append(mlines.Line2D([], [], color=color, linewidth=1.4))
        labels.append(lbl)

    ax_in.plot(t, Tw[:, 0],  color="k", linewidth=1.2)
    ax_out.plot(t, Tw[:, -1], color="k", linewidth=1.2)

    for ii, color in zip(idx, colors):
        ax_in.axvline(t[ii],  color=color, linewidth=0.7, alpha=0.7)
        ax_out.axvline(t[ii], color=color, linewidth=0.7, alpha=0.7)

    _style_axes(ax_in, ax_pro, ax_out, ylabel="Tw [K]")
    fig = _finalize(fig, handles, labels, save_path)
    return fig, (ax_in, ax_pro, ax_out)


def Graph_TgTw(
    col,
    plot_every: Union[int, List[int], List[float]] = 10,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """
    Gráfico combinado de temperatura del gas (línea sólida) y de la pared
    (línea discontinua) sobre los mismos ejes [K]. Solo con shell_tube activo.
    """
    t  = _require(col, "_t_results")
    Tg = _require(col, "_Tg_results")  # (n_t, N)
    Tw = _require(col, "_Tw_results")  # (n_t, N)
    z  = _require(col, "_z")

    idx    = _parse_plot_every(plot_every, t)
    colors = _make_colors(len(idx))

    fig, ax_in, ax_pro, ax_out = _build_figure("Temperatura gas (—) y pared (--)", figsize)

    handles, labels = [], []
    for ii, color in zip(idx, colors):
        lbl = f"t = {t[ii]:.1f} s"
        ax_pro.plot(z, Tg[ii, :], color=color, linestyle="-",  linewidth=1.4)
        ax_pro.plot(z, Tw[ii, :], color=color, linestyle="--", linewidth=1.1)
        handles.append(mlines.Line2D([], [], color=color, linewidth=1.4))
        labels.append(lbl)

    if hasattr(col, "_T_in_results"):
        ax_in.plot(t, np.asarray(col._T_in_results, dtype=float), color="k", linewidth=1.2)
    else:
        ax_in.plot(t, Tg[:, 0], color="k", linewidth=1.2)
    ax_in.plot(t,  Tw[:, 0],  color="k", linestyle="--", linewidth=1.0)
    ax_out.plot(t, Tg[:, -1], color="k", linewidth=1.2)
    ax_out.plot(t, Tw[:, -1], color="k", linestyle="--", linewidth=1.0)

    for ii, color in zip(idx, colors):
        ax_in.axvline(t[ii],  color=color, linewidth=0.7, alpha=0.7)
        ax_out.axvline(t[ii], color=color, linewidth=0.7, alpha=0.7)

    # Leyenda: tiempos + separador + tipo de línea
    handles += [
        mlines.Line2D([], [], color="none"),
        mlines.Line2D([], [], color="k", linestyle="-",  linewidth=1.4),
        mlines.Line2D([], [], color="k", linestyle="--", linewidth=1.1),
    ]
    labels += ["", "Tg [K]", "Tw [K]"]

    _style_axes(ax_in, ax_pro, ax_out, ylabel="T [K]")
    fig = _finalize(fig, handles, labels, save_path)
    return fig, (ax_in, ax_pro, ax_out)


def Graph_v(
    col,
    plot_every: Union[int, List[int], List[float]] = 10,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """
    Gráfico de velocidad superficial del gas [m/s].

    Nota: los valores negativos indican flujo en dirección −z.
    """
    t = _require(col, "_t_results")
    v = _require(col, "_v_results")   # (n_t, N)
    z = _require(col, "_z")

    idx    = _parse_plot_every(plot_every, t)
    colors = _make_colors(len(idx))

    fig, ax_in, ax_pro, ax_out = _build_figure("Velocidad superficial del gas", figsize)

    handles, labels = [], []
    for ii, color in zip(idx, colors):
        lbl = f"t = {t[ii]:.1f} s"
        ax_pro.plot(z, v[ii, :], color=color, linewidth=1.4)
        handles.append(mlines.Line2D([], [], color=color, linewidth=1.4))
        labels.append(lbl)

    # Velocidades exactas en cara de contorno (face 0 y face N)
    if hasattr(col, "_v_in_results"):
        ax_in.plot(t, np.asarray(col._v_in_results, dtype=float),  color="k", linewidth=1.2)
    else:
        ax_in.plot(t, v[:, 0], color="k", linewidth=1.2)
    if hasattr(col, "_v_out_results"):
        ax_out.plot(t, np.asarray(col._v_out_results, dtype=float), color="k", linewidth=1.2)
    else:
        ax_out.plot(t, v[:, -1], color="k", linewidth=1.2)

    for ii, color in zip(idx, colors):
        ax_in.axvline(t[ii],  color=color, linewidth=0.7, alpha=0.7)
        ax_out.axvline(t[ii], color=color, linewidth=0.7, alpha=0.7)

    # Línea de cero como referencia
    ax_pro.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")

    _style_axes(ax_in, ax_pro, ax_out, ylabel="v [m/s]")
    fig = _finalize(fig, handles, labels, save_path)
    return fig, (ax_in, ax_pro, ax_out)


# ═══════════════════════════════════════════════════════════════════════════════
# Funciones públicas — variables multicomponente
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_multicomp(
    col,
    attr: str,
    suptitle: str,
    ylabel: str,
    plot_every: Union[int, List[int], List[float]],
    log_scale: bool,
    save_path: Optional[str],
    figsize: Optional[Tuple[float, float]],
    inlet_data: Optional[np.ndarray] = None,
) -> Tuple[Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """
    Núcleo común para variables multicomponente (y, C, q).

    Colores  = tiempo (azul→rojo, mapa jet).
    Trazos   = especie (-, --, -., :).

    La leyenda combina entradas de tiempo y entradas de especie.
    """
    t      = _require(col, "_t_results")
    data   = _require(col, attr)          # (n_t, nc, N)
    z      = _require(col, "_z")

    if not hasattr(col, "_species") or col._species is None:
        nc = data.shape[1]
        species = [f"comp_{i}" for i in range(nc)]
    else:
        species = list(col._species)
        nc = len(species)

    if data.ndim != 3 or data.shape[1] != nc:
        raise ValueError(
            f"'{attr}' debe tener shape (n_t, {nc}, N); "
            f"se encontró {data.shape}"
        )

    idx    = _parse_plot_every(plot_every, t)
    colors = _make_colors(len(idx))

    fig, ax_in, ax_pro, ax_out = _build_figure(suptitle, figsize)

    if log_scale:
        ax_in.set_yscale("log")
        ax_pro.set_yscale("log")
        ax_out.set_yscale("log")

    # ── perfil axial: color=tiempo, linestyle=especie ─────────────────────────
    for ii, color in zip(idx, colors):
        for s, ls in zip(range(nc), _LINESTYLES):
            ax_pro.plot(
                z,
                data[ii, s, :],
                color=color,
                linestyle=ls,
                linewidth=1.3,
            )

    # ── series temporales en inlet/outlet ─────────────────────────────────────
    for s, ls in zip(range(nc), _LINESTYLES):
        in_vals = inlet_data[:, s] if inlet_data is not None else data[:, s, 0]
        ax_in.plot(t,  in_vals,         color="k", linestyle=ls, linewidth=1.1)
        ax_out.plot(t, data[:, s, -1],  color="k", linestyle=ls, linewidth=1.1)

    # Marcadores de tiempos seleccionados
    for ii, color in zip(idx, colors):
        ax_in.axvline(t[ii],  color=color, linewidth=0.7, alpha=0.7)
        ax_out.axvline(t[ii], color=color, linewidth=0.7, alpha=0.7)

    # ── leyenda: bloque de tiempos + bloque de especies ───────────────────────
    handles, labels = [], []

    # Entradas de tiempo (líneas coloreadas sólidas)
    for ii, color in zip(idx, colors):
        handles.append(
            mlines.Line2D([], [], color=color, linestyle="-", linewidth=1.4)
        )
        labels.append(f"t = {t[ii]:.1f} s")

    # Separador visual
    handles.append(mlines.Line2D([], [], color="none"))
    labels.append("")

    # Entradas de especie (líneas negras con linestyle)
    for s, sp_name in enumerate(species):
        ls = _LINESTYLES[s % len(_LINESTYLES)]
        handles.append(
            mlines.Line2D([], [], color="k", linestyle=ls, linewidth=1.4)
        )
        labels.append(sp_name)

    _style_axes(ax_in, ax_pro, ax_out, ylabel=ylabel)
    fig = _finalize(fig, handles, labels, save_path)
    return fig, (ax_in, ax_pro, ax_out)


def Graph_y(
    col,
    plot_every: Union[int, List[int], List[float]] = 10,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """Gráfico de fracción molar por especie [-]."""
    # Fracción molar en inlet derivada de la celda fantasma de concentración
    inlet_data = None
    if hasattr(col, "_C_in_results"):
        C_in = np.asarray(col._C_in_results, dtype=float)   # (n_t, nc)
        C_in_sum = C_in.sum(axis=1, keepdims=True)           # (n_t, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            inlet_data = np.where(C_in_sum > 0.0, C_in / C_in_sum, np.nan)
    return _plot_multicomp(
        col, "_y_results",
        suptitle="Fracción molar",
        ylabel="y [-]",
        plot_every=plot_every,
        log_scale=False,
        save_path=save_path,
        figsize=figsize,
        inlet_data=inlet_data,
    )


def Graph_C(
    col,
    plot_every: Union[int, List[int], List[float]] = 10,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    log_scale: bool = False,
) -> Tuple[Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """Gráfico de concentración molar por especie [mol/m³]."""
    inlet_data = None
    if hasattr(col, "_C_in_results"):
        inlet_data = np.asarray(col._C_in_results, dtype=float)  # (n_t, nc)
    return _plot_multicomp(
        col, "_C_results",
        suptitle="Concentración molar",
        ylabel="C [mol/m³]",
        plot_every=plot_every,
        log_scale=log_scale,
        save_path=save_path,
        figsize=figsize,
        inlet_data=inlet_data,
    )


def Graph_q(
    col,
    plot_every: Union[int, List[int], List[float]] = 10,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """Gráfico de carga adsorbida por especie [mol/kg]."""
    return _plot_multicomp(
        col, "_q_results",
        suptitle="Carga adsorbida",
        ylabel="q [mol/kg]",
        plot_every=plot_every,
        log_scale=False,
        save_path=save_path,
        figsize=figsize,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Funciones públicas — variables por especie individual
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_species_slice(
    col,
    attr: str,
    species_idx: int,
    suptitle_base: str,
    ylabel: str,
    plot_every: Union[int, List[int], List[float]],
    log_scale: bool,
    save_path: Optional[str],
    figsize: Optional[Tuple[float, float]],
    inlet_val=None,
) -> Tuple[Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """
    Núcleo común para gráficos de una única especie (y_i, C_i, q_i).

    El perfil axial muestra la variable a los tiempos seleccionados,
    coloreados de azul (t=0) a rojo (t=final) con el mapa jet.
    Las series temporales en inlet/outlet muestran solo la especie indicada.

    Parameters
    ----------
    col         : objeto de unidad con atributos estandarizados
    attr        : str — nombre del atributo (n_t, nc, N) en col
    species_idx : int — índice de la especie (0-based)
    suptitle_base : str — título base; se añade el nombre de la especie
    ylabel      : str — etiqueta del eje y
    plot_every  : int, list[int] o list[float]
    log_scale   : bool — escala logarítmica en y
    save_path   : str opcional
    figsize     : tuple opcional

    Returns
    -------
    (fig, (ax_inlet, ax_profile, ax_outlet))
    """
    t    = _require(col, "_t_results")
    data = _require(col, attr)          # (n_t, nc, N)
    z    = _require(col, "_z")

    if not hasattr(col, "_species") or col._species is None:
        nc      = data.shape[1]
        species = [f"comp_{i}" for i in range(nc)]
    else:
        species = list(col._species)
        nc      = len(species)

    if data.ndim != 3 or data.shape[1] != nc:
        raise ValueError(
            f"'{attr}' debe tener shape (n_t, {nc}, N); "
            f"se encontró {data.shape}"
        )
    if species_idx < 0 or species_idx >= nc:
        raise ValueError(
            f"species_idx={species_idx} fuera de rango [0, {nc - 1}]"
        )

    sp_name = species[species_idx]
    idx     = _parse_plot_every(plot_every, t)
    colors  = _make_colors(len(idx))

    fig, ax_in, ax_pro, ax_out = _build_figure(
        f"{suptitle_base} — {sp_name}", figsize
    )

    if log_scale:
        ax_in.set_yscale("log")
        ax_pro.set_yscale("log")
        ax_out.set_yscale("log")

    # Perfil axial: un trazo por tiempo seleccionado, coloreado por jet
    handles, labels = [], []
    for ii, color in zip(idx, colors):
        lbl = f"t = {t[ii]:.1f} s"
        ax_pro.plot(z, data[ii, species_idx, :], color=color, linewidth=1.4)
        handles.append(mlines.Line2D([], [], color=color, linewidth=1.4))
        labels.append(lbl)

    # Series temporales en inlet (contorno real o nodo 0) y outlet (nodo -1)
    in_vals = inlet_val if inlet_val is not None else data[:, species_idx, 0]
    ax_in.plot(t,  in_vals,                   color="k", linewidth=1.2)
    ax_out.plot(t, data[:, species_idx, -1],  color="k", linewidth=1.2)

    # Marcadores de tiempos seleccionados
    for ii, color in zip(idx, colors):
        ax_in.axvline(t[ii],  color=color, linewidth=0.7, alpha=0.7)
        ax_out.axvline(t[ii], color=color, linewidth=0.7, alpha=0.7)

    _style_axes(ax_in, ax_pro, ax_out, ylabel=ylabel)
    fig = _finalize(fig, handles, labels, save_path)
    return fig, (ax_in, ax_pro, ax_out)


def Graph_y_i(
    col,
    species_idx: int = 0,
    plot_every: Union[int, List[int], List[float]] = 10,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """
    Gráfico de fracción molar para una única especie [-].

    Parameters
    ----------
    col         : objeto de unidad con atributos estandarizados
    species_idx : int — índice de la especie a representar (0-based)
    plot_every  : int, list[int] o list[float]
    save_path   : str opcional
    figsize     : tuple opcional
    """
    inlet_val = None
    if hasattr(col, "_C_in_results"):
        C_in = np.asarray(col._C_in_results, dtype=float)   # (n_t, nc)
        C_in_sum = C_in.sum(axis=1)                          # (n_t,)
        with np.errstate(invalid="ignore", divide="ignore"):
            inlet_val = np.where(
                C_in_sum > 0.0,
                C_in[:, species_idx] / C_in_sum,
                np.nan,
            )
    return _plot_species_slice(
        col, "_y_results", species_idx,
        suptitle_base="Fracción molar",
        ylabel="y [-]",
        plot_every=plot_every,
        log_scale=False,
        save_path=save_path,
        figsize=figsize,
        inlet_val=inlet_val,
    )


def Graph_C_i(
    col,
    species_idx: int = 0,
    plot_every: Union[int, List[int], List[float]] = 10,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    log_scale: bool = False,
) -> Tuple[Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """
    Gráfico de concentración molar para una única especie [mol/m³].

    Parameters
    ----------
    col         : objeto de unidad con atributos estandarizados
    species_idx : int — índice de la especie a representar (0-based)
    plot_every  : int, list[int] o list[float]
    save_path   : str opcional
    figsize     : tuple opcional
    log_scale   : bool — usar escala logarítmica en y (útil para trazas)
    """
    inlet_val = None
    if hasattr(col, "_C_in_results"):
        C_in = np.asarray(col._C_in_results, dtype=float)
        inlet_val = C_in[:, species_idx]
    return _plot_species_slice(
        col, "_C_results", species_idx,
        suptitle_base="Concentración molar",
        ylabel="C [mol/m³]",
        plot_every=plot_every,
        log_scale=log_scale,
        save_path=save_path,
        figsize=figsize,
        inlet_val=inlet_val,
    )


def Graph_q_i(
    col,
    species_idx: int = 0,
    plot_every: Union[int, List[int], List[float]] = 10,
    save_path: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """
    Gráfico de carga adsorbida para una única especie [mol/kg].

    Parameters
    ----------
    col         : objeto de unidad con atributos estandarizados
    species_idx : int — índice de la especie a representar (0-based)
    plot_every  : int, list[int] o list[float]
    save_path   : str opcional
    figsize     : tuple opcional
    """
    return _plot_species_slice(
        col, "_q_results", species_idx,
        suptitle_base="Carga adsorbida",
        ylabel="q [mol/kg]",
        plot_every=plot_every,
        log_scale=False,
        save_path=save_path,
        figsize=figsize,
    )
