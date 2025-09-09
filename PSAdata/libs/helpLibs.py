# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 16:53:20 2025

@author: MiguelCamaraSanz
"""
import re,math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import dates as mdates



# (Constantes — aún no usadas, pero quedarán aquí para pasos posteriores)
R = 8.314462618 # J/(mol·K)
BAR_TO_PA = 1e5 # 1 bar = 1e5 Pa
MOL_PER_NM3 = 44.615 # mol por Nm³ (0°C, 1 atm)
# colores consistentes con el resto
COLORS = {
    "PRZ":   "#27ae60",  # verde
    "FEED":  "#e74c3c",  # rojo
    "BWD":   "#3498db",  # azul
    "PURGE": "#9b59b6",  # morado
}

STEP_COLORS = {
    "FEED":  "#e74c3c",
    "PRZ":   "#27ae60",
    "BWD":   "#3498db",
    "PURGE": "#9b59b6",
    "WAIT":  "#7f8c8d",
}

def getPSAdata(
    df_or_path,
    start: str | None = None,
    end:   str | None = None,
    *,
    # Steps / partners
    P_high: float = 7.0,
    P_low:  float = 1.0,
    dpos_thr: float = 0.010,     # bar/s → PRZ
    dneg_thr: float = -0.010,    # bar/s → BWD
    dflat_thr: float = 0.00999,  # |dP/dt| < dflat_thr → meseta
    eps: float = 0.02,           # tolerancia de P para emparejar PRZ/BWD
    prefer_neighbors: bool = True,
    min_seg_s: float = 15.0,     # fusiona “chispazos” de step
    smooth_substeps_s: float = 5.0,  # fusiona “micro-cortes” en substeps
    # Datos medidos
    keep_nans: bool = True,      # <<< NO convertir NaN a 0 en balances
    return_copy: bool = True,
):
    """
    Devuelve (df, meta) con:
      - DateTime (recortado a [start,end])
      - t_s, x01
      - Renombrados FEED_*, PRODUCT_AF_BYPASS_*, BYPASS_*, BED_DP_#
      - Flujos derivados:
          PRODUCT_BF_BYPASS_Q = PRODUCT_AF_BYPASS_Q - BYPASS_Q
          TAIL_AF_BYPASS_Q    = FEED_Q - PRODUCT_AF_BYPASS_Q
          TAIL_BF_BYPASS_Q    = TAIL_AF_BYPASS_Q + BYPASS_Q
      - b#_step ∈ {PRZ, FEED, BWD, PURGE, WAIT}
      - b#_partner ∈ {'FEED','BLOW','B#',''}
      - b#_inlet_mask, b#_outlet_mask
      - Válvulas booleanas:
          b#_Vfeed, b#_Vpurge, b#_Vprz_BX, b#_Vbwd_BX
    """

    # ---------------------
    # 0) Cargar / acotar
    # ---------------------
    if isinstance(df_or_path, (str, bytes, bytearray)):
        df = pd.read_csv(df_or_path, parse_dates=["DateTime"])
    else:
        df = df_or_path.copy() if return_copy else df_or_path

    if "DateTime" not in df.columns:
        raise ValueError("No se encuentra la columna 'DateTime' en el DataFrame.")

    if start is not None:
        df = df[df["DateTime"] >= pd.to_datetime(start)]
    if end is not None:
        df = df[df["DateTime"] <= pd.to_datetime(end)]
    df = df.sort_values("DateTime").reset_index(drop=True)
    if df.empty:
        raise ValueError("No hay datos en el rango solicitado.")

    # Auxiliares de tiempo
    t = pd.to_datetime(df["DateTime"])
    t_s = (t - t.iloc[0]).dt.total_seconds().astype(float)
    span = float(max(t_s.iloc[-1], 1e-12))
    df["t_s"] = t_s
    df["x01"] = t_s / span
    dt_s = float(pd.Series(t).diff().dt.total_seconds().median() or 10.0)

    # ---------------------
    # 1) Renombrados
    # ---------------------
    rename_map = {}
    # FEED
    rename_map.update({
        "PRC02_1_PI11101_Val": "FEED_P",
        "SIS_TI11102_Val":     "FEED_T",
        "PRC02_1_FY12504_Val": "FEED_Q",
    })
    # PRODUCT (medido DESPUÉS del bypass)
    rename_map.update({
        "PRC02_1_TI12501_Val": "PRODUCT_AF_BYPASS_T",
        "PRC02_1_PI12503_Val": "PRODUCT_AF_BYPASS_P",
        "PRC02_1_FY12505_Val": "PRODUCT_AF_BYPASS_Q",
    })
    # BYPASS
    rename_map.update({
        "PSA_FI12599B_Val":    "BYPASS_Q",
        "PRC02_1_PI12502_Val": "BYPASS_P",
        "PRC02_1_TI12502_Val": "BYPASS_T",
    })
    # Especies FEED
    for col in df.columns:
        m = re.match(r"^PRC02_2_AI11101_([A-Za-z0-9]+)_Val$", col)
        if m:
            rename_map[col] = f"FEED_{m.group(1).upper()}"
    # Especies PRODUCT (AFTER bypass)
    for col in df.columns:
        m = re.match(r"^PRC02_2_AI12502_([A-Za-z0-9]+)_Val$", col)
        if m:
            rename_map[col] = f"PRODUCT_AF_BYPASS_{m.group(1).upper()}"
    # Presiones lechos → BED_DP_#
    bed_tags_raw = [
        "PSA_PI12518_Val","PSA_PI12528_Val","PSA_PI12538_Val","PSA_PI12548_Val",
        "PSA_PI12558_Val","PSA_PI12568_Val","PSA_PI12578_Val","PSA_PI12588_Val",
    ]
    for i, tag in enumerate(bed_tags_raw, start=1):
        if tag in df.columns:
            rename_map[tag] = f"BED_DP_{i}"

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # ---------------------
    # 2) Balances (NaN se conserva si keep_nans=True)
    # ---------------------
    def _num(col):
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns \
               else pd.Series(np.nan, index=df.index, dtype=float)

    FEED_Q  = _num("FEED_Q")
    PROD_AF = _num("PRODUCT_AF_BYPASS_Q")
    BYP_Q   = _num("BYPASS_Q")

    if keep_nans:
        df["PRODUCT_BF_BYPASS_Q"] = PROD_AF - BYP_Q
        df["TAIL_AF_BYPASS_Q"]    = FEED_Q  - PROD_AF
        df["TAIL_BF_BYPASS_Q"]    = df["TAIL_AF_BYPASS_Q"] + BYP_Q
    else:
        df["PRODUCT_BF_BYPASS_Q"] = PROD_AF.fillna(0) - BYP_Q.fillna(0)
        df["TAIL_AF_BYPASS_Q"]    = FEED_Q.fillna(0)  - PROD_AF.fillna(0)
        df["TAIL_BF_BYPASS_Q"]    = df["TAIL_AF_BYPASS_Q"].fillna(0) + BYP_Q.fillna(0)

    # ---------------------
    # 3) Steps por cama
    # ---------------------
    bed_cols = [c for c in df.columns if re.fullmatch(r"BED_DP_\d+", c)]
    bed_cols.sort(key=lambda c: int(c.split("_")[-1]))
    if not bed_cols:
        raise ValueError("No hay presiones de lechos (BED_DP_#).")

    bed_code     = {tag: f"b{i+1}" for i, tag in enumerate(bed_cols)}
    bed_code_cap = {tag: f"B{i+1}" for i, tag in enumerate(bed_cols)}
    idx_of       = {tag: i for i, tag in enumerate(bed_cols)}

    def _rle_merge_short(labels: np.ndarray, times: pd.Series, min_s: float) -> np.ndarray:
        if min_s <= 0 or len(labels) <= 1:
            return labels
        segs = []
        s = 0
        for i in range(1, len(labels)+1):
            if i == len(labels) or labels[i] != labels[s]:
                segs.append([s, i]); s = i
        def dur(i0, i1):
            t0 = times.iloc[i0]; t1 = times.iloc[i1] if i1 < len(times) else times.iloc[-1]
            return (t1 - t0).total_seconds()
        changed = True
        while changed and len(segs) > 1:
            changed = False
            for k, (i0, i1) in enumerate(segs):
                if dur(i0, i1) < min_s:
                    if k == 0:
                        segs[k+1][0] = i0
                    elif k == len(segs)-1:
                        segs[k-1][1] = i1
                    else:
                        dprev = dur(segs[k-1][0], segs[k-1][1])
                        dnext = dur(segs[k+1][0], segs[k+1][1])
                        (segs[k+1][0] if dnext >= dprev else segs[k-1].__setitem__(1, i1))
                    del segs[k]; changed = True; break
        out = labels.copy()
        for i0, i1 in segs:
            out[i0:i1] = labels[i0]
        return out

    labels = {}
    for tag in bed_cols:
        P = pd.to_numeric(df[tag], errors="coerce").astype(float).values
        dPdt = np.r_[0.0, np.diff(P)] / max(dt_s, 1e-6)

        lab = np.empty(len(P), dtype=object); lab[:] = "WAIT"
        lab[dPdt >= dpos_thr] = "PRZ"
        lab[dPdt <= dneg_thr] = "BWD"
        flat = (np.abs(dPdt) < dflat_thr)
        lab[flat & (P >= P_high)] = "FEED"
        lab[flat & (P <= P_low )] = "PURGE"
        lab[flat & (P >  P_low) & (P <  P_high)] = "WAIT"

        lab = _rle_merge_short(lab, t, min_seg_s)   # alisado anti-chispazos
        labels[tag] = lab
        df[f"{bed_code[tag]}_step"] = lab

    # ---------------------
    # 4) Partners instante a instante
    # ---------------------
    Pmat = np.vstack([pd.to_numeric(df[tag], errors="coerce").astype(float).values for tag in bed_cols])
    N = Pmat.shape[1]
    partners = {tag: np.array([""]*N, dtype=object) for tag in bed_cols}

    for i in range(N):
        mode = {tag: labels[tag][i] for tag in bed_cols}
        in_PRZ = [tag for tag in bed_cols if mode[tag] == "PRZ"]
        in_BWD = [tag for tag in bed_cols if mode[tag] == "BWD"]

        # PRZ: donante (BWD) con P mayor; si no hay → FEED
        for tag in in_PRZ:
            k = idx_of[tag]; Pk = Pmat[k, i]
            cands = [(bj, Pmat[idx_of[bj], i]) for bj in in_BWD if Pmat[idx_of[bj], i] > Pk + eps]
            if cands:
                if prefer_neighbors:
                    bj = max(cands, key=lambda it: (it[1], -abs(idx_of[it[0]]-k)))[0]
                else:
                    bj = max(cands, key=lambda it: it[1])[0]
                partners[tag][i] = bed_code_cap[bj]
            else:
                partners[tag][i] = "FEED"

        # BWD: receptor (PRZ) con P menor; si no hay → BLOW
        for tag in in_BWD:
            k = idx_of[tag]; Pk = Pmat[k, i]
            cands = [(bj, Pmat[idx_of[bj], i]) for bj in in_PRZ if Pmat[idx_of[bj], i] < Pk - eps]
            if cands:
                if prefer_neighbors:
                    bj = min(cands, key=lambda it: (it[1],  abs(idx_of[it[0]]-k)))[0]
                else:
                    bj = min(cands, key=lambda it: it[1])[0]
                partners[tag][i] = bed_code_cap[bj]
            else:
                partners[tag][i] = "BLOW"

    # Suavizar micro-cortes de substeps (PRZ/BWD con el mismo partner)
    if smooth_substeps_s > 0:
        for tag in bed_cols:
            st = df[f"{bed_code[tag]}_step"].values
            pn = partners[tag].copy()
            # combinamos por (step, partner) con RLE y umbral
            key = np.array([f"{st_i}|{pn_i}" for st_i, pn_i in zip(st, pn)], dtype=object)
            key2 = _rle_merge_short(key, t, smooth_substeps_s)
            partners[tag] = np.array([k.split("|", 1)[1] if "|" in k else "" for k in key2], dtype=object)

    # ---------------------
    # 5) Volcar partner + IN/OUT + válvulas (sin fragmentar DF)
    # ---------------------
    add_cols = {}

    bed_caps = [f"B{i+1}" for i in range(len(bed_cols))]
    for tag in bed_cols:
        b = bed_code[tag]  # 'b1', ...
        step = df[f"{b}_step"].astype(object).values
        part = np.where(np.isin(step, ["PRZ","BWD"]), partners[tag], "")
        add_cols[f"{b}_partner"] = part

        in_mask  = ((step == "PRZ") & (part == "FEED")) | (step == "FEED")
        out_mask = ((step == "BWD") & (part == "BLOW")) | (step == "PURGE")
        add_cols[f"{b}_inlet_mask"]  = in_mask
        add_cols[f"{b}_outlet_mask"] = out_mask

        # (opcionales) etiquetas literales por substep
        in_kind  = np.where((step == "PRZ") & (part == "FEED"), "PRZ_FEED",
                     np.where(step == "FEED", "FEED", ""))
        out_kind = np.where((step == "BWD") & (part == "BLOW"), "BWD_PURGE",
                     np.where(step == "PURGE", "PURGE", ""))
        add_cols[f"{b}_in_kind"]  = in_kind
        add_cols[f"{b}_out_kind"] = out_kind

        # Válvulas
        add_cols[f"{b}_Vfeed"]  = ((step == "FEED") | ((step == "PRZ") & (part == "FEED"))).astype(bool)
        add_cols[f"{b}_Vpurge"] = ((step == "PURGE")| ((step == "BWD") & (part == "BLOW"))).astype(bool)
        for Bcap in bed_caps:
            if Bcap == f"B{int(b[1:])}":
                continue
            add_cols[f"{b}_Vprz_{Bcap}"] = ((step == "PRZ") & (part == Bcap)).astype(bool)
            add_cols[f"{b}_Vbwd_{Bcap}"] = ((step == "BWD") & (part == Bcap)).astype(bool)

    # Añadir todas de golpe para evitar fragmentación
    df = pd.concat([df, pd.DataFrame(add_cols, index=df.index)], axis=1)

    # ---------------------
    # 6) Meta mínima
    # ---------------------
    meta = dict(
        time_col="DateTime",
        dt_s=dt_s,
        beds=[bed_code[tag] for tag in bed_cols],
        bed_code=bed_code,
        bed_caps=bed_caps,
        params=dict(P_high=P_high, P_low=P_low, dpos_thr=dpos_thr, dneg_thr=dneg_thr,
                    dflat_thr=dflat_thr, eps=eps, prefer_neighbors=prefer_neighbors,
                    min_seg_s=min_seg_s, smooth_substeps_s=smooth_substeps_s,
                    keep_nans=keep_nans),
        n_rows=len(df),
    )

    return df, meta

def plot_raw_bed_pressure(
    df: pd.DataFrame,
    start: str|None = None,
    end:   str|None = None,
    *,
    columns: str | list[str] = "all",   # "all" o ["b1","b3",...]
    width: float = 12.0,
    ax: plt.Axes | None = None,
):
    # recorte temporal
    d = df
    if start is not None:
        d = d.loc[d["DateTime"] >= pd.to_datetime(start)]
    if end is not None:
        d = d.loc[d["DateTime"] <= pd.to_datetime(end)]
    d = d.reset_index(drop=True)
    if d.empty:
        raise ValueError("No hay datos en el rango solicitado.")

    # detectar columnas BED_DP_#
    bed_cols = [c for c in d.columns if re.fullmatch(r"BED_DP_\d+", c)]
    bed_cols = sorted(bed_cols, key=lambda c: int(c.split("_")[-1]))

    # filtrar por 'columns' (b1,b2,...)
    if isinstance(columns, str) and columns.lower() != "all":
        wanted = [s.strip().lower() for s in columns.split(",") if s.strip()]
        keep = []
        for w in wanted:
            m = re.fullmatch(r"b(\d+)", w)
            if m:
                tag = f"BED_DP_{int(m.group(1))}"
                if tag in bed_cols: keep.append(tag)
        bed_cols = keep
    elif isinstance(columns, list):
        keep = []
        for w in columns:
            m = re.fullmatch(r"b(\d+)", str(w).lower())
            if m:
                tag = f"BED_DP_{int(m.group(1))}"
                if tag in bed_cols: keep.append(tag)
        if keep: bed_cols = keep

    # crear eje si no lo pasan
    created_fig = False
    if ax is None:
        height = 2.2 + 0.7*max(1, len(bed_cols))
        fig, ax = plt.subplots(figsize=(32, 9))
        created_fig = True

    for tag in bed_cols:
        y = pd.to_numeric(d[tag], errors="coerce")
        ax.plot(d["DateTime"], y, linewidth=1.2, label=tag)

    ax.set_ylabel("Presión (bar)")
    ax.set_title("Presión en lechos")
    if bed_cols:
        ax.legend(ncol=2, fontsize=8)

    if created_fig:
        fig.tight_layout()
        plt.show()

    return ax

def plot_raw_header_pressures(
    df: pd.DataFrame,
    start: str|None = None,
    end:   str|None = None,
    *,
    width: float = 12.0,
):
    dfx = df.copy()
    if start is not None: dfx = dfx[dfx["DateTime"] >= pd.to_datetime(start)]
    if end   is not None: dfx = dfx[dfx["DateTime"] <= pd.to_datetime(end)]
    dfx = dfx.reset_index(drop=True)

    series = [
        ("FEED_P",               "FEED_P"),
        ("PRODUCT_AF_BYPASS_P",  "PRODUCT_P"),
        ("BYPASS_P",             "BYPASS_P"),
    ]
    present = [(c, name) for c, name in series if c in dfx.columns]
    if not present:
        raise ValueError("No hay columnas de presión: FEED_P / PRODUCT_AF_BYPASS_P / BYPASS_P.")

    height = 2.2 + 0.7*len(present)
    fig, ax = plt.subplots(figsize=(width, height))
    t = pd.to_datetime(dfx["DateTime"])
    for col, name in present:
        y = pd.to_numeric(dfx[col], errors="coerce")
        ax.plot(t, y, linewidth=1.2, label=name)

    ax.set_xlabel("DateTime")
    ax.set_ylabel("Presión (bar)")
    ax.set_title("Presiones — FEED / PRODUCT / BYPASS")
    ax.legend(ncol=2, fontsize=9)
    fig.autofmt_xdate(); fig.tight_layout(); plt.show()

def plot_raw_header_temperatures(
    df: pd.DataFrame,
    start: str|None = None,
    end:   str|None = None,
    *,
    width: float = 12.0,
):
    dfx = df.copy()
    if start is not None: dfx = dfx[dfx["DateTime"] >= pd.to_datetime(start)]
    if end   is not None: dfx = dfx[dfx["DateTime"] <= pd.to_datetime(end)]
    dfx = dfx.reset_index(drop=True)

    series = [
        ("FEED_T",               "FEED_T"),
        ("PRODUCT_AF_BYPASS_T",  "PRODUCT_T"),
        ("BYPASS_T",             "BYPASS_T"),
    ]
    present = [(c, name) for c, name in series if c in dfx.columns]
    if not present:
        raise ValueError("No hay columnas de temperatura: FEED_T / PRODUCT_AF_BYPASS_T / BYPASS_T.")

    height = 2.2 + 0.7*len(present)
    fig, ax = plt.subplots(figsize=(width, height))
    t = pd.to_datetime(dfx["DateTime"])
    for col, name in present:
        y = pd.to_numeric(dfx[col], errors="coerce")
        ax.plot(t, y, linewidth=1.2, label=name)

    ax.set_xlabel("DateTime")
    ax.set_ylabel("Temperatura (°C)")
    ax.set_title("Temperaturas — FEED / PRODUCT / BYPASS")
    ax.legend(ncol=2, fontsize=9)
    fig.autofmt_xdate(); fig.tight_layout(); plt.show()

def plot_raw_header_flows(
    df: pd.DataFrame,
    start: str|None = None,
    end:   str|None = None,
    *,
    width: float = 12.0,
):
    dfx = df.copy()
    if start is not None: dfx = dfx[dfx["DateTime"] >= pd.to_datetime(start)]
    if end   is not None: dfx = dfx[dfx["DateTime"] <= pd.to_datetime(end)]
    dfx = dfx.reset_index(drop=True)

    series = [
        ("FEED_Q",                "FEED_Q"),
        ("PRODUCT_AF_BYPASS_Q",   "PRODUCT_Q"),
        ("BYPASS_Q",              "BYPASS_Q"),
    ]
    present = [(c, name) for c, name in series if c in dfx.columns]
    if not present:
        raise ValueError("No hay columnas de caudal: FEED_Q / PRODUCT_AF_BYPASS_Q / BYPASS_Q.")

    height = 2.2 + 0.7*len(present)
    fig, ax = plt.subplots(figsize=(width, height))
    t = pd.to_datetime(dfx["DateTime"])
    for col, name in present:
        y = pd.to_numeric(dfx[col], errors="coerce")
        ax.plot(t, y, linewidth=1.2, label=name)

    ax.set_xlabel("DateTime")
    ax.set_ylabel("Caudal (Nm³/h)")
    ax.set_title("Caudales — FEED / PRODUCT / BYPASS")
    ax.legend(ncol=2, fontsize=9)
    fig.autofmt_xdate(); fig.tight_layout(); plt.show()

def plot_raw_species(
                    df: pd.DataFrame,
                    start: str | None = None,
                    end: str | None = None):
    """
    - Entrada: df, start, end.
    - Detecta especies en FEED_* y PRODUCT_AF_BYPASS_* (excluye P/T/Q).
    - Ordena subplots por promedio de FEED (desc) en el rango.
    - Máximo 3 columnas; si la última fila tiene 1 gráfico y hay 3 columnas, lo centra.
    - Eje X = DateTime sin normalizar.
    """
    # Recorte temporal
    d = df
    if start is not None:
        d = d.loc[d["DateTime"] >= pd.to_datetime(start)]
    if end is not None:
        d = d.loc[d["DateTime"] <= pd.to_datetime(end)]
    d = d.reset_index(drop=True)
    if d.empty:
        raise ValueError("No hay datos en el rango temporal seleccionado.")

    x = d["DateTime"]
    FEED_PREFIX = "FEED_"
    PROD_PREFIX = "PRODUCT_AF_BYPASS_"

    def is_species_col(col: str, prefix: str) -> bool:
        if not col.startswith(prefix):
            return False
        base = col[len(prefix):]
        return base not in ("P", "T", "Q")

    feed_cols = [c for c in d.columns if is_species_col(c, FEED_PREFIX)]
    prod_cols = [c for c in d.columns if is_species_col(c, PROD_PREFIX)]

    # conjunto de especies
    species = sorted(
        set(c[len(FEED_PREFIX):] for c in feed_cols) |
        set(c[len(PROD_PREFIX):] for c in prod_cols)
    )
    if not species:
        raise ValueError("No se encontraron columnas de especies FEED_* o PRODUCT_AF_BYPASS_*.")

    # Orden por promedio en FEED (descendente). Si no hay FEED para esa especie, va al final.
    def feed_mean(sp: str) -> float:
        col = f"{FEED_PREFIX}{sp}"
        if col not in d.columns:
            return -np.inf
        s = pd.to_numeric(d[col], errors="coerce")
        m = float(s.mean(skipna=True)) if s.notna().any() else -np.inf
        return m

    species.sort(key=feed_mean, reverse=True)

    # Layout: máximo 3 columnas
    n = len(species)
    n_cols = min(3, n)
    n_rows = math.ceil(n / n_cols)

    width = 12.0
    height = 2.0 + 2.0 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height), squeeze=False)

    # Por defecto, ocultar todos
    for ax in axes.ravel():
        ax.set_visible(False)

    # Posiciones: izquierda→derecha. Si última fila tiene 1 y n_cols==3, centrar en la columna 1.
    std_positions = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    if (n_cols == 3) and (n % n_cols == 1) and (n_rows >= 1):
        positions = std_positions[:n-1] + [(n_rows - 1, 1)]  # centra el último
    else:
        positions = std_positions[:n]

    # Pintar
    for sp, (r, c) in zip(species, positions):
        ax = axes[r][c]
        ax.set_visible(True)

        y_feed = d.get(f"{FEED_PREFIX}{sp}")
        y_prod = d.get(f"{PROD_PREFIX}{sp}")

        if y_feed is not None:
            ax.plot(x, pd.to_numeric(y_feed, errors="coerce"), linewidth=1.2, label="FEED")
        if y_prod is not None:
            ax.plot(x, pd.to_numeric(y_prod, errors="coerce"), linewidth=1.2, label="PRODUCT_AF_BYPASS")

        ax.set_title(sp)
        ax.set_xlabel("DateTime")
        ax.set_ylabel("Valor")
        if (y_feed is not None) and (y_prod is not None):
            ax.legend(loc="best", fontsize=8)

    fig.suptitle("Especies (ordenadas por promedio en FEED)", y=0.98)
    fig.tight_layout()
    plt.show()

def monitorData(
                df: pd.DataFrame,
                start: str | None = None,
                end: str | None = None):
    # Recorte temporal
    d = df
    if start is not None:
        d = d.loc[d["DateTime"] >= pd.to_datetime(start)]
    if end is not None:
        d = d.loc[d["DateTime"] <= pd.to_datetime(end)]
    d = d.reset_index(drop=True)
    if d.empty:
        raise ValueError("No hay datos en el rango temporal seleccionado.")

    fig = plt.figure(figsize=(32, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.25)

    # -------------------------
    # (1,1) Caudales
    # -------------------------
    ax_q = fig.add_subplot(gs[0, 0])
    flow_cols = [
        ("FEED_Q", "FEED_Q"),
        ("PRODUCT_BF_BYPASS_Q", "PRODUCT_BF_BYPASS_Q"),
        ("PRODUCT_AF_BYPASS_Q", "PRODUCT_AF_BYPASS_Q"),
        ("BYPASS_Q", "BYPASS_Q"),
        ("TAIL_BF_BYPASS_Q", "TAIL_BF_BYPASS_Q"),
        ("TAIL_AF_BYPASS_Q", "TAIL_AF_BYPASS_Q"),
    ]
    
    for col, lab in flow_cols:
        if col in d.columns:
            y = pd.to_numeric(d[col], errors="coerce").clip(lower=0.0)  # <— recorte a cero
            ax_q.plot(d["DateTime"], y, linewidth=1.2, label=lab)
    
    ax_q.set_title("Caudales")
    ax_q.set_ylabel("Nm³/h")
    if any(col in d.columns for col, _ in flow_cols):
        ax_q.legend(fontsize=8, ncol=2)

    # -------------------------
    # (2,1) Presión (izq) y Temperatura (dcha)
    # -------------------------
    ax_p = fig.add_subplot(gs[1, 0])
    if "FEED_P" in d.columns:
        ax_p.plot(d["DateTime"], pd.to_numeric(d["FEED_P"], errors="coerce"), linewidth=1.2, label="FEED_P")
    if "PRODUCT_AF_BYPASS_P" in d.columns:
        ax_p.plot(d["DateTime"], pd.to_numeric(d["PRODUCT_AF_BYPASS_P"], errors="coerce"), linewidth=1.2, label="PRODUCT_P")
    if "BYPASS_P" in d.columns:
        ax_p.plot(d["DateTime"], pd.to_numeric(d["BYPASS_P"], errors="coerce"), linewidth=1.2, label="TAIL_P")
    ax_p.set_ylabel("bar")
    ax_p.set_title("Presión (izq) y Temperatura (dcha)")

    ax_t = ax_p.twinx()
    if "FEED_T" in d.columns:
        ax_t.plot(d["DateTime"], pd.to_numeric(d["FEED_T"], errors="coerce"), linewidth=1.0, linestyle="--", label="FEED_T")
    if "PRODUCT_AF_BYPASS_T" in d.columns:
        ax_t.plot(d["DateTime"], pd.to_numeric(d["PRODUCT_AF_BYPASS_T"], errors="coerce"), linewidth=1.0, linestyle="--", label="PRODUCT_T")
    ax_t.set_ylabel("°C")

    h1, l1 = ax_p.get_legend_handles_labels()
    h2, l2 = ax_t.get_legend_handles_labels()
    if h1 or h2:
        ax_p.legend(h1 + h2, l1 + l2, fontsize=8, ncol=2, loc="best")

    # -------------------------
    # (1,2–3) Presiones de lechos
    # -------------------------
    ax_beds = fig.add_subplot(gs[0, 1:3])
    bed_cols = [c for c in d.columns if c.startswith("BED_DP_")]
    bed_cols.sort(key=lambda c: int(c.split("_")[-1]))
    for col in bed_cols:
        ax_beds.plot(d["DateTime"], pd.to_numeric(d[col], errors="coerce"), linewidth=1.0, label=col)
    ax_beds.set_title("Presión lechos")
    ax_beds.set_ylabel("bar")
    if bed_cols:
        ax_beds.legend(fontsize=8, ncol=min(4, len(bed_cols)))

    # -------------------------
    # (2,2–3) Especies (subrejilla 2×3)
    # -------------------------
    sub_gs = gs[1, 1:3].subgridspec(2, 3, hspace=0.35, wspace=0.25)
    axes_sp = sub_gs.subplots()

    FEED_PREFIX = "FEED_"
    PROD_PREFIX = "PRODUCT_AF_BYPASS_"

    def is_species(col: str, prefix: str) -> bool:
        if not col.startswith(prefix):
            return False
        key = col[len(prefix):]
        return key not in ("P", "T", "Q")

    feed_sp = [c[len(FEED_PREFIX):] for c in d.columns if is_species(c, FEED_PREFIX)]
    prod_sp = [c[len(PROD_PREFIX):] for c in d.columns if is_species(c, PROD_PREFIX)]
    species = sorted(set(feed_sp) | set(prod_sp))

    def rank_value(sp: str) -> float:
        a = pd.to_numeric(d.get(FEED_PREFIX + sp), errors="coerce")
        b = pd.to_numeric(d.get(PROD_PREFIX + sp), errors="coerce")
        m1 = float(a.mean(skipna=True)) if isinstance(a, pd.Series) else np.nan
        m2 = float(b.mean(skipna=True)) if isinstance(b, pd.Series) else np.nan
        return np.nanmax([m1, m2])

    species.sort(key=rank_value, reverse=True)
    top = species[:5]
    rest = species[5:]

    positions = [(r, c) for r in range(2) for c in range(3)]

    def plot_species(ax, sp: str, title: str | None = None):
        y1 = d.get(FEED_PREFIX + sp)
        y2 = d.get(PROD_PREFIX + sp)
        if y1 is not None:
            ax.plot(d["DateTime"], pd.to_numeric(y1, errors="coerce"), linewidth=1.0, label="FEED")
        if y2 is not None:
            ax.plot(d["DateTime"], pd.to_numeric(y2, errors="coerce"), linewidth=1.0, label="PRODUCT")
        ax.set_title(title or sp, fontsize=9)

    # Top 5 especies
    for i, sp in enumerate(top):
        r, c = positions[i]
        plot_species(axes_sp[r][c], sp, sp)

    # “Otros” (6º)
    if rest:
        r, c = positions[5]
        ax = axes_sp[r][c]
    
        cols_feed = [FEED_PREFIX + s for s in rest if FEED_PREFIX + s in d.columns]
        cols_prod = [PROD_PREFIX + s for s in rest if PROD_PREFIX + s in d.columns]
    
        plotted = False
    
        if cols_feed:
            y_other_feed = d[cols_feed].apply(pd.to_numeric, errors="coerce").sum(axis=1, skipna=True)
            ax.plot(d["DateTime"], y_other_feed, linewidth=1.0, label="FEED")
            plotted = True
    
        if cols_prod:
            y_other_prod = d[cols_prod].apply(pd.to_numeric, errors="coerce").sum(axis=1, skipna=True)
            ax.plot(d["DateTime"], y_other_prod, linewidth=1.0, label="PRODUCT")
            plotted = True
    
        if plotted:
            ax.set_title("Otros", fontsize=9)
            ax.legend(fontsize=7, loc="best")
        else:
            # Si por algún motivo no hay columnas válidas, ocultamos el eje
            ax.set_visible(False)

    # Oculta subplots no usados
    used = len(top) + (1 if rest else 0)
    for j in range(6):
        r, c = positions[j]
        if j >= used:
            axes_sp[r][c].set_visible(False)

    # Pequeñas leyendas en cada subgráfico de especies si corresponde
    for r in range(2):
        for c in range(3):
            ax = axes_sp[r][c]
            if ax.get_visible():
                h, l = ax.get_legend_handles_labels()
                if h:
                    ax.legend(fontsize=7, loc="best")

    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.99,
                    hspace=0.05, wspace=0.05)
    plt.show()

def plot_steps_gantt(
    df: pd.DataFrame,
    start: str|None = None,
    end:   str|None = None,
    *,
    mode: str = "steps",          # 'steps' | 'substeps' | 'inout'
    width: float = 12.0,
    label_min_seconds: float = 30.0,
    ax: plt.Axes | None = None,
):
    mode = mode.lower()
    if mode not in ("steps", "substeps", "inout"):
        raise ValueError("mode debe ser 'steps', 'substeps' o 'inout'.")
    if "DateTime" not in df.columns:
        raise ValueError("Falta 'DateTime'.")

    # recorte temporal
    d = df
    if start is not None:
        d = d.loc[d["DateTime"] >= pd.to_datetime(start)]
    if end is not None:
        d = d.loc[d["DateTime"] <= pd.to_datetime(end)]
    d = d.reset_index(drop=True)
    if d.empty:
        raise ValueError("No hay datos en el rango solicitado.")

    # detectar b1..bN
    bed_tags = []
    for c in d.columns:
        m = re.fullmatch(r"b(\d+)_step", c)
        if m:
            bed_tags.append(f"b{int(m.group(1))}")
    bed_tags.sort(key=lambda s: int(s[1:]))
    if not bed_tags:
        raise ValueError("No encuentro columnas 'b#_step'.")

    # helpers
    t = pd.to_datetime(d["DateTime"])
    x = mdates.date2num(t)
    def segs(vals: np.ndarray):
        out=[]; s=0
        for i in range(1, len(vals)+1):
            if i==len(vals) or vals[i]!=vals[s]:
                out.append((s, i, vals[s])); s=i
        return out
    def dur_s(i0, i1):
        i1 = min(i1, len(t)-1)
        return float((t.iloc[i1] - t.iloc[i0]).total_seconds())

    # crear eje si no lo pasan
    created_fig = False
    if ax is None:
        # height = 1.6 + 0.5*len(bed_tags)
        fig, ax = plt.subplots(figsize=(32, 9))
        created_fig = True

    COLORS = {"PRZ":"#27ae60","FEED":"#e74c3c","BWD":"#3498db","PURGE":"#9b59b6","WAIT":"#7f8c8d"}
    COLORS_INOUT = {"INLET":"#e67e22", "OUTLET":"#8e44ad"}
    TEXT_STYLE = dict(ha="center", va="center", fontsize=8, color="#111")
    h = 0.8

    yticks = []
    for k, b in enumerate(bed_tags):
        y0 = k
        yticks.append(y0 + h/2)
        steps = d[f"{b}_step"].astype(object).values

        if mode == "steps":
            for s,e,val in segs(steps):
                x0 = x[s]; x1 = x[max(s+1, e-1)]
                ax.add_patch(Rectangle((x0, y0), max(x1-x0, 1e-9), h,
                                       facecolor=COLORS.get(str(val), "#cccccc"),
                                       edgecolor="black", linewidth=0.2))
                if dur_s(s,e) >= label_min_seconds:
                    ax.text((x0+x1)/2, y0+h/2, str(val), **TEXT_STYLE)

        elif mode == "substeps":
            partner = d.get(f"{b}_partner", pd.Series([""]*len(d))).astype(object).values
            key = []
            for st, pn in zip(steps, partner):
                if st == "PRZ":
                    key.append(("PRZ", pn if pn else "FEED"))
                elif st == "BWD":
                    key.append(("BWD", pn if pn else "BLOW"))
                else:
                    key.append(None)
            segs2=[]; s=None
            for i, k2 in enumerate(key):
                if k2 is None:
                    if s is not None:
                        segs2.append((s, i, key[s][0], key[s][1])); s=None
                    continue
                if s is None: s=i
                elif k2 != key[s]:
                    segs2.append((s, i, key[s][0], key[s][1])); s=i
            if s is not None:
                segs2.append((s, len(key), key[s][0], key[s][1]))

            for s,e,st,pn in segs2:
                x0 = x[s]; x1 = x[max(s+1, e-1)]
                ax.add_patch(Rectangle((x0, y0), max(x1-x0, 1e-9), h,
                                       facecolor=COLORS.get(st, "#cccccc"),
                                       edgecolor="black", linewidth=0.2))
                if dur_s(s,e) >= label_min_seconds:
                    ax.text((x0+x1)/2, y0+h/2, str(pn), **TEXT_STYLE)

        else:  # inout
            partner = d.get(f"{b}_partner", pd.Series([""]*len(d))).astype(object).values
            in_mask  = ((steps == "PRZ") & (partner == "FEED")) | (steps == "FEED")
            out_mask = ((steps == "BWD") & (partner == "BLOW")) | (steps == "PURGE")
            lab = np.full(len(steps), "", dtype=object)
            lab[in_mask]  = "INLET"
            lab[out_mask] = "OUTLET"
            for s,e,val in segs(lab):
                if val == "": 
                    continue
                x0 = x[s]; x1 = x[max(s+1, e-1)]
                ax.add_patch(Rectangle((x0, y0), max(x1-x0, 1e-9), h,
                                       facecolor=COLORS_INOUT.get(val, "#cccccc"),
                                       edgecolor="black", linewidth=0.2))
                if dur_s(s,e) >= label_min_seconds:
                    ax.text((x0+x1)/2, y0+h/2, val, **TEXT_STYLE)

    # ejes
    ax.set_ylim(-0.2, len(bed_tags)-1 + 0.2 + h)
    ax.set_yticks(yticks)
    ax.set_yticklabels([b.upper() for b in bed_tags])
    ax.set_xlim(x[0], x[-1])
    ax.xaxis_date()
    ax.set_xlabel("Tiempo")
    ax.set_title({"steps":"Steps", "substeps":"Substeps (partner)", "inout":"INLET / OUTLET"}[mode])
    ax.grid(False)

    if created_fig:
        fig.tight_layout()
        plt.show()

    return ax

def compute_cycle_times(
    df: pd.DataFrame,
    start: str | None = None,
    end:   str | None = None,
    *,
    anchor_bed: str = "b1",
    min_cycle_s: float = 60.0,     # descarta ciclos demasiado cortos
    include_partial: bool = False, # incluir el último ciclo si no cierra con PRZ
    decimals: int = 1,
):
    """
    Requiere un DataFrame con:
      - 'DateTime' (datetime)
      - columnas 'b#_step' con valores {PRZ, FEED, BWD, PURGE, WAIT}
      - opcional 'b#_partner' (en PRZ/BWD: {'B#', 'FEED', 'BLOW', ''})

    Lógica de ciclo por cama:
      Un ciclo = [PRZ_k, PRZ_{k+1}) anclado al primer PRZ de 'anchor_bed'
      dentro del rango start/end.

    Substeps:
      PRZ_1, PRZ_2  → primeras dos subetapas de PRZ con partner ≠ FEED (sumando si hay varias)
      PRZ_FEED      → tramo de PRZ con partner == FEED (o vacío)
      BWD_1, BWD_2  → primeras dos subetapas de BWD con partner ≠ BLOW (sumando si hay varias)
      BWD_PURGE     → tramo de BWD con partner == BLOW (o vacío)

    Devuelve un dict con:
      - 'cycles'           : detalle por ciclo y cama
      - 'summary_steps'    : medias por cama (min) de steps + cycle_count
      - 'summary_substeps' : medias por cama (min) de substeps + cycle_count
      - 'summary_inout'    : medias por cama (min) de INLET/OUT
      - 'anchor_time'      : timestamp del ancla
    """
    # ---------- validaciones básicas ----------
    if "DateTime" not in df.columns:
        raise ValueError("Falta la columna 'DateTime' (datetime).")

    # recorte temporal
    d = df
    if start is not None:
        d = d.loc[d["DateTime"] >= pd.to_datetime(start)]
    if end is not None:
        d = d.loc[d["DateTime"] <= pd.to_datetime(end)]
    d = d.reset_index(drop=True)
    if d.empty:
        raise ValueError("No hay datos en el rango solicitado.")

    # detectar b1..bN
    bed_tags = []
    for c in d.columns:
        m = re.fullmatch(r"b(\d+)_step", c)
        if m:
            bed_tags.append(f"b{int(m.group(1))}")
    bed_tags.sort(key=lambda s: int(s[1:]))
    if not bed_tags:
        raise ValueError("No encuentro columnas 'b#_step' en el DataFrame.")

    if anchor_bed not in bed_tags:
        raise ValueError(f"anchor_bed='{anchor_bed}' no existe. Disponibles: {bed_tags}")

    # ---------- helpers ----------
    t = pd.to_datetime(d["DateTime"])

    def _segments(vals: np.ndarray):
        """[(i0, i1, value)] con runs contiguas."""
        out=[]; s=0
        for i in range(1, len(vals)+1):
            if i == len(vals) or vals[i] != vals[s]:
                out.append((s, i, vals[s])); s = i
        return out

    def _dur_s(i0: int, i1: int) -> float:
        """Duración (s) entre índices [i0, i1)."""
        i1 = min(i1, len(t)-1)
        return float((t.iloc[i1] - t.iloc[i0]).total_seconds())

    # ---------- buscar ancla: primer PRZ de anchor_bed ----------
    lab_anchor = d[f"{anchor_bed}_step"].astype(object).values
    i_start = 0
    if start is not None:
        i_start = int(np.argmax(t >= pd.to_datetime(start)))
    i_anchor = None
    for i in range(max(1, i_start), len(lab_anchor)):
        if lab_anchor[i] == "PRZ" and lab_anchor[i-1] != "PRZ":
            i_anchor = i
            break
    if i_anchor is None:
        raise ValueError("No se encuentra inicio de PRZ para el ancla en el rango.")

    # ---------- índices de inicio PRZ por cama (desde el ancla) ----------
    bed_prz_starts: dict[str, list[int]] = {}
    for b in bed_tags:
        lab = d[f"{b}_step"].astype(object).values
        idxs = []
        for i in range(max(1, i_anchor), len(lab)):
            if lab[i] == "PRZ" and lab[i-1] != "PRZ":
                idxs.append(i)
        bed_prz_starts[b] = idxs

    # ---------- recorrer ciclos ----------
    rows = []

    for b in bed_tags:
        steps = d[f"{b}_step"].astype(object).values
        partner = d.get(f"{b}_partner", pd.Series([""]*len(d))).astype(object).values

        starts = bed_prz_starts[b]
        if not starts:
            continue
        ends = starts[1:]
        if include_partial:
            ends = ends + [len(steps)]

        cyc_idx = 0
        for i0, i1 in zip(starts, ends):
            if i1 <= i0 + 1:
                continue
            dur_cycle = _dur_s(i0, i1)
            if dur_cycle < min_cycle_s:
                continue
            cyc_idx += 1

            # ---- STEPS (sumas por etiqueta) ----
            totals_steps = {"PRZ":0.0,"FEED":0.0,"BWD":0.0,"PURGE":0.0,"WAIT":0.0}
            segs = _segments(steps[i0:i1])
            segs = [(i0+s, i0+e, v) for (s,e,v) in segs]
            for s,e,v in segs:
                totals_steps[v] = totals_steps.get(v, 0.0) + _dur_s(s,e)

            # ---- SUBSTEPS por partner dentro de PRZ/BWD ----
            def _subsegs(code_name: str):
                out=[]
                for s,e,v in segs:
                    if v != code_name:
                        continue
                    # romper por partner constante
                    ss = s
                    while ss < e:
                        jj = ss + 1
                        while jj < e and partner[jj] == partner[ss]:
                            jj += 1
                        out.append((ss, jj, partner[ss]))
                        ss = jj
                return out

            # PRZ: primeros 2 donantes + tramo FEED
            prz_1 = prz_2 = prz_feed = 0.0
            donor_seen = 0
            for s,e,pn in _subsegs("PRZ"):
                dsec = _dur_s(s,e)
                if pn == "" or pn == "FEED":
                    prz_feed += dsec
                else:
                    if donor_seen == 0:
                        prz_1 += dsec; donor_seen = 1
                    else:
                        prz_2 += dsec

            # BWD: primeros 2 receptores + tramo BLOW
            bwd_1 = bwd_2 = bwd_purge = 0.0
            recip_seen = 0
            for s,e,pn in _subsegs("BWD"):
                dsec = _dur_s(s,e)
                if pn == "" or pn == "BLOW":
                    bwd_purge += dsec
                else:
                    if recip_seen == 0:
                        bwd_1 += dsec; recip_seen = 1
                    else:
                        bwd_2 += dsec

            # IN/OUT (min)
            inlet_min = (totals_steps["FEED"] + prz_feed) / 60.0
            out_min   = (totals_steps["PURGE"] + bwd_purge) / 60.0

            rows.append({
                "bed": b,
                "cycle_idx": cyc_idx,
                "t_start": t.iloc[i0],
                "t_end":   t.iloc[max(i1-1, i0)],
                "cycle_min": dur_cycle/60.0,
                # steps (min)
                "PRZ_min":   totals_steps["PRZ"]/60.0,
                "FEED_min":  totals_steps["FEED"]/60.0,
                "BWD_min":   totals_steps["BWD"]/60.0,
                "PURGE_min": totals_steps["PURGE"]/60.0,
                "WAIT_min":  totals_steps["WAIT"]/60.0,
                # substeps (min)
                "PRZ_1_min":       prz_1/60.0,
                "PRZ_2_min":       prz_2/60.0,
                "PRZ_FEED_min":    prz_feed/60.0,
                "BWD_1_min":       bwd_1/60.0,
                "BWD_2_min":       bwd_2/60.0,
                "BWD_PURGE_min":   bwd_purge/60.0,
                # in/out (min)
                "INLET_min": inlet_min,
                "OUT_min":   out_min,
            })

    cycles = pd.DataFrame(rows)
    if cycles.empty:
        raise ValueError("No se han detectado ciclos completos con los criterios dados.")

    # redondeo
    num_cols = cycles.select_dtypes(include=[float,int]).columns
    cycles[num_cols] = cycles[num_cols].round(decimals)

    # resumen steps
    summary_steps = (cycles
        .groupby("bed", as_index=False)[["cycle_min","PRZ_min","FEED_min","BWD_min","PURGE_min","WAIT_min"]]
        .mean())
    counts = (cycles.groupby("bed", as_index=False)["cycle_idx"]
              .count().rename(columns={"cycle_idx":"cycle_count"}))
    summary_steps = summary_steps.merge(counts, on="bed", how="left")
    summary_steps.iloc[:, 1:] = summary_steps.iloc[:, 1:].round(decimals)

    # resumen substeps
    summary_substeps = (cycles
        .groupby("bed", as_index=False)[
            ["PRZ_1_min","PRZ_2_min","PRZ_FEED_min","BWD_1_min","BWD_2_min","BWD_PURGE_min"]
        ].mean())
    summary_substeps = summary_substeps.merge(counts, on="bed", how="left")
    summary_substeps.iloc[:, 1:] = summary_substeps.iloc[:, 1:].round(decimals)

    # resumen IN/OUT
    summary_inout = (cycles
        .groupby("bed", as_index=False)[["INLET_min","OUT_min"]]
        .mean())
    summary_inout = summary_inout.merge(counts, on="bed", how="left")
    summary_inout.iloc[:, 1:] = summary_inout.iloc[:, 1:].round(decimals)

    # timestamp del ancla (en el rango recortado)
    anchor_time = t.iloc[i_anchor]

    return {
        "cycles": cycles.sort_values(["bed","cycle_idx"]).reset_index(drop=True),
        "summary_steps": summary_steps.sort_values("bed").reset_index(drop=True),
        "summary_substeps": summary_substeps.sort_values("bed").reset_index(drop=True),
        "summary_inout": summary_inout.sort_values("bed").reset_index(drop=True),
        "anchor_time": anchor_time,
    }

def _segments_from_bool(mask: np.ndarray):
    segs = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i + 1
            while j < n and mask[j]:
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1
    return segs

def _valve_color(name: str) -> str:
    if name.endswith("_Vfeed"):
        return STEP_COLORS["FEED"]
    if name.endswith("_Vpurge"):
        return STEP_COLORS["PURGE"]
    if "_Vprz_" in name:
        return STEP_COLORS["PRZ"]
    if "_Vbwd_" in name:
        return STEP_COLORS["BWD"]
    return "#7f8c8d"

def plot_bed_valves(
    df: pd.DataFrame,
    bed: str,                   # "b1", "b2", ...
    start: str | None = None,
    end:   str | None = None,
    *,
    figsize=(12, 3),
    band_height: float = 0.8,
    alpha_on: float = 0.85,
):
    if "DateTime" not in df.columns:
        raise ValueError("Se requiere columna 'DateTime' en el DataFrame.")

    # recorte temporal
    d = df
    if start is not None:
        d = d[d["DateTime"] >= pd.to_datetime(start)]
    if end is not None:
        d = d[d["DateTime"] <= pd.to_datetime(end)]
    d = d.sort_values("DateTime").reset_index(drop=True)
    if d.empty:
        raise ValueError("No hay datos en el rango solicitado.")

    t = pd.to_datetime(d["DateTime"])
    x = mdates.date2num(t)
    dx = np.diff(x)
    # ancho mínimo: el Δt mediano; si no existe, ~1 segundo en días
    min_width = float(np.nanmedian(dx)) if dx.size else (1.0 / 86400.0)

    # válvulas del lecho
    core = [f"{bed}_Vfeed", f"{bed}_Vpurge"]
    vpatt = re.compile(rf"^{bed}_(Vprz|Vbwd)_B\d+$")
    partners = [c for c in d.columns if vpatt.match(c)]
    valves = [c for c in core + sorted(partners) if c in d.columns and d[c].any()]
    if not valves:
        raise ValueError(f"No hay válvulas activas para {bed} en el rango.")

    fig, ax = plt.subplots(figsize=figsize)

    # pintar bandas
    for k, col in enumerate(valves):
        mask = d[col].astype(bool).values
        y0 = k  # pista por válvula
        for i0, i1 in _segments_from_bool(mask):
            x0 = x[i0]
            # usar i1-1 (índice válido); si el ancho es 0, aplicar min_width
            right_idx = max(i0, min(i1 - 1, len(x) - 1))
            x1 = x[right_idx]
            width = x1 - x0
            if width <= 0:
                width = min_width
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    width,
                    band_height,
                    facecolor=_valve_color(col),
                    edgecolor="black",
                    linewidth=0.3,
                    alpha=alpha_on,
                )
            )

    # formato
    ax.set_ylim(-0.2, len(valves) - 1 + band_height + 0.2)
    ax.set_yticks([i + band_height / 2 for i in range(len(valves))])
    ax.set_yticklabels(valves)
    ax.set_xlim(x[0], x[-1])
    ax.xaxis_date()
    ax.set_title(f"Válvulas — {bed.upper()}")
    ax.grid(False)
    fig.tight_layout()
    plt.show()

def plot_valves_grid(
    df: pd.DataFrame,
    start: str | None = None,
    end:   str | None = None,
    *,
    beds: list[str] | None = None,   # e.g. ["b1","b2",...]; si None, autodetectar
    nrows: int = 2,
    ncols: int = 4,
    figsize=(24, 10),
    band_height: float = 0.8,
    alpha_on: float = 0.85,
):
    if "DateTime" not in df.columns:
        raise ValueError("Se requiere columna 'DateTime'.")

    # recorte temporal común
    d = df
    if start is not None:
        d = d[d["DateTime"] >= pd.to_datetime(start)]
    if end is not None:
        d = d[d["DateTime"] <= pd.to_datetime(end)]
    d = d.sort_values("DateTime").reset_index(drop=True)
    if d.empty:
        raise ValueError("No hay datos en el rango solicitado.")

    # autodetectar beds si no se pasan
    if beds is None:
        beds = []
        for c in d.columns:
            m = re.fullmatch(r"b(\d+)_Vfeed", c)
            if m:
                beds.append(f"b{int(m.group(1))}")
        beds = sorted(set(beds), key=lambda s: int(s[1:]))

    total = min(len(beds), nrows * ncols)
    if total == 0:
        raise ValueError("No se detectaron lechos con válvulas en el DataFrame.")

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    # helper interno: dibujar en un axis (misma lógica que plot_bed_valves)
    def _draw_on_ax(ax, bed_name: str):
        t = pd.to_datetime(d["DateTime"])
        x = mdates.date2num(t)
        dx = np.diff(x)
        min_width = float(np.nanmedian(dx)) if dx.size else (1.0 / 86400.0)

        core = [f"{bed_name}_Vfeed", f"{bed_name}_Vpurge"]
        vpatt = re.compile(rf"^{bed_name}_(Vprz|Vbwd)_B\d+$")
        partners = [c for c in d.columns if vpatt.match(c)]
        valves = [c for c in core + sorted(partners) if c in d.columns and d[c].any()]

        if not valves:
            ax.text(0.5, 0.5, "Sin actividad", ha="center", va="center")
            ax.axis("off")
            return

        for k, col in enumerate(valves):
            mask = d[col].astype(bool).values
            y0 = k
            for i0, i1 in _segments_from_bool(mask):
                x0 = x[i0]
                right_idx = max(i0, min(i1 - 1, len(x) - 1))
                x1 = x[right_idx]
                width = x1 - x0
                if width <= 0:
                    width = min_width
                ax.add_patch(
                    Rectangle(
                        (x0, y0),
                        width,
                        band_height,
                        facecolor=_valve_color(col),
                        edgecolor="black",
                        linewidth=0.3,
                        alpha=alpha_on,
                    )
                )

        ax.set_ylim(-0.2, len(valves) - 1 + band_height + 0.2)
        ax.set_yticks([i + band_height / 2 for i in range(len(valves))])
        ax.set_yticklabels(valves, fontsize=8)
        ax.set_xlim(x[0], x[-1])
        ax.xaxis_date()
        ax.set_title(bed_name.upper(), fontsize=10)
        ax.grid(False)

    for i in range(total):
        r, c = divmod(i, ncols)
        _draw_on_ax(axes[r, c], beds[i])

    # apagar ejes sobrantes
    for j in range(total, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")

    fig.tight_layout()
    plt.show()

def compute_flows(
    df_ann: pd.DataFrame,
    start: str | None = None,
    end:   str | None = None,
    V=1.0,                         # escalar o dict {'b1':V1,...}
    *,
    T_col: str | None = "FEED_T",                 # °C para PRZ→FEED
    T_product_col: str | None = "PRODUCT_BF_BYPASS_T",  # °C para BWD/PURGE; si None → T_col
    T_const_K: float = 298.15,
    clamp_negative: bool = True,
    equal_split: bool = True,               # reparte cabecera FEED entre camas en FEED

    # --- correcciones/flags (FEED/PURGE) ---
    fix_equal_to_header: bool = True,       # si b_FEED ≈ feed_header → usa valor anterior
    fix_equal_to_outheader: bool = True,    # si b_PURGE ≈ columns_out → usa valor anterior
    shift_seconds_if_bad: float = 10.0,     # tamaño del desplazamiento (s)
    eps_abs: float = 0.0,                   # tolerancia absoluta Nm3/h (≈)
    eps_rel: float = 0.0,                   # tolerancia relativa (≈)

    # --- tags medidos (ya renombrados en getPSAdata) ---
    feed_header_tag:    str = "FEED_Q",                 # Nm3/h (cabecera feed)
    product_header_tag: str = "PRODUCT_BF_BYPASS_Q",    # Nm3/h (producto ANTES bypass)
    bypass_header_tag:  str = "BYPASS_Q",               # Nm3/h
) -> pd.DataFrame:
    """
    Devuelve un DataFrame con:
      - DateTime
      - feed_header_Nm3_h, product_header_Nm3_h, bypass_header_Nm3_h
      - tail_header_Nm3_h = max(feed_header - product_header, 0)
      - columns_from_headers_OUT_Nm3_h = max(tail_header - bypass_header, 0)
      - b#_PRZFEED_Nm3_h  (PV/RT con T_col)
      - b#_BWDPURGE_Nm3_h (PV/RT con T_product_col)
      - b#_FEED_Nm3_h     (split FEED con corrección ≈ header → valor anterior)
      - b#_PURGE_Nm3_h    (reparto leftover con corrección ≈ out_header → valor anterior)
      - total_PRZFEED_Nm3_h, total_OUT_Nm3_h, recon_error_OUT_Nm3_h
    """

    # ---------- helpers ----------
    def _timecol(df):
        if "DateTime" in df.columns: return "DateTime"
        if "ts" in df.columns: return "ts"
        raise ValueError("No encuentro columna temporal ('DateTime' o 'ts').")

    def _beds_from_dfann_local(df: pd.DataFrame) -> list[str]:
        beds = []
        for c in df.columns:
            m = re.match(r"^b(\d+)_step$", c)
            if m: beds.append(f"b{int(m.group(1))}")
        beds.sort(key=lambda s: int(s[1:]))
        if not beds:
            raise ValueError("No encuentro columnas 'b#_step'.")
        return beds

    def _pressure_map(df):
        BEDS_ALL = [f"BED_DP_{i}" for i in range(1, 9)]
        tags = [c for c in BEDS_ALL if c in df.columns]
        beds = [f"b{i+1}" for i in range(len(tags))]
        return dict(zip(beds, tags))

    def _col_as_series(df: pd.DataFrame, colname: str) -> pd.Series:
        if colname in df.columns:
            return pd.to_numeric(df[colname], errors="coerce")
        else:
            return pd.Series(np.nan, index=df.index, dtype=float)

    # ---------- recorte temporal ----------
    tcol = _timecol(df_ann)
    d = df_ann
    if start is not None:
        d = d.loc[d[tcol] >= pd.to_datetime(start)]
    if end is not None:
        d = d.loc[d[tcol] <= pd.to_datetime(end)]
    d = d.sort_values(tcol).reset_index(drop=True)
    if d.empty:
        raise ValueError("No hay datos en el rango solicitado.")

    # ---------- básicos ----------
    t = pd.to_datetime(d[tcol])
    beds = _beds_from_dfann_local(d)
    pmap = _pressure_map(d)

    # Temperaturas (K)
    if T_col and (T_col in d.columns):
        T_in_K = pd.to_numeric(d[T_col], errors="coerce").astype(float) + 273.15
        T_in_K = T_in_K.fillna(method="ffill").fillna(T_const_K)
    else:
        T_in_K = pd.Series(T_const_K, index=d.index, dtype=float)

    if T_product_col and (T_product_col in d.columns):
        T_out_K = pd.to_numeric(d[T_product_col], errors="coerce").astype(float) + 273.15
        T_out_K = T_out_K.fillna(method="ffill").fillna(T_const_K)
    else:
        T_out_K = T_in_K

    # Volúmenes por cama
    if isinstance(V, dict):
        V = {b: float(V.get(b, 1.0)) for b in beds}
    else:
        V = {b: float(V) for b in beds}

    # Δt en horas
    dt_h = t.diff().dt.total_seconds().astype(float) / 3600.0
    dt_h.iloc[0] = np.nan
    dt_h = dt_h.replace(0.0, np.nan)

    # shift (muestras) para correcciones ≈ header
    dt_s_median = float(pd.Series(t).diff().dt.total_seconds().median() or 10.0)
    if not np.isfinite(dt_s_median) or dt_s_median <= 0:
        dt_s_median = 10.0
    shift_n = max(1, int(round(shift_seconds_if_bad / dt_s_median)))

    out = pd.DataFrame({tcol: t})

    # ======================
    # 1) PRZ→FEED (inflow por PV/RT con T_in)
    # ======================
    for b in beds:
        step = d[f"{b}_step"].astype(object)
        partner = (d[f"{b}_partner"].astype(object).fillna("")
                   if f"{b}_partner" in d.columns
                   else pd.Series("", index=d.index, dtype=object))
        mask_pf = (step == "PRZ") & (partner == "FEED")

        if b not in pmap:
            out[f"{b}_PRZFEED_Nm3_h"] = np.nan
            continue

        P_bar = pd.to_numeric(d[pmap[b]], errors="coerce").astype(float)
        n_in = (P_bar * BAR_TO_PA) * V[b] / (R * T_in_K)  # mol en cama

        valid = mask_pf & mask_pf.shift(1, fill_value=False)
        dmol_h = (n_in - n_in.shift(1)) / dt_h
        q_in_mol_h = dmol_h.where(valid, np.nan)
        if clamp_negative:
            q_in_mol_h = q_in_mol_h.where(q_in_mol_h >= 0.0, 0.0)

        out[f"{b}_PRZFEED_Nm3_h"] = q_in_mol_h / MOL_PER_NM3

    prz_cols_nm3 = [c for c in out.columns if c.endswith("_PRZFEED_Nm3_h")]
    out["total_PRZFEED_Nm3_h"] = out[prz_cols_nm3].sum(axis=1, skipna=True)

    # ======================
    # 2) Cabeceras medidas (siempre Series)
    # ======================
    feed_header_nm3    = _col_as_series(d, feed_header_tag)
    product_header_nm3 = _col_as_series(d, product_header_tag)  # ya es BEFORE BYPASS (desde getPSAdata)
    bypass_header_nm3  = _col_as_series(d, bypass_header_tag)

    out["feed_header_Nm3_h"]    = feed_header_nm3
    out["product_header_Nm3_h"] = product_header_nm3
    out["bypass_header_Nm3_h"]  = bypass_header_nm3

    # TAIL total (cabecera) = feed - product  (≥0)
    tail_nm3 = (feed_header_nm3 - product_header_nm3).clip(lower=0.0)
    out["tail_header_Nm3_h"] = tail_nm3

    # ======================
    # 3) FEED (split a camas) + corrección ≈ feed_header
    # ======================
    bed_feed_masks = {b: (d[f"{b}_step"].astype(object) == "FEED") for b in beds}
    feed_counts = sum(bed_feed_masks.values())  # nº camas en FEED (Series)

    for b in beds:
        if equal_split:
            share = feed_header_nm3 / feed_counts.replace(0, np.nan)
            y_nm3 = share.where(bed_feed_masks[b], np.nan)
        else:
            # asignar cabecera completa a la cama en FEED (si hay una a la vez)
            y_nm3 = feed_header_nm3.where(bed_feed_masks[b], np.nan)

        # --- corrección “igual al header” → usa valor anterior ---
        if fix_equal_to_header:
            bad = y_nm3.notna() & feed_header_nm3.notna() & \
                  np.isclose(y_nm3, feed_header_nm3, rtol=eps_rel, atol=eps_abs)
            if bad.any():
                prev = y_nm3.shift(shift_n)
                y_nm3.loc[bad & prev.notna()] = prev.loc[bad & prev.notna()]

        # NaN → 0.0 para dejar explícito “sin FEED” en esa cama
        y_nm3 = y_nm3.fillna(0.0)

        out[f"{b}_FEED_Nm3_h"] = y_nm3

    feed_cols_nm3 = [f"{b}_FEED_Nm3_h" for b in beds]
    out["sum_beds_FEED_Nm3_h"] = out[feed_cols_nm3].sum(axis=1, skipna=True)

    # ======================
    # 4) BWD→PURGE (outflow por PV/RT con T_out)
    # ======================
    for b in beds:
        step = d[f"{b}_step"].astype(object)
        partner = (d[f"{b}_partner"].astype(object).fillna("")
                   if f"{b}_partner" in d.columns
                   else pd.Series("", index=d.index, dtype=object))
        mask_bp = (step == "BWD") & (partner == "BLOW")

        if b not in pmap:
            out[f"{b}_BWDPURGE_Nm3_h"] = np.nan
            continue

        P_bar = pd.to_numeric(d[pmap[b]], errors="coerce").astype(float)
        n_out = (P_bar * BAR_TO_PA) * V[b] / (R * T_out_K)  # mol en cama

        valid = mask_bp & mask_bp.shift(1, fill_value=False)
        dmol_h = (n_out - n_out.shift(1)) / dt_h      # negativo si sale gas
        q_out_mol_h = (-dmol_h).where(valid, np.nan)   # salida positiva
        if clamp_negative:
            q_out_mol_h = q_out_mol_h.where(q_out_mol_h >= 0.0, 0.0)

        out[f"{b}_BWDPURGE_Nm3_h"] = q_out_mol_h / MOL_PER_NM3

    bwd_cols_nm3 = [f"{b}_BWDPURGE_Nm3_h" for b in beds]
    total_bwd_nm3 = out[bwd_cols_nm3].sum(axis=1, skipna=True)

    # ======================
    # 5) PURGE (reparto del remanente) + corrección ≈ OUT_header
    #     columns_from_headers_OUT = max(tail_header − bypass, 0)
    #     leftover = columns_out − Σ(BWDPURGE) (≥0)
    #     repartir entre camas en PURGE
    # ======================
    columns_out_nm3 = (out["tail_header_Nm3_h"] - out["bypass_header_Nm3_h"]).clip(lower=0.0)
    out["columns_from_headers_OUT_Nm3_h"] = columns_out_nm3

    leftover_nm3 = (columns_out_nm3 - total_bwd_nm3).clip(lower=0.0)

    bed_purge_masks = {b: (d[f"{b}_step"].astype(object) == "PURGE") for b in beds}
    purge_counts = sum(bed_purge_masks.values())  # nº camas en PURGE (Series)

    for b in beds:
        share = leftover_nm3 / purge_counts.replace(0, np.nan)
        y_nm3 = share.where(bed_purge_masks[b], np.nan)

        # --- corrección “igual al OUT header (columns_out)” → valor anterior ---
        if fix_equal_to_outheader:
            bad = y_nm3.notna() & columns_out_nm3.notna() & \
                  np.isclose(y_nm3, columns_out_nm3, rtol=eps_rel, atol=eps_abs)
            if bad.any():
                prev = y_nm3.shift(shift_n)
                y_nm3.loc[bad & prev.notna()] = prev.loc[bad & prev.notna()]

        # NaN → 0.0 (sin PURGE en esa cama en ese instante)
        y_nm3 = y_nm3.fillna(0.0)

        out[f"{b}_PURGE_Nm3_h"] = y_nm3

    purge_cols_nm3 = [f"{b}_PURGE_Nm3_h" for b in beds]
    total_purge_nm3 = out[purge_cols_nm3].sum(axis=1, skipna=True)

    # ======================
    # 6) Totales de salida y reconciliación
    # ======================
    out["total_OUT_Nm3_h"] = total_bwd_nm3 + total_purge_nm3
    out["recon_error_OUT_Nm3_h"] = out["total_OUT_Nm3_h"] - out["columns_from_headers_OUT_Nm3_h"]

    return out

def monitorFlows(
    flow_df: pd.DataFrame,
    start: str | None = None,
    end:   str | None = None,
    *,
    beds: str | list[str] = "all",               # "all" o lista ["b1","b3",...]
    substeps: str | list[str] = "all",           # "all" o subset {"PRZFEED","FEED","BWDPURGE","PURGE"}
    show_out_as_negative: bool = True,
    squelch_zeros: bool = True,                  # oculta valores == 0
    eps_abs: float = 0.0,                        # tolerancia abs para “igual al header”
    eps_rel: float = 0.0,                        # tolerancia rel para “igual al header”
    shift_seconds_if_bad: float = 11.0,          # cuánto retroceder si “igual al header”
    width: float = 12.0,
    height: float = 4.5,
):
    """
    Requiere DataFrame devuelto por compute_flows (necesitamos:
      - 'DateTime'
      - b#_{PRZFEED|FEED|BWDPURGE|PURGE}_Nm3_h
      - 'feed_header_Nm3_h' y 'columns_from_headers_OUT_Nm3_h')
    """
    if "DateTime" not in flow_df.columns:
        raise ValueError("El DataFrame necesita columna 'DateTime'.")

    d = flow_df
    if start is not None:
        d = d.loc[d["DateTime"] >= pd.to_datetime(start)]
    if end is not None:
        d = d.loc[d["DateTime"] <= pd.to_datetime(end)]
    d = d.sort_values("DateTime").reset_index(drop=True)
    if d.empty:
        raise ValueError("No hay datos en el rango solicitado.")

    # dt medio para calcular shift_n
    dt_s_median = float(pd.Series(pd.to_datetime(d["DateTime"])).diff().dt.total_seconds().median() or 10.0)
    if not np.isfinite(dt_s_median) or dt_s_median <= 0:
        dt_s_median = 10.0
    shift_n = max(1, int(round(shift_seconds_if_bad / dt_s_median)))

    # detectar camas
    pat = re.compile(r"^b(\d+)_(PRZFEED|FEED|BWDPURGE|PURGE)_Nm3_h$")
    beds_found = sorted({f"b{m.group(1)}" for c in d.columns if (m := pat.match(c))},
                        key=lambda s: int(s[1:]))
    if not beds_found:
        raise ValueError("No encuentro columnas de caudal b#_*_Nm3_h.")

    # filtrar camas
    if isinstance(beds, str) and beds.lower() != "all":
        wanted = [s.strip().lower() for s in beds.split(",") if s.strip()]
        beds_pick = [b for b in beds_found if b in wanted]
    elif isinstance(beds, list):
        beds_pick = [str(b).lower() for b in beds if f"{b}".startswith("b")]
        beds_pick = [b for b in beds_found if b in beds_pick]
    else:
        beds_pick = beds_found[:]
    if not beds_pick:
        raise ValueError(f"No hay coincidencias de camas para {beds!r}.")

    # substeps
    all_steps = ["PRZFEED","FEED","BWDPURGE","PURGE"]
    if isinstance(substeps, str) and substeps.lower() == "all":
        steps_pick = all_steps
    elif isinstance(substeps, list):
        steps_pick = [s.upper() for s in substeps if s.upper() in all_steps]
        steps_pick = steps_pick or all_steps
    else:
        steps_pick = all_steps

    # series “header” para correcciones
    feed_header = d.get("feed_header_Nm3_h", pd.Series(np.nan, index=d.index, dtype=float))
    out_header  = d.get("columns_from_headers_OUT_Nm3_h", pd.Series(np.nan, index=d.index, dtype=float))

    # color por cama
    base_colors = ["#e74c3c","#3498db","#27ae60","#9b59b6","#f39c12","#16a085","#7f8c8d","#2c3e50"]
    bed_color = {b: base_colors[(int(b[1:])-1) % len(base_colors)] for b in beds_found}

    # marker por step
    step_marker = {"FEED":"o", "PRZFEED":"s", "BWDPURGE":"^", "PURGE":"v"}

    fig, ax = plt.subplots(figsize=(width, height))
    any_plotted = False

    for b in beds_pick:
        for st in steps_pick:
            col = f"{b}_{st}_Nm3_h"
            if col not in d.columns:
                continue

            y = pd.to_numeric(d[col], errors="coerce").astype(float).copy()

            # --- corrección “igual al header” ---
            if st in ("FEED", "PRZFEED"):
                # compara con feed_header
                if feed_header is not None:
                    bad = y.notna() & feed_header.notna() & np.isclose(y, feed_header, rtol=eps_rel, atol=eps_abs)
                    if bad.any():
                        prev = y.shift(shift_n)
                        y.loc[bad & prev.notna()] = prev.loc[bad & prev.notna()]
            elif st in ("PURGE", "BWDPURGE"):
                # compara con out_header (= columns_from_headers_OUT_Nm3_h)
                if out_header is not None:
                    bad = y.notna() & out_header.notna() & np.isclose(y, out_header, rtol=eps_rel, atol=eps_abs)
                    if bad.any():
                        prev = y.shift(shift_n)
                        y.loc[bad & prev.notna()] = prev.loc[bad & prev.notna()]

            # --- OUT en negativo si se pide ---
            if show_out_as_negative and st in ("BWDPURGE","PURGE"):
                y = -y

            # --- ocultar ceros (y +0.0/-0.0) ---
            if squelch_zeros:
                y = y.where(np.abs(y) > 1e-12)

            # no pintar si todo NaN
            if not y.notna().any():
                continue

            ax.plot(
                d["DateTime"], y,
                linewidth=1.2,
                marker=step_marker.get(st, None),
                markersize=3,
                label=f"{b.upper()} {st}",
                color=bed_color[b],
                alpha=0.95,
            )
            any_plotted = True

    ax.set_title("Caudales (Nm³/h)" + (" — +IN / −OUT" if show_out_as_negative else ""))
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Nm³/h")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(True, alpha=0.25)

    if any_plotted:
        ax.legend(ncol=3, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No hay series para mostrar",
                transform=ax.transAxes, ha="center", va="center")

    fig.tight_layout()
    plt.show()

#%%
# df=getPSAdata(DATA_PATH,t0,t1)
#%%
# t0 = "2025-04-20 00:00:01"   
# t1 = "2025-04-20 01:59:59" 
# t1 = "2025-04-20 00:13:59"   
# plot_raw_bed_pressure(df,t0,t1)
# plot_raw_header_pressures(df,t0,t1)
# plot_raw_header_temperatures(df,t0,t1)
# plot_raw_header_flows(df,t0,t1)
# plot_raw_species(df,t0,t1)
# plot_steps_gantt(df,t0,t1, mode="steps")
# plot_steps_gantt(df,t0,t1, mode="substeps")
# plot_steps_gantt(df,t0,t1, mode="inout")
# 
# t0 = "2025-04-20 00:00:01"   
# t1 = "2025-04-20 00:13:59"  

# monitorData(df,t0,t1)
# monitorSteps(df,t0,t1)
