import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo
import re
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D


def _make_auto_time_formatter(t: pd.Series) -> FuncFormatter:
    """
    Devuelve un FuncFormatter que adapta el formato de etiquetas del eje X
    según la duración del rango temporal t (serie de timestamps ya recortada):
      - < 1 hora  -> mm:ss (relativo al inicio)
      - ≤ 1 día   -> HH:MM:SS
      - >  1 día  -> YYYY-mm-dd HH:MM
    """
    if len(t) == 0:
        # fallback neutro (no debería pasar)
        return FuncFormatter(lambda frac, pos: "")

    t0 = pd.to_datetime(t.iloc[0])
    t1 = pd.to_datetime(t.iloc[-1])
    span_s = max((t1 - t0).total_seconds(), 1e-9)  # evita 0

    def _fmt(frac: float, pos=None) -> str:
        # clamp del eje normalizado
        f = 0.0 if frac < 0 else (1.0 if frac > 1 else float(frac))
        t_tick = t0 + (t1 - t0) * f

        if span_s < 3600:  # < 1 hora -> mm:ss relativo al inicio
            delta = (t_tick - t0).total_seconds()
            mm = int(delta // 60)
            ss = int(round(delta % 60))
            # corrige 60s por redondeo
            if ss == 60:
                mm += 1
                ss = 0
            return f"{mm:02d}:{ss:02d}"

        elif span_s > 86400:  # > 1 día -> día + hora
            return t_tick.strftime("%Y-%m-%d %H:%M")

        else:  # entre 1 h y 1 día -> hora con segundos
            return t_tick.strftime("%H:%M:%S")

    return FuncFormatter(_fmt)


def _slice_and_normalize_time(df: pd.DataFrame, start: str|None, end: str|None):
    """recorte por fechas + x en [0,1] y formateador de ticks a hora real"""
    def _time_col(df: pd.DataFrame) -> str:
        if "DateTime" in df.columns: return "DateTime"
        if "ts" in df.columns: return "ts"
        raise ValueError("No encuentro columna temporal ('DateTime' o 'ts').")
    tcol = "DateTime" if "DateTime" in df.columns else "ts"
    t_all = pd.to_datetime(df[tcol])
    m = np.ones(len(df), dtype=bool)
    if start is not None: m &= (t_all >= pd.Timestamp(start))
    if end   is not None: m &= (t_all <= pd.Timestamp(end))
    if not m.any(): raise ValueError("El rango start/end no solapa con los datos.")

    df2 = df.loc[m].reset_index(drop=True)
    t = pd.to_datetime(df2[tcol])

    tnum = mdates.date2num(t)
    span = float(tnum[-1] - tnum[0]) if len(tnum) > 1 else 1.0
    xnorm = (tnum - tnum[0]) / (span if span != 0 else 1.0)

    fmt = _make_auto_time_formatter(t)

    dt_s = float(np.median(np.diff(tnum))) * 24*3600 if len(tnum) > 1 else 10.0
    return df2, t, xnorm, fmt, dt_s

def _beds_from_dfann(df: pd.DataFrame) -> list[str]:
    """detecta b1..b8 a partir de columnas b#_step"""
    beds = []
    for c in df.columns:
        m = re.match(r"^b(\d+)_step$", c)
        if m: beds.append(f"b{int(m.group(1))}")
    beds.sort(key=lambda s: int(s[1:]))
    if not beds:
        raise ValueError("No encuentro columnas 'b#_step'. ¿Has corrido la función de anotado?")
    return beds

def _figsize_for_beds(n_beds: int, width: float = 12.0, base: float = 2.2, per_bed: float = 0.7):
    """misma receta para ambas figuras → misma altura"""
    return (width, base + per_bed * n_beds)

def _set_midnight_xticks(
    ax,
    t: pd.Series,           # serie de timestamps ya recortada que usaste para x
    x: np.ndarray,          # eje normalizado 0–1 que ya calculas
    fmt: FuncFormatter,     # el formatter que ya devuelves en _slice_and_normalize_time
    *,
    n_base: int = 6,        # nº de ticks base uniformes (0..1)
    show_vlines: bool = True,
    date_fmt: str = "%Y-%m-%d",  # formato para la etiqueta de medianoche
    snap_left_sec: float = 60.0  # si 00:00 cae <60 s antes del primer punto, lo “pegamos” a x=0
):
    """
    Coloca ticks base + ticks en cada comienzo de día (00:00 local).
    Si el 00:00 del primer día cae ligeeeramente antes del rango (p.ej. t0=00:00:01),
    lo “pegamos” al borde izquierdo (x=0) para que SIEMPRE aparezca.
    """
    if len(t) == 0:
        return
    t0 = pd.to_datetime(t.iloc[0])
    t1 = pd.to_datetime(t.iloc[-1])
    tnum0 = mdates.date2num(t0)
    tnum1 = mdates.date2num(t1)
    span = (tnum1 - tnum0) if (tnum1 != tnum0) else 1.0

    # ticks base uniformes
    base = list(np.linspace(0.0, 1.0, n_base))

    # medianoches dentro del rango (y “snap” si están un pelín antes)
    midnights = []
    d0_mid = t0.normalize()  # 00:00 del primer día visible
    # ¿00:00 justo antes del primer dato? pegamos si está muy cerca
    if d0_mid < t0 and (t0 - d0_mid).total_seconds() <= snap_left_sec:
        midnights.append((0.0, d0_mid))  # en el borde
        cur = d0_mid + pd.Timedelta(days=1)
    else:
        cur = d0_mid if d0_mid >= t0 else d0_mid + pd.Timedelta(days=1)

    while cur <= t1:
        xmid = (mdates.date2num(cur) - tnum0) / span
        if 0.0 <= xmid <= 1.0:
            midnights.append((float(xmid), cur))
        cur += pd.Timedelta(days=1)

    # unir ticks base + midnights (evitar duplicados cercanos)
    xs = base[:]
    for xv, _ in midnights:
        if all(abs(xv - xi) > 1e-3 for xi in xs):
            xs.append(xv)
    xs = sorted(xs)

    # labels: para midnights -> "YYYY-mm-dd 00:00"; resto -> fmt (hora real)
    def _label_for(xv: float) -> str:
        # ¿es uno de medianoche?
        for xm, tm in midnights:
            if abs(xv - xm) <= 1e-3:
                return f"{tm.strftime(date_fmt)} 00:00"
        # si no, hora con tu formatter
        return fmt(xv, None)

    labels = [_label_for(xv) for xv in xs]

    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=0)

    # líneas verticales finas en medianoche
    if show_vlines:
        for xv, _ in midnights:
            ax.axvline(x=xv, color="#bbbbbb", linewidth=0.8, linestyle=":", zorder=1)


#------------------------------------------------------------------------------
# =============================================================================
# # GETDATA
# =============================================================================
def getPSAdata(
    df_or_path,
    start: str | None = None,   # "YYYY-mm-dd HH:MM:SS" (hora local Europe/Madrid)
    end:   str | None = None,
    *,
    tz: str = "Europe/Madrid",
    time_col: str | None = None,       # autodetecta "DateTime" o "ts" si None
    P_high: float = 7.0,
    P_low:  float = 1.0,
    dpos_thr: float = 0.010,           # bar/s → PRZ
    dneg_thr: float = -0.010,          # bar/s → BWD
    dflat_thr: float = 0.00999,        # |dP/dt| < dflat_thr → “meseta”
    eps: float = 0.02,                 # tolerancia de P para emparejar PRZ/BWD
    prefer_neighbors: bool = True,     # en empate, prioriza camas vecinas por índice
    min_seg_s: float = 15.0,           # fusiona “chispazos” de step más cortos
    return_copy: bool = True
) -> tuple[pd.DataFrame, dict]:
    """
    Anota el DataFrame con:
      - b#_step ∈ {PRZ, FEED, BWD, PURGE, WAIT}
      - b#_partner en PRZ/BWD: {'B#', 'FEED', 'BLOW'} (vacío en otros steps)
      - b#_in_mask / b#_in_kind   (PRZ_FEED, FEED)
      - b#_out_mask / b#_out_kind (BWD_PURGE, PURGE)
      - DateTime_local, t_s, x01 (tiempo normalizado 0–1)

    Devuelve (df_anotado, meta).
    """

    # ---- helpers ----
    def _find_time_col(_path, cands=("DateTime","ts")):
        head = pd.read_csv(_path, nrows=0)
        for c in cands:
            if c in head.columns:
                return c
        raise ValueError(f"No encuentro columna temporal entre {cands}. Columnas: {list(head.columns)}")

    def _to_local_naive(s: pd.Series, tzname: str) -> pd.Series:
        tzinfo = ZoneInfo(tzname)
        sdt = pd.to_datetime(s, errors="coerce")
        if getattr(sdt.dt, "tz", None) is None:
            sdt = sdt.dt.tz_localize(tzinfo, nonexistent="shift_forward", ambiguous="NaT")
        else:
            sdt = sdt.dt.tz_convert(tzinfo)
        return sdt.dt.tz_localize(None)

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

    # ---- cargar / localizar columna tiempo ----
    if isinstance(df_or_path, (str, bytes, bytearray)):
        path = str(df_or_path)
        if time_col is None:
            time_col = _find_time_col(path)
        df = pd.read_csv(path, parse_dates=[time_col])
    else:
        df = df_or_path.copy() if return_copy else df_or_path
        if time_col is None:
            time_col = "DateTime" if "DateTime" in df.columns else ("ts" if "ts" in df.columns else None)
            if time_col is None:
                raise ValueError("No encuentro columna temporal ('DateTime' o 'ts').")

    # ---- tiempo local naive + recorte ----
    t_local = _to_local_naive(df[time_col], tz)
    df[time_col] = t_local
    if start is not None:
        df = df.loc[df[time_col] >= pd.to_datetime(start)].reset_index(drop=True)
    if end is not None:
        df = df.loc[df[time_col] <= pd.to_datetime(end)].reset_index(drop=True)
    if df.empty:
        raise ValueError("El rango start/end no solapa con los datos.")

    t = df[time_col]
    t_s = (t - t.iloc[0]).dt.total_seconds().astype(float)
    span_s = float(max(t_s.iloc[-1], 1e-12))
    x01 = (t_s / span_s).astype(float)

    df["DateTime_local"] = t
    df["t_s"] = t_s
    df["x01"] = x01

    # ---- camas ----
    BEDS_ALL = [
        "PSA_PI12518_Val","PSA_PI12528_Val","PSA_PI12538_Val","PSA_PI12548_Val",
        "PSA_PI12558_Val","PSA_PI12568_Val","PSA_PI12578_Val","PSA_PI12588_Val",
    ]
    BEDS = [b for b in BEDS_ALL if b in df.columns]
    if not BEDS:
        raise ValueError("No hay presiones de lechos en df.")
    bed_code     = {b: f"b{i+1}" for i, b in enumerate(BEDS)}
    bed_code_cap = {b: f"B{i+1}" for i, b in enumerate(BEDS)}
    idx_of       = {b: i for i, b in enumerate(BEDS)}

    # dt medio (s) para dP/dt
    dt_s = float(pd.Series(t).diff().dt.total_seconds().median() or 10.0)

    # ---- steps por cama ----
    labels = {}
    for b in BEDS:
        P = pd.to_numeric(df[b], errors="coerce").astype(float).values
        d = np.r_[0.0, np.diff(P)] / max(dt_s, 1e-6)
        lab = np.empty(len(P), dtype=object); lab[:] = "WAIT"
        lab[d >= dpos_thr] = "PRZ"
        lab[d <= dneg_thr] = "BWD"
        flat = (np.abs(d) < dflat_thr)
        lab[flat & (P >= P_high)] = "FEED"
        lab[flat & (P <= P_low )] = "PURGE"
        lab[flat & (P >  P_low) & (P <  P_high)] = "WAIT"
        lab = _rle_merge_short(lab, t, min_seg_s)  # alisado
        labels[b] = lab
        df[f"{bed_code[b]}_step"] = lab

    # ---- partners instante a instante ----
    Pmat = np.vstack([pd.to_numeric(df[b], errors="coerce").astype(float).values for b in BEDS])
    N = Pmat.shape[1]
    partners = {b: np.array([""]*N, dtype=object) for b in BEDS}

    for i in range(N):
        mode = {b: labels[b][i] for b in BEDS}
        in_PRZ = [b for b in BEDS if mode[b] == "PRZ"]
        in_BWD = [b for b in BEDS if mode[b] == "BWD"]

        # PRZ: donante con P mayor; si no hay -> FEED
        for b in in_PRZ:
            k = idx_of[b]; Pk = Pmat[k, i]
            cands = [(bj, Pmat[idx_of[bj], i]) for bj in in_BWD if Pmat[idx_of[bj], i] > Pk + eps]
            if cands:
                bj = max(cands, key=lambda it: (it[1], -abs(idx_of[it[0]]-k)))[0] if prefer_neighbors \
                     else max(cands, key=lambda it: it[1])[0]
                partners[b][i] = bed_code_cap[bj]
            else:
                partners[b][i] = "FEED"

        # BWD: receptor con P menor; si no hay -> BLOW
        for b in in_BWD:
            k = idx_of[b]; Pk = Pmat[k, i]
            cands = [(bj, Pmat[idx_of[bj], i]) for bj in in_PRZ if Pmat[idx_of[bj], i] < Pk - eps]
            if cands:
                bj = min(cands, key=lambda it: (it[1],  abs(idx_of[it[0]]-k)))[0] if prefer_neighbors \
                     else min(cands, key=lambda it: it[1])[0]
                partners[b][i] = bed_code_cap[bj]
            else:
                partners[b][i] = "BLOW"

    # volcar partner y máscaras IN/OUT
    for b in BEDS:
        tag = bed_code[b]
        step = df[f"{tag}_step"].astype(object).values
        part = np.where(np.isin(step, ["PRZ","BWD"]), partners[b], "")
        df[f"{tag}_partner"] = part

        # in_mask = ((step == "PRZ") & (part == "FEED")) | (step == "FEED")
        # in_kind = np.where((step == "PRZ") & (part == "FEED"), "PRZ_FEED",
        #                    np.where(step == "FEED", "FEED", ""))

        # out_mask = ((step == "BWD") & (part == "BLOW")) | (step == "PURGE")
        # out_kind = np.where((step == "BWD") & (part == "BLOW"), "BWD_PURGE",
        #                     np.where(step == "PURGE", "PURGE", ""))

        # df[f"{tag}_in_mask"]  = in_mask
        # df[f"{tag}_in_kind"]  = in_kind
        # df[f"{tag}_out_mask"] = out_mask
        # df[f"{tag}_out_kind"] = out_kind

        # INLET = PRZ_FEED + FEED
        in_mask = ((step == "PRZ") & (part == "FEED")) | (step == "FEED")
        # OUT = BWD_PURGE + PURGE
        
        out_mask = ((step == "BWD") & (part == "BLOW")) | (step == "PURGE")
        # alias explícitos solicitados
        df[f"{tag}_inlet_mask"]  = in_mask      # INLET = PRZ_FEED + FEED
        df[f"{tag}_outlet_mask"] = out_mask     # OUT   = BWD_PURGE + PURGE

    meta = dict(
        BEDS=BEDS,
        bed_code=bed_code,
        dt_s=dt_s,
        time_col=time_col,
        tz=tz,
        params=dict(P_high=P_high, P_low=P_low, dpos_thr=dpos_thr, dneg_thr=dneg_thr,
                    dflat_thr=dflat_thr, eps=eps, prefer_neighbors=prefer_neighbors,
                    min_seg_s=min_seg_s)
    )
    return df, meta

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# =============================================================================
# # PLOT RAW DATA
# =============================================================================
#HELPERS
#________________________ 
def plot_raw_flows(
    df: pd.DataFrame,
    start: str | None = None,
    end:   str | None = None,
    *,
    units: str = "Nm3_h",                   # 'Nm3_h' o 'mol_h'
    figsize: tuple[float, float] = (12.0, 4.2),
    # posibles aliases por si la columna tiene otros nombres
    feed_candidates:   tuple[str,...] = ("PRC02_1_FY12504_Val","FY-12504","FY12504"),
    product_candidates:tuple[str,...] = ("PRC02_1_FY12505_Val","FY-12505","FY12505"),
    bypass_candidates: tuple[str,...] = ("PSA_FI12599B_Val","FI-12599","FI12599"),
):
    """
    Traza los caudales de cabecera de Feed, Product y Bypass (FY) en Nm³/h o mol/h
    con el mismo eje X normalizado que los otros gráficos raw.
    """
    # helpers ya presentes en tu módulo
    def _timecol(df):
        if "DateTime" in df.columns: return "DateTime"
        if "ts" in df.columns: return "ts"
        raise ValueError("No encuentro columna temporal ('DateTime' o 'ts').")

    def _slice_and_normalize_time(df: pd.DataFrame, start: str|None, end: str|None):
        tcol = _timecol(df)
        t_all = pd.to_datetime(df[tcol])
        m = np.ones(len(df), dtype=bool)
        if start is not None: m &= (t_all >= pd.Timestamp(start))
        if end   is not None: m &= (t_all <= pd.Timestamp(end))
        if not m.any(): raise ValueError("El rango start/end no solapa con los datos.")
        df2 = df.loc[m].reset_index(drop=True)
        t = pd.to_datetime(df2[tcol])
        tnum = mdates.date2num(t)
        span = float(tnum[-1] - tnum[0]) if len(tnum) > 1 else 1.0
        xnorm = (tnum - tnum[0]) / (span if span != 0 else 1.0)

        # formateo adaptable: >1 día → d/h, <1h → mm:ss, otro → HH:MM
        duration_s = (t.iloc[-1] - t.iloc[0]).total_seconds() if len(t) > 1 else 0
        def _tick_fmt(frac, pos):
            frac = min(max(frac, 0.0), 1.0)
            tt = t.iloc[0] + pd.to_timedelta(frac * duration_s, unit="s")
            if duration_s >= 24*3600:
                return tt.strftime("%d %H:%M")
            elif duration_s <= 3600:
                return tt.strftime("%M:%S")
            else:
                return tt.strftime("%H:%M")
        fmt = FuncFormatter(_tick_fmt)

        return df2, t, xnorm, fmt

    def _first_existing(df, names):
        for n in names:
            if n in df.columns:
                return n
        return None

    df2, t, x, fmt = _slice_and_normalize_time(df, start, end)

    tag_feed    = _first_existing(df2, feed_candidates)
    tag_product = _first_existing(df2, product_candidates)
    tag_bypass  = _first_existing(df2, bypass_candidates)

    if not any([tag_feed, tag_product, tag_bypass]):
        raise ValueError("No encuentro ninguna señal FY de feed/product/bypass.")

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    def _maybe_convert(series_nm3):
        y = pd.to_numeric(series_nm3, errors="coerce")
        if units.lower() == "mol_h":
            return y * MOL_PER_NM3
        return y

    # colores suaves
    if tag_feed:
        ax.plot(x, _maybe_convert(df2[tag_feed]),   linewidth=1.3, label="Feed (FY)",   color="#e74c3c", alpha=0.95)
    if tag_product:
        ax.plot(x, _maybe_convert(df2[tag_product]),linewidth=1.3, label="Product (FY)",color="#3498db", alpha=0.95)
    if tag_bypass:
        ax.plot(x, _maybe_convert(df2[tag_bypass]), linewidth=1.3, label="Bypass (FY)", color="#7f8c8d", alpha=0.95)

    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.xaxis.set_major_formatter(fmt)
    ax.set_xlabel("Tiempo (normalizado)")

    ax.set_ylabel("Caudal (mol/h)" if units.lower()=="mol_h" else "Caudal (Nm³/h)")
    ax.set_title("Caudales de cabecera — Feed / Product / Bypass")
    ax.legend(loc="best", ncol=3, frameon=True)
    fig.tight_layout()
    plt.show()




def plot_raw_pressure(
    df: pd.DataFrame,
    start: str|None = None,
    end:   str|None = None,
    *,
    width: float = 12.0,
):
    df2, t, x, fmt, _ = _slice_and_normalize_time(df, start, end)
    # detectamos cuántas camas hay para igualar altura
    BED_TAGS = [
        "PSA_PI12518_Val","PSA_PI12528_Val","PSA_PI12538_Val","PSA_PI12548_Val",
        "PSA_PI12558_Val","PSA_PI12568_Val","PSA_PI12578_Val","PSA_PI12588_Val",
    ]
    tags = [c for c in BED_TAGS if c in df2.columns]
    if not tags:
        raise ValueError("No encuentro presiones de lechos en df.")

    fig = plt.figure(figsize=_figsize_for_beds(len(tags), width=width))
    ax = fig.add_subplot(111)
    for tag in tags:
        y = pd.to_numeric(df2[tag], errors="coerce").astype(float).values
        ax.plot(x, y, label=tag, linewidth=1.1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks(np.linspace(0,1,6))
    ax.xaxis.set_major_formatter(fmt)  
    # _set_midnight_xticks(ax, t, x, fmt, n_base=6, show_vlines=True, date_fmt="%m-%d")
    ax.set_ylabel("bar")
    ax.set_title("Presiones de lechos vs tiempo (X normalizado)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    plt.show()

    plt.show()


def plot_raw_temperature(
    df: pd.DataFrame,
    start: str|None = None,
    end:   str|None = None,
    *,
    width: float = 12.0,
    tags: list[str] | None = None,   # opcional: lista de columnas a pintar
):
    """
    Pinta temperaturas vs tiempo normalizado (0–1) con ticks en hora real.
    Si `tags` es None, detecta columnas de temperatura típicas (…_TI… o …TI#####…).
    """
    df2, t, x, fmt, _ = _slice_and_normalize_time(df, start, end)

    # -- descubrir columnas de temperatura si no se pasan --
    if tags is None:
        cand = []
        for c in df2.columns:
            # típicos: SIS_TI11102_Val, PRC02_1_TI12501_Val, etc.
            if re.search(r"(?:^|_)TI\d{4,6}(?:_|$)", c) or re.search(r"(?:^|_)TI(?:_|$)", c):
                cand.append(c)
        # filtrar a numéricos reales (y que varíen)
        tags = []
        for c in cand:
            s = pd.to_numeric(df2[c], errors="coerce")
            if s.notna().any():
                tags.append(c)
    if not tags:
        raise ValueError("No encuentro columnas de temperatura (p.ej. 'SIS_TI11102_Val', 'PRC02_1_TI12501_Val').")

    # altura proporcional al nº de series para mantener estética uniforme
    fig = plt.figure(figsize=_figsize_for_beds(len(tags), width=width))
    ax = fig.add_subplot(111)

    for tag in tags:
        y = pd.to_numeric(df2[tag], errors="coerce").astype(float).values
        ax.plot(x, y, label=tag, linewidth=1.2)

    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks(np.linspace(0,1,6))
    ax.xaxis.set_major_formatter(fmt)  
    # _set_midnight_xticks(ax, t, x, fmt, n_base=6, show_vlines=True, date_fmt="%m-%d")
    ax.set_ylabel("°C")
    ax.set_title("Temperaturas vs tiempo (X normalizado)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    plt.show()


def plot_raw_species(
    df: pd.DataFrame,
    start: str|None = None,
    end:   str|None = None,
    *,
    width: float = 12.0,
    tags: list[str] | None = None,   # opcional: lista exacta de columnas a pintar
):
    """
    Pinta señales de especies de analizadores (AI) vs tiempo normalizado.
    Si `tags` es None, detecta columnas típicas de AI por especie: *_H2, *_CO2, *_CO, *_N2, *_CH4, *_O2, *_C1..*_C6, *_H2O, *_Ar, *_He.
    """
    df2, t, x, fmt, _ = _slice_and_normalize_time(df, start, end)

    # -- detección automática si no se pasan 'tags' --
    if tags is None:
        species_pat = r"(H2|CO2|CO|N2|CH4|O2|C[1-6]|H2O|Ar|He)"
        ai_cols = []
        for c in df2.columns:
            # ejemplos esperados: SIS_AI11101_H2, PRC02_1_AI12502_CO2, etc.
            if re.search(r"(?:^|_)AI\d{4,6}(?:_|$)", c) and re.search(species_pat + r"(?:_|$)", c, re.IGNORECASE):
                ai_cols.append(c)
        # fallback: cualquier columna con _H2/_CO2/... aunque no tenga AI explícito
        if not ai_cols:
            for c in df2.columns:
                if re.search(species_pat + r"(?:_|$)", c, re.IGNORECASE):
                    ai_cols.append(c)
        # filtrar a numéricos
        tags = []
        for c in ai_cols:
            s = pd.to_numeric(df2[c], errors="coerce")
            if s.notna().any():
                tags.append(c)

    if not tags:
        raise ValueError("No encuentro columnas de especies (AI). Esperaba tags tipo '...AI#####_H2', '...AI#####_CO2', etc.")

    # Colores consistentes por especie
    base_colors = {
        "H2":"#2ecc71", "CO2":"#9b59b6", "CO":"#e67e22", "N2":"#3498db", "CH4":"#e74c3c",
        "O2":"#1abc9c", "H2O":"#16a085", "Ar":"#34495e", "He":"#7f8c8d",
        "C1":"#95a5a6","C2":"#c0392b","C3":"#8e44ad","C4":"#2980b9","C5":"#27ae60","C6":"#d35400"
    }
    def _species_name(col: str) -> str:
        m = re.search(r"(H2|CO2|CO|N2|CH4|O2|H2O|Ar|He|C[1-6])(?:_|$)", col, re.IGNORECASE)
        return m.group(1).upper() if m else col

    fig = plt.figure(figsize=_figsize_for_beds(len(tags), width=width))
    ax = fig.add_subplot(111)

    for col in tags:
        y = pd.to_numeric(df2[col], errors="coerce").astype(float).values
        specie = _species_name(col)
        color = base_colors.get(specie.upper(), None)
        ax.plot(x, y, label=f"{col}", linewidth=1.2, color=color)

    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks(np.linspace(0,1,6))
    ax.xaxis.set_major_formatter(fmt)  
    # _set_midnight_xticks(ax, t, x, fmt, n_base=6, show_vlines=True, date_fmt="%m-%d")
    ax.set_ylabel("% (vol)")   # ajusta si tus AI dan otras unidades
    ax.set_title("Especies (analizadores AI) vs tiempo (X normalizado)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    plt.show()

#------------------------------------------------------------------------------   
#------------------------------------------------------------------------------




# =============================================================================
#  CALCULO DEL TIEMPO PROMEDIO DE CICLO 
# =============================================================================
def compute_cycle_times(
    df_ann: pd.DataFrame,
    start: str | None = None,
    end:   str | None = None,
    *,
    anchor_bed: str = "b1",
    min_cycle_s: float = 60.0,     # descartar ciclos demasiado cortos
    include_partial: bool = False, # incluir último ciclo si no cierra con PRZ?
    decimals: int = 1,
):
    """
    Usa el DF anotado por psa_prepare (con b#_step y b#_partner) y calcula tiempos por ciclo.
    - Ancla el cómputo al PRIMER PRZ de anchor_bed dentro del rango start/end.
    - Un ciclo de una cama = [PRZ_k, PRZ_{k+1}) para esa cama tras el ancla.
    - SUBSTEPS:
        PRZ_1, PRZ_2  = primeras dos subetapas de PRZ con partner distinto de FEED (sumando si hay varias),
        PRZ_FEED      = tramo final de PRZ con partner == FEED (si existe, si no 0).
        BWD_1, BWD_2  = primeras dos subetapas de BWD con partner distinto de BWD (sumando si hay varias),
        BWD_PURGE     = tramo final de BWD con partner == BWD (si existe, si no 0).

    Devuelve: dict con
      - 'cycles': DataFrame detalle por cama y ciclo,
      - 'summary_steps': medias por cama (min) de steps + cycle_count,
      - 'summary_substeps': medias por cama (min) de substeps + cycle_count,
      - 'anchor_time': timestamp del ancla.
    """
    def _get_time_col(df: pd.DataFrame) -> str:
        if "DateTime" in df.columns: return "DateTime"
        if "ts" in df.columns: return "ts"
        raise ValueError("No encuentro columna temporal ('DateTime' o 'ts').")
        
    def _find_bed_codes(df: pd.DataFrame) -> list[str]:
        # columnas tipo b1_step, b2_step...
        pat = re.compile(r"^b(\d+)_step$")
        beds = []
        for c in df.columns:
            m = pat.match(c)
            if m:
                beds.append(f"b{int(m.group(1))}")
        beds.sort(key=lambda s: int(s[1:]))  # orden b1..b8
        if not beds:
            raise ValueError("No encuentro columnas 'b#_step'. ¿Has ejecutado antes psa_prepare?")
        return beds
    
    def _segments_from_series(vals: np.ndarray):
        """Devuelve segmentos [s,e, val] contiguos para un array 1D."""
        segs = []
        if len(vals) == 0: return segs
        s = 0
        for i in range(1, len(vals)+1):
            if i == len(vals) or vals[i] != vals[s]:
                segs.append((s, i, vals[s]))
                s = i
        return segs
    
    def _duration_seconds(t: pd.Series, i0: int, i1: int) -> float:
        """Duración entre índices [i0, i1) en segundos."""
        i1 = min(i1, len(t)-1)
        return float((t.iloc[i1] - t.iloc[i0]).total_seconds())

    time_col = _get_time_col(df_ann)
    t = pd.to_datetime(df_ann[time_col])

    mask = np.ones(len(df_ann), dtype=bool)
    if start is not None: mask &= (t >= pd.Timestamp(start))
    if end   is not None: mask &= (t <= pd.Timestamp(end))
    if not mask.any():    raise ValueError("El rango start/end no solapa con los datos.")

    df = df_ann.loc[mask].reset_index(drop=True)
    t  = pd.to_datetime(df[time_col])

    beds = _find_bed_codes(df)
    if anchor_bed not in beds:
        raise ValueError(f"anchor_bed='{anchor_bed}' no existe en este DF. Disponibles: {beds}")

    lab_anchor = df[f"{anchor_bed}_step"].values.astype(object)
    i_start = int(np.argmax(t >= pd.Timestamp(start))) if start is not None else 0
    i_anchor = None
    for i in range(max(1, i_start), len(lab_anchor)):
        if lab_anchor[i] == "PRZ" and lab_anchor[i-1] != "PRZ":
            i_anchor = i; break
    if i_anchor is None:
        raise ValueError("No encuentro inicio de PRZ para el ancla dentro del rango.")

    bed_prz_starts = {}
    for b in beds:
        lab = df[f"{b}_step"].values.astype(object)
        starts = []
        for i in range(max(1, i_anchor), len(lab)):
            if lab[i] == "PRZ" and lab[i-1] != "PRZ":
                starts.append(i)
        bed_prz_starts[b] = starts

    cycles_rows = []

    for b in beds:
        lab  = df[f"{b}_step"].values.astype(object)
        part = df.get(f"{b}_partner", pd.Series([""]*len(df))).values.astype(object)

        starts = bed_prz_starts[b]
        if not starts: continue
        ends = starts[1:]
        if include_partial: ends = ends + [len(lab)]

        cyc_idx = 0
        for i0, i1 in zip(starts, ends):
            if i1 <= i0 + 1: continue
            dur_cyc_s = _duration_seconds(t, i0, i1)
            if dur_cyc_s < min_cycle_s: continue
            cyc_idx += 1

            # STEPS
            totals_steps = {"PRZ":0.0,"FEED":0.0,"BWD":0.0,"PURGE":0.0,"WAIT":0.0}
            segs = _segments_from_series(lab[i0:i1])
            segs = [(i0+s, i0+e, v) for (s,e,v) in segs]
            for s, e, v in segs:
                dsec = _duration_seconds(t, s, e)
                totals_steps[v] = totals_steps.get(v, 0.0) + dsec

            # SUBSTEPS por partner dentro de PRZ/BWD
            def substeps_segments(code_name: str):
                out = []
                for s,e,v in segs:
                    if v == code_name:
                        ss = s
                        while ss < e:
                            jj = ss + 1
                            while jj < e and part[jj] == part[ss]:
                                jj += 1
                            out.append((ss, jj, part[ss]))  # (i0,i1,partner)
                            ss = jj
                return out

            prz_segs = substeps_segments("PRZ")
            prz_1 = prz_2 = prz_feed = 0.0; donor_seen = 0
            for s,e,pn in prz_segs:
                dsec = _duration_seconds(t, s, e)
                if pn == "" or pn == "FEED":
                    prz_feed += dsec
                else:
                    if donor_seen == 0: prz_1 += dsec; donor_seen = 1
                    else:               prz_2 += dsec

            bwd_segs = substeps_segments("BWD")
            bwd_1 = bwd_2 = bwd_purge = 0.0; recip_seen = 0
            for s,e,pn in bwd_segs:
                dsec = _duration_seconds(t, s, e)
                if pn == "" or pn == "BLOW":
                    bwd_purge += dsec
                else:
                    if recip_seen == 0: bwd_1 += dsec; recip_seen = 1
                    else:               bwd_2 += dsec

            # IN/OUT (min): INLET = FEED + PRZ_FEED ; OUT = PURGE + BWD_PURGE
            inlet_min = (totals_steps["FEED"] + prz_feed) / 60.0
            out_min   = (totals_steps["PURGE"] + bwd_purge) / 60.0

            row = {
                "bed": b,
                "cycle_idx": cyc_idx,
                "t_start": t.iloc[i0],
                "t_end":   t.iloc[i1-1],
                "cycle_min": dur_cyc_s/60.0,
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
            }
            cycles_rows.append(row)

    cycles = pd.DataFrame(cycles_rows)
    if cycles.empty:
        raise ValueError("No se han detectado ciclos completos con los criterios dados.")

    # redondeo
    num_cols = cycles.select_dtypes(include=[float,int]).columns
    cycles[num_cols] = cycles[num_cols].round(decimals)

    # summary steps
    agg_steps = (cycles
                 .groupby("bed", as_index=False)[["cycle_min","PRZ_min","FEED_min","BWD_min","PURGE_min","WAIT_min"]]
                 .mean())
    counts = cycles.groupby("bed", as_index=False)["cycle_idx"].count().rename(columns={"cycle_idx":"cycle_count"})
    summary_steps = agg_steps.merge(counts, on="bed", how="left")
    summary_steps.iloc[:, 1:] = summary_steps.iloc[:, 1:].round(decimals)

    # summary substeps
    agg_sub = (cycles
               .groupby("bed", as_index=False)[
                   ["PRZ_1_min","PRZ_2_min","PRZ_FEED_min","BWD_1_min","BWD_2_min","BWD_PURGE_min"]
               ].mean())
    summary_substeps = agg_sub.merge(counts, on="bed", how="left")
    summary_substeps.iloc[:, 1:] = summary_substeps.iloc[:, 1:].round(decimals)

    # summary IN/OUT
    agg_inout = (cycles
                 .groupby("bed", as_index=False)[["INLET_min","OUT_min"]]
                 .mean())
    summary_inout = agg_inout.merge(counts, on="bed", how="left")
    summary_inout.iloc[:, 1:] = summary_inout.iloc[:, 1:].round(decimals)

    anchor_time = t.iloc[i_anchor]

    return {
        "cycles": cycles.sort_values(["bed","cycle_idx"]).reset_index(drop=True),
        "summary_steps": summary_steps.sort_values("bed").reset_index(drop=True),
        "summary_substeps": summary_substeps.sort_values("bed").reset_index(drop=True),
        "summary_inout": summary_inout.sort_values("bed").reset_index(drop=True),
        "anchor_time": anchor_time,
    }

# ------------------------------------------------------------
# 2) PLOT DE BARRAS (STEPS o SUBSTEPS)
# ------------------------------------------------------------
def plot_steps_gantt(
    df_ann: pd.DataFrame,
    start: str|None = None,
    end:   str|None = None,
    *,
    mode: str = "steps",               # 'steps' | 'substeps' | 'inout'
    label_min_seconds: float = 30.0,
    merge_bridge_max_s: float = 30.0,  # solo para 'substeps'
    bridge_gap_s: float = 20.0,        # <-- NUEVO: fusionar INLET/OUT separados por huecos cortos
    width: float = 12.0,
):
    mode = mode.lower()
    if mode not in ("steps", "substeps", "inout"):
        raise ValueError("mode debe ser 'steps', 'substeps' o 'inout'.")

    df, t, x, fmt, dt_s = _slice_and_normalize_time(df_ann, start, end)
    beds = _beds_from_dfann(df)

    fig = plt.figure(figsize=_figsize_for_beds(len(beds), width=width))
    ax = fig.add_subplot(111)

    COLORS = {"PRZ":"#27ae60","FEED":"#e74c3c","BWD":"#3498db","PURGE":"#9b59b6","WAIT":"#7f8c8d"}
    COLORS_INOUT = {"INLET":"#e74c3c", "OUT":"#9b59b6"}

    TEXT_STYLE = dict(ha="center", va="center", fontsize=9, fontweight="bold",
                      color="#111", bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, boxstyle="round,pad=0.25"),
                      zorder=3)

    def _segs(vals):
        out=[]; s=0
        for i in range(1, len(vals)+1):
            if i==len(vals) or vals[i]!=vals[s]:
                out.append((s,i,vals[s])); s=i
        return out

    def _dur_s(i0, i1):
        i1 = min(i1, len(t)-1)
        return float((t.iloc[i1] - t.iloc[i0]).total_seconds())

    def _merge_equal(segs):
        if not segs: return segs
        out=[segs[0]]
        for s,e,c in segs[1:]:
            S,E,C = out[-1]
            if C==c: out[-1] = (S,e,C)
            else:    out.append((s,e,c))
        return out

    def _merge_equal_with_partner(segs):
        if not segs: return segs
        out=[segs[0]]
        for s,e,c,p in segs[1:]:
            S,E,C,P = out[-1]
            if C==c and P==p: out[-1]=(S,e,C,P)
            else:             out.append((s,e,c,p))
        return out

    def _merge_ABA_with_partner(segs, max_bridge_s):
        changed=True
        while changed and len(segs)>=3:
            changed=False; i=0; new=[]
            while i < len(segs):
                if i <= len(segs)-3:
                    a,b,c = segs[i], segs[i+1], segs[i+2]
                    if a[2]==c[2] and a[3]==c[3] and _dur_s(b[0], b[1]) <= max_bridge_s:
                        new.append((a[0], c[1], a[2], a[3])); i+=3; changed=True; continue
                new.append(segs[i]); i+=1
            segs = _merge_equal_with_partner(new)
        return segs

    h = 0.8
    centers = []

    for k, b in enumerate(beds):
        y0 = k
        centers.append(y0 + h/2)
        steps = df[f"{b}_step"].astype(object).values

        if mode == "steps":
            for s,e,code in _segs(steps):
                x0, x1 = x[s], x[min(e-1, len(x)-1)]
                ax.add_patch(Rectangle((x0, y0), max(x1-x0, 1e-6), h,
                                       facecolor=COLORS.get(code,"#ccc"),
                                       edgecolor="black", linewidth=0.2, zorder=2))
                if _dur_s(s,e) >= label_min_seconds:
                    ax.text((x0+x1)/2, y0+h/2, str(code).upper(), **TEXT_STYLE)

        elif mode == "substeps":
            part = df[f"{b}_partner"].astype(object).values
            key=[]
            for c,p in zip(steps, part):
                if c in ("PRZ","BWD"): key.append((c, p if p else ("FEED" if c=="PRZ" else "BLOW")))
                else:                  key.append(None)
            segs=[]; s=None
            for i,k2 in enumerate(key):
                if k2 is None:
                    if s is not None: segs.append((s,i,key[s][0],key[s][1])); s=None
                    continue
                if s is None: s=i
                elif k2 != key[s]: segs.append((s,i,key[s][0],key[s][1])); s=i
            if s is not None: segs.append((s,len(key),key[s][0],key[s][1]))
            segs = _merge_equal_with_partner(segs)
            if merge_bridge_max_s > 0: segs = _merge_ABA_with_partner(segs, merge_bridge_max_s)

            for s,e,code,ptxt in segs:
                x0, x1 = x[s], x[min(e-1, len(x)-1)]
                ax.add_patch(Rectangle((x0, y0), max(x1-x0, 1e-6), h,
                                       facecolor=COLORS.get(code,"#ccc"),
                                       edgecolor="black", linewidth=0.2, zorder=2))
                if _dur_s(s,e) >= label_min_seconds:
                    ax.text((x0+x1)/2, y0+h/2, ptxt, **TEXT_STYLE)

        else:  # mode == "inout"
            part = df.get(f"{b}_partner", pd.Series([""]*len(df))).astype(object).values
            in_mask  = ((steps == "PRZ") & (part == "FEED")) | (steps == "FEED")
            out_mask = ((steps == "BWD") & (part == "BLOW")) | (steps == "PURGE")

            # Serie de etiquetas: "", "INLET", "OUT"
            lab = np.full(len(steps), "", dtype=object)
            lab[in_mask]  = "INLET"
            lab[out_mask] = "OUT"

            segs = _segs(lab)                 # [(s,e,label)]
            segs = _merge_equal(segs)         # une consecutivos iguales

            # --- NUEVO: fusionar A–""–A si el hueco "" ≤ bridge_gap_s
            if bridge_gap_s > 0 and len(segs) >= 3:
                changed = True
                while changed and len(segs) >= 3:
                    changed = False
                    new = []
                    i = 0
                    while i < len(segs):
                        if i <= len(segs) - 3:
                            a, gap, c = segs[i], segs[i+1], segs[i+2]
                            if a[2] == c[2] and gap[2] == "" and _dur_s(gap[0], gap[1]) <= bridge_gap_s:
                                # fusionar en un solo bloque A
                                new.append((a[0], c[1], a[2]))
                                i += 3
                                changed = True
                                continue
                        new.append(segs[i])
                        i += 1
                    segs = _merge_equal(new)

            # pintar solo INLET/OUT
            for s,e,val in segs:
                if val == "": 
                    continue
                x0, x1 = x[s], x[min(e-1, len(x)-1)]
                ax.add_patch(Rectangle((x0, y0), max(x1-x0, 1e-6), h,
                                       facecolor=COLORS_INOUT.get(val,"#cccccc"),
                                       edgecolor="black", linewidth=0.2, zorder=2))
                if _dur_s(s,e) >= label_min_seconds:
                    ax.text((x0+x1)/2, y0+h/2, val, **TEXT_STYLE)

    ax.set_yticks(centers)
    ax.set_yticklabels([b.upper() for b in beds])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.2, len(beds)-1 + 0.2 + h)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Tiempo (normalizado)")
    ax.set_title({"steps":"Steps", "substeps":"Substeps (partner)", "inout":"INLET / OUT"}[mode])
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks(np.linspace(0,1,6))
    ax.xaxis.set_major_formatter(fmt)  
    # _set_midnight_xticks(ax, t, x, fmt, n_base=6, show_vlines=True, date_fmt="%m-%d")
    fig.tight_layout()
    plt.show()

def _timecol(df):
    if "DateTime" in df.columns: return "DateTime"
    if "ts" in df.columns: return "ts"
    raise ValueError("No encuentro columna temporal ('DateTime' o 'ts').")

def _pressure_map(df):
    BEDS_ALL = [
        "PSA_PI12518_Val","PSA_PI12528_Val","PSA_PI12538_Val","PSA_PI12548_Val",
        "PSA_PI12558_Val","PSA_PI12568_Val","PSA_PI12578_Val","PSA_PI12588_Val",
    ]
    tags = [c for c in BEDS_ALL if c in df.columns]
    beds = [f"b{i+1}" for i in range(len(tags))]
    return dict(zip(beds, tags))

# =========================
# 1) compute_flows (FIX)
# =========================
R = 8.314462618       # J/(mol·K)
BAR_TO_PA = 1e5       # 1 bar = 1e5 Pa
MOL_PER_NM3 = 44.615  # mol por Nm³ (0°C, 1 atm)

def compute_flows(
    df_ann: pd.DataFrame,
    volumes_m3=1.0,                        # escalar o dict {'b1':V1,...}
    T_col: str | None = "SIS_TI11102_Val", # °C (si None usa T_const_K)
    T_const_K: float = 298.15,
    clamp_negative: bool = True,
    equal_split: bool = True,              # reparte el caudal de cabecera entre camas en FEED
    # --- correcciones/flags ---
    fix_equal_to_header: bool = True,      # si b_FEED ≈ header → usar valor anterior
    shift_seconds_if_bad: float = 10.0,    # segundos hacia atrás
    eps_abs: float = 0.0,                  # tolerancia absoluta Nm3/h
    eps_rel: float = 0.0,                  # tolerancia relativa (fracción)
    # tags medidos (puedes cambiarlos si tu DF usa otros nombres)
    feed_header_tag: str = "PRC02_1_FY12504_Val",
):
    """
    Devuelve un DataFrame con:
      - t (misma columna temporal que df_ann)
      - b#_PRZFEED_mol_h, b#_PRZFEED_Nm3_h  (Δ(PV/RT)/Δt solo en PRZ→FEED)
      - b#_FEED_Nm3_h, b#_FEED_mol_h       (reparto del header en FEED; NaN→0.0)
      - total_PRZFEED_mol_h, total_PRZFEED_Nm3_h
      - feed_header_Nm3_h, feed_header_mol_h
      - sum_beds_FEED_Nm3_h, sum_beds_FEED_mol_h
    """
    # --- helpers ya existentes en tu módulo ---
    def _timecol(df):
        if "DateTime" in df.columns: return "DateTime"
        if "ts" in df.columns: return "ts"
        raise ValueError("No encuentro columna temporal ('DateTime' o 'ts').")

    def _beds_from_dfann(df: pd.DataFrame) -> list[str]:
        beds = []
        for c in df.columns:
            m = re.match(r"^b(\d+)_step$", c)
            if m: beds.append(f"b{int(m.group(1))}")
        beds.sort(key=lambda s: int(s[1:]))
        if not beds:
            raise ValueError("No encuentro columnas 'b#_step'. ¿Has corrido getPSAdata?")
        return beds

    def _pressure_map(df):
        BEDS_ALL = [
            "PSA_PI12518_Val","PSA_PI12528_Val","PSA_PI12538_Val","PSA_PI12548_Val",
            "PSA_PI12558_Val","PSA_PI12568_Val","PSA_PI12578_Val","PSA_PI12588_Val",
        ]
        tags = [c for c in BEDS_ALL if c in df.columns]
        beds = [f"b{i+1}" for i in range(len(tags))]
        return dict(zip(beds, tags))

    tcol = _timecol(df_ann)
    t = pd.to_datetime(df_ann[tcol])
    beds = _beds_from_dfann(df_ann)
    pmap = _pressure_map(df_ann)

    # Temperatura (K)
    if T_col and (T_col in df_ann.columns):
        T_K = pd.to_numeric(df_ann[T_col], errors="coerce").astype(float) + 273.15
        T_K = T_K.fillna(method="ffill").fillna(T_const_K)
    else:
        T_K = pd.Series(T_const_K, index=df_ann.index, dtype=float)

    # Volúmenes
    if isinstance(volumes_m3, dict):
        V = {b: float(volumes_m3.get(b, 1.0)) for b in beds}
    else:
        V = {b: float(volumes_m3) for b in beds}

    # Δt en horas
    dt_h = t.diff().dt.total_seconds().astype(float) / 3600.0
    dt_h.iloc[0] = np.nan
    dt_h = dt_h.replace(0.0, np.nan)

    out = pd.DataFrame({tcol: t})

    # ===== PRZ→FEED =====
    for b in beds:
        step = df_ann[f"{b}_step"].astype(object)
        partner = df_ann[f"{b}_partner"].astype(object).fillna("") if f"{b}_partner" in df_ann.columns \
                  else pd.Series("", index=df_ann.index, dtype=object)
        mask_pf = (step == "PRZ") & (partner == "FEED")

        if b not in pmap:
            out[f"{b}_PRZFEED_mol_h"] = np.nan
            out[f"{b}_PRZFEED_Nm3_h"] = np.nan
            continue

        P_bar = pd.to_numeric(df_ann[pmap[b]], errors="coerce").astype(float)  # bar
        n = (P_bar * BAR_TO_PA) * V[b] / (R * T_K)                             # mol en cama

        valid = mask_pf & mask_pf.shift(1, fill_value=False)
        q_mol_h = (n - n.shift(1)) / dt_h
        q_mol_h[~valid] = np.nan
        if clamp_negative:
            q_mol_h = q_mol_h.where(q_mol_h >= 0.0, 0.0)

        out[f"{b}_PRZFEED_mol_h"] = q_mol_h
        out[f"{b}_PRZFEED_Nm3_h"] = q_mol_h / MOL_PER_NM3

    prz_cols_mol = [c for c in out.columns if c.endswith("_PRZFEED_mol_h")]
    prz_cols_nm3 = [c for c in out.columns if c.endswith("_PRZFEED_Nm3_h")]
    out["total_PRZFEED_mol_h"] = out[prz_cols_mol].sum(axis=1, skipna=True)
    out["total_PRZFEED_Nm3_h"] = out[prz_cols_nm3].sum(axis=1, skipna=True)

    # ===== FEED (reparto header) =====
    feed_header_nm3 = pd.to_numeric(df_ann.get(feed_header_tag, np.nan), errors="coerce")
    out["feed_header_Nm3_h"] = feed_header_nm3
    out["feed_header_mol_h"] = feed_header_nm3 * MOL_PER_NM3

    bed_feed_masks = {b: (df_ann[f"{b}_step"].astype(object) == "FEED") for b in beds}
    feed_counts = sum(bed_feed_masks.values())  # nº camas en FEED

    # tamaño del shift en muestras
    # (usamos dt medio de df_ann; si no, asumimos 10 s)
    dt_s_median = float(np.nanmedian(t.diff().dt.total_seconds())) if len(t) > 1 else 10.0
    if not np.isfinite(dt_s_median) or dt_s_median <= 0:
        dt_s_median = 10.0
    shift_n = max(1, int(round(shift_seconds_if_bad / dt_s_median)))

    for b in beds:
        if equal_split:
            share = feed_header_nm3 / feed_counts.replace(0, np.nan)
            y_nm3 = share.where(bed_feed_masks[b], np.nan)
        else:
            y_nm3 = feed_header_nm3.where(bed_feed_masks[b], np.nan)

        # corrección: si b_FEED ≈ header → usar valor anterior
        if fix_equal_to_header:
            bad = y_nm3.notna() & feed_header_nm3.notna() & \
                  np.isclose(y_nm3, feed_header_nm3, rtol=eps_rel, atol=eps_abs)
            if bad.any():
                prev = y_nm3.shift(shift_n)
                y_nm3.loc[bad & prev.notna()] = prev.loc[bad & prev.notna()]

        # *** NUEVO: NaN → 0.0 ***
        y_nm3 = y_nm3.fillna(0.0)

        out[f"{b}_FEED_Nm3_h"] = y_nm3
        out[f"{b}_FEED_mol_h"] = y_nm3 * MOL_PER_NM3

    feed_cols_nm3 = [f"{b}_FEED_Nm3_h" for b in beds if f"{b}_FEED_Nm3_h" in out.columns]
    out["sum_beds_FEED_Nm3_h"] = out[feed_cols_nm3].sum(axis=1, skipna=True)
    out["sum_beds_FEED_mol_h"] = out["sum_beds_FEED_Nm3_h"] * MOL_PER_NM3

    return out

# =========================



# 2) plot_flows (FIX unpack + alineación)
# =========================
def plot_flows(
    df_ann: pd.DataFrame,
    flows_df: pd.DataFrame,
    start: str | None = None,
    end:   str | None = None,
    *,
    beds: str | list[str] = "all",          # 'all' | 'ball' | 'bedsall' | ['b1','b2',...]
    steps: str | list[str] = "all",         # 'all' | ['prz-feed','feed','bwd-purge','purge','headers','totals']
    units: str = "Nm3_h",                   # 'Nm3_h' o 'mol_h'
    figsize: tuple[float, float] = (12.0, 4.2),
):
    import numpy as np
    import re
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from matplotlib import dates as mdates
    from matplotlib.lines import Line2D

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
            raise ValueError("No encuentro columnas 'b#_step'. ¿Has corrido la función de anotado?")
        return beds

    beds_detect = globals().get("_beds_from_dfann", _beds_from_dfann_local)

    slicer = globals().get("_slice_and_normalize_time")
    if slicer is None:
        raise RuntimeError("Falta el helper '_slice_and_normalize_time' en el módulo.")

    # ---------- parseo de beds ----------
    all_beds = beds_detect(df_ann)
    if isinstance(beds, str):
        beds_norm = beds.strip().lower()
        if beds_norm in ("all", "ball", "bedsall"):
            beds = all_beds
        else:
            beds = [s.strip() for s in beds.split(",") if s.strip()]
    else:
        beds = sorted(set(beds), key=lambda b: int(b[1:]))

    # ---------- parseo de steps ----------
    steps_lookup = {
        "prz-feed": "PRZFEED",
        "feed": "FEED",
        "bwd-purge": "BWDPURGE",
        "purge": "PURGE",
        "headers": "HEADERS",
        "totals": "TOTALS",
    }
    if isinstance(steps, str):
        st_norm = steps.strip().lower()
        if st_norm in ("all", "stepsall"):
            steps = ["prz-feed","feed","bwd-purge","purge"]
        else:
            steps = [s.strip().lower() for s in st_norm.split(",") if s.strip()]
    bad = [s for s in steps if s not in steps_lookup]
    if bad:
        raise ValueError(f"Paso(s) no reconocido(s): {bad}. Usa {list(steps_lookup.keys())}.")
    steps_key = [steps_lookup[s] for s in steps]

    # ---------- normalización temporal ----------
    flows, t_f, x, fmt, *_ = slicer(flows_df, start, end)
    tcol = _timecol(flows)

    ann, _, _, _, *_ = slicer(df_ann, start, end)
    if len(ann) != len(flows):
        ann = ann.set_index(pd.to_datetime(ann[tcol]))
        flows = flows.set_index(pd.to_datetime(flows[tcol]))
        ann = ann.reindex(flows.index).reset_index(drop=True)
        flows = flows.reset_index(drop=True)
        t_f = pd.to_datetime(flows[tcol])

    # ---------- estilos y colores ----------
    cmap = plt.get_cmap("tab10")
    colors = {b: cmap((i % 10)) for i, b in enumerate(all_beds)}

    step_style = {
        "PRZFEED":  dict(linestyle="--", marker=None, linewidth=1.9),
        "FEED":     dict(linestyle="-",  marker="*",  linewidth=1.6, markersize=4.5),
        "BWDPURGE": dict(linestyle="--", marker=None, linewidth=1.9),
        "PURGE":    dict(linestyle="-",  marker="*",  linewidth=1.6, markersize=4.5),
    }
    thin_header_style = dict(linestyle="-", linewidth=1.0, alpha=0.85, color="#444444")
    thin_total_style  = dict(linestyle="-.", linewidth=1.1, alpha=0.95, color="#222222")

    # nombres de columnas por unidad
    def colname(b: str, family: str) -> str:
        if units == "Nm3_h":
            return f"{b}_{family}_Nm3_h"
        elif units == "mol_h":
            base = f"{b}_{family}_mol_h"
            if base in flows.columns:
                return base
            nm3 = f"{b}_{family}_Nm3_h"
            if nm3 in flows.columns:
                # convierte on-the-fly si existe Nm3/h
                flows[base] = pd.to_numeric(flows[nm3], errors="coerce") * MOL_PER_NM3
                return base
            return base
        else:
            raise ValueError("units debe ser 'Nm3_h' o 'mol_h'.")

    def header_cols():
        cols = []
        if units == "Nm3_h":
            for c in ("feed_header_Nm3_h","tail_header_Nm3_h","bypass_header_Nm3_h"):
                if c in flows.columns: cols.append(c)
        else:
            for c in ("feed_header_mol_h","tail_header_mol_h","bypass_header_mol_h"):
                if c in flows.columns: cols.append(c)
        return cols

    def total_cols():
        cols = []
        if units == "Nm3_h":
            for c in ("total_PRZFEED_Nm3_h","total_OUT_Nm3_h"):
                if c in flows.columns: cols.append(c)
        else:
            for c in ("total_PRZFEED_mol_h","total_OUT_mol_h"):
                if c in flows.columns: cols.append(c)
        return cols

    # ---------- figura ----------
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # 1) series por cama/step
    for b in beds:
        # *** FIX: usar escalar para crear la serie “vacía” ***
        step_series = ann[f"{b}_step"].astype(object) if f"{b}_step" in ann.columns \
                      else pd.Series("", index=ann.index, dtype=object)

        for fam in ("PRZFEED","FEED","BWDPURGE","PURGE"):
            if fam not in steps_key:
                continue
            cname = colname(b, fam)
            if cname not in flows.columns:
                continue
            y = pd.to_numeric(flows[cname], errors="coerce")

            # enmascarar para dibujar sólo dentro del step correspondiente
            # if fam == "FEED":
            #     y = y.where(step_series == "FEED", np.nan)
            # elif fam == "PURGE":
            #     y = y.where(step_series == "PURGE", np.nan)
            # PRZFEED/BWDPURGE ya vienen NaN fuera de tramo

            style = step_style[fam].copy()
            ax.plot(x, y, color=colors[b], **style)

    # 2) headers
    if "HEADERS" in steps_key:
        nice = {
            "feed_header":   "Header — FEED",
            "tail_header":   "Header — PRODUCT",
            "bypass_header": "Header — BYPASS",
        }
        for c in header_cols():
            ax.plot(x, pd.to_numeric(flows[c], errors="coerce"),
                    label=nice.get(c.split("_")[0], c), **thin_header_style)

    # 3) totals
    if "TOTALS" in steps_key:
        for c in total_cols():
            ax.plot(x, pd.to_numeric(flows[c], errors="coerce"), label=c, **thin_total_style)

    # Eje X normalizado con ticks reales
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.xaxis.set_major_formatter(fmt)
    ax.set_xlabel("Tiempo (normalizado)")

    # Etiquetas y título
    ylab = "Caudal (Nm³/h)" if units == "Nm3_h" else "Caudal (mol/h)"
    ax.set_ylabel(ylab)
    ax.set_title("Caudales por columna — selección de steps/columns")

    # Leyenda de colores (columnas)
    color_handles = [Line2D([0],[0], color=colors[b], lw=2.0, label=b.upper()) for b in beds]
    if color_handles:
        leg1 = ax.legend(handles=color_handles, title="Columnas", ncol=min(4, len(color_handles)),
                         loc="upper left", frameon=True)
        ax.add_artist(leg1)

    # Leyenda de estilos
    style_map_title = {
        "PRZFEED":  "PRZ→FEED",
        "FEED":     "FEED",
        "BWDPURGE": "BWD→PURGE",
        "PURGE":    "PURGE",
    }
    style_handles = []
    for fam in ("PRZFEED","FEED","BWDPURGE","PURGE"):
        if fam in steps_key:
            st = step_style[fam]
            style_handles.append(
                Line2D([0],[0], color="black", lw=st.get("linewidth",1.6),
                       linestyle=st.get("linestyle","-"),
                       marker=st.get("marker", None),
                       markersize=st.get("markersize", None),
                       label=style_map_title[fam])
            )
    if style_handles:
        ax.legend(handles=style_handles, title="Step", loc="upper right", frameon=True)

    # Si hay headers/totals, mini leyenda
    if ("HEADERS" in steps_key) or ("TOTALS" in steps_key):
        ax.legend(loc="lower right", frameon=True, fontsize=8)

    ax.grid(False)
    fig.tight_layout()
    plt.show()
