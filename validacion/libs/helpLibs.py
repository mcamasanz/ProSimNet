import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo
import re
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

# ------------------------------------------------------------
# GETDATA
# ------------------------------------------------------------
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


# =============================================================================
#  UTILIDAD PARA LOS PLOTS
# =============================================================================
def _slice_and_normalize_time(df: pd.DataFrame, start: str|None, end: str|None):
    """recorte por fechas + x en [0,1] y formateador de ticks a hora real"""
    def _time_col(df: pd.DataFrame) -> str:
        if "DateTime" in df.columns: return "DateTime"
        if "ts" in df.columns: return "ts"
        raise ValueError("No encuentro columna temporal ('DateTime' o 'ts').")
    tcol = _time_col(df)
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

    t0, t1 = t.iloc[0], t.iloc[-1]
    def _tick_fmt(frac, pos):
        frac = 0.0 if frac < 0 else (1.0 if frac > 1 else frac)
        return (t0 + (t1 - t0) * frac).strftime("%H:%M:%S")
    fmt = FuncFormatter(_tick_fmt)

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
    ax.set_xticks(np.linspace(0,1,6))
    ax.xaxis.set_major_formatter(fmt)
    fig.tight_layout()
    plt.show()

# --------------------------
# 2) SOLO PRESIONES (eje X normalizado 0–1)
# --------------------------
def plot_beds_pressure(
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
    ax.set_ylabel("bar")
    ax.set_title("Presiones de lechos vs tiempo (X normalizado)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    plt.show()

    plt.show()
    
R = 8.314462618  # J/mol/K
BAR_TO_PA = 1e5
N_PER_NM3 = 44.615  # mol / Nm3 (0 °C, 1 atm)

R = 8.314462618   # J/mol/K
BAR_TO_PA = 1e5
N_PER_NM3 = 44.615  # mol/Nm3

def _timecol(df):
    if "DateTime" in df.columns: return "DateTime"
    if "ts" in df.columns: return "ts"
    raise ValueError("No encuentro columna temporal ('DateTime' o 'ts').")

def _beds_from_dfann(df):
    beds = sorted({c[:-5] for c in df.columns if c.endswith("_step")},
                  key=lambda s: int(s[1:]))
    if not beds:
        raise ValueError("No encuentro columnas 'b#_step'. ¿Has ejecutado getPSAdata?")
    return beds

def _pressure_map(df):
    BEDS_ALL = [
        "PSA_PI12518_Val","PSA_PI12528_Val","PSA_PI12538_Val","PSA_PI12548_Val",
        "PSA_PI12558_Val","PSA_PI12568_Val","PSA_PI12578_Val","PSA_PI12588_Val",
    ]
    tags = [c for c in BEDS_ALL if c in df.columns]
    beds = [f"b{i+1}" for i in range(len(tags))]
    return dict(zip(beds, tags))



def compute_przfeed_flow(
    df_ann: pd.DataFrame,
    *,
    volumes_m3 = 1.0,                       # escalar o dict {'b1':V1, ...}
    T_col: str | None = "SIS_TI11102_Val",  # °C (si None usa T_const_K)
    T_const_K: float = 298.15,
    clamp_negative: bool = True,
    # Condiciones “normales” para convertir a Nm3/h:
    N_ref_P_bar: float = 1.00,              # usa 1.01325 si tu “N”=1 atm
    N_ref_T_K: float = 273.15,
) -> pd.DataFrame:
    """
    Caudal PRZ→FEED por cama usando Δ(PV/RT)/Δt.
    Devuelve un DataFrame con:
      - columna de tiempo (DateTime/ts)
      - b#_PRZFEED_mol_h  (mol/h)
      - b#_PRZFEED_Nm3_h  (Nm^3/h @ P=N_ref_P_bar, T=N_ref_T_K)
      - total_PRZFEED_mol_h / total_PRZFEED_Nm3_h
      - feed_header_Nm3_h / product_header_Nm3_h / bypass_header_Nm3_h (medidores)
    """
    # --- constantes ---
    R = 8.314462618          # J/mol/K
    BAR_TO_PA = 1e5

    # --- helpers existentes ---
    tcol = _timecol(df_ann)            # igual a tu _get_time_col / _timecol
    t = pd.to_datetime(df_ann[tcol])
    beds = _beds_from_dfann(df_ann)     # detecta ['b1','b2',...]
    pmap = _pressure_map(df_ann)        # {'b1':'PSA_PI12518_Val', ...}

    # --- Temperatura (K) ---
    if T_col and (T_col in df_ann.columns):
        T_K = pd.to_numeric(df_ann[T_col], errors="coerce").astype(float) + 273.15
        T_K = T_K.fillna(method="ffill").fillna(T_const_K)
    else:
        T_K = pd.Series(T_const_K, index=df_ann.index, dtype=float)

    # --- Volúmenes por cama ---
    if isinstance(volumes_m3, dict):
        V = {b: float(volumes_m3.get(b, 1.0)) for b in beds}
    else:
        V = {b: float(volumes_m3) for b in beds}

    # --- Δt en horas (evitar 0) ---
    dt_h = t.diff().dt.total_seconds().astype(float) / 3600.0
    dt_h.iloc[0] = np.nan
    dt_h = dt_h.replace(0.0, np.nan)

    out = pd.DataFrame({tcol: t})

    # factor mol -> Nm3 a condiciones “N”
    Pn_Pa = N_ref_P_bar * BAR_TO_PA
    mol_to_Nm3 = (R * N_ref_T_K) / Pn_Pa  # Nm3/mol

    # --- PRZ→FEED por cama ---
    for b in beds:
        step = df_ann[f"{b}_step"].astype(object)
        partner = df_ann.get(f"{b}_partner", pd.Series([""]*len(df_ann), index=df_ann.index)).astype(object)
        mask_pf = (step == "PRZ") & (partner == "FEED")

        if b not in pmap:
            out[f"{b}_PRZFEED_mol_h"] = np.nan
            out[f"{b}_PRZFEED_Nm3_h"] = np.nan
            continue

        # n(t) = P*V / (R*T)
        P_bar = pd.to_numeric(df_ann[pmap[b]], errors="coerce").astype(float)  # bar
        n = (P_bar * BAR_TO_PA) * V[b] / (R * T_K)                              # mol

        # derivada solo dentro de PRZ_FEED (i e i-1 válidos)
        valid = mask_pf & mask_pf.shift(1, fill_value=False)
        q_mol_h = (n - n.shift(1)) / dt_h
        q_mol_h[~valid] = np.nan
        if clamp_negative:
            q_mol_h = q_mol_h.where(q_mol_h >= 0.0, 0.0)

        q_Nm3_h = q_mol_h * mol_to_Nm3

        out[f"{b}_PRZFEED_mol_h"] = q_mol_h
        out[f"{b}_PRZFEED_Nm3_h"] = q_Nm3_h

    # --- Totales PRZ→FEED (sumatorio camas) ---
    mol_cols = [c for c in out.columns if c.endswith("_PRZFEED_mol_h")]
    vol_cols = [c for c in out.columns if c.endswith("_PRZFEED_Nm3_h")]
    out["total_PRZFEED_mol_h"] = out[mol_cols].sum(axis=1, skipna=True)
    out["total_PRZFEED_Nm3_h"] = out[vol_cols].sum(axis=1, skipna=True)

    # --- Medidores en Nm3/h (si existen) ---
    def _pick(colnames):
        for c in colnames:
            if c in df_ann.columns:
                return c
        return None

    col_feed    = _pick(["PRC02_1_FY12504_Val", "FY12504", "PRC02_1_FY12504"])
    col_product = _pick(["PRC02_1_FY12505_Val", "FY12505", "PRC02_1_FY12505"])
    col_bypass  = _pick([
        "PSA_FI12599B_Val", "FI12599B",
        "PRC02_1_FI12599_Val", "PRC02_1_FY12599_Val", "FY12599", "PSA_FY12599_Val"
    ])

    out["feed_header_Nm3_h"]    = pd.to_numeric(df_ann[col_feed], errors="coerce") if col_feed else np.nan
    out["product_header_Nm3_h"] = pd.to_numeric(df_ann[col_product], errors="coerce") if col_product else np.nan
    out["bypass_header_Nm3_h"]  = pd.to_numeric(df_ann[col_bypass], errors="coerce") if col_bypass else np.nan

    return out


def plot_przfeed_flow(
    flows_df: pd.DataFrame,
    *,
    units: str = "Nm3",           # "Nm3" o "mol"
    start: str | None = None,
    end:   str | None = None,
    normalize_time: bool = True,  # X normalizado 0–1 (ticks con hora real)
    width: float = 12.0,
    height: float = 4.6,
    show_total: bool = True,
    show_headers: bool = True,
    N_ref_P_bar: float = 1.00,    # si units="mol" y quieres convertir headers
    N_ref_T_K: float = 273.15,
):
    """
    Dibuja caudales PRZ→FEED por cama (salida de compute_przfeed_flow).
      - units: "Nm3" o "mol" → elige *_PRZFEED_Nm3_h o *_PRZFEED_mol_h
      - start/end: recorte visual (opcional)
      - normalize_time: eje X en [0,1] con ticks mapeados a hora real
      - show_total: línea del total PRZ→FEED
      - show_headers: líneas de medidores de cabecera (feed/product/bypass)

    Requisitos: flows_df debe contener:
      - columna temporal (DateTime o ts)
      - b#_PRZFEED_mol_h / b#_PRZFEED_Nm3_h
      - total_PRZFEED_mol_h / total_PRZFEED_Nm3_h
      - feed_header_Nm3_h / product_header_Nm3_h / bypass_header_Nm3_h (opcionales)
    """
    # -------- localizar columna de tiempo --------
    if "DateTime" in flows_df.columns:
        tcol = "DateTime"
    elif "ts" in flows_df.columns:
        tcol = "ts"
    else:
        raise ValueError("No encuentro columna temporal ('DateTime' o 'ts') en flows_df.")

    flows = flows_df.copy()

    # -------- recorte temporal --------
    t_all = pd.to_datetime(flows[tcol])
    mask = np.ones(len(flows), dtype=bool)
    if start is not None:
        mask &= (t_all >= pd.Timestamp(start))
    if end is not None:
        mask &= (t_all <= pd.Timestamp(end))
    if not mask.any():
        raise ValueError("El rango start/end no solapa con los datos en flows_df.")
    flows = flows.loc[mask].reset_index(drop=True)
    t = pd.to_datetime(flows[tcol])

    # -------- elegir columnas por unidades --------
    units = units.strip().lower()
    suf = "PRZFEED_Nm3_h"
    total_col = "total_PRZFEED_Nm3_h"
    ylab = "Nm³/h"
    

    # columnas de camas: SOLO b\d+_PRZFEED_... (evita 'total_...')
    bed_cols = [c for c in flows.columns if re.match(rf"^b\d+_{suf}$", c)]
    if not bed_cols:
        raise ValueError(f"No encuentro columnas '*_{suf}' en flows_df.")

    beds = sorted({re.match(r"^(b\d+)_", c).group(1) for c in bed_cols},
                  key=lambda s: int(s[1:]))

    # -------- eje X (normalizado o tiempo real) --------
    if normalize_time:
        # normalizar a [0,1] y formatear ticks como hora real
        tnum = mdates.date2num(t)
        span = float(tnum[-1] - tnum[0]) if len(tnum) > 1 else 1.0
        x = (tnum - tnum[0]) / (span if span != 0 else 1.0)
        t0, t1 = t.iloc[0], t.iloc[-1]
        def _tick_fmt(frac, pos):
            frac = 0.0 if frac < 0 else (1.0 if frac > 1 else frac)
            return (t0 + (t1 - t0) * frac).strftime("%H:%M:%S")
        xfmt = FuncFormatter(_tick_fmt)
        xlim = (-0.02, 1.02)
    else:
        x = t
        loc = mdates.AutoDateLocator()
        xfmt = mdates.ConciseDateFormatter(loc)
        xlim = None  # lo ajusta Matplotlib

    # -------- preparar figura --------
    fig, ax = plt.subplots(1, 1, figsize=(width, height))

    # líneas por cama
    for b in beds:
        col = f"{b}_{suf}"
        if col in flows.columns:
            y = pd.to_numeric(flows[col], errors="coerce").astype(float)
            ax.plot(x, y, linewidth=1.3, label=b.upper(), alpha=0.9)

    # total
    if show_total and (total_col in flows.columns):
        ytot = pd.to_numeric(flows[total_col], errors="coerce").astype(float)
        ax.plot(x, ytot, linewidth=2.2, linestyle="--", label="TOTAL PRZ→FEED", alpha=0.95)

    # medidores cabecera (si units='Nm3' los mostramos tal cual; si 'mol', convertimos)
    if show_headers:
        # disponibles:
        fcol = "feed_header_Nm3_h"     if "feed_header_Nm3_h"     in flows.columns else None
        pcol = "product_header_Nm3_h"  if "product_header_Nm3_h"  in flows.columns else None
        bcol = "bypass_header_Nm3_h"   if "bypass_header_Nm3_h"   in flows.columns else None

        if any([fcol, pcol, bcol]):
            if units.startswith("mol"):
                # convertir Nm3/h → mol/h a condiciones "N" indicadas
                R = 8.314462618
                BAR_TO_PA = 1e5
                Pn = N_ref_P_bar * BAR_TO_PA
                Nm3_to_mol = Pn / (R * N_ref_T_K)  # mol/Nm3

                if fcol:
                    ax.plot(x, pd.to_numeric(flows[fcol], errors="coerce") * Nm3_to_mol,
                            label="Feed (medido)", linewidth=1.8, alpha=0.9)
                if pcol:
                    ax.plot(x, pd.to_numeric(flows[pcol], errors="coerce") * Nm3_to_mol,
                            label="Product (medido)", linewidth=1.8, alpha=0.9)
                if bcol:
                    ax.plot(x, pd.to_numeric(flows[bcol], errors="coerce") * Nm3_to_mol,
                            label="Bypass/Off-gas (medido)", linewidth=1.8, alpha=0.9)
            else:
                if fcol:
                    ax.plot(x, pd.to_numeric(flows[fcol], errors="coerce"),
                            label="Feed (medido)", linewidth=1.8, alpha=0.9)
                if pcol:
                    ax.plot(x, pd.to_numeric(flows[pcol], errors="coerce"),
                            label="Product (medido)", linewidth=1.8, alpha=0.9)
                if bcol:
                    ax.plot(x, pd.to_numeric(flows[bcol], errors="coerce"),
                            label="Bypass/Off-gas (medido)", linewidth=1.8, alpha=0.9)

    # decoración
    ax.set_xlabel("Tiempo normalizado" if normalize_time else "Tiempo")
    ax.set_ylabel(ylab)
    if normalize_time:
        ax.set_xlim(xlim)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.xaxis.set_major_formatter(xfmt)
    else:
        loc = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(xfmt)

    ax.set_title(f"Caudales PRZ→FEED por cama ({ylab})")
    ax.grid(True, alpha=0.15)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    plt.show()
