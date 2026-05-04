"""
@module profiling
@brief Utilidad de profiling de funciones para modo desarrollo.

@details
Dos flags globales controlan el comportamiento:

    _develop   : bool — habilita el modo desarrollo (análogo al antiguo _develop de columnv2).
    _benchmark : bool — activa el registro de llamadas y tiempos por función.

Ambos deben ser True para que el decorador `profiled` registre datos.
Si cualquiera es False el decorador es un pass-through puro: coste cero en producción.

Uso típico en notebook o script:

    import src.utils.profiling as prof
    prof.set_develop(True)
    prof.set_benchmark(True)
    prof.reset_benchmark()

    # ... ejecutar simulación ...

    prof.print_benchmark_functions()

Unidades de tiempo: segundos internamente, milisegundos en el informe.
"""

from __future__ import annotations

import time
from functools import wraps
from typing import Callable

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# Flags globales
# ═══════════════════════════════════════════════════════════════════════════════

_develop:   bool = False
_benchmark: bool = False

# Almacén de datos: { qualname → {"calls": int, "total_s": float} }
_PROFILE_DATA: dict = {}


# ═══════════════════════════════════════════════════════════════════════════════
# API de control
# ═══════════════════════════════════════════════════════════════════════════════

def set_develop(value: bool) -> None:
    """Activa o desactiva el modo desarrollo."""
    global _develop
    _develop = bool(value)


def set_benchmark(value: bool) -> None:
    """Activa o desactiva el registro de profiling (requiere _develop=True)."""
    global _benchmark
    _benchmark = bool(value)


def reset_benchmark() -> None:
    """Limpia todos los datos acumulados de profiling."""
    _PROFILE_DATA.clear()


def is_active() -> bool:
    """Devuelve True si el profiling está activo."""
    return _develop and _benchmark


# ═══════════════════════════════════════════════════════════════════════════════
# Decorador
# ═══════════════════════════════════════════════════════════════════════════════

def profiled(fn: Callable) -> Callable:
    """
    Decorador que registra el número de llamadas y el tiempo acumulado de `fn`.

    Si `_develop` o `_benchmark` son False, la función se ejecuta directamente
    sin ningún overhead de medición.

    Parameters
    ----------
    fn : Callable — función a instrumentar

    Returns
    -------
    Callable — función envuelta (o la original si profiling está inactivo)
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not (_develop and _benchmark):
            return fn(*args, **kwargs)

        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        dt = time.perf_counter() - t0

        key = fn.__qualname__
        if key not in _PROFILE_DATA:
            _PROFILE_DATA[key] = {"calls": 0, "total_s": 0.0, "module": fn.__module__}
        _PROFILE_DATA[key]["calls"]   += 1
        _PROFILE_DATA[key]["total_s"] += dt

        return result

    return wrapper


# ═══════════════════════════════════════════════════════════════════════════════
# Informe
# ═══════════════════════════════════════════════════════════════════════════════

def print_benchmark_functions(top_n: int = None) -> pd.DataFrame:
    """
    Imprime y devuelve el informe de profiling ordenado por tiempo total descendente.

    Parameters
    ----------
    top_n : int, opcional — si se indica, muestra solo las top_n funciones más lentas.

    Returns
    -------
    pd.DataFrame con columnas:
        función, módulo, llamadas, total [ms], media [ms], % tiempo
    """
    if not _PROFILE_DATA:
        print("No hay datos de profiling. ¿Se activaron set_develop(True) y set_benchmark(True)?")
        return pd.DataFrame()

    rows = []
    for qualname, v in _PROFILE_DATA.items():
        rows.append({
            "función":    qualname,
            "módulo":     v.get("module", ""),
            "llamadas":   v["calls"],
            "total [ms]": v["total_s"] * 1e3,
            "media [ms]": v["total_s"] / max(v["calls"], 1) * 1e3,
        })

    df = pd.DataFrame(rows).sort_values("total [ms]", ascending=False).reset_index(drop=True)

    total_ms = df["total [ms]"].sum()
    df["% tiempo"] = (df["total [ms]"] / max(total_ms, 1e-30) * 100.0).round(2)

    if top_n is not None:
        df = df.head(int(top_n))

    # ── Imprimir ──────────────────────────────────────────────────────────────
    n_calls_total = df["llamadas"].sum()
    print("=" * 72)
    print(f"  BENCHMARK DE FUNCIONES  —  {len(df)} funciones  |  {n_calls_total:,} llamadas totales")
    print(f"  Tiempo instrumentado total: {total_ms:.2f} ms")
    print("=" * 72)

    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.max_rows", None)
    print(df.to_string(index=True))
    print("=" * 72)

    return df
