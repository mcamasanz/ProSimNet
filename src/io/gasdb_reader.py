"""
@module gasdb_reader
@brief Lector de la base de datos de propiedades puras de gases en formato NDJSON.

@details
Este módulo contiene la única función responsable de leer el archivo gasdb.txt
(Newline-Delimited JSON). Su existencia como módulo independiente garantiza que
ninguna otra parte del proyecto duplique esta lógica de I/O.

La base de datos indexa cada especie por su fórmula química (ej. "N2", "CO2").
Cada línea del archivo es un objeto JSON independiente con la estructura:
  {
    "formula": "N2",
    "molar_mass": {"value": 0.028014},
    "reference": {"Tref_K": 298.15},
    "limits": {"Tmax_K": 2000.0},
    "transport": {"lj": {"sigma_A": ..., "epsilon_over_k_K": ...}},
    "polynomials": {"properties": {"mu": ..., "k": ..., "cp": ..., "h": ...}}
  }

Unidades de salida: las que estén almacenadas en la BD (ver convenciones en CLAUDE.md).
"""

from __future__ import annotations

import json
import os
from typing import Dict

from src.utils.profiling import profiled


@profiled
def read_gasdb(db_path: str) -> Dict[str, dict]:
    """
    @brief
    Lee el archivo gasdb.txt como NDJSON e indexa los registros por fórmula química.

    @details
    Recorre el archivo línea a línea (formato NDJSON: un JSON por línea).
    Valida que cada línea sea un dict con la clave "formula" y que no haya
    duplicados. Lanza excepciones descriptivas ante cualquier error de formato
    o ausencia del archivo.

    Esta función es I/O pura: no tiene estado, no modifica nada, y devuelve
    siempre el mismo resultado para el mismo archivo.

    Parameters
    ----------
    db_path : str
        Ruta al archivo gasdb.txt. Puede ser relativa o absoluta.

    Returns
    -------
    data : dict[str, dict]
        Diccionario indexado por fórmula química. Cada valor es el registro
        completo tal como aparece en el archivo JSON.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe en la ruta indicada.
    ValueError
        Si el archivo está vacío, una línea tiene fórmula vacía, o hay
        fórmulas duplicadas.
    KeyError
        Si algún registro no contiene la clave "formula".
    TypeError
        Si algún registro no es un dict JSON válido.

    Notes
    -----
    - El archivo debe estar codificado en UTF-8.
    - Las líneas vacías se ignoran silenciosamente.
    - La función no valida el contenido de las propiedades físicas; esa
      responsabilidad recae en build_pure_gas_properties (pure_gas.py).
    """
    db_path_str = str(db_path).strip()
    if not db_path_str:
        raise ValueError("db_path must be a non-empty string.")
    if not os.path.isfile(db_path_str):
        raise FileNotFoundError(f"gasdb not found: {db_path_str}")

    data: Dict[str, dict] = {}

    with open(db_path_str, "r", encoding="utf-8") as fobj:
        for line_num, raw_line in enumerate(fobj, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"gasdb: JSON parse error at line {line_num}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise TypeError(
                    f"gasdb: record at line {line_num} is not a JSON object."
                )
            if "formula" not in record:
                raise KeyError(
                    f"gasdb: record at line {line_num} is missing key 'formula'."
                )

            formula = str(record["formula"]).strip()
            if not formula:
                raise ValueError(
                    f"gasdb: empty 'formula' field at line {line_num}."
                )
            if formula in data:
                raise ValueError(
                    f"gasdb: duplicated formula '{formula}' at line {line_num}."
                )

            data[formula] = record

    if not data:
        raise ValueError(f"gasdb is empty: {db_path_str}")

    return data
