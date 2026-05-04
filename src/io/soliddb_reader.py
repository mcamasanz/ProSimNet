"""
@module soliddb_reader
@brief Lector de la base de datos de propiedades de materiales sólidos en formato NDJSON.

@details
Módulo análogo a gasdb_reader.py pero para materiales sólidos (metales, cerámicas).
Lee el archivo soliddb.txt (Newline-Delimited JSON), indexado por "id" de material.

Estructura de cada registro:
  {
    "id": "SS316L",
    "name_es": "acero inoxidable 316L",
    "name_en": "stainless steel 316L",
    "category": "metal",
    "reference": {"Tref_K": 298},
    "limits": {"Tmax_K": 1273},
    "cap_at_tmax": {"T_K": 1273, "rho_kg_per_m3": ..., "cp_J_per_kgK": ..., "k_W_per_mK": ...},
    "polynomials": {
      "basis": "deltaT",
      "degree_max": 7,
      "properties": {
        "rho": {"units": "kg/m3", "a0_to_a7": [...]},
        "cp":  {"units": "J/kg/K", "a0_to_a7": [...]},
        "k":   {"units": "W/m/K",  "a0_to_a7": [...]}
      }
    }
  }

Unidades de salida: las almacenadas en la BD (SI).
"""

from __future__ import annotations

import json
import os
from typing import Dict

from src.utils.profiling import profiled


@profiled
def read_soliddb(db_path: str) -> Dict[str, dict]:
    """
    @brief
    Lee el archivo soliddb.txt como NDJSON e indexa los registros por id de material.

    Parameters
    ----------
    db_path : str
        Ruta al archivo soliddb.txt (relativa o absoluta).

    Returns
    -------
    data : dict[str, dict]
        Diccionario indexado por id de material. Cada valor es el registro completo.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe en la ruta indicada.
    ValueError
        Si el archivo está vacío, un id está vacío, o hay ids duplicados.
    KeyError
        Si algún registro no contiene la clave "id".
    TypeError
        Si algún registro no es un dict JSON válido.
    """
    db_path_str = str(db_path).strip()
    if not db_path_str:
        raise ValueError("db_path must be a non-empty string.")
    if not os.path.isfile(db_path_str):
        raise FileNotFoundError(f"soliddb not found: {db_path_str}")

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
                    f"soliddb: JSON parse error at line {line_num}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise TypeError(
                    f"soliddb: record at line {line_num} is not a JSON object."
                )
            if "id" not in record:
                raise KeyError(
                    f"soliddb: record at line {line_num} is missing key 'id'."
                )

            mat_id = str(record["id"]).strip()
            if not mat_id:
                raise ValueError(
                    f"soliddb: empty 'id' field at line {line_num}."
                )
            if mat_id in data:
                raise ValueError(
                    f"soliddb: duplicated id '{mat_id}' at line {line_num}."
                )

            data[mat_id] = record

    if not data:
        raise ValueError(f"soliddb is empty: {db_path_str}")

    return data
