# Case Card ProSimNet — B3 · Shahbeig & Nosrati (2020)

> **Tipo de uso:** Cinéticas + caracterización de lodo de depuradora municipal para fuel YAML
> **Referencia:** Renewable and Sustainable Energy Reviews 119 (2020) 109567 — Universidad Tarbiat Modares, Irán
> **NO es un caso de gasificador**: TGA + Py-GC/MS + evaluación tecno-económica

---

## 1. Combustible — Lodo de depuradora municipal (MSS, Teherán)

Secado al sol 72h, granulado 3–5 mm, almacenado en bolsas herméticas.

| Análisis | Valor | Método |
|---|---|---|
| VM (volátiles) | ver rango bibliográfico: 36–75% (dry) | ASTM E870 |
| FC (carbono fijo) | ~8% media bibliográfica | calculado |
| Cenizas | 15–61% (dry) | ASTM |
| **HHV (calculado)** | **16.47 ± 0.03 MJ/kg** | Correlación Nhuchhen-Salam |
| C | 45–55% | elemental |
| H | 6–10% | elemental |
| N | 5–12% | elemental |
| S | 0.5–1.5% | elemental |
| O | 25–40% | por diferencia |

> ⚠ El artículo da rangos bibliográficos, no los valores exactos medidos de ESTE lodo.

---

## 2. TGA — Condiciones experimentales

- 4 velocidades de calentamiento: **5, 10, 30, 50 °C/min**
- Rango: temperatura ambiente → no especificado (>600°C)
- Atmósfera: N₂

**3 etapas identificadas:**
1. Zona de secado: T < 200°C
2. Pirólisis activa (zona principal): 200–600°C
3. Descomposición del char: T > 600°C

---

## 3. Parámetros cinéticos (métodos isoconversionales)

| Método | Ea media [kJ/mol] | Rango |
|---|---|---|
| **FWO** | **136.92** | (vs. B2: 225.92) |
| **KAS** | **126.62** | (vs. B2: 218.04) |
| **Starink** | ~131 | — |

> ⚠ **Diferencia significativa con B2**: B3 da Ea≈127–137 kJ/mol, mientras B2 da Ea≈218–226 kJ/mol para otro lodo. Esto refleja la heterogeneidad composicional del lodo entre regiones/tratamientos.
> La diferencia en ΔH entre Ea y ΔH es ~5 kJ/mol (indica formación favorable de productos).

---

## 4. Py-GC/MS a 700°C — identificación de productos volátiles

Productos principales (~60% del total):
- Benceno y derivados (aromáticos)
- Compuestos C₇+ (hidrocarburos >7 átomos de C)
- También: compuestos nitrogenados, alcoholes, furanos, azufrados

> No se cuantifican rendimientos másicos por especie — solo identificación cualitativa.

---

## 5. Parámetros para fuel YAML (disponibles)

```yaml
# sewage_sludge_municipal_tehran_2020
fuel_id: "sewage_sludge_municipal_tehran_2020"
description: "Lodo depuradora municipal Teherán — Shahbeig & Nosrati 2020"

heating_values:
  biomass: 16.47   # MJ/kg (HHV calculado)

kinetics:
  drying:
    A: ~1e6       # s⁻¹ (estimar)
    E: 88000      # J/mol

  pyrolysis:
    # Cinética global aparente (promedio métodos isoconversionales B3)
    A: ~1e9        # s⁻¹ (estimar de Ea=131 kJ/mol y T_pico≈450°C=723K)
    E: 131000      # J/mol  (media FWO+KAS: (136920+126620)/2 ≈ 131770)
    n: 1           # aproximación

# ⚠ Datos NO disponibles en este artículo:
#   pyrolysis_yields (char/tar/CO/CO2/H2/H2O/CH4): no cuantificados
#   Cp(T) del lodo y del char: no reportados
#   k del lodo: no reportado
#   rho_particle: no reportado
#   dp_initial: 3-5 mm (sí reportado: 3e-3 a 5e-3 m)
```

---

## 6. Valor para construir el fuel YAML del lodo

| Dato | Disponible en B2 | Disponible en B3 | Uso en ProSimNet |
|---|---|---|---|
| Ea pirólisis | 218–226 kJ/mol | 127–137 kJ/mol | Rango real; usar promedio o calibrar con experimento propio |
| HHV | ❌ | ✅ 16.47 MJ/kg | `heating_values.biomass` |
| Orden de reacción | F3 (orden 3) | no determinado | Confirma no es 1er orden |
| Productos (qualitative) | ❌ | ✅ (Py-GC/MS) | Solo identificación, no yields |
| Cp(T), k | ❌ | ❌ | Buscar en literatura específica de lodo |
| dp | ❌ | ✅ 3–5 mm | `physical.dp_initial` |

**Conclusión:** B2+B3 juntos dan: Ea (rango), HHV, dp. No dan: yields de pirólisis, Cp(T), k ni rendimiento de char. El fuel YAML del lodo NO puede completarse solo con estos dos artículos — requiere datos de B4/B5 o medición propia del bioestabilizado.
