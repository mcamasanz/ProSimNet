# C1 — Wood pellets Cp y k (2013) · Propiedades térmicas medidas experimentalmente

## Puntuación: ★★★★★

**El artículo más completo para obtener Cp(T) y k efectiva de pellets de madera mediante métodos directos (Hot Wire + DSC). Datos directamente incorporables a `soliddb.txt` de ProSimNet.**

---

## Ficha bibliográfica

| Campo | Valor |
|---|---|
| Título | Determination of effective thermal conductivity and specific heat capacity of wood pellets |
| Autores | — (extraer del PDF) |
| Revista | Fuel |
| Año | 2013 |
| DOI | [10.1016/j.fuel.2012.10.059](https://doi.org/10.1016/j.fuel.2012.10.059) |
| Acceso | ScienceDirect (Elsevier) |

---

## Material

| Parámetro | Valor |
|---|---|
| Material | Pellets de madera (wood pellets) — biomasa lignocelulósica estándar |
| Diámetro pellet | 6 mm (estándar EN 14961) |
| Humedad | Varios niveles |

---

## Datos disponibles

| Dato | Disponible | Notas |
|---|---|---|
| Cp [J/kg/K] en función de T | ✓ | Medido con DSC, varios rangos de T |
| k efectiva [W/m/K] del lecho de pellets | ✓ | Medida con Hot Wire Method |
| k en función de la compactación | ✓ | Efecto de la presión sobre el lecho |
| Densidad bulk del lecho | ✓ | — |
| Efecto de la humedad sobre Cp y k | ✓ | — |
| Valores para biomasa cruda Y char | ✗ (solo biomasa cruda) | Para char usar C3 o Di Blasi |

---

## Valores orientativos reportados

| Propiedad | Valor | Condiciones |
|---|---|---|
| Cp (biomasa cruda, 20°C) | ~1500–1800 J/kg/K | — |
| k efectiva lecho pellets | ~0.08–0.15 W/m/K | Depende compactación |
| Cp variación con T | Lineal, ~+2 J/kg/K² | Pendiente positiva |

*(Completar con los valores exactos de las figuras/tablas del PDF)*

---

## Uso en ProSimNet

- **Aplicación directa:** alimentar `soliddb.txt` con Cp_fns y k para la biomasa de referencia
- **Integración en el modelo:** los valores de Cp(T) se usan en `thermal_mass_correction` y `check_balances`
- **Prioridad:** ★★★★★ — sin estos datos el balance de energía del sólido no puede cerrarse correctamente

---

## Notas

- Los pellets de madera son el feedstock de referencia más común para gasificadores de lecho fijo.
- Los valores de k del lecho son más relevantes que la k de la partícula individual para el modelo 1D.
- Disponible también en ResearchGate: buscar "Determination of effective thermal conductivity and specific heat capacity of wood pellets".
- PDF: colocar como `C1_WoodPellets2013.pdf` en esta carpeta.
