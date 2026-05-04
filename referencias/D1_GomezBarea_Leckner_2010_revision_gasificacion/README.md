# D1 — Gómez-Barea & Leckner (2010) · Revisión de modelos de gasificación en lecho fluidizado

## Puntuación: ★★★★★

**La revisión más completa y citada sobre modelado de gasificación de biomasa en lecho fluidizado. Fuente consolidada de cinéticas heterogéneas del char (Boudouard, water-gas, combustión) y correlaciones de transferencia. Referencia obligatoria para cualquier modelo de gasificación.**

---

## Ficha bibliográfica

| Campo | Valor |
|---|---|
| Título | Modeling of biomass gasification in fluidized bed |
| Autores | Gómez-Barea, A.; Leckner, B. |
| Revista | Progress in Energy and Combustion Science |
| Año | 2010 |
| Volumen / Páginas | 36(4), 444–509 |
| DOI | [10.1016/j.pecs.2009.12.002](https://doi.org/10.1016/j.pecs.2009.12.002) |
| Acceso | ScienceDirect (Elsevier) — revisar acceso institucional CIRCE |

---

## Contenido

Esta revisión cubre exhaustivamente:

### Cinéticas del char — parámetros de referencia

| Reacción | Nombre | Ea [kJ/mol] | Notas |
|---|---|---|---|
| C + O2 → CO2 | Combustión directa | ~130–150 | Muy rápida, zona de combustión |
| C + CO2 → 2CO | Boudouard | ~200–250 | Dominante en zona de reducción |
| C + H2O → CO + H2 | Water-gas | ~180–220 | Activa con vapor |
| CO + H2O ↔ CO2 + H2 | Water-gas shift | ~80–100 | Reacción homogénea |
| C + 2H2 → CH4 | Methanación | ~200 | Lenta, solo a alta P |

*(Confirmar valores exactos en las tablas del artículo — extracción prioritaria al descargar PDF)*

### Correlaciones de transferencia de calor y masa incluidas

| Correlación | Aplicación |
|---|---|
| Ranz-Marshall | h_gas-sólido para partículas esféricas |
| Nusselt vs. Re para lechos | h efectiva del lecho |
| Ergun | Caída de presión en lecho fijo |

### Modelos de partícula del char incluidos

| Modelo | Descripción |
|---|---|
| Shrinking Core Model (SCM) | Reacción en la superficie externa |
| Uniform Conversion Model (UCM) | Reacción homogénea en toda la partícula |
| Shrinking Particle Model (SPM) | Partícula que se reduce en tamaño |

---

## Datos disponibles

| Dato | Disponible | Notas |
|---|---|---|
| Cinéticas heterogéneas del char (Ea, A) | ✓ | Tabla comparativa de múltiples estudios |
| Cinéticas homogéneas gas (WGS, oxidación CO, CH4) | ✓ | — |
| Correlaciones de transferencia h_gs y h_gw | ✓ | — |
| Survey de modelos de gasificación 1D/2D publicados | ✓ | ~50+ modelos comparados |
| Validación experimental de modelos | ✓ | Comparativa entre modelos |
| Cinéticas de pirólisis | ✓ (parcial) | Modelos simplificados |

---

## Uso en ProSimNet

- **Aplicación principal:** fuente de referencia para todos los parámetros cinéticos de las reacciones del char en `rhs_gasifier.py`
- **Correlaciones de transporte:** validar las correlaciones de Ranz-Marshall usadas en `compute_transfer_coefficients`
- **Modelos de partícula:** seleccionar entre SCM/UCM/SPM según el tamaño de partícula del feedstock
- **Prioridad:** ★★★★★ — lectura obligatoria antes de implementar cualquier reacción heterogénea del char

---

## Notas

- >1000 citas en Google Scholar — el estándar de referencia del campo.
- El artículo cubre lecho fluidizado pero los parámetros cinéticos del char son válidos para lecho fijo también.
- Los autores son del grupo de Chalmers (Suecia) + Universidad de Sevilla (España).
- PDF: colocar como `D1_GomezBarea_Leckner2010.pdf` en esta carpeta.
