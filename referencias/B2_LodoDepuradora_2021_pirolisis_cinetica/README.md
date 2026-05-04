# B2 — Lodo de depuradora (2021) · Pirólisis lenta — cinéticas y termodinámica

## Puntuación: ★★★☆☆

**Pirólisis lenta de lodo de depuradora urbano con cinéticas completas por tres métodos isoconversionles y análisis de gases evolucionados. Biorresiduo urbano de alta disponibilidad.**

---

## Ficha bibliográfica

| Campo | Valor |
|---|---|
| Título | Thermodynamics, kinetics and thermal decomposition characteristics of sewage sludge during slow pyrolysis |
| Autores | — (extraer del PDF) |
| Revista | Journal of Environmental Management |
| Año | 2021 |
| DOI | [10.1016/j.jenvman.2021.111802](https://doi.org/10.1016/j.jenvman.2021.111802) |
| Acceso | ScienceDirect (Elsevier) |

---

## Combustible

| Parámetro | Valor |
|---|---|
| Combustible | Lodo de depuradora municipal (sewage sludge) |
| Origen | Planta de tratamiento de aguas residuales urbanas |
| Relevancia urbana | ★★★★★ — disponibilidad masiva en ciudades |

---

## Datos disponibles

| Dato | Disponible | Notas |
|---|---|---|
| Curvas TGA + DTG | ✓ | 3 rampas de calentamiento (5, 10, 20 K/min) |
| Análisis elemental e inmediato | ✓ | — |
| Cinéticas por KAS, FWO, Starink | ✓ | Ea y A por método |
| 3 etapas de descomposición identificadas | ✓ | 180–220°C, 220–650°C, 650–780°C |
| Gases identificados (TG-MS) | ✓ | H2O, H2, CH4, CO2, CO, NO, SO2 |
| Termodinámica (ΔH, ΔG, ΔS) | ✓ | Por etapa |
| Propiedades físicas (Cp, k) | ✗ | No reportadas |
| Datos de reactor real | ✗ | Solo TGA |

---

## Parámetros cinéticos reportados

| Método | Ea media [kJ/mol] |
|---|---|
| KAS | 413.4 |
| FWO | 419.6 |
| Starink | 416.3 |

*(Valores por etapa disponibles en las tablas del artículo)*

---

## Etapas de descomposición identificadas

| Etapa | Rango T [°C] | Fracción | Explicación |
|---|---|---|---|
| 1 | 180–220 | Deshidratación | Evaporación de agua ligada |
| 2 | 220–650 | Pirólisis principal | Descomposición de materia orgánica |
| 3 | 650–780 | Descomposición final | Carbonatos, inorgánicos |

---

## Uso en ProSimNet

- **Aplicación:** cinéticas de pirólisis para el feedstock de lodo de depuradora
- **Limitación:** las Ea elevadas (>400 kJ/mol) son características del lodo; validar vs. otras fuentes antes de implementar
- **Complemento necesario:** C1 o C2 para Cp y k del sólido; A1 para datos de reactor

---

## Notas

- El lodo de depuradora tiene mayor contenido en N, S y metales que la biomasa lignocelulósica → las cinéticas y los productos de gas son distintos.
- Relevante para el objetivo de ProSimNet de tratar biorresiduos urbanos heterogéneos.
- PDF: colocar como `B2_LodoDepuradora2021.pdf` en esta carpeta.
