# A3 — Co-gasificación biomasa+carbón (2022) · Lecho fijo updraft

## Puntuación: ★★★★☆

**Datos experimentales de composición del syngas y relación H2/CO en gasificador updraft a escala piloto. Útil para validar el modo updraft con biomasa lignocelulósica.**

---

## Ficha bibliográfica

| Campo | Valor |
|---|---|
| Título | Co-gasification of biomass and coal in a top-lit updraft fixed bed gasifier: Syngas composition and its interchangeability with natural gas for combustion applications |
| Autores | — (extraer del PDF) |
| Revista | Fuel |
| Año | 2022 |
| DOI | [10.1016/j.fuel.2022.123489](https://doi.org/10.1016/j.fuel.2022.123489) |
| Acceso | ScienceDirect (Elsevier) |

---

## Combustible y reactor

| Parámetro | Valor |
|---|---|
| Combustibles | Biomasa lignocelulósica (palm kernel shell) + carbón (70/85/100% biomasa) |
| Tipo de reactor | Lecho fijo updraft (top-lit) |
| Escala | Laboratorio / Piloto |
| Agente gasificante | Aire |

---

## Datos disponibles para validación

| Dato | Disponible | Notas |
|---|---|---|
| Composición syngas (H2, CO, CH4, CO2) | ✓ | Para 3 proporciones biomasa/carbón |
| Relación H2/CO | ✓ | 0.42–0.59 según proporción biomasa |
| Temperatura de operación | ✓ (parcial) | Zona de combustión |
| Rendimiento energético del gas | ✓ | LHV del syngas |
| Perfiles axiales de temperatura | ✗ | No reportados detalladamente |
| Cinéticas | ✗ | No es el foco del artículo |

---

## Parámetros de syngas reportados

| Proporción biomasa | H2/CO | CO [%] | H2 [%] | CH4 [%] | CO2 [%] |
|---|---|---|---|---|---|
| 70% biomasa | 0.57–0.59 | — | — | — | — |
| 85% biomasa | 0.49–0.51 | — | — | — | — |
| 100% biomasa | 0.42–0.46 | — | — | — | — |

*(Completar con valores exactos del artículo)*

---

## Uso en ProSimNet

- **Modo de validación:** `updraft` — comparar composición del syngas a la salida
- **Limitación:** no hay perfiles axiales de temperatura; validación solo de salida
- **Complemento:** usar con A1 (Di Blasi) para los perfiles axiales

---

## Notas

- Interesante para estudiar el efecto de la mezcla biomasa/carbón en la composición del gas.
- PDF: colocar como `A3_CoGasificacion2022.pdf` en esta carpeta.
