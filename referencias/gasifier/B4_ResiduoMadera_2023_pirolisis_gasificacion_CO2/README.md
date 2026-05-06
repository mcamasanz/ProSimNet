# B4 — Residuo de madera (2023) · Pirólisis + gasificación con CO2 — modelos IPR y DAEM

## Puntuación: ★★★★☆

**Uno de los pocos artículos que proporciona cinéticas de pirólisis Y gasificación del char en un único estudio, usando dos modelos cinéticos comparados. Residuo forestal/industrial de madera.**

---

## Ficha bibliográfica

| Campo | Valor |
|---|---|
| Título | A comparison between two kinetic models for the pyrolysis and gasification of woody wastes under a carbon dioxide atmosphere |
| Autores | — (extraer del PDF) |
| Revista | Cleaner Materials |
| Año | 2023 |
| DOI | [10.1016/j.clema.2023.100188](https://doi.org/10.1016/j.clema.2023.100188) |
| Acceso | ScienceDirect (Elsevier) |

---

## Combustible

| Parámetro | Valor |
|---|---|
| Combustibles | Residuo de pino forestal + residuo industrial de madera |
| Tipo | Biorresiduos lignocelulósicos |
| Mezclas | Pino puro, residuo industrial puro, mezcla 50/50 |

---

## Datos disponibles

| Dato | Disponible | Notas |
|---|---|---|
| TGA pirólisis en N2 | ✓ | Hasta 900°C |
| TGA gasificación en CO2 | ✓ | 800–1000°C |
| Modelo IPR (Independent Parallel Reactions) | ✓ | Ea, A, factor de peso por fracción |
| Modelo DAEM (Distributed Activation Energy) | ✓ | Distribución de Ea |
| Comparación IPR vs DAEM | ✓ | Error relativo de ajuste reportado |
| Análisis elemental e inmediato | ✓ | — |
| Composición del char residual | ✓ | — |
| Datos de reactor real | ✗ | Solo TGA |
| Propiedades físicas (Cp, k) | ✗ | — |

---

## Parámetros cinéticos reportados

### Pirólisis (modelo IPR — 3 pseudo-componentes)
| Pseudo-componente | Ea [kJ/mol] | A [s⁻¹] |
|---|---|---|
| Hemicelulosa | A determinar | — |
| Celulosa | A determinar | — |
| Lignina | A determinar | — |

### Gasificación del char con CO2 (Boudouard)
| Temperatura [°C] | Tasa de reacción | Notas |
|---|---|---|
| 800–1000 | A determinar | Cinética de primer orden reportada |

*(Completar con los valores de las tablas al descargar el PDF)*

---

## Uso en ProSimNet

- **Aplicación:** parámetros IPR para pirólisis + cinéticas Boudouard para el char
- **Ventaja clave:** el artículo cubre toda la ruta pirólisis → gasificación del char en un único experimento → coherencia interna de parámetros
- **Modo de uso:** alimentar tanto `r_pyr_*` como `r_char_boudouard` en `rhs_gasifier.py`

---

## Notas

- La comparación IPR vs DAEM es valiosa para elegir el modelo más adecuado según la fidelidad requerida.
- CO2 como agente gasificante es relevante para gasificación en atmósfera enriquecida o procesos BECCS.
- PDF: colocar como `B4_ResiduoMadera2023.pdf` en esta carpeta.
