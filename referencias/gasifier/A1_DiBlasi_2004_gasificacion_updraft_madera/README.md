# A1 — Di Blasi (2004) · Gasificación updraft de madera en lecho fijo

## Puntuación: ★★★★★

> ⚠️ **AVISO — PDF INCORRECTO**: El archivo `DiBlasi_2004_AIChE.pdf` descargado NO corresponde a este artículo.
> El PDF descargado es: **Galgano & Di Blasi (2004), Combustion and Flame 139:16–27** —
> "Modeling the propagation of drying and decomposition fronts in wood" — modelo de partícula única (pirólisis), NO el modelo de gasificador updraft.
> Se necesita descargar el artículo correcto: **Di Blasi (2004), AIChE Journal 50(9):2306–2319**, DOI 10.1002/aic.10189.

**Referencia canónica para validar el modo updraft del gasificador 1D de ProSimNet.**
Proporciona perfiles axiales completos de temperatura y composición del gas productor para un reactor de escala laboratorio con madera.

---

## Ficha bibliográfica

| Campo | Valor |
|---|---|
| Título | Modeling wood gasification in a countercurrent fixed-bed reactor |
| Autores | Di Blasi, C. |
| Revista | AIChE Journal |
| Año | 2004 |
| Volumen / Páginas | 50(9), 2306–2319 |
| DOI | [10.1002/aic.10189](https://doi.org/10.1002/aic.10189) |
| Acceso | Wiley / ResearchGate (preprint libre) |

---

## Combustible y reactor

| Parámetro | Valor |
|---|---|
| Combustible | Madera (hardwood/softwood, trozos ~30 mm) |
| Tipo de reactor | Lecho fijo contracorriente (updraft) |
| Escala | Laboratorio |
| Agente gasificante | Aire |
| Caudal de aire | Variado (parámetro del estudio) |

---

## Datos disponibles para validación

| Dato | Disponible | Notas |
|---|---|---|
| Perfiles axiales de temperatura | ✓ | Gas y sólido por separado |
| Composición gas productor (CO, CO2, H2, CH4) | ✓ | A la salida del reactor |
| Conversión del char | ✓ | Perfil axial |
| Tasa de producción de gas | ✓ | |
| Cinéticas de pirólisis | ✓ | Modelo de 3 reacciones competitivas (celulosa, hemicelulosa, lignina) |
| Cinéticas gasificación del char (Boudouard, water-gas) | ✓ | Parámetros Ea y A tabulados |
| Cinética combustión del char | ✓ | |
| Propiedades físicas de la madera (Cp, k, ρ) | ✓ (parcial) | Valores constantes; complementar con C1 para Cp(T) |

---

## Parámetros cinéticos reportados

### Pirólisis (tres reacciones competitivas)
| Reacción | Ea [kJ/mol] | A [s⁻¹] |
|---|---|---|
| Madera → char | ~125 | — |
| Madera → gas | ~112 | — |
| Madera → tar | ~140 | — |

*(Valores exactos en la Tabla 1 del artículo — extraer al descargar el PDF)*

### Gasificación heterogénea del char
| Reacción | Ea [kJ/mol] |
|---|---|
| Boudouard (C + CO2 → 2CO) | ~200 |
| Water-gas (C + H2O → CO + H2) | ~196 |
| Combustión (C + O2 → CO2) | ~130 |

*(Confirmar con Tabla del artículo)*

---

## Uso en ProSimNet

- **Modo de validación:** `updraft` con `mode="updraft"` en `build_boundary_c_config`
- **Variables a comparar:** perfiles axiales de Tg, Ts, fracciones molares CO/CO2/H2/CH4 vs. posición axial z
- **Geometría de referencia:** reactor laboratorio ~0.3–0.5 m de longitud, Di ~0.1 m

---

## Notas

- Artículo más citado del área para validar modelos 1D de gasificadores updraft.
- El modelo del artículo es 1D+transiente; ProSimNet puede comparar el estado estacionario.
- Preprint disponible libremente en ResearchGate.
- PDF: colocar como `A1_DiBlasi2004.pdf` en esta carpeta.
