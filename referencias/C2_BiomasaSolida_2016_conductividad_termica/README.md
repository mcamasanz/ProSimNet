# C2 — Conductividad térmica biomasa sólida (2016) · Estudio comparativo de 8+ tipos

## Puntuación: ★★★★☆

**Medición comparativa de k para múltiples tipos de biomasa sólida incluyendo paja de trigo, miscanthus, astillas y pellets torrefactados. Cubre biorresiduos urbanos y agrícolas relevantes para ProSimNet.**

---

## Ficha bibliográfica

| Campo | Valor |
|---|---|
| Título | Comparative Study of the Thermal Conductivity of Solid Biomass Fuels |
| Autores | — (extraer del PDF) |
| Revista | Energy & Fuels (ACS Publications) |
| Año | 2016 |
| DOI | [10.1021/acs.energyfuels.5b02261](https://doi.org/10.1021/acs.energyfuels.5b02261) |
| Acceso | ACS Publications |

---

## Materiales estudiados

| Biomasa | Tipo | Relevancia para RSU |
|---|---|---|
| Paja de trigo | Residuo agrícola | ★★★★☆ |
| Miscanthus | Cultivo energético | ★★★☆☆ |
| Astilla de madera | Residuo forestal/industria maderera | ★★★★☆ |
| Pellets torrefactados | Biomasa pretratada | ★★★☆☆ |
| Pellets estándar | Biomasa lignocelulósica | ★★★★★ |

---

## Datos disponibles

| Dato | Disponible | Notas |
|---|---|---|
| k efectiva [W/m/K] para cada biomasa | ✓ | Tabla comparativa principal |
| Efecto de la temperatura sobre k | ✓ (parcial) | Para algunos tipos |
| Efecto de la humedad sobre k | ✓ | — |
| k en dirección axial vs. radial (anisotropía) | ✓ (parcial) | Para madera: k_axial ≈ 0.1 W/m/K, k_radial ≈ 0.05 W/m/K |
| Cp | ✗ | Referencia a otros artículos — usar C1 |
| Propiedades del char | ✗ | — |

---

## Valores orientativos reportados

| Biomasa | k [W/m/K] | Notas |
|---|---|---|
| Madera (axial) | ~0.10 | A temperatura ambiente, base seca |
| Madera (radial) | ~0.05 | — |
| Paja de trigo | ~0.05–0.08 | — |
| Astilla de madera | ~0.08–0.12 | Depende de densidad |
| Pellets compactados | ~0.10–0.15 | Mayor k por mejor contacto |

*(Completar con los valores exactos de la Tabla comparativa del PDF)*

---

## Uso en ProSimNet

- **Aplicación:** seleccionar k para el feedstock específico del caso de validación
- **Complemento a C1:** C1 da Cp(T) de pellets; C2 amplía k a otros tipos de biomasa
- **Integración:** el valor de k del sólido entra en el coeficiente de transferencia efectivo del lecho

---

## Notas

- La anisotropía de la madera (k axial ≠ k radial) es relevante para partículas grandes; para el modelo 1D se usa un valor efectivo.
- Para biomasa húmeda, k aumenta significativamente — considerar efecto de la humedad.
- PDF: colocar como `C2_BiomasaThermalConductivity2016.pdf` en esta carpeta.
