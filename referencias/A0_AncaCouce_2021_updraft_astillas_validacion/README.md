# A0 — Anca-Couce et al. (2021) · Modelo 1D updraft con validación experimental completa

## Puntuación: ★★★★★  PRIORIDAD MÁXIMA

**Artículo de acceso abierto (CC BY) con modelo 1D updraft completo, todos los parámetros cinéticos y físicos tabulados, y validación experimental contra 5 casos con astillas de madera (humedad 8–30%). Es la referencia principal para validar ProSimNet.**

> PDF disponible en: [`../modelo1dcinetico.pdf`](../modelo1dcinetico.pdf)

---

## Ficha bibliográfica

| Campo | Valor |
|---|---|
| Título | Modelling fuel flexibility in fixed-bed biomass conversion with a low primary air ratio in an updraft configuration |
| Autores | Anca-Couce, A.; Archan, G.; Buchmayr, M.; Essl, M.; Hochenauer, C.; Scharler, R. |
| Revista | Fuel |
| Año | 2021 |
| Volumen / Páginas | 296, 120687 |
| DOI | [10.1016/j.fuel.2021.120687](https://doi.org/10.1016/j.fuel.2021.120687) |
| Licencia | CC BY 4.0 — acceso abierto |
| Institución | Graz University of Technology (Austria) + Hargassner GesmbH |

---

## Combustible y reactor

| Parámetro | Valor |
|---|---|
| Combustible | Astillas de madera (wood chips — softwood, spruce) |
| Tipo de reactor | Lecho fijo contracorriente (updraft), baja razón de aire primario |
| Escala | Caldera comercial pequeña (~30 kW) |
| Agente gasificante | Aire (λ ≈ 0.21) + recirculación de humos en algunos casos |

---

## Casos experimentales de validación (Tabla 3 del artículo)

| Caso | Humedad [% m.b.] | Aire primario [kg/h] | Recirculación | T entrada gas [°C] |
|---|---|---|---|---|
| M8 | 8 | 7.4 | No | 500 |
| M16 | 16.5 | 7.5 | No | 500 |
| M30 | 30.5 | 7.75 | No | 500 |
| M8-reci | 8 | 6.1 | 5.1 kg/h | 350 |
| M16-reci | 16.5 | 6.2 | 5.7 kg/h | 325 |

Potencia de entrada biomasa: 30.3 kW (todos los casos).

---

## Parámetros cinéticos (Tabla 1 del artículo) ← EXTRAÍDOS DEL PDF

| Reacción | Estequiometría | A | E [kJ/mol] | Fuente |
|---|---|---|---|---|
| **Secado** | H₂O(l) → H₂O(g) | 5.6×10⁶ s⁻¹ | 88 | [Mandl 2010] |
| **Pirólisis** | Biomasa → 0.234 Char + 0.082 CO + 0.114 H₂O + 0.124 CO₂ + 0.006 H₂ + 0.016 CH₄ + 0.013 C₂H₄ + 0.411 Tar | 2×10⁸ s⁻¹ | 133 | Cinética [Gronli 1996]; Composición [RAC scheme] |
| **Combustión char** | Char + (1−η/2+α/4−β/2) O₂ → η CO + (1−η) CO₂ + α/2 H₂O | 5.7×10⁷ m/s | 160 | [Di Blasi 2004] |
| **Gasificación CO₂** (Boudouard) | Char + CO₂ → 2CO + (α/2−β) H₂ + β H₂O | 1×10⁷ m/s | 220 | [Di Blasi 2004] |
| **Gasificación H₂O** (water-gas) | Char + (1−β) H₂O → CO + (1+α/2−β) H₂ | 1×10⁷ m/s | 220 | [Di Blasi 2004] |

**Composición del char:** α = 0.3934 (mol H/mol C), β = 0.0484 (mol O/mol C)

**Composición del tar:** C₂.₃₄₆₆H₃.₉₆₇₁O₁.₅₂₉₆

**Ratio CO/CO₂ en combustión del char (Ecuación 11):**
```
η = 12·exp(−3300/Ts) / (1 + 12·exp(−3300/Ts))
```

---

## Propiedades físicas (Tabla 2 del artículo) ← EXTRAÍDAS DEL PDF

### Geometría del reactor

| Propiedad | Valor | Unidades |
|---|---|---|
| Longitud reactor | 0.25 | m |
| Diámetro reactor | 0.25 | m |
| Porosidad del lecho (εr) | 0.6 | — |
| Densidad partícula biomasa seca | 430 | kg/m³ |
| Diámetro inicial de partícula | 0.01 | m |
| Permeabilidad del lecho (κs) | 1×10⁻⁸ | m² |

### Propiedades de transporte

| Propiedad | Valor | Unidades |
|---|---|---|
| Viscosidad gas (μg) | 3×10⁻⁵ | kg/m/s |
| Conductividad térmica biomasa (λ_biomasa) | 0.17 | W/m/K |
| Conductividad térmica char (λ_char) | 0.10 | W/m/K |
| Conductividad térmica gas (λg) | 0.0258 | W/m/K |
| Emisividad del sólido | 0.9 | — |

### Entalpías de combustión (calores de reacción)

| Especie | Δh [MJ/kg] |
|---|---|
| Biomasa seca | 19.13 |
| Char | 32.37 |
| Tar | 21.44 |

### Capacidades caloríficas Cp = f(T) [J/kg/K]

| Especie | Cp(T) [J/kg/K] |
|---|---|
| Biomasa seca | 1500 + 1.0·T |
| Char | 420 + 2.09·T − 0.000685·T² |
| Humedad | 4200 (constante) |
| CO | 979.7 + 0.193·T |
| CO₂ | 594.3 + 0.977·T − 0.000331·T² |
| H₂O | 1648 + 0.64·T |
| H₂ | 14346 − 0.2679·T + 0.000917·T² |
| CH₄ | 1327 + 3.144·T |
| C₂H₄ | 238.2 + 4.854·T − 0.00176·T² |
| Tar | −100 + 4.4·T − 0.00157·T² |
| O₂ | 807 + 0.399·T − 0.000117·T² |
| N₂ | 976.4 + 0.183·T |

> **Nota:** T en Kelvin. Válidas en [300–1500 K] (base NIST).

---

## Datos de validación disponibles

| Dato | Disponible | Descripción |
|---|---|---|
| Perfiles axiales de Tgas y Tsólido | ✓ | 3 posiciones de termopar por caso → Fig. 3 |
| Composición gas a la salida | ✓ | CO, CO₂, H₂O, H₂, CxHy, Tar en % masa → Fig. 4 |
| Composición gas a mitad de reactor (M30) | ✓ | Tabla 7: CO=30.3%, CO₂=4.2%, H₂=6.6%, CH₄=2.0% vol. seco |
| Flujos másicos de sólido y gas a lo largo del reactor | ✓ | Fig. 1 (M30) |
| Temperaturas máximas en la parrilla | ✓ | Tabla 6 para todos los casos |
| Balances de masa y energía | ✓ | Tabla 4 (M30): error menor |

### Composición del gas productor a la salida — caso M30 (Tabla 5)

| Especie | Medido [% masa] | Modelo [% masa] |
|---|---|---|
| CO | 18.2 | 19.6 |
| CO₂ | 7.3 | 6.9 |
| H₂O | 22.4 | 20.9 |
| H₂ | 0.3 | 0.4 |
| CH₄ | 0.7 | 0.6 |
| C₂H₄ + CxHy | 0.4 | 0.5 |
| Tar | 14.4 | 15.1 |

### Temperaturas máximas en parrilla (Tabla 6)

| Caso | T_max modelo [°C] | T medida a 5 cm [°C] | Diferencia [°C] |
|---|---|---|---|
| M8 | 1179 | 1079 | −100 |
| M16 | 1184 | 1080 | −104 |
| M30 | 1197 | 1106 | −90 |
| M8-reci | 1060 | 947 | −112 |
| M16-reci | 1054 | 960 | −94 |

---

## Uso en ProSimNet

### Parámetros directamente incorporables

1. **`rhs_gasifier.py`** — todas las constantes cinéticas (Tabla 1): A, E para secado, pirólisis, combustión char, Boudouard, water-gas
2. **`soliddb.txt`** — Cp(T) para biomasa seca y char (Tabla 2): polinomios lineales/cuadráticos en T
3. **`gasdb.txt`** — Cp(T) para CO, CO₂, H₂O, H₂, CH₄, C₂H₄, O₂, N₂ (Tabla 2): a verificar vs. los polinomios grado-7 existentes
4. **`build_initial_conditions`** — geometría de referencia: L=0.25 m, D=0.25 m, εr=0.6, dp=0.01 m, ρs=430 kg/m³
5. **`build_boundary_c_config`** — condiciones de contorno de los 5 casos experimentales

### Variables a comparar en el test de validación

```python
# Perfiles axiales a estado estacionario:
T_gas(z),  T_solid(z)          # Fig. 1 y Fig. 3
y_CO(z),   y_CO2(z),  y_H2(z)  # Fig. 2
# A la salida:
y_exit_CO, y_exit_CO2, ...     # Tabla 5 (M30), Fig. 4 (todos los casos)
# Temperatura máxima en parrilla:
T_max_solid                     # Tabla 6
```

### Caso de inicio recomendado

**M30** (humedad 30%): caso más desafiante y con más datos publicados (perfiles axiales completos en Fig. 1 y Fig. 2, composición en el medio del reactor en Tabla 7).

---

## Notas importantes

- El modelo del artículo usa el **Shrinking Core Model (SCM)** para el char — ProSimNet usa el mismo enfoque con `a_p` dinámico.
- Las reacciones en fase gas (WGS, cracking de tar) **no se incluyen** en el modelo del artículo para este reactor — justificado por la baja temperatura de los volátiles.
- El ratio CO/CO₂ en la combustión del char es variable (Ecuación 11) — esto es más riguroso que un valor constante.
- La conductividad efectiva del lecho (λs) se calcula con el modelo de Tsotsas (VDI Heat Atlas) — referencia para la correlación de λ_efect del lecho.
- PDF en acceso abierto CC BY: [`../modelo1dcinetico.pdf`](../modelo1dcinetico.pdf)
