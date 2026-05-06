# Case Card ProSimNet — A0 · Anca-Couce et al. (2021)

> **Tipo de caso:** 1D updraft, lecho fijo contracorriente, biomasa (astillas de pícea/spruce)
> **Referencia:** Fuel 296 (2021) 120687 — open access CC BY
> **5 casos experimentales:** M8, M16, M30 (sin recirculación) · M8-reci, M16-reci (con recirculación de gases de combustión)

---

## 1. Geometría del reactor

| Parámetro | Nombre ProSimNet | Valor | Unidades |
|---|---|---|---|
| Longitud del lecho | `dz * N` | 0.25 | m |
| Diámetro interno | `Di` | 0.25 | m |
| Sección transversal | `Ai = π/4 · Di²` | 0.04909 | m² |
| Perímetro interno | `Pi = π · Di` | 0.7854 | m |
| Porosidad del lecho | `epsi_r` | 0.6 | — |
| Permeabilidad del lecho | — | 1×10⁻⁸ | m² |
| Nº de celdas (recomendado) | `N` | 50–100 | — |

```python
Di   = 0.25       # [m]
L    = 0.25       # [m]
N    = 78         # (original del artículo; 50 es suficiente para ProSimNet)
dz   = L / N      # [m]
Ai   = 3.14159 / 4 * Di**2   # = 0.04909 m²
Pi   = 3.14159 * Di           # = 0.7854 m
epsi_r = 0.6
```

---

## 2. Combustible — fuel YAML (spruce / astillas de pícea)

### 2.1. Propiedades físicas de la partícula

| Parámetro YAML | Valor | Unidades |
|---|---|---|
| `physical.rho_particle` | 430.0 | kg/m³ |
| `physical.dp_initial` | 0.010 | m |

### 2.2. Valores caloríficos

| Parámetro YAML | Valor | Unidades |
|---|---|---|
| `heating_values.biomass` | 19.13 | MJ/kg |
| `heating_values.char` | 32.37 | MJ/kg |
| `heating_values.tar` | 21.44 | MJ/kg |

### 2.3. Propiedades térmicas del sólido

| Especie | `Cp_poly_T` [a₀, a₁, a₂] | Cp(T) = a₀ + a₁·T + a₂·T² [J/kg/K] | `k` [W/m/K] |
|---|---|---|---|
| **biomass** | `[1500.0, 1.0]` | 1500 + 1.0·T | 0.17 |
| **char** | `[420.0, 2.09, -6.85e-4]` | 420 + 2.09·T − 6.85×10⁻⁴·T² | 0.10 |
| **moisture** | `[4200.0]` | 4200 (constante) | 0.60 |

### 2.4. Composición del char

| Parámetro YAML | Valor | Significado |
|---|---|---|
| `char_composition.alpha` | 0.3934 | mol H / mol C en CHₐOβ |
| `char_composition.beta` | 0.0484 | mol O / mol C en CHₐOβ |

### 2.5. Tar — pseudo-componente

| Parámetro YAML | Valor | Unidades |
|---|---|---|
| `tar.a` | 2.3466 | átomos C en C_a H_b O_c |
| `tar.b` | 3.9671 | átomos H |
| `tar.c` | 1.5296 | átomos O |
| `tar.MW` | ~56.60 | g/mol (calculado: 2.3466×12 + 3.9671×1 + 1.5296×16) |
| `tar.Cp_poly_T` | `[-100.0, 4.4, -1.57e-3]` | J/kg/K |

### 2.6. Rendimientos de pirólisis (fracciones másicas, suma = 1)

| Especie | Fracción másica |
|---|---|
| char | 0.234 |
| CO | 0.082 |
| H₂O | 0.114 |
| CO₂ | 0.124 |
| H₂ | 0.006 |
| CH₄ | 0.016 |
| C₂H₄ | 0.013 |
| tar | 0.411 |
| **SUMA** | **1.000** |

### 2.7. Cp de todas las especies gaseosas [J/kg/K] — de Tabla 2

| Especie | a₀ | a₁ | a₂ | Cp(T) |
|---|---|---|---|---|
| CO | 979.7 | 0.193 | — | 979.7 + 0.193·T |
| CO₂ | 594.3 | 0.977 | −3.31×10⁻⁴ | cuadrático |
| H₂O | 1648.0 | 0.64 | — | lineal |
| H₂ | 14346.0 | −0.2679 | 9.17×10⁻⁴ | cuadrático |
| O₂ | 807.0 | 0.399 | −1.17×10⁻⁴ | cuadrático |
| CH₄ | 1327.0 | 3.144 | — | lineal |
| C₂H₄ | 238.2 | 4.854 | −1.76×10⁻³ | cuadrático |
| tar | −100.0 | 4.4 | −1.57×10⁻³ | cuadrático |
| N₂ | 976.4 | 0.183 | — | lineal |

> Nota: Los Cp del gasdb.txt de ProSimNet son polinomios en base (T−Tref). Los valores anteriores son en base T absoluta (= como los usa A0). Al construir el fuel YAML o validar gasdb, verificar la base del polinomio.

---

## 3. Cinéticas de reacción — Tabla 1

| Reacción | A | Ea [kJ/mol] | Unidades A | Modelo |
|---|---|---|---|---|
| Secado | 5.6×10⁶ | 88 | s⁻¹ | Arrhenius 1er orden en ρ_moisture |
| Pirólisis | 2×10⁸ | 133 | s⁻¹ | Arrhenius 1er orden en ρ_biomass |
| Char oxidation (C+O₂) | 5.7×10⁷ | 160 | m/s | SCM + Ranz-Marshall |
| CO₂ gasification (Boudouard) | 1×10⁷ | 220 | m/s | SCM + Ranz-Marshall |
| Steam gasification (Water-gas) | 1×10⁷ | 220 | m/s | SCM + Ranz-Marshall |

**Relación CO/CO₂ en oxidación del char (Ecuación 11):**
```
η = 12·exp(−3300/Ts) / (1 + 12·exp(−3300/Ts))
→ en ProSimNet: co_co2_ratio.model = "anca_couce_2017", C1=12.0, C2=3300.0
```

> Sin reacciones homogéneas (WGS, tar cracking) — el artículo justifica explícitamente su omisión para esta tecnología.

---

## 4. Condiciones de contorno por caso — Tabla 3

### 4.1. Parámetros de entrada comunes

```python
# Composición del aire (molar)
y_air = np.zeros(9)   # [CO, CO2, H2O, H2, O2, CH4, C2H4, tar, N2]
y_air[4] = 0.21       # O2
y_air[8] = 0.79       # N2

# Reactor dimensions
Ai = 0.04909  # m²

# Conversión ṁ_aire [kg/h] → v_gas_in [m/s] a T_in y P=1 atm
# C_total = P/(R·T_in)
# v_in = (ṁ_aire/3600/MW_air) / (C_total · Ai)
# MW_air = 0.02897 kg/mol

# Conversión ṁ_biomasa [kg/h] → v_solid [m/s]
# rho_bulk_dry = rho_particle × (1 - epsi_r) = 430 × 0.4 = 172 kg/m³_bed
# rho_moisture_bulk = rho_bulk_dry × (MC_wb / (1 - MC_wb))
# v_solid = (ṁ_biomasa_wet/3600) / ((rho_biomass_bulk + rho_moisture_bulk) × Ai)
```

### 4.2. Tabla resumen de condiciones de entrada

| Caso | ṁ_biomasa [kg/h] | MC [% wb] | ṁ_aire [kg/h] | ṁ_FGR [kg/h] | T_gas_in [°C] | ER (aire) | Q_bottom [kW] | Q_top [kW] |
|---|---|---|---|---|---|---|---|---|
| **M8** | 6.25 | 8.0 | 7.40 | — | 500 | 0.213 | 2.0 | 0.75 |
| **M16** | 7.00 | 16.5 | 7.50 | — | 500 | 0.213 | 2.0 | 0.75 |
| **M30** | 8.70 | 30.5 | 7.75 | — | 500 | 0.213 | 2.0 | 0.75 |
| **M8-reci** | 6.25 | 8.0 | 6.10 | 5.10 | 350 | 0.176 | 1.8 | 0.75 |
| **M16-reci** | 7.00 | 16.5 | 6.20 | 5.70 | 325 | 0.176 | 1.8 | 0.75 |

### 4.3. Velocidades y densidades de entrada calculadas

| Caso | v_gas_in [m/s] | T_gas_in [K] | y_gas_in (molar) | v_solid [m/s] | ρ_biomass_in [kg/m³] | ρ_moisture_in [kg/m³] |
|---|---|---|---|---|---|---|
| **M8** | 0.0918 | 773.15 | O₂=0.21, N₂=0.79 | 1.89×10⁻⁴ | 172 | 15.0 |
| **M16** | 0.0930 | 773.15 | O₂=0.21, N₂=0.79 | 1.92×10⁻⁴ | 172 | 34.0 |
| **M30** | 0.0960 | 773.15 | O₂=0.21, N₂=0.79 | 1.99×10⁻⁴ | 172 | 75.5 |
| **M8-reci** | 0.1115 | 623.15 | O₂=0.141, CO₂=0.057, H₂O=0.050, N₂=0.752 | 1.89×10⁻⁴ | 172 | 15.0 |
| **M16-reci** | — | 598.15 | O₂=0.141, CO₂=0.056, H₂O=0.055, N₂=0.748 | 1.92×10⁻⁴ | 172 | 34.0 |

> **Cálculo v_gas_in:** `v = (ṁ_total_mol/s) / (P/(R·T_in) · Ai)`
> **Cálculo FGR mole fractions:** Conversión masa→mol de la composición másica del FGR:
> M8-reci FGR (% masa): O₂=6.4%, CO₂=18.9%, H₂O=6.8%, N₂=67.9% → molar: O₂≈5.83%, CO₂≈12.52%, H₂O≈11.01%, N₂≈70.66%

### 4.4. bc_config template en ProSimNet

```python
# Ejemplo para M8 (sin FGR)
y_gas_in = np.zeros(9)   # [CO, CO2, H2O, H2, O2, CH4, C2H4, tar, N2]
y_gas_in[4] = 0.21       # O2
y_gas_in[8] = 0.79       # N2

bc_config = build_bc_config(
    n_comp        = 9,
    P_out_bar     = 1.01325,
    v_gas_in      = 0.0918,        # [m/s] aire a 500°C, 1 atm
    T_gas_in      = 773.15,        # [K]
    y_gas_in      = y_gas_in,
    v_out         = None,          # isobárico
    v_solid       = 1.89e-4,       # [m/s]
    direction     = "updraft",
    rho_solid_in  = np.array([172.0, 0.0, 15.0]),  # [kg/m³_bed]: [biomass, char, moisture]
    T_solid_in    = 293.15,        # [K] temperatura ambiente
    inlet_mode    = "prescribed",
)
```

```python
# Ejemplo para M8-reci (con FGR)
# FGR + aire → mezclar antes de entrar
y_mix = np.zeros(9)
y_mix[1] = 0.057   # CO2
y_mix[2] = 0.050   # H2O
y_mix[4] = 0.141   # O2
y_mix[8] = 0.752   # N2
# v_gas_in calculado con flujo molar total de (aire + FGR) a T=350°C
v_gas_in_reci = 0.1115  # [m/s]  — recalcular con flujo exacto

bc_config_reci = build_bc_config(
    n_comp      = 9,
    P_out_bar   = 1.01325,
    v_gas_in    = v_gas_in_reci,
    T_gas_in    = 623.15,          # [K] 350°C
    y_gas_in    = y_mix,
    v_out       = None,
    v_solid     = 1.89e-4,
    direction   = "updraft",
    rho_solid_in = np.array([172.0, 0.0, 15.0]),
    T_solid_in  = 293.15,
    inlet_mode  = "prescribed",
)
```

---

## 5. Condiciones de contorno térmicas

El artículo define pérdidas de calor como:
- **Q_bottom**: calor cedido a la parrilla/parte inferior [kW] — actúa como pérdida en z=0
- **Q_top**: calor aportado por radiación desde la cámara de combustión [kW] — actúa como entrada en z=L
- **Lateral**: resistencia térmica del aislante = 1/15 K·m²/W

En ProSimNet, esto se puede aproximar con `thermal_bc_mode = "ambient_htc"` o con una condición de flujo prescrito.

```python
# Opción 1: adiabático (primera simulación, ignorar pérdidas)
thermal_bc_config = build_thermal_bc_config(mode="adiabatic", Di=0.25, ...)

# Opción 2: pérdidas laterales con resistencia de aislante
# h_ext = 1 / (R_ins × Pi × L) = ... [W/m²/K] → calcular
# Temperatura exterior: agua enfriadora ≈ 50°C = 323 K
```

> Para la primera validación se recomienda empezar con modo **adiabático** y añadir pérdidas en un segundo paso.

---

## 6. Condiciones iniciales (para obtener estado estacionario)

El artículo da resultados en estado estacionario. ProSimNet debe integrarse hasta convergencia (t → ∞). Condiciones iniciales sugeridas:

```python
# Inicialización con atmósfera de N2 y cama fría
T_init = 900.0    # [K] — temperatura inicial uniforme (arranque caliente)
P_init = 1.01325  # [bar]

y_init = np.zeros(9)
y_init[8] = 1.0   # N2 puro al inicio

rho_biomass_init = 172.0   # [kg/m³_bed] — lecho lleno de biomasa
rho_char_init    = 0.0     # [kg/m³_bed]
rho_moisture_init = 15.0   # [kg/m³_bed] (para M8)
```

---

## 7. Datos de validación — qué comparar

### 7.1. Temperatura máxima del sólido (Tabla 6)

| Caso | Ts_max modelo [°C] | T_medida a 5 cm sobre parrilla [°C] | Diferencia |
|---|---|---|---|
| M8 | 1179 | 1079 | −100 |
| M16 | 1184 | 1080 | −104 |
| M30 | 1197 | 1106 | −90 |
| M8-reci | 1060 | 947 | −112 |
| M16-reci | 1054 | 960 | −94 |

> La diferencia sistémica (~100°C) se debe a que el modelo predice el máximo interior, no el punto de medición experimental.

### 7.2. Composición del gas productor en salida (Tabla 5 — caso M30, % masa húmedo)

| Especie | Medido [% masa] | Modelado [% masa] |
|---|---|---|
| CO | 18.2 | 19.6 |
| CO₂ | 7.3 | 6.9 |
| H₂O | 22.4 | 20.9 |
| H₂ | 0.3 | 0.4 |
| CH₄ | 0.7 | 0.6 |
| C₂H₄ + CxHy | 0.4 | 0.5 |
| Tar | 14.4 | 15.1 |

> Para los casos M8, M16, M8-reci, M16-reci: datos disponibles como gráficas de barras (Fig. 4) — similares al M30 para CO y CO₂, con variación en fracción de vapor.

### 7.3. Composición en mitad del reactor — caso M30 (Tabla 7, % vol seco)

| Especie | Medido [% vol seco] | Modelado [% vol seco] |
|---|---|---|
| CO | 30.3 ± 2.3 | 29.0 |
| CO₂ | 4.2 ± 0.7 | 2.6 |
| H₂ | 6.6 ± 2.9 | 5.3 |
| CH₄ | 2.0 ± 1.3 | 0.03 |
| Tar | ≈ 0 | 0.2 |

> Posición: 0.11 m sobre la parrilla (equivalente a 0.10 m en el modelo por capa de cenizas de 0.01 m no modelada).

### 7.4. Posiciones de los termopares

Los termopares estaban en **3 posiciones** dentro del reactor. Las curvas de comparación Tg/Ts vs. z están en **Fig. 3** para los 5 casos.

---

## 8. Notas importantes para la implementación

1. **Sin WGS ni tar cracking**: el artículo lo justifica explícitamente — para esta tecnología (reactor compacto, alta presencia de tar, temperaturas < 500°C en la zona de pirólisis) estas reacciones son irrelevantes. ProSimNet puede prescindir de ellas para este caso.

2. **Capa de cenizas**: el artículo asume 0.01 m de capa de cenizas en la parrilla durante los experimentos. Esto desplaza efectivamente el origen del lecho en 1 cm. Al comparar perfiles de temperatura, ajustar la posición axial.

3. **Potencia de entrada biomasa ≈ 30.3 kW** para todos los casos (promedio del artículo).

4. **ER efectivo**: sin FGR = 0.213 (media experimental). Con FGR (solo aire) = 0.176, pero ER real incluyendo FGR ≈ 0.22.

5. **Calor de reacción**: el artículo lo calcula como diferencia de calores de combustión (∆h = LHV_reactivos − LHV_productos) y lo asigna íntegramente a la fase sólida.
