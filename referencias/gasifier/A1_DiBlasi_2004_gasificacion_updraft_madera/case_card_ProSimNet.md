# Case Card ProSimNet — A1 · Di Blasi (2004)

> **Tipo de caso:** 1D updraft, lecho fijo contracorriente, madera (hardwood/softwood)
> **Referencia:** AIChE Journal 50(9) (2004) 2306–2319 — Universidad de Nápoles, Italia
> **DOI:** 10.1002/aic.10189

---

## ⚠ Estado del PDF

El archivo `DiBlasi_2004_AIChE.pdf` en esta carpeta es **incorrecto**.
Contiene: Galgano & Di Blasi (2004), *Combustion and Flame* 139:16–27 — modelo de partícula única.
Se necesita: **Di Blasi (2004), AIChE Journal 50(9):2306** — modelo de gasificador 1D.

**Los campos marcados con `???` deben completarse una vez descargado el PDF correcto.**
Los campos sin `???` provienen de citas directas de A0 (Anca-Couce 2021) que atribuye explícitamente esos valores a Di Blasi [13].

---

## 1. Geometría del reactor

> ??? — extraer de la sección "Model description" o "Results" del artículo.

| Parámetro | Valor estimado | Fuente | Campo ProSimNet |
|---|---|---|---|
| Longitud del lecho | ??? (~0.5–1.0 m) | típico gasificadores lab | `dz * N` |
| Diámetro interno | ??? (~0.1 m) | típico gasificadores lab | `Di` |
| Porosidad del lecho | ??? | — | `epsi_r` |
| Tamaño de partícula | ??? (~30 mm) | README A1 | `dp0` |
| Densidad de partícula seca | ??? (madera ~400–600 kg/m³) | — | `rho_particle` |

```python
# A rellenar tras leer el PDF correcto
Di     = ???   # m
L      = ???   # m
N      = ???   # celdas (el artículo usa discretización fina en frentes)
dz     = L / N
Ai     = 3.14159 / 4 * Di**2
epsi_r = ???
```

---

## 2. Combustible — fuel YAML (madera, hardwood/softwood)

### 2.1. Propiedades físicas

| Parámetro YAML | Valor | Fuente |
|---|---|---|
| `physical.rho_particle` | ??? | Extraer del artículo |
| `physical.dp_initial` | ??? (~0.030 m) | README A1 |

### 2.2. Propiedades térmicas del sólido

> ??? — Di Blasi 2004 usa probablemente propiedades constantes de la literatura (Gronli 1996).

| Especie | `Cp_poly_T` [J/kg/K] | `k` [W/m/K] |
|---|---|---|
| **biomass** | ??? | ??? |
| **char** | ??? | ??? |
| **moisture** | [4200.0] | 0.60 |

> Referencia alternativa si el artículo usa valores constantes: usar los de A0 (Anca-Couce 2021):
> Cp_biomass ≈ 1500+1.0·T, Cp_char ≈ 420+2.09·T−6.85×10⁻⁴·T², k_biomass=0.17, k_char=0.10 W/m/K

### 2.3. Valores caloríficos

| Parámetro YAML | Valor | Fuente |
|---|---|---|
| `heating_values.biomass` | ??? MJ/kg | Extraer del artículo |
| `heating_values.char` | ??? MJ/kg | Extraer del artículo |
| `heating_values.tar` | ??? MJ/kg | Extraer del artículo |

### 2.4. Rendimientos de pirólisis

> ??? — Di Blasi 2004 usa probablemente pirólisis de 3 reacciones competitivas (celulosa, hemicelulosa, lignina). ProSimNet necesita rendimientos totales en fracción másica (suma = 1).

```python
# Extraer de la tabla de reacciones del artículo
pyrolysis_yields = {
    "char":  ???,
    "CO":    ???,
    "CO2":   ???,
    "H2O":   ???,
    "H2":    ???,
    "CH4":   ???,
    "C2H4":  ???,
    "tar":   ???,
}
```

### 2.5. Composición del char y tar

| Parámetro YAML | Valor | Fuente |
|---|---|---|
| `char_composition.alpha` | ??? | Extraer del artículo |
| `char_composition.beta` | ??? | Extraer del artículo |
| `tar.MW` | ??? | Extraer del artículo |

---

## 3. Cinéticas de reacción

### 3.1. Cinéticas del char — **CONFIRMADAS** (citadas en A0, Tabla 1, con atribución a Di Blasi [13])

| Reacción | A | Ea [kJ/mol] | Unidades A |
|---|---|---|---|
| **Char oxidation** | **5.7×10⁷** | **160** | m/s (SCM) |
| **CO₂ gasification (Boudouard)** | **1×10⁷** | **220** | m/s (SCM) |
| **Steam gasification (Water-gas)** | **1×10⁷** | **220** | m/s (SCM) |

> Fuente: Anca-Couce et al. (2021), Tabla 1, cita explícita: "The surface kinetics from Di Blasi [13] are employed".

### 3.2. Relación CO/CO₂ en oxidación del char

> ??? — A0 usa la correlación de Anca-Couce 2017 [ref 30], NO la de Di Blasi 2004. Di Blasi 2004 puede usar una correlación diferente. Extraer del artículo.

```python
# Extraer del artículo
co_co2_ratio = {"model": "???", "C1": ???, "C2": ???}
```

### 3.3. Secado

| Parámetro | Valor | Fuente |
|---|---|---|
| A [s⁻¹] | ??? | Extraer del artículo |
| Ea [J/mol] | ??? | Extraer del artículo |

### 3.4. Pirólisis (3 reacciones competitivas)

> Di Blasi 2004 describe la pirólisis con 3 reacciones paralelas (celulosa, hemicelulosa, lignina). ProSimNet actualmente implementa 1 solo paso. Para este artículo, usar los valores equivalentes globales o esperar a implementar multi-paso.

| Reacción | Ea [kJ/mol] (estimado) | A [s⁻¹] (estimado) | Fuente |
|---|---|---|---|
| Madera → char | ~125 | ??? | README A1 (estimado) |
| Madera → gas | ~112 | ??? | README A1 (estimado) |
| Madera → tar | ~140 | ??? | README A1 (estimado) |

> ??? — Confirmar valores exactos con el PDF correcto. Los de arriba son estimaciones previas a la lectura del artículo.

---

## 4. Condiciones de contorno (bc_config)

> ??? — Extraer de la sección "Simulated cases" o "Boundary conditions" del artículo.

Di Blasi 2004 estudia el efecto de variar el caudal de aire (parámetro principal del estudio).

```python
# Template a rellenar tras leer el PDF
y_air = np.zeros(9)  # [CO, CO2, H2O, H2, O2, CH4, C2H4, tar, N2]
y_air[4] = 0.21      # O2
y_air[8] = 0.79      # N2

bc_config = build_bc_config(
    n_comp       = 9,
    P_out_bar    = 1.01325,
    v_gas_in     = ???,     # [m/s] a calcular de ṁ_aire y T_in
    T_gas_in     = ???,     # [K]
    y_gas_in     = y_air,
    v_out        = None,    # isobárico
    v_solid      = ???,     # [m/s] — del caudal de biomasa
    direction    = "updraft",
    rho_solid_in = np.array([???, 0.0, ???]),  # [biomass, char, moisture]
    T_solid_in   = ???,     # [K]
    inlet_mode   = "prescribed",
)
```

### Condiciones esperadas del estudio paramétrico

Di Blasi 2004 varía el **caudal de aire** como parámetro principal. Se esperan varios casos con distintos `v_gas_in`. Extraer la tabla de casos del artículo.

---

## 5. Condiciones térmicas de la pared

> ??? — Extraer del artículo. Di Blasi 2004 puede usar reactor adiabático o con pérdidas laterales.

```python
# Primera aproximación: adiabático
thermal_bc_config = build_thermal_bc_config(mode="adiabatic", Di=???, ...)
```

---

## 6. Condiciones iniciales

```python
# Condiciones de arranque recomendadas (estado estacionario por integración larga)
T_init        = 900.0   # K — temperatura inicial uniforme
P_init        = 1.01325 # bar
y_init        = [0]*9; y_init[8] = 1.0  # N2 puro
rho_biomass_0 = rho_particle * (1 - epsi_r)  # kg/m³_bed
rho_char_0    = 0.0
rho_moisture_0 = ???  # según MC del combustible
```

---

## 7. Datos de validación esperados

Di Blasi 2004 proporciona (según README y bibliografía):

| Magnitud | Formato esperado | Notas |
|---|---|---|
| Perfiles axiales de Tg(z) y Ts(z) | Figura(s) de temperatura vs. altura | Varios caudales de aire |
| Composición del gas productor en salida | CO, CO₂, H₂, CH₄ [% vol o masa] | Función del caudal de aire |
| Perfil axial de composición del gas | Figura de y_i(z) | Para al menos un caso |
| Conversión del char vs. altura | Figura | — |
| Tasa de producción de gas | kg/h vs. caudal de aire | — |

---

## 8. Pasos para completar este case card

1. Descargar el PDF correcto: **Di Blasi (2004), AIChE J 50(9):2306** — DOI 10.1002/aic.10189
2. Renombrar a `DiBlasi_2004_AIChE.pdf` (reemplazar el actual incorrecto)
3. Ejecutar `conda run -n base python _extract_all.py` para regenerar `_txt_A1.txt`
4. Leer las secciones: "Mathematical model", "Model properties" (tabla), "Simulated cases", "Results"
5. Rellenar todos los campos `???` de este case card
6. Verificar que las cinéticas del char (ya confirmadas) coinciden con los valores de la Tabla de A0
