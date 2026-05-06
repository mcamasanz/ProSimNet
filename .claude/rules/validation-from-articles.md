# Metodología: de artículos científicos a base de datos de validación

> Este documento recoge la metodología desarrollada durante la validación del gasificador ProSimNet.
> Aplicar el mismo proceso para cualquier equipo nuevo que se valide.

---

## 1. Objetivo

Antes de ejecutar una sola simulación de validación, cada artículo debe transformarse en una **ficha de caso** (`case_card_ProSimNet.md`) que contenga toda la información necesaria para configurar el modelo directamente, sin volver a leer el artículo.

La ficha de caso **no es un resumen del artículo**. Es un documento de ingeniería con los valores exactos de cada parámetro del modelo, en las unidades correctas, con el nombre del campo en ProSimNet.

---

## 2. Estructura de carpetas

```
referencias/
└── <equipo>/                           ← un subfolder por equipo
    ├── README.md                        ← catálogo con tabla de prioridades
    ├── <cod>_<AutorAño>_<descripcion>/
    │   ├── README.md                    ← ficha bibliográfica del artículo
    │   ├── case_card_ProSimNet.md       ← ← datos listos para el modelo
    │   └── <AutorAño>.pdf              ← PDF si está disponible
    └── _txt_<cod>.txt                  ← texto extraído del PDF (pypdf)
```

### Convención de código

| Categoría | Código | Contenido |
|-----------|--------|-----------|
| Modelo 1D completo + validación experimental | A0, A1, A2, ... | Artículos con gasificador real, perfiles T y composición |
| Cinéticas de pirólisis/gasificación | B1, B2, ... | Artículos TGA o FBR con Ea, A, rendimientos |
| Propiedades térmicas del sólido | C1, C2, ... | Mediciones de Cp(T), k, ρ |
| Revisiones de referencia | D1, D2, ... | Reviews exhaustivos de cinéticas y correlaciones |
| Documentos internos | E1, E2, ... | Informes propios, modelos teóricos del proyecto |

---

## 3. Proceso paso a paso

### Paso 1 — Selección de artículos

Criterios de puntuación:
- ★★★★★: perfiles axiales de T + composición de gas + cinéticas + propiedades sólido en un solo artículo
- ★★★★☆: la mayoría de datos; alguna propiedad requiere otra fuente
- ★★★☆☆: solo cinéticas o solo propiedades físicas

**Priorizar artículos de acceso abierto** (CC BY) para evitar bloqueos en la extracción. Identificar antes de descargar si el artículo existe en ResearchGate o arXiv.

### Paso 2 — Extracción del texto (pypdf)

Crear un script `_extract_all.py` en la carpeta del equipo:

```python
import pypdf, os

BASE = os.path.dirname(os.path.abspath(__file__))

PDFS = {
    "A0": "A0_.../AutorAño.pdf",
    "A1": "A1_.../AutorAño.pdf",
    # ...
}

for key, rel in PDFS.items():
    path = os.path.join(BASE, rel)
    out  = os.path.join(BASE, f"_txt_{key}.txt")
    if not os.path.exists(path):
        print(f"NOT FOUND: {path}"); continue
    reader = pypdf.PdfReader(path)
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"Pages: {len(reader.pages)}\n\n")
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            f.write(f"\n{'='*60}\nPage {i+1}\n{'='*60}\n{txt}")
    print(f"OK {key}: {len(reader.pages)} pages -> {os.path.basename(out)}")
```

Ejecutar: `conda run -n base python _extract_all.py`

**Problema habitual:** pypdf extrae texto con artefactos en PDFs escaneados o con fórmulas. En ese caso leer el PDF página a página con la herramienta `Read` de Claude Code.

### Paso 3 — Lectura y clasificación de artículos

Antes de extraer datos, clasificar cada artículo en:

| Tipo | Qué aporta | Uso en ProSimNet |
|------|-----------|------------------|
| Gasificador 1D completo | Geometría + BC + cinéticas + propiedades + resultados | `case_card_ProSimNet.md` completa para simulación directa |
| Cinéticas TGA/FBR | Ea, A, orden de reacción por especie | Parámetros para fuel YAML |
| Propiedades sólido | Cp(T), k, ρ | Campos del fuel YAML |
| Solo datos experimentales (sin modelo) | T(z), y_i en salida | Datos de validación (comparar contra simulación propia) |

### Paso 4 — Verificación del PDF descargado

**Siempre verificar** que el PDF descargado corresponde al artículo esperado:
1. Comprobar el nombre de la revista en la primera página del texto extraído
2. Comprobar el título y los autores
3. Comprobar el DOI si aparece

**Caso real (gasificador A1):** Se descargó `DiBlasi_2004_AIChE.pdf` que resultó ser Galgano & Di Blasi (2004), *Combustion and Flame* — un modelo de partícula única, no el gasificador updraft de AIChE Journal. El mismo autor publicó dos artículos el mismo año sobre temas distintos.

Si el PDF es incorrecto: añadir aviso `⚠️ PDF INCORRECTO` al README y al case_card antes de proceder.

### Paso 5 — Extracción de datos para el case_card

Leer el artículo sección por sección y extraer en este orden:

#### 5.1 Geometría del reactor

Buscar en: "Experimental setup", "Materials and methods", tablas de propiedades.

| Dato a buscar | Campo ProSimNet | Unidades SI |
|---|---|---|
| Longitud / altura del lecho | `dz * N` | m |
| Diámetro interno | `Di` | m |
| Porosidad del lecho | `epsi_r` | — |
| Tamaño de partícula | `dp0` | m |
| Densidad de partícula (seca) | `rho_particle` | kg/m³ |

#### 5.2 Condiciones de contorno (bc_config)

**Gas de entrada:**
- Caudal másico [kg/h o kg/s] → convertir a velocidad superficial [m/s] usando `v_in = ṁ / (ρ_gas_in × Ai)`
- Temperatura de entrada [°C] → convertir a [K]
- Composición (aire: y_O₂=0.21, y_N₂=0.79 mol/mol)
- Tipo de outlet: sellado / válvula / isobárico

**Sólido de entrada (si hay flujo):**
- Caudal másico [kg/h] → velocidad superficial del sólido [m/s] usando `v_s = ṁ_wet / (ρ_bulk_total × Ai)`
- ρ_bulk = ρ_particle × (1 − epsi_r)
- Temperatura de entrada del sólido [K]
- Densidades de entrada: `[ρ_biomass, 0, ρ_moisture]` [kg/m³_bed]

**Cálculo de ρ_moisture a partir del contenido de humedad:**
```
MC_wb = humedad en base húmeda (fracción)
ρ_moisture_bulk = ρ_biomass_bulk × (MC_wb / (1 − MC_wb))
```

**Cálculo de v_gas_in a partir del caudal de aire:**
```python
MW_air = 0.02897   # kg/mol
n_dot  = mass_flow_air_kg_s / MW_air    # mol/s
C_tot  = P_Pa / (R_GAS * T_in_K)       # mol/m³ (gas ideal)
v_gas_in = n_dot / (C_tot * Ai)        # m/s
```

**Cálculo de v_solid a partir del caudal de biomasa:**
```python
rho_bulk_dry  = rho_particle * (1 - epsi_r)   # kg/m³_bed
rho_moisture  = rho_bulk_dry * (MC_wb / (1 - MC_wb))
rho_bulk_total = rho_bulk_dry + rho_moisture
v_solid = mass_flow_wet_kg_s / (rho_bulk_total * Ai)  # m/s
```

#### 5.3 Cinéticas de reacción

Buscar en: "Reactions", "Kinetic parameters", tabla de reacciones.

| Reacción | Parámetros a extraer | Unidades |
|---|---|---|
| Secado | A [s⁻¹], Ea [kJ/mol] | Convertir Ea a J/mol |
| Pirólisis | A [s⁻¹], Ea [kJ/mol], rendimientos (Σ = 1) | — |
| Char oxidation | A [m/s], Ea [kJ/mol] | SCM |
| Boudouard | A [m/s], Ea [kJ/mol] | SCM |
| Water-gas | A [m/s], Ea [kJ/mol] | SCM |
| CO/CO₂ ratio | modelo (e.g. anca_couce_2017), C1, C2 | — |

**Cuidado con las unidades de A:** la pirólisis y el secado usan s⁻¹; las reacciones del char con SCM usan m/s.

#### 5.4 Propiedades térmicas del sólido

Buscar en: tabla de propiedades del modelo, sección "Model properties".

| Propiedad | Cómo se reporta | Conversión para fuel YAML |
|---|---|---|
| Cp_biomass(T) | polinomio en T [J/kg/K] | `Cp_poly_T = [a0, a1, a2, ...]` |
| Cp_char(T) | polinomio en T [J/kg/K] | `Cp_poly_T = [a0, a1, a2, ...]` |
| k_biomass, k_char | constante [W/m/K] | `k` directamente |
| HHV biomasa, char, tar | [MJ/kg] | `heating_values.*` en MJ/kg |

#### 5.5 Rendimientos de pirólisis

Buscar en: estequiometría de la reacción de pirólisis en la tabla de reacciones.
La suma de fracciones másicas debe ser exactamente 1.0.

Componentes esperados en ProSimNet: char, CO, CO₂, H₂O, H₂, CH₄, C₂H₄, tar.

#### 5.6 Composición del char y del tar

- **Char:** CHₐOβ → extraer α (mol H/mol C) y β (mol O/mol C)
- **Tar:** fórmula C_aH_bO_c → extraer a, b, c y calcular MW = 12a + b + 16c

#### 5.7 Datos de validación (resultados del artículo)

Buscar en: "Results", tablas y figuras de comparación modelo-experimento.

| Dato | Dónde buscarlo | Formato para validación |
|---|---|---|
| T(z) en estado estacionario | Figura de perfiles de temperatura | T [°C o K] vs. z [m], por posición de termopar |
| Composición del gas productor (salida) | Tabla o figura de composición | % másico o % vol (seco o húmedo — especificar) |
| Composición en mitad del reactor | Tabla si disponible | % vol seco preferiblemente |
| Temperatura máxima | Tabla | °C + posición |
| Balance de masa/energía | Tabla si disponible | kW o kg/s |

---

## 4. Estructura del case_card_ProSimNet.md

```markdown
# Case Card ProSimNet — <Código> · <AutorAño>

> Tipo de caso: [1D updraft / batch pirólisis / CSTR / etc.]
> Referencia: [Revista Vol (Año) pág] — [Institución]
> N casos simulados: [número]

## 1. Geometría del reactor
[tabla con Di, L, epsi_r, dp0, Ai, Pi — y código Python]

## 2. Combustible — fuel YAML
[todas las secciones: rho_particle, Cp_poly_T, k, heating_values, rendimientos, char, tar]

## 3. Cinéticas de reacción
[tabla A, Ea por reacción + CO/CO₂ ratio]

## 4. Condiciones de contorno por caso
[tabla resumen + código bc_config para cada caso]

## 5. Condiciones térmicas de la pared
[modo adiabático / fixed_twall / ambient_htc + valores]

## 6. Condiciones iniciales recomendadas
[valores para arranque en frío o en caliente]

## 7. Datos de validación
[tablas con valores medidos vs. modelados — T, composición, eficiencias]

## 8. Notas de implementación
[reacciones que el artículo NO incluye, hipótesis clave, ajustes específicos]
```

---

## 5. Campos que con frecuencia NO aparecen en los artículos

Cuando un campo no está en el artículo, documentarlo explícitamente con `???` y la fuente alternativa:

| Campo ausente | Fuente alternativa típica |
|---|---|
| Cp_biomass(T) para biomasas no estándar | Gronli (1996), A0 Tabla 2, o DSC propio |
| k del char | A0: k_char=0.10 W/m/K para madera; medir para materiales nuevos |
| HHV del tar | Estimar de fórmula molecular (Channiwala-Parikh) |
| Rendimientos de pirólisis | Py-GC/MS o FBR del artículo; si no disponible, usar valores típicos de biomasa similar |
| α, β del char | Calcular de análisis elemental del char: α=H/C, β=O/C (molares) |
| v_solid (si el artículo solo da potencia en kW) | Calcular: `P = ṁ_biomass × LHV_biomass → ṁ → v_solid` |

---

## 6. Señales de alerta durante la extracción

| Señal | Qué hacer |
|---|---|
| PDF arranca con nombre de revista diferente al esperado | Verificar DOI; añadir `⚠️ PDF INCORRECTO` al README |
| El artículo usa TGA pero no FBR ni reactor real | Clasificar como cinéticas TGA, no como caso de gasificador |
| El modelo del artículo no tiene balance de energía | No es un modelo 1D completo; usar solo para cinéticas |
| Las cinéticas están en [min⁻¹] o [kJ/mol] | Convertir siempre a [s⁻¹] y [J/mol] antes de anotar |
| Las fracciones másicas de pirólisis no suman 1.0 | El artículo puede usar base libre de cenizas; ajustar |
| Datos solo en figuras (sin tabla numérica) | Digitalizar con WebPlotDigitizer o anotar como estimación visual |

---

## 7. Resumen del proceso (checklist)

### Por cada artículo

- [ ] Verificar que el PDF descargado corresponde al artículo correcto (revista, año, autores)
- [ ] Clasificar el artículo: gasificador 1D / cinéticas / propiedades / solo datos
- [ ] Crear carpeta `<código>_<AutorAño>_<descripcion>/`
- [ ] Crear `README.md` con ficha bibliográfica y puntuación (★1–5)
- [ ] Extraer texto con pypdf → `_txt_<código>.txt`
- [ ] Crear `case_card_ProSimNet.md` con todos los campos en formato ProSimNet
- [ ] Marcar con `???` los campos ausentes e indicar fuente alternativa
- [ ] Anotar explícitamente qué reacciones/efectos el artículo NO incluye

### Para el catálogo del equipo

- [ ] Actualizar `README.md` del equipo con el nuevo artículo en la tabla de prioridades
- [ ] Indicar si el PDF está disponible o pendiente de descarga
- [ ] Indicar qué artículos son replicables directamente con ProSimNet y cuáles requieren extensiones

---

## 8. Ejemplo de caso completo (referencia canónica: A0 gasificador)

El artículo de referencia para esta metodología es:

**A0 — Anca-Couce et al. (2021)**, Fuel 296:120687

- Acceso abierto (CC BY) → extracción sin restricciones
- Contiene TODO en un solo artículo: modelo 1D, cinéticas, propiedades, 5 casos experimentales
- Case card completo en: `referencias/gasifier/A0_AncaCouce_2021.../case_card_ProSimNet.md`

Este case card es el estándar de referencia para los case cards de otros equipos.
