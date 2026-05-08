# Proyecto OFE-2024-1077 — Contexto y antecedentes

**P9 — Aseguramiento De La Economía Circular Mediante La Gasificación Avanzada Para La Producción De Biochar Y Singás**

---

## 1. Marco contractual

| Campo | Valor |
|---|---|
| **Referencia** | OFE-2024-1077 |
| **Cliente** | UTE NIVARIA — URBASER S.A.U. + FCC MEDIO AMBIENTE S.A. (gestión RSU Isla de Tenerife) |
| **Ejecutor** | Fundación CIRCE — Centro de Investigación de Recursos y Consumos Energéticos, Zaragoza |
| **Presupuesto** | 418 500 € (sin IGIC) / 447 795 € (con IGIC) |
| **Investigadora principal** | Clara Ángela Jarauta Córdoba |
| **Contactos CIRCE** | Alfonso Juan Romance (ajuan@fcirce.es) · Diego Redondo Taberner (dredondo@fcirce.es) |
| **Hito clave** | Entregables del primer año ejecutados antes del 30/11/2025 |
| **Firmado** | 13/02/2025 |

### Contexto del encargo

UTE NIVARIA es adjudicataria del contrato de servicio público de gestión de residuos de la Isla de Tenerife (Cabildo de Tenerife). Entre sus obligaciones contractuales está la realización de proyectos IDi (Investigación, Desarrollo e Innovación). Este proyecto es uno de los proyectos IDi financiados bajo ese marco.

---

## 2. Objetivo del proyecto

Demostrar la viabilidad técnica de la **gasificación avanzada de material bioestabilizado** (fracción orgánica del RSU tras tratamiento biológico-mecánico) para producir simultáneamente:

- **Biochar** — producto principal: enmienda de suelos, secuestro de carbono, aplicaciones industriales
- **Syngas** — producto secundario: aprovechamiento energético en el proceso

El proyecto forma parte de la estrategia de **economía circular** del Cabildo de Tenerife: transformar un residuo difícil de gestionar (bioestabilizado) en productos de valor mediante una tecnología de alta eficiencia.

---

## 3. Tecnología seleccionada (E2.1)

Tras una evaluación multicriterio (matriz de decisión ponderada) entre gasificación en **lecho fijo** y **lecho fluidizado**, se seleccionó:

**→ Gasificador de lecho fijo en configuración updraft (ascendente)**

| Criterio | Peso | Razón de selección |
|---|---|---|
| Rendimiento en biochar | 5 (máx.) | Lecho fijo updraft produce mayor fracción sólida |
| Calidad del biochar | 5 | Temperatura controlada favorece estabilidad del char |
| Facilidad de separación | 5 | Biochar en la parte inferior, fácil recuperación y enfriamiento rápido (evita PAHs) |
| Versatilidad (variaciones de alimentación) | 4 | Tolera alta humedad y granulometría variable |
| Calidad del syngas | 4 | Composición típica: CO 20-30%, H₂ 10-20%, CO₂ 10-20%, CH₄ 2-5% |
| Complejidad operativa | 4 | Diseño robusto y simple; menor coste operativo |
| Escalabilidad | 3 | Adecuado para escala pequeña-media |
| Costes (CAPEX+OPEX) | 3 | CAPEX estimado: 2 000-5 000 €/(kg/h) escala pequeña |
| Estabilidad a largo plazo | 2 | — |
| TRL / experiencia | 2 | Tecnología probada |

**CAPEX orientativo:** una planta de 1 000 kg/h requeriría 1,5-3 M€.

---

## 4. Combustible — Bioestabilizado urbano (E1.1)

El bioestabilizado es la fracción sólida resultante del tratamiento biológico-mecánico del RSU. No es biomasa lignocelulósica clásica: su composición es heterogénea e incluye fracciones vegetales, restos alimentarios, papel/cartón y materiales inorgánicos.

### 4.1 Análisis del bioestabilizado **afinado** (laboratorio externo, mayo 2025)

| Parámetro | Valor | Base | Implicación para el modelo |
|---|---|---|---|
| **Humedad total** | 44,0 ± 1,0 % | s.r. | Muy alta → pretratamiento de secado obligatorio antes de gasificación |
| Materias volátiles | 54,5 % | b.s. | Buena para gasificación — generará gases combustibles |
| Carbono fijo | 10,0 % | b.s. | Moderado → contribuye al char y reacciones heterogéneas |
| **Cenizas** | 35,5 ± 1,8 % | b.s. | Alto → inerte en el proceso, ρ_ash elevada |
| **Densidad aparente** | 540 ± 40 kg/m³ | s.r. | Entrada al modelo: `rho_particle` (estimado ~900 kg/m³ partícula seca) |
| **C** | 35,3 % | b.s. | — |
| **H** | 5,1 ± 0,9 % | b.s. | H/C ≈ 1,73 (favorable para H₂ y CH₄) |
| **N** | 2,11 ± 0,32 % | b.s. | Riesgo NOx — monitoreado en syngas |
| **S** | 0,415 % | b.s. | Riesgo corrosión menor |
| **Cl** | 0,962 % | b.s. | ⚠ Riesgo corrosión de equipos — vigilar |
| **O** | 20,7 % | b.s. | O/C ≈ 0,44 |
| **PCS** | 16,5 ± 0,4 MJ/kg | b.s. | `heating_values.biomass` (dry basis) |
| **PCI** | 15,4 ± 0,4 MJ/kg | b.s. | — |
| **PCI (s.r.)** | 7,6 MJ/kg ≈ 2,1 kWh/kg | s.r. | Valor real con la humedad del material recibido |

### 4.2 Temperaturas de fusibilidad de cenizas (riesgo de escorificación)

| Temperatura | Valor | Incertidumbre |
|---|---|---|
| Contracción (SST) | 1 060 °C | ± 180 °C |
| Deformación inicial (DT) | 1 180 °C | ± 60 °C |
| Hemiesfera (HT) | 1 260 °C | ± 65 °C |
| Fluida (FT) | 1 280 °C | ± 70 °C |

> ⚠ La zona de oxidación del gasificador updraft puede superar los 1 000-1 100 °C. Existe riesgo de escorificación (slagging) en la parrilla. El modelo debe predecir si la temperatura máxima del sólido supera SST.

### 4.3 Bioestabilizado salida de reactor (análisis CIRCE)

| Parámetro | Valor |
|---|---|
| Humedad relativa | 14,4 % |
| Densidad aparente (s.r.) | 360,3 kg/m³ |
| Densidad aparente (b.s.) | 308,4 kg/m³ |
| Cenizas | 49,73 % b.s. |

> El bioestabilizado de salida de reactor (sin afinado) tiene mayor tamaño de partícula y mayor contenido en inertes (vidrios, metales, cerámicas, plásticos). Requiere pretratamiento adicional antes de gasificación.

### 4.4 Diferencias clave respecto a biomasa lignocelulósica estándar

| Propiedad | Biomasa estándar (abeto A0) | Bioestabilizado |
|---|---|---|
| Humedad (s.r.) | 8-30 % | **44 %** — mucho más alta |
| Cenizas | 1-3 % | **35,5 %** — orden de magnitud mayor |
| PCS | 19,1 MJ/kg | 16,5 MJ/kg |
| Cl | <0,1 % | **0,96 %** — riesgo de corrosión |
| N | <0,5 % | **2,1 %** — riesgo NOx |
| Heterogeneidad | Baja | **Alta** — cinéticas multi-componente |

---

## 5. Estructura de entregables del proyecto

```
E1 — Caracterización y pretratamientos
│   E1.1  Caracterización del bioestabilizado (completado, mayo 2025)
│   E1.2  Pretratamientos para alimentación al reactor (v1 y v3 disponibles)
│
E2 — Tecnología y reactor experimental
│   E2.1  Evaluación tecnológica — decisión lecho fijo updraft (completado)
│   E2.2  Adaptaciones del reactor experimental de CIRCE
│
E3 — Modelo teórico ← ProSimNet
│   E3.1  Desarrollo del modelo teórico del gasificador updraft (en desarrollo)
│
E4 — Biochar: calidad y regulación
│   E4.1  Requisitos de calidad del biochar y casos de uso
│   E4.2  Marco regulatorio del biochar
│
E6 — Economía circular
    E6.5  Marco para medición y evaluación del desempeño circular
```

---

## 6. Rol de ProSimNet en el proyecto

**ProSimNet es el entregable E3.1** — el modelo físico-matemático del gasificador.

### Objetivos del modelo

1. **Calibración experimental (Bloque II):** El modelo se ajustará con los datos del reactor batch de laboratorio de CIRCE. Los parámetros cinéticos (Ea, A) y las propiedades térmicas del bioestabilizado se calibrarán experimentalmente — no son transferibles directamente desde bibliografía de biomasa lignocelulósica.

2. **Escalado a operación continua:** Una vez calibrado en batch, el modelo se aplica al gasificador updraft continuo para predecir perfiles de temperatura, composición del syngas y rendimiento a biochar bajo distintas condiciones operativas.

3. **Comparación con CFD:** Los resultados 1D de ProSimNet sirven como referencia y condiciones de contorno para un modelo CFD detallado del proceso (también en desarrollo en el proyecto).

4. **Optimización operativa:** Identificar las condiciones óptimas (ER, caudal de sólido, temperatura de pared) para maximizar la calidad del biochar y minimizar el contenido en tar del syngas.

### Capacidades necesarias del modelo para este proyecto

| Necesidad del proyecto | Capacidad de ProSimNet |
|---|---|
| Alta humedad del bioestabilizado (44%) | ✅ Secado como ODE explícita |
| Alto contenido en cenizas (35.5%) | ⚠ Cenizas como inertes implícitos en ρ_char residual |
| Temperatura máxima (riesgo slagging >1060°C) | ✅ Predicción de Ts_max |
| Rango de ER (equivalence ratio) | ✅ Variable a través de v_gas_in y y_gas_in |
| Agente gasificante: aire (fase 1), vapor/CO₂ (futuro) | ✅ Composición y_gas_in configurable |
| Perfiles axiales T(z) y y_i(z) | ✅ Modo 1D con N celdas |
| Modo batch (calibración en laboratorio) | ✅ Modo batch implementado |
| Modo updraft continuo (escala industrial) | ✅ Modo updraft conveyor |
| Cinéticas multi-componente del bioestabilizado | ⚠ Requiere calibración experimental (Bloque II) |

---

## 7. Documentos de referencia en esta carpeta

| Fichero | Tipo | Contenido |
|---|---|---|
| `20250213. P9. CIRCE. Biochar. Contrato_vf.pdf` | Contractual | Contrato CIRCE-UTE NIVARIA, cláusulas, presupuesto, equipo |
| `E1.1 - Informe de caracterización...pdf` | Técnico | Análisis físico-químico del bioestabilizado (datos de entrada al modelo) |
| `E1.2 - Pretratamientos...v1/v3.pdf` | Técnico | Pretratamientos necesarios antes de gasificación |
| `E2.1 - Estudio de evaluación tecnológica.pdf` | Técnico | Comparativa lecho fijo vs. fluidizado; decisión updraft |
| `E2.2 - Implementación de adaptaciones...pdf` | Técnico | Modificaciones al reactor experimental de CIRCE |
| `E3.1 - Desarrollo del modelo teórico...pdf` | Técnico | Marco teórico de ProSimNet: ecuaciones, hipótesis, estrategia de validación |
| `E4.1 - Identificación de requisitos de calidad del biochar.pdf` | Técnico | Qué calidad debe tener el biochar y para qué aplicaciones |
| `E4.2 - Evaluación del marco regulatorio del biochar.pdf` | Técnico | Normativa aplicable al biochar |
| `E6.5 - Marco para la medición y evaluación del desempeño circular.pdf` | Técnico | Indicadores de economía circular del proceso |
