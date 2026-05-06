# Case Card ProSimNet — E3 · Modelo Teórico Gasificador CIRCE (2025)

> **Tipo:** Documento de validación primaria del proyecto OFE-2024-1077
> **Referencia:** "E3.1 — Desarrollo del modelo teórico de un gasificador updraft", CIRCE, nov. 2025, v1.0
> **Combustible objetivo:** Bioestabilizado urbano (fracción orgánica de RSU tratada térmicamente)
> **Estado:** Sin datos experimentales aún — Bloque II (reactor lab) generará los datos

---

## 1. Contexto y estrategia de validación

E3 define dos escenarios de operación para ProSimNet:

| Escenario | Reactor | Modo ProSimNet | Estado de datos |
|---|---|---|---|
| **Laboratorio** | Lecho fijo batch (escala pequeña) | `batch` / `semibatch` | **Pendiente** — Bloque II del proyecto |
| **Industrial** | Lecho fijo continuo updraft | `updraft` conveyor | **Pendiente** — escalar tras calibración batch |

**Estrategia de doble validación:**
1. **Validación experimental:** datos del reactor batch de laboratorio → calibrar cinéticas y propiedades del bioestabilizado
2. **Validación CFD:** comparar perfiles 1D de ProSimNet contra el modelo CFD del propio proyecto E3

---

## 2. Combustible — Bioestabilizado urbano

El bioestabilizado es la fracción orgánica del RSU tras tratamiento biológico/mecánico. Es materialmente diferente a la biomasa lignocelulósica clásica:

| Característica | Bioestabilizado | Implicación para ProSimNet |
|---|---|---|
| Composición | Fracción vegetal + restos alimentarios + papel/cartón + sintéticos | Cinéticas multi-paso necesarias |
| Contenido mineral | Elevado (cenizas) | Mayor ρ_ash, menor reactividad del char |
| Humedad inicial | Potencialmente alta | Secado relevante, capa secado amplia |
| Heterogeneidad | Alta (varía con lote/temporada) | Calibración experimental imprescindible |

**Datos del combustible:** NO disponibles hasta realización de los análisis del Bloque II.

---

## 3. Lo que E3 define para el modelo

### 3.1. Etapas del proceso (Tabla 1 del documento)

| Etapa | Enfoque de modelado | Implementación en ProSimNet |
|---|---|---|
| Calentamiento | Transporte de calor | Balance de energía gas-sólido ✅ |
| Secado | Cinética física + transporte | Arrhenius 1er orden ✅ |
| Pirólisis | Cinética (desvolatilización) + transporte | Arrhenius 1er orden (multi-paso futuro) ✅/⚠ |
| Oxidación parcial | Cinética heterogénea/homogénea | SCM char oxidation ✅ |
| Reducción (gasificación) | Cinética heterogénea | SCM Boudouard + Water-gas ✅ |
| Tar | Modelo lumped | Tar como pseudo-componente ✅ |

### 3.2. Agentes gasificantes contemplados

- Aire (ya implementado)
- CO₂ (ya soportado — es una especie del gas)
- Vapor (H₂O — ya en el modelo)
- Mezclas (combinaciones de los anteriores)

### 3.3. Tipo de reactor objetivo (industrial)

Lecho fijo contracorriente (updraft), operación continua con alimentación continua de sólido por la parte superior.

En ProSimNet: modo `updraft` con `direction="updraft"` e `inlet_mode="computed"` (caudal de sólido fresco → velocidad calculada internamente).

---

## 4. Template de bc_config para el reactor lab batch (a completar cuando lleguen datos)

```python
# TEMPLATE — rellenar cuando se tengan datos experimentales del Bloque II

# Geometría del reactor de laboratorio (a confirmar con Bloque II)
Di_lab = ???          # m
L_lab  = ???          # m
Ai_lab = 3.14159/4 * Di_lab**2

# bc_config batch sellado (primer modo para calibración)
bc_config_batch = build_bc_config(
    n_comp    = 9,
    P_out_bar = 1.01325,
    v_gas_in  = None,    # batch — sin flujo de gas de entrada
    v_out     = 0.0,     # sellado
    v_solid   = 0.0,
)

# bc_config semibatch (con válvula de alivio para mantener presión)
bc_config_semibatch = build_bc_config(
    n_comp    = 9,
    P_out_bar = 1.01325,
    v_gas_in  = None,
    v_out     = ???,     # m/s — ajustar según válvula
    v_solid   = 0.0,
)

# bc_config updraft continuo (escala industrial)
bc_config_updraft = build_bc_config(
    n_comp       = 9,
    P_out_bar    = 1.01325,
    v_gas_in     = ???,   # m/s — a determinar por ER objetivo
    T_gas_in     = ???,   # K
    y_gas_in     = ???,   # composición del agente gasificante
    v_out        = None,  # isobárico
    v_solid      = ???,   # m/s — calculado de caudal de alimentación
    direction    = "updraft",
    inlet_mode   = "computed",
    rho_solid_fresh_total = ???,  # kg/m³_bed — del bioestabilizado
    mc_wb        = ???,           # fracción másica de humedad wb
)
```

---

## 5. Fuel YAML del bioestabilizado — estructura esperada (a rellenar con datos del Bloque II)

```yaml
fuel_id: "biostabilized_msw_circe_2025"
description: "Bioestabilizado urbano — proyecto OFE-2024-1077, CIRCE, calibrado con datos Bloque II"

physical:
  rho_particle: ???   # kg/m³  — medir (pieza seca compactada)
  dp_initial:   ???   # m — granulometría real del material

heating_values:
  biomass: ???   # MJ/kg — medir con bomba calorimétrica
  char:    ???   # MJ/kg
  tar:     ???   # MJ/kg (estimar o medir)

solid_thermal:
  biomass:
    Cp_poly_T: [???, ???]   # J/kg/K — medir con DSC
    k: ???                   # W/m/K
  char:
    Cp_poly_T: [???, ???]
    k: ???
  moisture:
    Cp_poly_T: [4200.0]
    k: 0.60

pyrolysis_yields:   # a calibrar con experimentos TGA/FBR del Bloque II
  char:  ???
  CO:    ???
  CO2:   ???
  H2O:   ???
  H2:    ???
  CH4:   ???
  C2H4:  ???
  tar:   ???

kinetics:
  drying:    {A: ???, E: ???}   # J/mol
  pyrolysis: {A: ???, E: ???}   # calibrar con TGA del Bloque II
  char_oxidation:     {A: ???, E: ???}
  boudouard:          {A: ???, E: ???}
  steam_gasification: {A: ???, E: ???}

co_co2_ratio:
  model: "anca_couce_2017"   # punto de partida; calibrar con experimento
  C1: 12.0
  C2: 3300.0
```

---

## 6. Datos de validación esperados del Bloque II

| Magnitud | Instrumento | Uso en ProSimNet |
|---|---|---|
| Pérdida de masa vs. tiempo | Balanza + TGA | Calibrar cinéticas drying + pirólisis |
| Temperatura en múltiples posiciones | Termopares | Comparar Tg(z,t) y Ts(z,t) |
| Composición del gas producido | Micro-GC | Comparar y_CO, y_CO₂, y_H₂, y_CH₄ |
| Rendimiento de char (masa final) | Gravimetría | Calibrar pyrolysis_yields.char |
| Rendimiento de tar | Trampas + pesada | Calibrar pyrolysis_yields.tar |
| Proximal + ultimate del bioestabilizado | ASTM/EN | Completar fuel YAML |

---

## 7. Secuencia de trabajo recomendada

```
Bloque II Experimentos batch  →  Calibrar fuel YAML (cinéticas + yields)
                                          ↓
                          test_gasifier_01_batch_biostabilized.ipynb
                                          ↓
                   Ampliar a updraft continuo con mismo fuel YAML
                                          ↓
                   Comparar con modelo CFD del proyecto E3
                                          ↓
                        Publicar como validación del modelo 1D
```
