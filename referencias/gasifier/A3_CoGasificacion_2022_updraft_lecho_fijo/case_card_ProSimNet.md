# Case Card ProSimNet — A3 · Quintero-Coronel et al. (2022)

> **Tipo de caso:** TLUD (Top-Lit Updraft) — cogasificación PKS + carbón bituminoso, lecho fijo
> **Referencia:** Fuel 316 (2022) 123394 — Universidad del Norte, Colombia
> **9 condiciones experimentales:** 3 velocidades de aire × 3 proporciones de carbón (0/15/30 vol%)
> **Utilidad en ProSimNet:** Datos experimentales de validación (T y composición syngas) — SIN modelo cinético propio

---

## ⚠ Limitaciones del artículo

1. **Sin modelo cinético**: solo datos experimentales — no se pueden extraer Ea, A ni rendimientos de pirólisis del artículo.
2. **Cogasificación** (biomasa + carbón): ProSimNet actualmente modela un solo combustible sólido. Replicar los casos puros de PKS (VP_c=0%) es posible; los casos con carbón requerirían extender el modelo.
3. **Modo TLUD**: frente de ignición descendente sobre lecho fijo de biomasa + aire ascendente. Conceptualmente similar al updraft de ProSimNet pero con dinámica transitoria del frente de ignición.
4. **Datos de composición en figuras**: los valores exactos de H₂, CO, CO₂, CH₄ solo están disponibles en gráficas de barras (Fig. 5), no en tablas numéricas.

---

## 1. Geometría del reactor

| Parámetro | Valor | Unidades | Parámetro ProSimNet |
|---|---|---|---|
| Diámetro interno | 152.4 | mm = **0.1524 m** | `Di` |
| Altura total | 900 | mm = **0.90 m** | `dz × N` |
| Aislamiento | 25 mm fiberglass | — | `e_wall` |
| Sección transversal | π/4 × 0.1524² | **0.01824 m²** | `Ai` |
| Nº termopares | 6 | — | posiciones de validación |

```python
Di = 0.1524   # m
L  = 0.90     # m  (altura de sólidos cargada)
Ai = 3.14159/4 * Di**2  # = 0.01824 m²
Pi = 3.14159 * Di        # = 0.4786 m
```

---

## 2. Combustibles — Tabla 2 (composición elemental)

### PKS — Palm Kernel Shell (biomasa pura)

| Parámetro | Valor | Base |
|---|---|---|
| C | 53.8 % | dafb |
| H | 6.13 % | dafb |
| N | 0.88 % | dafb |
| S | 0.11 % | dafb |
| O | 39.0 % | dafb |
| H/C molar | 1.36 | — |
| O/C molar | 0.54 | — |
| Volátiles (VM) | 81.6 % | db |
| Carbono fijo (FC) | 14.6 % | db |
| Cenizas | 3.78 % | db |
| **HHV** | **21 073 kJ/kg** | db |
| Humedad | 6.0 % | wb |
| Tamaño de partícula (d₅₀) | 4.9 ± 2.3 mm | — |

### HVBC — High-Volatile Bituminous Coal (carbón)

| Parámetro | Valor | Base |
|---|---|---|
| C | 74.6 % | dafb |
| H | 6.07 % | dafb |
| N | 0.05 % | dafb |
| S | 1.91 % | dafb |
| O | 17.4 % | dafb |
| Volátiles (VM) | 33.7 % | db |
| Carbono fijo (FC) | 45.4 % | db |
| Cenizas | 20.9 % | db |
| **HHV** | **25 781 kJ/kg** | db |
| Humedad | 2.0 % | wb |
| Tamaño de partícula | 4.7–9.5 mm | — |

---

## 3. Condiciones de operación — Tabla 1 (diseño factorial 3²)

| Experimento | v_s [m/s] | VP_c [%] | Nota |
|---|---|---|---|
| 1 | 0.096 | 15 | |
| 2 | 0.082 | 30 | |
| 3 | 0.069 | 15 | |
| 4 | 0.082 | 0 | PKS puro |
| 5 | 0.096 | 0 | PKS puro |
| 6 | 0.096 | 30 | |
| 7 | 0.069 | 30 | |
| 8 | 0.069 | 0 | PKS puro |
| 9 | 0.082 | 15 | |

> `v_s` = velocidad superficial del aire a condiciones normales (101.325 kPa, 15°C).
> Cada experimento tiene su réplica (denominada 1-1, 2-2, etc.).
> El rango de ER obtenido fue **Φ = 0.26–0.34** en todos los experimentos.

---

## 4. Datos de validación disponibles

### 4.1. Ratio H₂/CO (dato numérico extraído del texto)

| VP_c | H₂/CO |
|---|---|
| 0% (PKS puro) | **0.42–0.46** |
| 15% carbón | **0.49–0.51** |
| 30% carbón | **0.57–0.59** |

> Con mayor % de carbón, más H₂ y menos CO.

### 4.2. LHV del syngas (dato numérico del texto)

| Condición | LHV_g [MJ/Nm³] |
|---|---|
| PKS puro, v_s=0.096 m/s | **~3.70** |
| Resto de condiciones | 3.4–3.7 (approx.) |

### 4.3. Eficiencias (Fig. 7 — valores aproximados)

| VP_c | CGE [%] | CCE [%] |
|---|---|---|
| 0% (PKS puro) | 34–46 | n/d |
| 15% carbón | 35–45 | 66–83 |
| 30% carbón | 37–44 | 59–67 |

> CGE aumenta con v_s en todos los casos.

### 4.4. Composición del syngas (Fig. 5 — solo gráficas, sin tabla numérica)

Del texto: "CO₂ had the highest share (after N₂), followed by CO and H₂, while CH₄ < 4% vol".
- CO₂: ~12–16% vol estimado (dominante entre combustibles)
- CO: ~10–14% vol estimado
- H₂: ~5–8% vol estimado
- CH₄: < 4% vol
- N₂: balance (~60–70% vol)

> ⚠ Estos valores son estimaciones visuales de las figuras. Para usar en validación, leer directamente los datos digitalizados de Fig. 5.

### 4.5. Perfil de temperatura (Fig. 2 — solo gráficas)

6 termopares a lo largo de la altura del reactor. El frente de ignición produce un pico de temperatura que desciende con el tiempo (proceso batch TLUD). La temperatura máxima es similar en todos los experimentos. No hay datos tabulados de T(z).

---

## 5. Equivalencia en ProSimNet — solo casos PKS puro (VP_c=0%)

Para replicar los 3 casos de PKS puro (experimentos 4, 5, 8):

```python
# Geometría
Di = 0.1524; L = 0.90; Ai = 0.01824

# Gas inlet (aire a condiciones normales → convertir a condiciones de T_in real)
# El artículo no da T_in del aire; asumir temperatura ambiente (≈25°C = 298 K)
# v_s en el artículo es a condiciones normales (15°C, 101.325 kPa)
# Corrección: v_in_real = v_s_normal × (T_real/T_normal) = v_s × (298/288) ≈ v_s × 1.035

y_air = np.zeros(9)
y_air[4] = 0.21  # O2
y_air[8] = 0.79  # N2

# Experimento 4 (v_s=0.082 m/s, PKS puro)
v_gas_in = 0.082 * (298.15/288.15)  # ≈ 0.0848 m/s a 25°C
T_gas_in = 298.15                    # K (asumir temperatura ambiente)

bc_config = build_bc_config(
    n_comp    = 9,
    P_out_bar = 1.01325,
    v_gas_in  = v_gas_in,
    T_gas_in  = T_gas_in,
    y_gas_in  = y_air,
    v_out     = None,   # isobárico (o v_out=0 para modo batch TLUD)
    v_solid   = 0.0,    # TLUD: sólido estático (batch)
    direction = None,
)
```

> **Nota clave:** El TLUD es fundamentalmente un proceso **batch** (el sólido no se alimenta continuamente). El frente de ignición se mueve hacia abajo mientras el aire sube. ProSimNet puede aproximarlo como un batch con gas entrando desde abajo (CSTR sin flujo de sólido). NO se puede replicar la dinámica del frente de ignición descendente con el modelo 1D actual — el modelo convergerá a un estado cuasi-estacionario diferente al TLUD real.

---

## 6. Resumen de capacidad de réplica

| Funcionalidad necesaria | ¿Disponible? | Nota |
|---|---|---|
| Geometría del reactor | ✅ | Di=0.1524m, L=0.90m |
| PKS como combustible | ⚠ | Requiere crear fuel YAML con datos de Tabla 2 |
| Cogasificación PKS+carbón | ❌ | ProSimNet tiene 1 sólido; no hay multi-fuel |
| Dinámica TLUD (frente descendente) | ❌ | Proceso batch con frente propagante, no modelable como 1D estacionario |
| Validación de composición de syngas | ⚠ | Solo valores de figuras (no tablas numéricas) |
| Validación de temperatura (6 posiciones) | ⚠ | Solo figuras, sin datos digitalizados |

**Recomendación:** A3 no es el caso de validación más adecuado para el estado actual de ProSimNet. Puede usarse para validar el orden de magnitud de la composición de syngas de PKS puro en condiciones de gasificación updraft con aire, pero la dinámica TLUD es fundamentalmente diferente del updraft continuo.
