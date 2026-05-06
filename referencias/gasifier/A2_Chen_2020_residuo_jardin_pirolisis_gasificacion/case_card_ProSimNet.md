# Case Card ProSimNet — A2 · Gupta & Mahajani (2020)

> **Tipo de caso:** Batch pirólisis — zona de pirólisis de gasificador downdraft
> **Referencia:** Energy 208 (2020) 118427 — IIT Bombay, India
> **Combustibles:** Residuo de jardín: hojas secas de Jackfruit, Mango, Eucalipto y Raintree
> **Utilidad en ProSimNet:** Cinéticas de pirólisis + datos de rendimientos → fuel YAML para residuos urbanos verdes

---

## ⚠ Limitación del artículo

Este artículo NO proporciona un modelo de gasificador 1D completo. El modelo de zona de pirólisis que presentan (compartment model, Fig. 7) asume el perfil de temperatura experimental directamente (sin balance de energía). Su utilidad para ProSimNet es:
1. Cinéticas de pirólisis (Tabla 4) para 4 especies de residuo de jardín
2. Rendimientos másicos de pirólisis (Tabla 3) — para configurar `pyrolysis_yields` en el fuel YAML
3. Caracterización del combustible (Tabla 2)

---

## 1. Caracterización del combustible — Tabla 2

| Parámetro | Jackfruit | Mango | Eucalipto | Raintree |
|---|---|---|---|---|
| **Volátiles** [% masa] | 63 | 67 | 76 | 75 |
| **Carbono fijo** [% masa] | 18 | 23 | 20 | 20 |
| **Cenizas** [% masa] | 19 | 10 | 4 | 5 |
| **C** [% masa] | 37 | 45 | 52 | 51 |
| **H** [% masa] | 5 | 6 | 7 | 8 |
| **N** [% masa] | 0.7 | 0.6 | 1.3 | 2.6 |
| **O** [% masa, por diferencia] | 38 | 38 | 35 | 33 |
| **Celulosa** [% masa] | 25±5 | 40±1 | 37±2 | 28±2 |
| **Lignina** [% masa] | 19±4 | 29±1 | 23±3 | 38±2 |
| **Hemicelulosa** [% masa] | 3±1 | 11±1 | 14±1 | 19±1 |
| **HHV** [MJ/kg] | 16 | 18 | 20 | 22 |

**Caracterización del tar producido (análisis elemental, Tabla 2):**

| | Jackfruit | Mango | Eucalipto | Raintree |
|---|---|---|---|---|
| C [%] | 70.3 | 69.8 | 75.2 | 73.1 |
| H [%] | 8.4 | 9.1 | 10.6 | 9.7 |
| N [%] | 2.3 | 1.6 | 1.2 | 4.9 |
| O [%] | 18.2 | 18.5 | 12.4 | 11.7 |
| **LHV tar** [MJ/kg] | 31–35 | 31–35 | 31–35 | 31–35 |

---

## 2. Rendimientos de pirólisis — Tabla 3 (a 20 K/min)

Datos en % del total de gases no condensables (% de los gases permanentes dentro de la fracción gaseosa total):

| Combustible | CO₂ [% masa] | CO [% masa] | H₂ [% masa] | CH₄ [% masa] | Tar [% masa] | Char [% masa] |
|---|---|---|---|---|---|---|
| Jackfruit | 20.2 | 10.0 | ~1 | 2.3 | 18 | 47.5 |
| Mango | 21.0 | 9.5 | ~1 | 2.4 | 22 | 43.0 |
| Eucalipto | 21.0 | 8.3 | ~1 | 2.7 | 34 | 31.5 |
| Raintree | 22.0 | 8.1 | ~1 | 2.7 | 32 | 33.0 |

> Nota: Los % son sobre la muestra inicial seca. El artículo reporta un ejemplo numérico: 54 g de biomasa seca → CO₂=12g, CO=4.4g, H₂=0.6g, CH₄=1.5g, Tar=17.4g, Char=18g. Resto no cuantificado (~0.1g de H₂O y otros).

**Conversión a fracciones para fuel YAML** (ejemplo con Eucalipto, % → fracción respecto al total):

```python
# Eucalipto — a 20 K/min
# Total masa = 100% (base seca)
# Fracción de gas permanente ≈ 21.0+8.3+1.0+2.7 = 33%
# Tar = 34%, Char = 31.5%, ~1.5% no cuantificado

pyrolysis_yields = {
    "char": 0.315,
    "CO":   0.083,
    "CO2":  0.210,
    "H2":   0.010,
    "H2O":  0.05,   # estimado (no cuantificado directamente)
    "CH4":  0.027,
    "C2H4": 0.005,  # estimado
    "tar":  0.300,  # ajustado para que la suma = 1.0
}
# ⚠ Verificar que suma = 1.0 exacto antes de introducir en el YAML
```

---

## 3. Cinéticas de pirólisis — Tabla 4 (modelo nth orden, a 20 K/min)

Modelo: `dxi/dT = (Ai/β) · exp(−Ea/RT) · (1−x)ⁿ`

donde `xi` es la conversión acumulada del gas i, `β = 20 K/min`.

| Especie | Jackfruit | Mango | Eucalipto | Raintree |
|---|---|---|---|---|
| **CO₂** | A=2300 min⁻¹, Ea=60.7 kJ/mol, n=1.6 | A=8200 min⁻¹, Ea=67.5 kJ/mol, n=1.48 | A=213.7 min⁻¹, Ea=45.3 kJ/mol, n=1.57 | A=1.3×10⁴ min⁻¹, Ea=61.5 kJ/mol, n=2.17 |
| **CO** | A=9.9×10³, Ea=77.2, n=2.5 | A=36.7, Ea=42.7, n=1.21 | A=2.31, Ea=28.2, n=0.46 | A=5.31, Ea=31.1, n=1.01 |
| **CH₄** | A=450, Ea=61.3, n=1.09 | A=5.4×10⁶, Ea=123, n=1.83 | A=953, Ea=67.9, n=1.09 | A=5100, Ea=74.2, n=1.56 |
| **H₂** | A=50.8, Ea=55.4, n=0.013 | A=51.7, Ea=54.2, n=0.24 | A=917, Ea=77.6, n=0.29 | A=34, Ea=51.1, n=0.27 |
| **Tar** | A=3.9×10⁴, Ea=75.3, n=1.3 | A=1.21, Ea=15.8, n=0.57 | A=213, Ea=48.9, n=0.89 | A=105, Ea=43.1, n=0.68 |

> **Conversión de unidades:** A en min⁻¹ → dividir por 60 para obtener s⁻¹. Ea en kJ/mol → multiplicar por 1000 para obtener J/mol.
>
> **Importante:** estas son cinéticas "aparentes" (lumped) que incluyen efectos de transferencia de calor y masa en pellets a escala reactor (FBR). No son cinéticas intrínsecas de TGA.

### Conversión a ProSimNet (ejemplo Eucalipto, CO₂)

```python
# CO2 — Eucalipto, a 20 K/min (cinética de gas específico, nth orden)
A_CO2  = 213.7 / 60.0   # s⁻¹ = 3.56 s⁻¹
Ea_CO2 = 45.3e3          # J/mol
n_CO2  = 1.57

# ProSimNet actualmente implementa solo 1er orden (n=1) en pirólisis global
# Para integrar estas cinéticas se necesitaría ampliar el módulo de pirólisis
# a reacciones paralelas por especie gaseosa.
```

> **Limitación de implementación actual:** ProSimNet tiene pirólisis de 1 solo paso que genera todos los productos proporcionalmente según `pyrolysis_yields`. Estas cinéticas nth-orden por especie requieren extender el módulo de pirólisis a reacciones paralelas independientes.

---

## 4. Modo de uso en ProSimNet

### 4a. Uso inmediato (caso batch pirólisis, modo 0D)

Configurar un caso batch con el fuel YAML de Eucalipto o Raintree (las más interesantes por menor contenido en cenizas) y comparar:
- Tiempo de conversión a diferentes temperaturas finales
- Rendimientos de gas vs. temperatura de pirolización
- **No hay datos de temperatura axial** — solo datos globales

```python
# bc_config para batch 0D
bc_config = build_bc_config(
    n_comp    = 9,
    P_out_bar = 1.01325,
    v_gas_in  = None,     # batch
    v_out     = 0.0,      # sellado
    v_solid   = 0.0,
)
```

### 4b. Uso futuro (cuando ProSimNet tenga pirólisis multi-paso)

Con las cinéticas de la Tabla 4, reproducir la evolución de composición de la zona de pirólisis del downdraft gasifier de IIT Bombay:
- **Geometría zona de pirólisis:** altura ~32 cm (del reactor completo de 67 cm), Di=22 cm superior / 11 cm sección reducida
- **Temperatura:** perfil desde ~300°C hasta ~700°C a 20 K/min
- **Caudal de biomasa:** 12 kg/h (density 650 kg/m³)

---

## 5. Resumen de capacidad de réplica

| Funcionalidad necesaria | ¿Disponible en ProSimNet? | Acción |
|---|---|---|
| Batch 0D con pirólisis y secado | ✅ | Crear fuel YAML |
| Pirólisis global un solo paso | ✅ | Usar rendimientos de Tabla 3 |
| Pirólisis nth-orden por especie gas | ❌ | Futura extensión |
| Zona de pirólisis 1D con T prescrita | Parcial | Requiere perfil T como BC |
| Gasificador completo (combustión + reducción) | No aplica | Artículo no lo modela |
