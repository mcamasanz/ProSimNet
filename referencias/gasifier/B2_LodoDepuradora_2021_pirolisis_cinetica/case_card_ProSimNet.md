# Case Card ProSimNet — B2 · Mphahlele et al. (2021)

> **Tipo de uso:** Cinéticas de pirólisis lenta para fuel YAML de lodo de depuradora (sewage sludge)
> **Referencia:** J. Environmental Management 284 (2021) 112006 — Vaal University of Technology, SA
> **NO es un caso de gasificador**: solo TGA, sin datos de reactor

---

## 1. Combustible — Lodo de depuradora de Gauteng (GSS)

GSS (Gauteng Sewage Sludge) — digerido anaeróbicamente, deshidratado.

| Análisis | Valor | Base |
|---|---|---|
| Humedad | elevada (muestra pre-secada) | — |
| Volatiles (VM) | — | secado previo |
| Carbono fijo (FC) | — | — |
| Cenizas | elevado (lodo mineral) | — |

**TGA:** 3 velocidades de calentamiento (10, 20, 30 °C/min), rango 30–900°C, N₂ (99.995%).
Partícula < 75 μm (minimizar resistencias de transferencia de calor).

---

## 2. Comportamiento térmico

**3 etapas de degradación:**
1. **Secado:** hasta ~150°C — pérdida de humedad libre
2. **Pirólisis activa:** 150–570°C — descomposición principal de materia orgánica
3. **Char residual:** > 570°C — degradación lenta de estructuras aromáticas

Mayor velocidad de calentamiento → desplazamiento de picos a temperaturas más altas.
Tasa máxima de descomposición: **1.10 %/min·mg** a 30°C/min.

---

## 3. Parámetros cinéticos (métodos isoconversionales)

### Energías de activación medias

| Método | Ea media [kJ/mol] |
|---|---|
| **FWO (Flynn-Wall-Ozawa)** | **225.92** |
| **KAS (Kissinger-Akahira-Sunose)** | **218.04** |
| **Starink** | **218.97** |

> La variación de Ea con la conversión indica proceso multi-paso (mecanismo complejo).

### Mecanismo de reacción (Criado master plots)

- Mecanismo dominante: **orden 3** (F3) y **difusión 3D** (D4 Jander)
- Estos mecanismos son complejos y no directamente mapeables a un primer orden simple.

### Propiedades termodinámicas de la pirólisis

| Propiedad | Rango |
|---|---|
| ΔH [kJ/mol] | ~212–221 |
| ΔG [kJ/mol] | ~aproximadamente ΔH |
| ΔS [J/mol/K] | negativo o próximo a cero |

---

## 4. Parámetros para fuel YAML (parciales)

```yaml
# sewage_sludge_gauteng_2021
fuel_id: "sewage_sludge_gauteng_2021"
description: "Lodo depuradora Gauteng SA — digerido anaeróbicamente, Mphahlele 2021"

kinetics:
  drying:
    A: ~1e6    # s⁻¹ (estimado)
    E: 88000   # J/mol (valor típico, no reportado en el artículo)

  pyrolysis:
    # Cinética global aparente (promedio isoconversional)
    A: ~1e15   # s⁻¹ (estimar de Ea y temperatura de pico)
    E: 221000  # J/mol (media FWO: 225920, media KAS: 218040)
    n: 1       # aproximación 1er orden (modelo real es más complejo)

# ⚠ pyrolysis_yields: NO disponibles — requieren medición cuantitativa adicional
# ⚠ HHV, densidad, Cp, k del lodo: NO reportados en este artículo
```

---

## 5. Lo que falta para un fuel YAML completo del lodo

| Dato necesario | Fuente recomendada |
|---|---|
| Rendimientos de pirólisis (char/tar/gas) | B3 (Shahbeig 2020) reporta algunos datos |
| HHV del combustible | B3 reporta HHV=16.47 MJ/kg |
| Cp(T) del lodo y del char | No en B2 — buscar en literatura de lodo |
| k del lodo y char | No disponible — estimar |
| Composición elemental completa | B3 o medir directamente |
