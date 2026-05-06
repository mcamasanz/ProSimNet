# Case Card ProSimNet — B1 · Márquez et al. (2023)

> **Tipo de uso:** Cinéticas de pirólisis para fuel YAML de RSU (Residuos Sólidos Urbanos)
> **Referencia:** Waste Management 172 (2023) 171–181 — CIEMAT, Madrid
> **NO es un caso de gasificador**: solo TGA, sin datos de reactor ni perfiles de T o composición

---

## 1. Combustible

**RSU tratado térmicamente** (fracción sólida tras hidrólisis térmica a presión, ECONWARD, Madrid):

| Análisis | Valor |
|---|---|
| Fracción celulósica (glucanos, xylanos, etc.) | presente |
| Fracción plástica | presente (distinguible en TGA) |
| Contenido de cenizas | elevado (tratamiento previo) |

**TGA:** 6 velocidades de calentamiento (5, 10, 15, 20, 25, 40 °C/min), rango 30–900°C, atmósfera N₂.

---

## 2. Comportamiento térmico (proceso multi-paso)

El RSU muestra **4 etapas de degradación** diferenciadas:
- **Etapa I (biomásica):** 150–400°C, pico 300–350°C → celulosa, hemicelulosa, lignin
- **Etapa II (plástica):** 400–520°C → fracción plástica del RSU
- Etapas I y III menos activas (humedad y residuo mineral)

---

## 3. Parámetros cinéticos — métodos isoconversionales + IPR

### Energías de activación medias (métodos isoconversionales)

| Método | Ea etapa I (biomasa) [kJ/mol] | Ea etapa II (plástico) [kJ/mol] |
|---|---|---|
| KAS (Kissinger-Akahira-Sunose) | ~240 | ~250 |
| OFW (Ozawa-Flynn-Wall) | ~240 | ~250 |
| Starink | ~240 | ~250 |
| Friedman | ~240 | ~250 |
| Vyazovkin avanzado | ~240 | ~250 |

### Modelo de reacción (Criado master plots)

- Etapa I (fracción biomásica): **orden de reacción 3** (modelo F3)
- Etapa II (fracción plástica): **orden de reacción 1** (modelo F1)

### Factores preexponenciales (IPR — Independent Parallel Reactions)

> No se tabulan explícitamente en el artículo — se obtienen del ajuste IPR con MATLAB/Openkinetics.
> Valor estimado a partir de Ea conocida y temperatura de pico:
> Para Ea≈240 kJ/mol y T_pico≈600 K → A ≈ exp(Ea/RT_pico + ln(β/Tp²)) → del orden de 10¹⁵–10¹⁸ s⁻¹.

---

## 4. Identificación de productos de pirólisis (Py-GC/MS)

Composición aproximada de productos volátiles (% total de productos):
- Hidrocarburos totales: ~64% (de los cuales ~24% aromáticos)
- Compuestos oxigenados: ~20% (cetonas, furanos, ácidos)
- Resto: nitrogenados y otros

**Aplicación al fuel YAML:** Los rendimientos de Py-GC/MS no están en el formato kg_especie/kg_biomasa que necesita `pyrolysis_yields`. Se necesitaría hacer una integración de cromatogramas para obtener rendimientos cuantitativos por especie.

---

## 5. Parámetros para fuel YAML (parciales)

```yaml
# RSU_CIEMAT_2023 — datos disponibles
fuel_id: "msw_ciemat_2023"
description: "RSU tratado (hidrolizado) — ECONWARD Madrid, Márquez et al. 2023"

kinetics:
  drying:
    A: ~1e6    # s⁻¹ (estimar)
    E: ~88000  # J/mol (usar valor típico de biomasa)

  pyrolysis:
    # Dos reacciones paralelas:
    # Etapa I (biomasa, orden 3)
    biomass_fraction:
      A: ~1e16   # s⁻¹ (estimar de IPR)
      E: 240000  # J/mol
      n: 3
    # Etapa II (plástico, orden 1)
    plastic_fraction:
      A: ~1e17   # s⁻¹ (estimar de IPR)
      E: 250000  # J/mol
      n: 1

# ⚠ pyrolysis_yields: NO disponibles del artículo — necesitan medición separada
```

---

## 6. Uso en ProSimNet

| Qué aporta | Para qué sirve | Estado |
|---|---|---|
| Ea etapa I=240 kJ/mol (biomasa RSU) | Cinética pirólisis para fuel YAML RSU | Disponible |
| Ea etapa II=250 kJ/mol (plásticos) | Pirólisis de fracción plástica (futura extensión multi-paso) | Disponible |
| Orden reacción: F3 (biomasa), F1 (plástico) | Confirma que pirólisis RSU NO es primer orden | Disponible |
| Productos Py-GC/MS | Identificación cualitativa (no cuantitativa para yields) | Parcial |
| HHV, densidad, Cp del RSU | **NO disponibles en este artículo** | Ausente |

**Conclusión:** B1 aporta Ea y orden de reacción para las dos fracciones del RSU, pero NO es suficiente por sí solo para construir el fuel YAML completo. Necesita complementarse con datos de rendimientos (qué fracción se convierte en char, tar, CO, etc.) de otro estudio.
