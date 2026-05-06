# B1 — MSW multi-step kinetics (2023) · Pirólisis de residuos sólidos urbanos

## Puntuación: ★★★★☆

**Cinéticas de pirólisis de la fracción orgánica de RSU mediante modelo multi-step con reacciones paralelas independientes (IPR). Compatible con la arquitectura de reacciones del gasificador ProSimNet.**

---

## Ficha bibliográfica

| Campo | Valor |
|---|---|
| Título | Pyrolysis of municipal solid waste: A kinetic study through multi-step reaction models |
| Autores | — (extraer del PDF) |
| Revista | Waste Management |
| Año | 2023 |
| DOI | [10.1016/j.wasman.2023.10.022](https://doi.org/10.1016/j.wasman.2023.10.022) |
| Acceso | ScienceDirect (Elsevier) |

---

## Combustible

| Parámetro | Valor |
|---|---|
| Combustible | Residuos sólidos urbanos (fracción orgánica) |
| Composición típica | Papel, madera, residuo alimentario, plástico ligero, textil |
| Relevancia urbana | ★★★★★ |

---

## Datos disponibles

| Dato | Disponible | Notas |
|---|---|---|
| Curvas TGA (TG + DTG) | ✓ | Múltiples rampas de calentamiento |
| Análisis elemental e inmediato | ✓ | C, H, O, N, S; humedad, cenizas, volátiles |
| Cinéticas por método isoconversional | ✓ | KAS, OFW, Starink, Friedman, Vyazovkin avanzado |
| Modelo IPR (reacciones paralelas independientes) | ✓ | Compatible con arquitectura ProSimNet |
| Modelo DAEM | ✓ | Alternativa al IPR |
| Temperatura de pico de descomposición por fracción | ✓ | |
| Calor de pirólisis (ΔH_pyrolysis) | ✓ (parcial) | Mediante DSC |
| Datos de reactor real | ✗ | Solo TGA |

---

## Parámetros cinéticos reportados

### Energías de activación por etapa de descomposición
| Etapa | Rango T [°C] | Ea [kJ/mol] | Fracción dominante |
|---|---|---|---|
| Etapa I | 200–380 | 180–200 | Papel, madera, residuo alimentario |
| Etapa II | 380–550 | 268–360 | Plásticos, lignina residual |

*(Valores exactos y A pre-exponencial en las tablas del artículo — extraer al descargar el PDF)*

---

## Uso en ProSimNet

- **Aplicación:** alimentar los parámetros de pirólisis en `rhs_gasifier.py` para RSU
- **Compatibilidad:** el modelo IPR del artículo mapea directamente a la estructura de reacciones paralelas del gasificador
- **Limitación:** no incluye datos de reactor; solo cinéticas de TGA → combinar con A1 o A2 para validación completa

---

## Notas

- Los RSU tienen una variabilidad composicional alta; el artículo usa muestra representativa.
- La heterogeneidad de los RSU hace que los modelos multi-step sean más robustos que los single-step.
- PDF: colocar como `B1_MSW2023.pdf` en esta carpeta.
