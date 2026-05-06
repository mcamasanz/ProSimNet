# A2 — Chen et al. (2020) · Pirólisis y gasificación de residuo de jardín en lecho fijo downdraft

## Puntuación: ★★★★☆

**Biorresiduo urbano relevante (residuo de jardín) con datos experimentales de pirólisis y gasificación en reactor a escala de laboratorio. Combina cinéticas TGA con datos del reactor real.**

---

## Ficha bibliográfica

| Campo | Valor |
|---|---|
| Título | Kinetic studies in pyrolysis of garden waste in the context of downdraft gasification: Experiments and modeling |
| Autores | Chen et al. |
| Revista | Energy |
| Año | 2020 |
| DOI | [10.1016/j.energy.2020.118591](https://doi.org/10.1016/j.energy.2020.118591) |
| Acceso | ScienceDirect (Elsevier) |

---

## Combustible y reactor

| Parámetro | Valor |
|---|---|
| Combustible | Residuo de jardín (biorresiduo urbano — ramas, hojas, poda) |
| Tipo de reactor | Lecho fijo downdraft |
| Escala | Laboratorio |
| Agente gasificante | Aire |

---

## Datos disponibles para validación

| Dato | Disponible | Notas |
|---|---|---|
| Curvas TGA a múltiples rampas de calentamiento | ✓ | Necesario para ajuste cinético |
| Cinéticas de pirólisis (Ea, A) | ✓ | Por método isoconversional |
| Evolución de gases con T (CO, CO2, H2, CH4) | ✓ | Durante pirólisis |
| Perfiles de temperatura en el reactor | ✓ (parcial) | En zonas del reactor downdraft |
| Composición del syngas a la salida | ✓ | |
| Rendimiento de char y gas | ✓ | |
| Propiedades físicas del residuo (Cp, k, ρ) | ✗ | Requiere complementar con C1/C2 |

---

## Parámetros cinéticos reportados

| Método | Ea [kJ/mol] | Notas |
|---|---|---|
| KAS | A determinar del PDF | — |
| OFW | A determinar del PDF | — |
| Friedman | A determinar del PDF | — |

*(Extraer valores exactos de las tablas del artículo al descargar el PDF)*

---

## Uso en ProSimNet

- **Modo de validación:** `downdraft` en `build_boundary_c_config`
- **Variables a comparar:** composición syngas (CO, CO2, H2, CH4), temperatura de las zonas
- **Ventaja:** biorresiduo urbano → directamente aplicable al caso de uso de valorización de RSU

---

## Notas

- Es uno de los pocos artículos que combina TGA + datos de reactor real para un biorresiduo urbano específico.
- El residuo de jardín es un componente importante de los RSU en España (Directiva 2008/98/CE).
- PDF: colocar como `A2_Chen2020.pdf` en esta carpeta.
