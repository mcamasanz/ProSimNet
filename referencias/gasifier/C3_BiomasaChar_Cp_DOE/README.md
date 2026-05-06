# C3 — Cp biomasa y char (DOE/OSTI) · Valores de capacidad calorífica para 21 biomasas

## Puntuación: ★★★☆☆

**Reporte técnico del DOE con valores de Cp(T) para 21 tipos de biomasa y sus chars correspondientes. Útil para validar el rango esperado de Cp y para biomasas no cubiertas por C1.**

---

## Ficha bibliográfica

| Campo | Valor |
|---|---|
| Título | Low-Order Modeling of Internal Heat Transfer in Biomass Particles |
| Autores | — (extraer del documento) |
| Fuente | OSTI / DOE (Department of Energy, USA) |
| Año | ~2016 |
| URL | [https://www.osti.gov/servlets/purl/1261553](https://www.osti.gov/servlets/purl/1261553) |
| Acceso | Libre (acceso abierto OSTI) |

---

## Datos disponibles

| Dato | Disponible | Notas |
|---|---|---|
| Cp(T) para 21 tipos de biomasa | ✓ | Relación lineal con T en [313–353 K] |
| Cp para chars de diferentes biomasas | ✓ | ~1000 J/kg/K independiente de la biomasa |
| Rango de Cp biomasa cruda | ✓ | 1300–2000 J/kg/K |
| Modelo Cp = f(T) | ✓ | Lineal, Cp = a + b·T |
| k del lecho | ✗ | Usar C1 y C2 |
| Cinéticas | ✗ | No es el foco |

---

## Valores clave

| Material | Cp [J/kg/K] | Notas |
|---|---|---|
| Biomasa cruda (rango) | 1300–2000 | Varía con tipo y humedad |
| Char (universal) | ~1000 | Independiente del precursor |
| Variación lineal con T | ~+2 J/kg/K² | Pendiente positiva media |

---

## Uso en ProSimNet

- **Aplicación principal:** valor de Cp para el char en el balance de energía del sólido
- **Regla práctica útil:** Cp_char ≈ 1000 J/kg/K (relativamente constante entre biomasas)
- **Complemento:** C1 para Cp(T) detallado de wood pellets; C2 para k

---

## Notas

- El hecho de que Cp_char ≈ constante a ~1000 J/kg/K simplifica la implementación en el modelo.
- El documento es de libre acceso en el servidor OSTI del DOE.
- Descargar directamente desde la URL de OSTI (documento PDF libre).
- PDF: colocar como `C3_BiomasaChar_Cp_DOE.pdf` en esta carpeta.
