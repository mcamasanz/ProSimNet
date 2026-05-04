# Referencias — Validación de ProSimNet

Catálogo de artículos científicos seleccionados para validar el modelo 1D de gasificación/pirólisis de biorresiduos.
Cada subcarpeta contiene un `README.md` con la ficha detallada del artículo y, cuando esté disponible, el PDF.

## Criterios de puntuación

| Criterio | Descripción |
|---|---|
| ★★★★★ | Ideal: perfiles axiales de T, composición de gas, cinéticas y propiedades físicas en un único artículo |
| ★★★★☆ | Muy útil: la mayoría de datos presentes; alguna propiedad requiere completarse de otra fuente |
| ★★★☆☆ | Útil parcialmente: aporta cinéticas o propiedades físicas, pero no perfiles de reactor completos |

---

## Catálogo por categorías

### Categoría A — Modelos 1D con validación experimental completa
> Perfiles axiales de temperatura + composición del gas productor. Prioritarios para validar el runner del gasificador.

| Carpeta | Artículo | Combustible | Reactor | Puntuación |
|---|---|---|---|---|
| [A1](A1_DiBlasi_2004_gasificacion_updraft_madera/) | Di Blasi (2004) — AIChE Journal | Madera | Lecho fijo updraft | ★★★★★ |
| [A2](A2_Chen_2020_residuo_jardin_pirolisis_gasificacion/) | Chen et al. (2020) — Energy | Residuo de jardín | Lecho fijo downdraft | ★★★★☆ |
| [A3](A3_CoGasificacion_2022_updraft_lecho_fijo/) | Co-gasif. biomasa+carbón (2022) — Fuel | Biomasa lignocelulósica | Lecho fijo updraft | ★★★★☆ |

### Categoría B — Cinéticas de pirólisis/gasificación de biorresiduos urbanos
> Parámetros cinéticos (Ea, A, orden de reacción) para drying, pirólisis y gasificación del char.

| Carpeta | Artículo | Combustible | Datos cinéticos | Puntuación |
|---|---|---|---|---|
| [B1](B1_MSW_2023_pirolisis_cinetica_multistep/) | MSW multi-step kinetics (2023) — Waste Management | Residuos sólidos urbanos | Ea etapa I/II, IPR, DAEM | ★★★★☆ |
| [B2](B2_LodoDepuradora_2021_pirolisis_cinetica/) | Sewage sludge slow pyrolysis (2021) — J. Env. Management | Lodo depuradora | Ea, 3 etapas TGA, gases | ★★★☆☆ |
| [B3](B3_LodoDepuradora_2019_bioenergy/) | Sewage sludge bioenergy (2019) — RSER | Lodo depuradora | TGA, TG-MS, rendimientos | ★★★☆☆ |
| [B4](B4_ResiduoMadera_2023_pirolisis_gasificacion_CO2/) | Woody waste + CO2 gasif. (2023) — Cleaner Materials | Residuo pino + madera industrial | Pirólisis + gasif. char IPR/DAEM | ★★★★☆ |
| [B5](B5_BiomasaIndustrial_2022_cinetica/) | Industrial biomass kinetics (2022) — J. Env. Management | Poplar, pino, maíz (paja) | Cinéticas comparadas 3 biomases | ★★★☆☆ |

### Categoría C — Propiedades térmicas del sólido (Cp, k, ρ)
> Datos físicos para la fase sólida: necesarios para `soliddb.txt` y el balance de energía del sólido.

| Carpeta | Artículo | Combustible | Propiedades | Puntuación |
|---|---|---|---|---|
| [C1](C1_WoodPellets_2013_Cp_k/) | Wood pellets Cp & k (2013) — Fuel | Pellets de madera | Cp(T) y k efectiva medidos | ★★★★★ |
| [C2](C2_BiomasaSolida_2016_conductividad_termica/) | Thermal conductivity biomass (2016) — Energy & Fuels | 8 tipos de biomasa (paja, astilla, miscanthus) | k comparativa vs tipo y condición | ★★★★☆ |
| [C3](C3_BiomasaChar_Cp_DOE/) | Biomass & char Cp survey — DOE/OSTI | 21 biomases + chars | Cp(T) lineal, ~1000 J/kg/K para char | ★★★☆☆ |

### Categoría D — Revisiones de referencia
> Fuentes consolidadas de cinéticas y correlaciones; base para extraer parámetros que no estén en los artículos de validación.

| Carpeta | Artículo | Contenido | Puntuación |
|---|---|---|---|
| [D1](D1_GomezBarea_Leckner_2010_revision_gasificacion/) | Gómez-Barea & Leckner (2010) — Prog. Energy Combust. Sci. | Revisión exhaustiva cinéticas heterogéneas del char, correlaciones, modelos | ★★★★★ |

---

## Prioridad de descarga y uso

Los 6 artículos más urgentes para arrancar la validación:

| Orden | Artículo | Por qué es urgente |
|---|---|---|
| 1 | **A1** Di Blasi 2004 | Valida directamente el runner del gasificador updraft con datos axiales completos |
| 2 | **C1** Wood pellets Cp+k | Propiedades físicas medidas; alimentan soliddb directamente |
| 3 | **C2** Thermal conductivity biomass | Amplía C1 a más tipos de biomasa; necesario para el rango de k |
| 4 | **B1** MSW multi-step 2023 | Cinéticas pirólisis RSU compatibles con el modelo de reacciones del gasificador |
| 5 | **B4** Woody waste + CO2 gasif. 2023 | Cinéticas pirólisis + gasificación del char en un solo artículo |
| 6 | **D1** Gómez-Barea & Leckner 2010 | Fuente consolidada para todas las cinéticas heterogéneas del char |

---

## Notas de uso

- Cada carpeta tiene su `README.md` con la ficha completa: referencia, DOI, datos disponibles, variables extractables y notas de uso en el modelo.
- Al descargar un PDF, colocarlo en la carpeta correspondiente con el nombre: `[codigo]_[AutorAnyo].pdf` (ej. `A1_DiBlasi2004.pdf`).
- Los datos extraídos (cinéticas, Cp, k) se registran en el README de la carpeta para trazabilidad.
