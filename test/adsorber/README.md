# Tests — Adsorber (PSA)

Tests del adsorbedor de columna 1D con modelo LDF.
Cubren ajuste de isotermas, pasos individuales de ciclo PSA, benchmarks de configuración,
convergencia espacial, escalado de especies y modelo dinámico de pared.

Datos de adsorbentes en `data/` (CMS-3K.xlsx, Zeolite13X.csv/.xlsx, Activated_Alumina.xlsx).

| Archivo | Tipo | Descripción |
|---------|------|-------------|
| `test_psa_00_isoLibs_zeolites.ipynb` | Técnico | Ajuste de isotermas DSL para 7 especies sobre CMS-3K desde datos experimentales; compara métodos de mezcla naive, IAST y RAST. |
| `test_psa_01_steps_adsorption.ipynb` | Tutorial | Paso único de adsorción de 600 s en columna de 21 celdas (Zeolite 13X, CO₂/CH₄); verificación completa de balances de masa y energía con perfiles axiales y temporales. |
| `test_psa_07_config_benchmark.ipynb` | Benchmark | 24 combinaciones de configuración (modos de propiedades de gas × frecuencia de actualización × modelo de pared); mide tiempo de cómputo y cierre de balances por caso. |
| `test_psa_08_nodes_benchmark.ipynb` | Benchmark | Convergencia espacial: variación de N ∈ {1, 5, 10, 20, 50, 100, 200} para la misma configuración PSA; mide error L₂ respecto a referencia y tiempo de pared. |
| `test_psa_09_species_benchmark.ipynb` | Benchmark | Escalado de nc=2 a nc=6 especies (añadiendo N₂, O₂, H₂, CO no adsorbentes); evalúa coste computacional y cierre de balances en función del número de componentes. |
| `test_psa_10_shell_tube.ipynb` | Técnico | Validación del modelo dinámico de pared (Tw como ODE): 5 casos (exterior adiabático, pérdidas al ambiente, comparación con/sin pared, conducción axial, compatibilidad hacia atrás). |
