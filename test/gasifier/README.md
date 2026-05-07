# Tutoriales — Gasifier

Serie completa de tutoriales del gasificador de lecho fijo 0D/1D.
Progresión de lo más simple (reactor concentrado, sin flujo) hasta lo más complejo
(lecho móvil, optimización con redes neuronales).

Ver `.claude/equipment/gasifier_modes.md` para la contextualización conceptual de los modos.

> **Numeración actualizada:** Tutorial 03 es ahora `parametric_sweep` (nuevo).
> Los antiguos 03→04, 04→05, 05→06.

---

## Bloque 0 — Configuración (sin integración)

| Archivo | Tipo | Descripción |
|---------|------|-------------|
| `tutorial_gasifier_00_config_survey.ipynb` | Técnico | Catálogo completo de configuraciones: todos los modos de contorno, opciones de BC térmico, modelos de transporte y modos de propiedades de gas; catálogo de señales BC (ramp, step, pulse, piecewise, proportional, onoff) — sin integrar ODEs. |

---

## Bloque 1 — Reactor concentrado 0D (N=1)

> Representación del gasificador como volumen perfectamente mezclado. Sin gradientes axiales.
> Concepto clave: qué ocurre cuando N=1 y qué significa no tener velocidades de cara.

| Archivo | Tipo | Descripción |
|---------|------|-------------|
| `tutorial_gasifier_01_0D_batch.ipynb` | Tutorial | Pirólisis de biomasa en reactor cerrado con calefacción externa (T_wall=800 °C). Muestra evolución del sólido, gases producidos y cierre de balances en 0D. |
| `tutorial_gasifier_02_0D_semibatch.ipynb` | Tutorial | Pirólisis 0D con 4 modos de outlet: batch sellado, venteo proporcional, isobaro exacto (`v_out=None`) y válvula ISA-75.01 (`Cv`). Análisis paramétrico de T_wall, mc_wb, T_MAX (agotamiento biomasa+char) y P_out. |
| `tutorial_gasifier_03_0D_parametric_sweep.ipynb` | Tutorial | Barrido paramétrico de T_wall (8 casos) y análisis de sensibilidad (4 parámetros ±10 %). Comparación de tiempos serie vs paralelo (n_jobs=1,2,4,8). Introduce `parametric_sweep` y `sensitivity_analysis`. |
| `tutorial_gasifier_04_0D_signals.ipynb` | Tutorial | BC con señales variables: ramp/step/pulse/piecewise de T_wall y Qwall; control proporcional de T_wall por Ts_mean. Salida Cv=0.5. Verificación de balances con señales callable. |
| `tutorial_gasifier_05_0D_control.ipynb` | Tutorial | **Pendiente.** Controlador PID en runner, barrido paramétrico, optimización de setpoints y análisis de sensibilidad (requiere `src.control.optimization`). |
| `tutorial_gasifier_06_0D_cstr.ipynb` | Tutorial | Inyección de agente gasificante (aire/vapor) en reactor 0D con sólido fijo. Introduce el balance gas-sólido con flujo externo. |

---

## Bloque 2 — Lecho fijo 1D (N>1, sólido estático)

> Transición de 0D a 1D: aparecen gradientes axiales y velocidades de cara.
> Concepto clave: cómo varía el resultado al aumentar N.

| Archivo | Tipo | Descripción |
|---------|------|-------------|
| `tutorial_gasifier_10_1D_batch.ipynb` | Tutorial | Lecho fijo en batch con N=10. Muestra los gradientes axiales de temperatura y composición que el 0D no puede capturar. |
| `tutorial_gasifier_11_1D_cstr.ipynb` | Tutorial | Gas atravesando un lecho fijo 1D (N=10). Comparación con el caso 0D equivalente para observar el efecto de la discretización espacial. |

---

## Bloque 3 — Gasificador con flujo (sólido en movimiento)

> Modos con transporte de sólido: contra-corriente, co-corriente y tornillo.
> Solo tienen sentido en 1D (requieren dirección axial).

| Archivo | Tipo | Descripción |
|---------|------|-------------|
| `tutorial_gasifier_12_1D_updraft.ipynb` | Tutorial | Gas sube (aire por la parte inferior), sólido baja por gravedad. Gasificador tipo updraft clásico. |
| `tutorial_gasifier_13_1D_downdraft.ipynb` | Tutorial | Gas y sólido descienden juntos. Gasificador tipo downdraft con zona de oxidación definida. |
| `tutorial_gasifier_14_1D_conveyor.ipynb` | Tutorial | Sólido transportado por tornillo sin fin a velocidad controlada. Tiempo de residencia del sólido independiente del gas. |

---

## Bloque 4 — Térmica avanzada

> Modos de gestión del calor en la pared: desde adiabático hasta pared con dinámica propia.

| Archivo | Tipo | Descripción |
|---------|------|-------------|
| `tutorial_gasifier_20_wall_models.ipynb` | Técnico | Comparación de los 4 modos de BC térmico (adiabático, heatfluxwall, fixed_twall, ambient_htc) sobre un mismo caso base. |
| `tutorial_gasifier_21_shell_tube.ipynb` | Técnico | Activación del modelo dinámico de pared (Tw como ODE). Comparación con y sin pared dinámica. |

---

## Bloque 5 — Validación y análisis

| Archivo | Tipo | Descripción |
|---------|------|-------------|
| `tutorial_gasifier_30_balances.ipynb` | Técnico | Verificación sistemática de cierres de balances (★ masa, ★ energía gas, ★ energía sólido) en todos los modos de operación. |
| `tutorial_gasifier_31_convergence_0D_1D.ipynb` | Benchmark | Convergencia espacial N ∈ {1, 2, 5, 10, 20, 50}: cuándo el 0D es suficiente y cuándo se necesita 1D. |

---

## Bloque 6 — Optimización (futuro)

| Archivo | Tipo | Descripción |
|---------|------|-------------|
| `tutorial_gasifier_40_parametric.ipynb` | Benchmark | Estudio paramétrico de variables de diseño: temperatura de pared, caudal de agente, contenido de humedad. |
| `tutorial_gasifier_50_surrogate.ipynb` | Tutorial | Construcción de un modelo sustituto (ROM) a partir de simulaciones del gasificador. |
| `tutorial_gasifier_60_nn_optimization.ipynb` | Tutorial | Optimización operacional y de diseño del gasificador mediante redes neuronales entrenadas sobre el ROM. |

---

## Tutoriales legacy (antes de la reestructuración)

| Archivo | Estado | Nota |
|---------|--------|------|
| `tutorial_gasifier_01_LEGACY.ipynb` | Archivado | Batch + semibatch con 8 combinaciones BC. Reemplazado por 01, 02 y 20. |
| `tutorial_gasifier_02_LEGACY.ipynb` | Archivado | CSTR/plug-flow. Reemplazado por 03 y 11. |
