# Tutoriales del Gasificador — Catálogo y metodología

> Estado: 6 tutoriales implementados y validados (dev/gasifier).
> Referencia para planificar nuevos tutoriales y para replicar la estructura en otros equipos.

---

## Progresión didáctica

La serie sigue una escalada en tres ejes simultáneos:
- **Complejidad física:** 0D batch → semibatch → señales temporales → control
- **Herramientas de análisis:** caso único → comparación manual → barrido sistemático → sensibilidad
- **Modo de ejecución:** serie → serie con barras → paralelo con timing

Esta progresión es deliberada. Cada tutorial introduce **un concepto nuevo** sobre una base
ya validada; nunca dos conceptos a la vez si se puede evitar.

---

## Tabla de tutoriales

| Archivo | Concepto nuevo | BC | Herramientas usadas |
|---------|---------------|-----|---------------------|
| `tutorial_00_config_survey` | Config builders sin integrar | — | builders, `_validate_params` |
| `tutorial_01_0D_batch` | Pirólisis 0D, balances básicos | fija | `run_step`, `check_balances` |
| `tutorial_02_0D_semibatch` | 4 modos de outlet + barrido manual | fija | `parametric_sweep`, `plot_sweep_*` |
| `tutorial_03_0D_parametric_sweep` | Barrido sistemático + timing paralelo | fija | `parametric_sweep`, `sensitivity_analysis` |
| `tutorial_04_0D_signals` | BC dinámicas (ramp, step, pulse, piecewise, ctrl P, ctrl onoff) | callable | `resolve`, signals, controllers |
| `tutorial_05_0D_control` | PID, optimize_bc *(pendiente)* | callable | `optimize_bc` |
| `tutorial_06_0D_cstr` | Inyección de agente gasificante *(pendiente)* | entrada de gas | `run_step` |

---

## Qué demuestra cada tutorial

### tutorial_00 — Config survey
- Todos los `build_*_config` funcionan sin integrar ODEs.
- Catálogo vivo de opciones: modos BC, modelos transporte, tipos de señal.
- Primer checkpoint antes de cualquier simulación.

### tutorial_01 — Batch 0D
- Pirólisis en reactor cerrado: secado → pirólisis → char.
- Balances ★ cierran con residual < 1 %.
- Introduce el patrón `run_step` → `build_gasifier_results` → `check_balances` → `display_balances`.

### tutorial_02 — Semibatch 0D
**Bloque principal:** 4 modos de outlet (batch, venteo proporcional, isobaro, válvula ISA).
- Demuestra que el modo de outlet no rompe los balances.
- Introduce `Cv` (válvula ISA-75.01) como BC.

**Sección 11 — Análisis paramétrico:**
- 6 estudios con `parametric_sweep`: T_wall, mc_wb, T_MAX, v_out, Cv, P_out.
- Patrón `run_fn` + `metrics_fn` + `plot_sweep_*`.
- Funciones de postproceso: `plot_sweep_profiles`, `plot_sweep_composition`, `plot_sweep_solid`, `plot_sweep_pressure`, `plot_sweep_metrics`.

**Conclusión clave:** el modo de salida del gas no afecta la conversión de biomasa
(las reacciones son las mismas); lo que cambia es la presión y la concentración absoluta.

### tutorial_03 — Parametric sweep + timing
- `parametric_sweep` sobre 8 valores de T_wall con timing n_jobs={1,2,4,8}.
- `sensitivity_analysis` sobre 4 parámetros con patcher para T_wall.
- Comparación speedup vs eficiencia paralela.
- Plots: tiempo, speedup, eficiencia; tabla de sensibilidad normalizada.

**Conclusión clave sobre paralelismo:**
- Para casos de >10 s/simulación: eficiencia ≥ 80 % con n_jobs=4-8.
- `return_results=True` siempre en bucles de timing — nunca intentar desempaquetar
  el retorno de `parametric_sweep` si `return_results=False` (devuelve un DataFrame, no tupla).
- El caso base de `sensitivity_analysis` ya no es serial forzado: está en el lote paralelo.

### tutorial_04 — Señales BC
6 casos con la **misma geometría y física**, solo cambia el tipo de señal BC:

| Caso | Señal | Observación principal |
|------|-------|-----------------------|
| Ref | `float` constante | Referencia: pirólisis desde t=0 |
| 4A ramp | `callable(t)` | Retardo de pirólisis proporcional a la pendiente |
| 4B step | `callable(t)` | Secado antes del escalón; pirólisis brusca con el salto |
| 4C pulse | `callable(t)` | Energía total inyectada determina conversión; sin continuidad después |
| 4D piecewise | `callable(t)` | Velocidad de conversión controlada por protocolo de etapas |
| 4E ctrl P | `callable(t, snap)` | Error estacionario inevitables sin integral; T_wall suave |
| 4F ctrl onoff | `callable(t, snap)` | Ts oscila dentro de la banda; T_wall binaria; sin error SS |

**Conclusión clave:** los tres tipos de señal (`float`, `callable(t)`, `callable(t, snap)`)
son transparentes para el RHS y el runner — no requieren ningún cambio en la física.
El solver BDF maneja correctamente discontinuidades (escalón, pulso).

**Comparación 4E vs 4F (mismo SP=700 K):**

| Aspecto | On-off | Proporcional |
|---------|--------|--------------|
| Error estacionario | ≈ 0 (cicla) | Permanente |
| T_wall | Binaria | Suave |
| Oscilaciones Ts | Sí (±25 K) | No |
| Ajuste | `band` | `gain` + `output_bias` |

---

## Patrones establecidos en estos tutoriales

### Patrón `make_params(tbc)`
Para tutoriales con múltiples casos que solo cambian el `thermal_bc_config`:
```python
def make_params(tbc):
    return {**params_base, "thermal_bc_config": tbc, "_cache": {}}
```

### Patrón `run_fn` para `parametric_sweep`
```python
def run_X(params):
    # 1. Leer el parámetro del barrido
    val = params["PARAM_KEY"]
    # 2. Reconstruir solo lo que cambia
    tbc = build_thermal_bc_config(..., T_wall=val, ...)
    bc  = build_bc_config(...)
    p   = {**params_base, "bc_config": bc, "thermal_bc_config": tbc, "_cache": {}}
    # 3. Integrar
    t_arr, _, g = run_step(sv0=sv0, t_max=T_MAX, params=p, ...,
                           show_progress=bool(params.get("_show_progress", False)))
    g._t = t_arr   # adjuntar para plot_sweep_profiles (que usa col._t_results)
    return g
```

### Patrón `patcher` para `sensitivity_analysis`
```python
def patcher(params, name, value):
    p = {**params, "_cache": {}}
    if name == "T_wall_K":
        p["thermal_bc_config"] = build_thermal_bc_config(..., T_wall=float(value), ...)
    else:
        p[name] = value   # parámetros top-level: dp0, epsi_r, rho_char0...
    return p
```

### Anti-patrón: desempaquetar con return_results condicional
```python
# INCORRECTO — falla cuando return_results=False (devuelve DataFrame, no tupla)
df, res = parametric_sweep(..., return_results=(n_jobs == last), ...)

# CORRECTO — siempre return_results=True; los resultados intermedios se sobreescriben
df, res = parametric_sweep(..., return_results=True, ...)
df_final, res_final = df, res   # guardar al salir del bucle
```

---

## Estructura de un tutorial completo (referencia)

```
Celda 0:  Título + tabla resumen de casos
Celda 1:  Introducción: qué fenómeno nuevo introduce
Celda 2:  Imports (incluyendo postprocessing y señales si aplica)
Celda 3:  Setup: geometría, combustible, sv0, params_base
Celda 4:  Caso base / referencia (con balance)
——— Bloque de casos ———
Celda N:  Markdown: qué hace este caso y qué observar
Celda N+1: Código: definir señal/BC → run_step → plots → check_balances
——— Final ———
Celda -3: Resumen comparativo (todos los casos superpuestos)
Celda -2: Tabla de métricas finales
Celda -1: Conclusiones
```

---

## Qué replicar en un equipo nuevo

1. `tutorial_00` — config survey: verificar todos los builders sin integrar.
2. Caso base con balance — validar el caso más simple antes de añadir complejidad.
3. Sección de análisis paramétrico con `parametric_sweep` + `plot_sweep_*` del equipo.
4. Casos de señales BC si el equipo admite BC dinámicas.
5. Timing benchmark con `n_jobs={1,2,4,8}` para caracterizar el comportamiento paralelo.

La referencia de implementación de señales es `signal-bc-integration.md`.
La referencia de `parametric_sweep` / `sensitivity_analysis` es `howto.md`.
