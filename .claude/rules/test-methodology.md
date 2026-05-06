# Metodología para tests y tutoriales de equipos industriales 0D/1D

> v1.0 — Primera versión. Actualizar conforme la metodología evolucione.

---

## Clasificación del trabajo

Antes de escribir o modificar cualquier test, clasificarlo en:

### A. Tutorial-oriented test
Objetivo: enseñar al usuario cómo configurar y ejecutar el modelo.
Ejemplos: configurar gasificador desde cero, interpretar balances, visualizar perfiles.

### B. Pure technical test
Objetivo: verificación, benchmarking o regresión.
Ejemplos: comparar tiempos, verificar estabilidad numérica, validar tolerancias.

### C. Hybrid test
Usar solo cuando sea necesario. Si el resultado se vuelve confuso, dividir en:
- un tutorial didáctico,
- uno o más tests técnicos separados.

---

## Principio de diseño obligatorio: un caso principal a la vez

- SIEMPRE preferir un caso principal por sección.
- NO combinar muchos casos en el mismo test/tutorial salvo que el propósito explícito sea comparación.
- Si se comparan varios casos:
  - explicar cada uno independientemente primero,
  - crear una sección de comparación separada,
  - indicar claramente qué cambia de un caso a otro.

---

## Archivo índice por carpeta (obligatorio)

Cada carpeta de tests **DEBE** contener un archivo `README.md` que:

1. Liste todos los notebooks de la carpeta en orden numérico.
2. Indique para cada uno: nombre del archivo, tipo (tutorial / técnico / benchmark), y una descripción de 1-2 frases.
3. Se actualice cada vez que se añada, elimine o renombre un test.

Formato obligatorio:

```markdown
# Tests — <NombreEquipo>

| Archivo | Tipo | Descripción |
|---------|------|-------------|
| `test_equipo_00_nombre.ipynb` | Tutorial | Qué hace y para qué sirve. |
| `test_equipo_01_nombre.ipynb` | Técnico  | Qué verifica. |
| `bench_equipo_01_nombre.ipynb` | Benchmark | Qué mide. |
```

Tipos válidos: **Tutorial** (orientado a enseñar), **Técnico** (verificación/regresión), **Benchmark** (rendimiento/sensibilidad).

El `README.md` **no** es documentación de usuario — es un índice de navegación para el desarrollador. Máximo 2 frases por test.

---

## Naming convention para archivos

```
test_<equipo>_<nn>_<descripcion_breve>.ipynb
```

Ejemplos:
- `test_gasifier_00_config_survey.ipynb` — encuesta de configuraciones (sin integración)
- `test_gasifier_01_batch.ipynb`         — operación batch
- `test_gasifier_02_semibatch.ipynb`     — operación semibatch
- `test_gasifier_03_updraft.ipynb`       — modo updraft con flujo de gas
- `test_gasifier_10_balances.ipynb`      — foco en verificación de balances
- `bench_gasifier_01_performance.ipynb`  — benchmarks de rendimiento (prefijo bench_)

Rangos recomendados:
- `00–09`: configuración, propiedades, casos fundamentales
- `10–19`: balances, validación
- `20–29`: benchmarks, sensibilidad
- `30+`: estudios paramétricos, comparaciones con experimentos

---

## test_<equipo>_00_config_survey — template obligatorio para todo equipo nuevo

El primer test de cualquier equipo **debe ser** un catálogo de configuraciones
(`test_<equipo>_00_config_survey.ipynb`) que verifique todos los builders de config
sin ejecutar ninguna integración temporal.

**Referencia canónica:** `test/gasifier/test_gasifier_00_config_survey.ipynb`
Usar ese notebook como plantilla directa al crear el test_00 de un equipo nuevo.

### Propósito

- Verificar que todos los `build_*` del equipo funcionan sin errores
- Documentar visualmente (tablas) los valores que produce cada builder
- Validar el dict `params` completo con `_validate_<equipo>_params` sin integrar
- Servir de documentación viva de todas las opciones de configuración del equipo

### Estructura obligatoria del test_00

```
Celda 0   Título + descripción del equipo y ejes de configuración
Celda 1   Tabla de modos de operación (qué parámetros determinan cada modo)
Celda 2   Imports + paths a bases de datos + geometría de referencia
─────────────────────────────────────────────────────────────────────────
TEST 1    build_bc_config — todos los modos; tabla resumen con _v_str()
TEST 2    build_thermal_bc_config — 4 modos × con/sin shell_tube; validación
TEST 3    build_transport_config — constant + correlation; tabla con valores
          a un perfil de temperatura representativo (N nodos)
TEST 4    build_gas_prop_config — modos disponibles; tabla de propiedades
          invariantes + tabla a T de referencia
TEST 5    Params completo — base_params() para cada modo; _validate_*_params()
          sin integrar; tabla con sv0.shape, C_N2(c0), rho_bio(c0), Ts(c0)
```

### Reglas específicas del test_00

- **Sin `run_step()`** — ninguna celda llama al integrador ODE
- **Variables con descripción**: cada constante incluye qué es, no solo unidades:
  ```python
  V_IN = 0.05   # [m/s]  velocidad superficial del gas de entrada al reactor
  ```
- **Índices de especies con `species.index()`**, no con números hardcoded:
  ```python
  y_in[species.index("O2")] = 0.21
  ```
- **Verificar scope de sv0**: mostrar layout explícito y shape correcto (`16×N` para gasifier)
- **Valores de correlación visibles**: en TEST 3, calcular h_bed y h_wall a T_nodes
  para que la tabla de modo `correlation` no quede en blanco
- Cuando la composición del gas de entrada es una convención (p.ej. aire), explicar
  en un comentario por qué se usa ese valor de referencia

### Checklist antes de marcar test_00 como completo

- [ ] Todos los builders del equipo tienen su TEST dedicado
- [ ] Todas las tablas tienen valores (ninguna columna con "None" sin explicación)
- [ ] `_validate_*_params` llamado para cada modo en TEST 5
- [ ] sv0 shape y layout documentados correctamente
- [ ] SOLID_DB, GAS_DB y FUEL_PATH definidos en la celda de imports (no en celdas internas)
- [ ] Sin código zombie ni imports sin usar

---

## Estructura obligatoria (notebook tutorial)

```
[cell 0]  Título (Markdown)
[cell 1]  Introducción (Markdown)
[cell 2]  Imports
[cell 3]  Parámetros físicos y geométricos
[cell 4]  Condiciones iniciales
[cell 5]  Condiciones de contorno
[cell 6]  Configuración del solver
[cell 7]  Ejecutar simulación
[cell 8]  Construir objeto de resultados
[cell 9]  Perfiles espaciales en instantes seleccionados
[cell 10] Evolución temporal de variables clave
[cell 11] Verificación de balances (check_balances)
[cell 12] Interpretación de balances (Markdown)
[cell 13] Conclusiones (Markdown)
```

Celdas opcionales:
- `[cell 14]` Comparación entre dos configuraciones
- `[cell 15]` Análisis de sensibilidad

---

## Plantillas de código por sección

### 3.1. Imports

```python
# Standard
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Project
from src.solvers.runner_<equipo> import run_step
from src.units.<equipo>.state_extraction import build_<equipo>_results
from src.postprocessing.<equipo>_balances import check_balances

# Utilities (opcional)
from src.utils.profiling import print_benchmark_functions
```

### 3.2. Parámetros físicos y geométricos

```python
# Geometría
L  = 1.5      # [m] longitud del lecho
Di = 0.1      # [m] diámetro interno

# Condiciones de operación
T_bed = 1073.15  # [K] temperatura inicial (800 °C)
P_in  = 1.01325  # [bar] presión de entrada
```

Incluir unidades como comentario inline para todo valor numérico.

### 3.4. Condiciones de contorno

```python
bc_config = {
    "mode": "updraft",
    "v_in": v_in,       # [m/s] velocidad superficial entrada
    "T_in": T_in,       # [K]   temperatura del gas de entrada
    "C_in": C_in,       # (nc,) [mol/m³_gas] concentraciones entrada
}
```

### 3.5. Configuración del solver

```python
solver_config = {
    "t_end":        3600.0,   # [s]  tiempo final
    "max_step":       10.0,   # [s]  paso máximo (controla resolución temporal)
    "rtol":           1e-4,   # tolerancia relativa
    "atol":           1e-6,   # tolerancia absoluta por variable de estado
    "method":        "BDF",   # recomendado para sistemas rígidos
    "progress_bar":   True,   # mostrar barra de progreso
    "t_eval":         None,   # dejar al solver elegir instantes
}
```

**Por qué BDF**: adecuado para EDOs rígidas (sistemas con constantes de tiempo muy distintas).
**Tolerancias**: empezar con rtol=1e-4, atol=1e-6. Apretar si el balance no cierra.

### 3.6. Limpiar caché antes de cada caso independiente

```python
# Limpiar caché del solver antes de un nuevo caso (evita warm-start de otro caso)
params.pop("_cache", None)
```

Aplicar siempre cuando se lance más de un caso en el mismo notebook.

### 3.7. Ejecutar simulación y comprobar estado

```python
print("=" * 60)
print(f"Simulando: {caso_descripcion}")
print("=" * 60)

result = run_step(params, solver_config)

# Verificar que el solver convergió
if result.status != 0:
    print(f"⚠ Solver status={result.status}: {result.message}")
else:
    print(f"✓ Completado. Pasos: {len(result.t_arr)}")
```

**Status codes de solve_ivp:**
- `0`: éxito,
- `1`: alcanzado el tiempo final,
- `-1`: fallo de integración — investigar tolerancias o RHS.

---

## Estándares científicos obligatorios

### Unidades

- Todo parámetro y variable de estado debe incluir unidad en comentario o label.
- Plots: unidades en etiquetas de ejes.
- Balances energéticos en `[J/m³_bed]`, másicos en `[kg/m³_bed]`.

### Verificación de balances

- Todo test **DEBE** llamar a `check_balances(col, params)`.
- Resultado etiquetado como ★ (cierre numérico, debe ser ≈ 0) o ✗ (fuente física, ≠ 0).
- Umbral para cierre numérico: **|residual/término_mayor| < 1%**.
- Si un balance no cierra, el test **NO** se considera pasado.

### Celda de interpretación de balances (obligatoria)

```markdown
## Interpretación de balances

★ = cierre numérico (residual debe ser ≈ 0, umbral: < 1% del término mayor)
✗ = fuente física (residual = cantidad producida/consumida, se espera ≠ 0)

Si |residual| > 1%: investigar causa raíz:
  - max_step demasiado grande → reducir max_step
  - tolerancias flojas → apretar atol/rtol
  - bug físico en el RHS → revisar q_masstransfer, thermal_correction
  - condición de contorno desbalanceada → revisar entrada/salida
```

### Reproducibilidad

- Las semillas aleatorias o parámetros derivados de bases de datos deben referenciarse por ruta.
- Condiciones iniciales: físicamente consistentes (C ≥ 0, T > 273 K, P > 0).
- Condiciones de contorno: compatibles con el modo del modelo.

---

## Anti-patrones a evitar

| Anti-patrón | Por qué es un problema | Qué hacer en su lugar |
|---|---|---|
| Varios casos en un solo loop | Oculta parámetros clave, dificulta depuración | Un caso por sección |
| Figuras sin etiquetas de ejes ni unidades | No se puede verificar corrección física | Siempre añadir unidades |
| Simulación sin verificar balances | No se puede validar el modelo | Llamar siempre a check_balances |
| Omitir celda de interpretación | El usuario no sabe si el resultado es bueno | Añadir celda dedicada |
| Mezclar configuración, simulación y análisis en una celda | No se identifica dónde está el error | Separar en celdas dedicadas |
| Números mágicos sin comentarios | Código ilegible | Inline: valor + unidad + contexto |
| Ignorar warnings del solver | Puede indicar mala convergencia | Documentar y comprobar status |
| No limpiar caché entre casos | Warm-start de otro caso contamina el resultado | `params.pop("_cache", None)` |
| No comprobar `result.status` | Fallo silencioso del integrador | Siempre comprobar status ≠ -1 |

---

## Estructura de comparación entre casos (cuando sea necesario)

```
[caso A]   Configuración completa del caso A (caso base)
[caso A]   Simulación del caso A
[caso A]   Resultados del caso A

[caso B]   Diferencias respecto al caso A (solo lo que cambia)
[caso B]   Simulación del caso B
[caso B]   Resultados del caso B

[comparación]  Gráficas superpuestas + tabla de diferencias
[comparación]  Interpretación de la comparación
```

---

## Conclusiones (celda obligatoria al final de cada tutorial)

```python
# --- Conclusiones ---
# Qué demuestra este caso:
# - {aprendizaje principal 1}
# - {aprendizaje principal 2}
#
# Qué probar a continuación:
# - cambiar este parámetro: ...
# - comparar con este otro caso: ...
```

---

## Lista de verificación final antes de marcar un test como completo

- [ ] Título e introducción presentes
- [ ] Unidades en todos los parámetros y etiquetas de plots
- [ ] Condiciones iniciales físicamente consistentes
- [ ] Caché limpiada antes de cada caso independiente
- [ ] `result.status` verificado tras la simulación
- [ ] `check_balances` llamado y resultado mostrado
- [ ] Balances ★ con residual < 1%
- [ ] Celda de interpretación de balances presente
- [ ] Celda de conclusiones presente
- [ ] Sin código zombie (imports sin usar, variables sin referenciar)
- [ ] `README.md` de la carpeta actualizado con el nuevo test
