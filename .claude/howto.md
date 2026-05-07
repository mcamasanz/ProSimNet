# Guía de operaciones — cómo hacer X en este entorno

Referencia práctica para encontrar el camino correcto a la primera, sin probar variantes.
Actualizar cuando se descubra un patrón nuevo o una restricción del entorno.

---

## Entorno de ejecución

| Variable | Valor |
|----------|-------|
| OS | Windows 11, shell bash (Git Bash) |
| Python | `/c/ProgramData/anaconda3/python.exe` (Python 3.11.4) |
| `python` / `python3` en bash | ❌ No funciona — alias de Microsoft Store sin instalación |
| numpy / scipy / matplotlib | ✓ Disponibles en el Python de Anaconda |
| Conda envs adicionales | Ninguno (solo base) |

---

## Ejecutar Python

```bash
# CORRECTO
/c/ProgramData/anaconda3/python.exe script.py
/c/ProgramData/anaconda3/python.exe -c "import numpy; ..."

# INCORRECTO — no funciona en este entorno
python script.py
python3 script.py
```

---

## Leer contenido de notebooks Jupyter (.ipynb)

Los notebooks son JSON. Estrategias en orden de preferencia:

### 1. Títulos e introducciones (primera opción)
Usar el subagente **Explore** con instrucción de leer los primeros 60 líneas de cada notebook.
Es la forma más eficiente cuando hay varios archivos a la vez.

### 2. Extraer campos concretos con Python (segunda opción)
Cuando se necesita contenido estructurado (todas las celdas markdown, fuentes de código, etc.):

```bash
/c/ProgramData/anaconda3/python.exe -c "
import json
with open('path/to/notebook.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
for cell in nb['cells'][:5]:
    if cell['cell_type'] == 'markdown':
        print(''.join(cell['source'])[:300])
"
```

### 3. Grep en notebooks (tercera opción)
Para buscar un patrón concreto dentro de notebooks (funcionan como ficheros de texto JSON):

```
Grep(pattern="DATA_PATH|os.getcwd", glob="**/*.ipynb")
```

### 4. Read directo (última opción para un solo notebook)
`Read` sobre un `.ipynb` devuelve el JSON raw — útil solo si se sabe exactamente
en qué línea está la información buscada.

---

## Mover archivos

| Situación | Herramienta | Ejemplo |
|-----------|-------------|---------|
| Archivo trackeado por git | `git mv` | `git mv test/foo.ipynb test/bar/foo.ipynb` |
| Archivo NO trackeado (xlsx, csv, etc.) | `mv` directo | `mv test/data/file.csv test/adsorber/data/` |
| Directorio NO trackeado | `mv` directo | `mv test/data test/adsorber/data` |
| Directorio git vacío por git mv | `rmdir` o se elimina solo | `rmdir test/data` |

⚠ `git mv` falla con "source directory is empty" si el directorio contiene solo
archivos no trackeados. En ese caso usar `mv` directo.

---

## PowerShell desde bash

PowerShell funciona vía `powershell -Command "..."`. Restricciones conocidas:

- El flag `-Directory` en `Get-ChildItem` no existe en esta versión de PS → usar filtro alternativo.
- Las variables de entorno se acceden como `$env:USERPROFILE` dentro de la cadena PS.
- Para bloques multi-línea, poner el comando entre comillas dobles externas con escapado.

Preferir bash puro cuando sea posible; recurrir a PowerShell solo para operaciones
específicas de Windows (rutas de registro, certificados, etc.).

---

## Buscar archivos y símbolos

| Necesidad | Herramienta correcta |
|-----------|----------------------|
| Listar archivos por patrón | `Glob(pattern="**/*.py")` |
| Buscar texto en archivos | `Grep(pattern="...", glob="*.py")` |
| Exploración abierta multi-archivo | Subagente `Explore` |
| Localización de función específica | `Grep` directo (más rápido que Explore) |

**No usar** `find` ni `grep` de bash — las herramientas dedicadas dan mejor experiencia.

---

## Regla obligatoria antes de cada commit

**SIEMPRE verificar la rama activa antes de `git add` + `git commit`:**

```bash
git branch --show-current   # debe mostrar dev/gasifier (o la rama correcta)
```

Si el resultado no es el esperado, hacer `git checkout <rama-correcta>` antes de continuar.

Síntomas de haberse equivocado de rama:
- `git push` falla con "no upstream branch"
- El mensaje del commit lleva el nombre de otra rama: `[dev/reactor ...]`

Corrección si el commit fue a la rama equivocada (y NO se ha hecho push):
```bash
git checkout <rama-correcta>
git cherry-pick <hash-del-commit-erróneo>
git checkout <rama-equivocada>
git reset --hard origin/<rama-equivocada>   # ejecutar en terminal propia, no en Claude Code
git checkout <rama-correcta>
git push
```

---

## Git en este proyecto

- Rama principal: `main`
- Commits: siempre con `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`
- `.claude/` está en `.gitignore` — añadir con `git add -f` si es necesario rastrear algo ahí
- `.ipynb_checkpoints/` no están trackeados (ignorados)
- Los datos de adsorbentes (`test/adsorber/data/*.xlsx|csv`) tampoco están trackeados

---

## Leer archivos grandes

- Notebooks grandes: usar `Read` con `limit` y `offset` para leer por tramos.
- Para extraer solo celdas concretas de un notebook, mejor Python que leer líneas raw.
- Archivos de código Python largos: `Read` con rango de líneas específico o `Grep` para localizar primero.

---

## Bytecode `.pyc` cacheado en Windows + Jupyter

**Síntoma:** después de editar un archivo `.py` con Claude Code, el notebook sigue ejecutando
código antiguo. El traceback muestra el nombre de función viejo aunque el fuente tenga el nuevo.

**Causa:** Windows puede no actualizar el `mtime` del archivo con suficiente granularidad, y el
kernel de Jupyter mantiene el módulo cargado en memoria. Python usa el `.pyc` cacheado.

**Corrección inmediata:**

```bash
# 1. Eliminar el .pyc del módulo modificado
rm src/units/gasifier/config/__pycache__/thermal_bc.cpython-311.pyc

# 2. Reiniciar el kernel de Jupyter (Kernel → Restart)
```

**Prevención:** cuando Claude Code modifica un archivo `.py` que un notebook ya importó,
**siempre reiniciar el kernel** antes de volver a ejecutar las celdas que usan ese módulo.

**Regla en Claude Code:** después de editar cualquier módulo de `src/`, borrar el `.pyc`
correspondiente si hay un kernel de Jupyter activo que lo haya importado.

```bash
# Borrar todos los .pyc del módulo modificado (forma segura):
rm -f src/units/gasifier/config/__pycache__/<modulo>.cpython-311.pyc
# O borrar toda la caché del paquete (solo si se han editado varios archivos):
rm -rf src/units/gasifier/config/__pycache__/
```

---

## Análisis paramétrico y sensibilidad — `src/utils/optimization.py`

### Cuándo usar cada utilidad

| Utilidad | Uso | Casos generados |
|----------|-----|----------------|
| `parametric_sweep` | Barrer valores explícitos de uno o más parámetros | producto cartesiano de `sweep_vars` |
| `sensitivity_analysis` | Medir impacto de ±delta_pct por parámetro | `len(param_specs) * 2 + 1` |
| `optimize_bc` | Maximizar/minimizar un objetivo continuo | llamadas del optimizador scipy |

### Contrato de `run_fn`

La `run_fn` recibe el dict `params` parchado y devuelve el objeto resultado (`col`/`gasifier`).
Siempre reconstruir solo lo que cambia; no modificar `params_base` original.

```python
def run_fn(params):
    # 1. Leer el parámetro barrido/parchado
    val = params["MI_PARAM"]
    # 2. Reconstruir solo lo necesario
    tbc = build_thermal_bc_config(..., T_wall=val, ...)
    bc  = build_bc_config(...)
    p   = {**params_base, "bc_config": bc, "thermal_bc_config": tbc, "_cache": {}}
    # 3. Respetar la inyección de show_progress (parametric_sweep la gestiona)
    t_arr, _, g = run_step(..., show_progress=bool(params.get("_show_progress", False)))
    g._t = t_arr   # requerido por plot_sweep_profiles (usa _t_results internamente)
    return g
```

**Claves especiales inyectadas automáticamente por `parametric_sweep`:**
- `params["_show_progress"]` — True en serie + show_sim_progress=True; False en paralelo
- `params["_case_desc"]`     — etiqueta del caso (la imprime el worker en paralelo)
- `params["_cache"]`         — siempre {} (evita warm-start entre casos independientes)

### Contrato de `patcher` (opcional)

Necesario cuando el parámetro barrido requiere reconstruir un sub-dict (e.g. `thermal_bc_config`).
Sin patcher, `parametric_sweep` hace `params[name] = value` (solo top-level).

```python
def patcher(params, name, value):
    p = {**params, "_cache": {}}
    if name == "T_wall_K":
        p["thermal_bc_config"] = build_thermal_bc_config(..., T_wall=float(value), ...)
    else:
        p[name] = value   # dp0, epsi_r, rho_char0... top-level directo
    return p
```

### Contrato de `objective_fn`

- `parametric_sweep`: `callable(col) → dict[str, float]` — métricas para el DataFrame.
- `sensitivity_analysis` / `optimize_bc`: `callable(col) → float` — escalar a minimizar.

```python
def metrics(g):
    return {
        "conv_bio":  round(1.0 - float(g._rho_solid_results[-1, 0, 0]) / rho_bio_0, 3),
        "Ts_fin_C":  round(float(g._Ts_results[-1, 0]) - 273.15, 1),
        "P_max_bar": round(float(g._P_results.max()), 4),
    }
```

### Ejecución paralela — reglas prácticas

```python
# Siempre funciona (no requiere joblib):
parametric_sweep(..., n_jobs=1)

# Requiere: pip install joblib
# Rentable cuando cada simulación dura > 10 s:
parametric_sweep(..., n_jobs=-1)   # todos los cores
parametric_sweep(..., n_jobs=4)    # 4 workers

# En parallel, show_sim_progress se ignora (siempre False para evitar caos de output)
```

| Tiempo/caso | n_jobs recomendado | Eficiencia esperada |
|-------------|-------------------|---------------------|
| < 1 s | 1 (serie) | overhead domina |
| 1–10 s | 2–4 | 60–80 % |
| > 10 s | 4–8 | 80–95 % |

**Primer arranque de workers (loky/Windows):** la primera llamada paralela incluye
~2-5 s de overhead de arranque. Siempre hacer un warmup antes de medir tiempos:
```python
parametric_sweep(..., n_jobs=2, sweep_vars={"X": [v1, v2]}, verbose=False)
```

### Anti-patrón: desempaquetar retorno condicional

```python
# INCORRECTO — cuando return_results=False devuelve DataFrame (no tupla)
# → ValueError: too many values to unpack
df, res = parametric_sweep(..., return_results=(n_jobs == last))

# CORRECTO — siempre return_results=True en bucles de timing
for n_jobs in NJOBS_LIST:
    df, res = parametric_sweep(..., return_results=True, ...)
# df y res son los del último n_jobs al salir del bucle
```

### Funciones de postproceso de barridos — `postprocessing/gasifier_plots.py`

Todas aceptan `(df, results, sweep_col)` donde `df` viene de `parametric_sweep`
y `results` es la lista de objetos `col`.

| Función | Cuándo usarla |
|---------|--------------|
| `plot_sweep_profiles(df, results, attr_fn, ylabel, title, sweep_col)` | Cualquier perfil temporal — pasar `attr_fn=lambda g: g._Ts_results[:,0]` |
| `plot_sweep_composition(df, results, sweep_col, species_show=[...])` | Composición del gas final como barras agrupadas |
| `plot_sweep_solid(df, results, sweep_col)` | ρ_bio, ρ_char, ρ_moisture en el tiempo |
| `plot_sweep_pressure(df, results, sweep_col, P_ref=...)` | P(t) y v_out(t) lado a lado |
| `plot_sweep_metrics(df, metric_cols, sweep_col)` | Métricas escalares del DataFrame como barras |

**Colores automáticos:**
- Sweep numérico → gradiente viridis
- Sweep con None/NaN o `use_tab10=True` → paleta discreta tab10

**label_fn:** `lambda row: f"{row['T_wall_K']-273.15:.0f} °C"` — formatea leyenda a gusto.

**Nota:** estas funciones usan `col._t_results` (atributo del objeto resultado).
Las `run_fn` de barridos añaden `g._t = t_arr` como alias, pero las funciones de
postproceso solo necesitan el objeto `col` completo — no el `t_arr` separado.
