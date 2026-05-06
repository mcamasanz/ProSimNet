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
