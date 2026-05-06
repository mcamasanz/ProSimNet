# Flujo de trabajo Git — ProSimNet

## Modelo de ramas

```
main              ← código estable y validado; solo recibe merges de dev/<equipo>
 │
 ├── dev/gasifier  ← desarrollo activo del gasificador
 ├── dev/reactor   ← desarrollo activo del reactor (futuro)
 ├── dev/adsorber  ← desarrollo activo del adsorbedor (futuro)
 └── ...
```

**No existe rama `develop`.** Las ramas de equipo se sincronizan directamente con `main`.

---

## Regla de oro: scope de cada rama

Cada `dev/<equipo>` puede modificar **únicamente** estos ficheros:

```
src/solvers/rhs/rhs_<equipo>.py
src/solvers/runner_<equipo>.py
src/units/<equipo>/                          (state, config, state_extraction)
src/boundary_conditions/<equipo>_boundary.py
src/postprocessing/<equipo>_balances.py
src/postprocessing/<equipo>_plots.py
test/<equipo>/                               (notebooks + README.md)
.claude/equipment/<equipo>.md
```

**NUNCA puede tocar:**

```
src/physics/          ← librerías comunes de física
src/discretization/   ← esquemas numéricos comunes
src/io/               ← lectores de bases de datos
src/utils/            ← utilidades comunes
CLAUDE.md             ← solo desde main
.claude/rules/        ← solo desde main
.claude/physics/      ← solo desde main
```

Si durante el desarrollo de un equipo se detecta un bug o una mejora en las
librerías comunes: **abrir un issue mental, cambiar a main, hacer el fix, merge a main,
luego sincronizar la rama de equipo.**

---

## Crear una nueva rama de equipo

```bash
# 1. Partir siempre desde main actualizado
git checkout main
git pull

# 2. Crear la rama
git checkout -b dev/<equipo>

# 3. Crear los ficheros propios del equipo (si no existen aún)
#    src/solvers/rhs/rhs_<equipo>.py
#    src/solvers/runner_<equipo>.py
#    src/units/<equipo>/  (usar /new-equipment como plantilla)
#    src/boundary_conditions/<equipo>_boundary.py
#    src/postprocessing/<equipo>_balances.py
#    src/postprocessing/<equipo>_plots.py
#    test/<equipo>/README.md

# 4. Primer commit en la rama
git add src/solvers/rhs/rhs_<equipo>.py ...
git commit -m "feat(<equipo>): scaffold new equipment — empty RHS, runner, balances, plots"
```

---

## Flujo diario de desarrollo

```bash
# Trabajar siempre en la rama del equipo
git checkout dev/<equipo>

# Commits frecuentes y atómicos
git add src/solvers/rhs/rhs_gasifier.py test/gasifier/test_gasifier_01.ipynb
git commit -m "feat(gasifier): implement 1D updraft mode with solid transport"

# Sincronizar con main cuando haya cambios en librerías comunes
git merge main    # o: git rebase main
```

---

## Criterios de merge a main (checklist obligatorio)

Un `dev/<equipo>` solo puede mergearse a `main` cuando se cumplen TODOS:

### Funcionalidad
- [ ] Todos los modos de operación del equipo implementados y ejecutables
- [ ] `run_step()` funciona sin errores para todos los modos planificados

### Tests y validación
- [ ] Al menos un test por cada bloque (0D, 1D, modos con flujo si aplica)
- [ ] `check_balances()` llamado en todos los tests
- [ ] Balances ★ con residual < 1 % en todos los casos probados
- [ ] `result.status == 0` en todas las simulaciones

### Código
- [ ] `<equipo>_plots.py` implementado con las funciones básicas de visualización
- [ ] `<equipo>_balances.py` completo con todos los balances aplicables
- [ ] Sin modificaciones en `src/physics/`, `src/discretization/`, `src/io/`, `src/utils/`
- [ ] Sin código zombie (funciones sin usar, imports sin referenciar)

### Documentación
- [ ] `test/<equipo>/README.md` actualizado con todos los tests
- [ ] `.claude/equipment/<equipo>.md` refleja el estado actual

---

## Merge a main

```bash
# Asegurarse de que la rama está sincronizada
git checkout dev/<equipo>
git merge main

# Cambiar a main y mergear con --no-ff para preservar la historia
git checkout main
git merge --no-ff dev/<equipo> -m "feat: merge dev/<equipo> — <descripción breve>"

# Etiquetar el estado del equipo
git tag <equipo>-v1.0
```

---

## Cambios en librerías comunes (solo desde main)

```bash
git checkout main

# Hacer el cambio en src/physics/ o src/discretization/ o src/io/ o src/utils/
# ... editar fichero ...

git add src/physics/transport/transfer_coefficients.py
git commit -m "fix(physics): clip Re to zero before fractional power in Ranz-Marshall"

# Sincronizar todas las ramas de equipo activas
git checkout dev/gasifier && git merge main
git checkout dev/reactor  && git merge main   # si existe
git checkout main
```

---

## Worktrees — desarrollo paralelo de equipos

Cada equipo tiene su propio worktree en disco. Los chats de Claude Code nunca comparten
directorio de trabajo ni pelean por el mismo branch.

### Crear el worktree de un nuevo equipo

```bash
# Desde ProSimNet/ (el repo principal, siempre en dev/gasifier o main)
git worktree add "../ProSimNet-<equipo>" dev/<equipo>

# Ejemplo: reactor
git worktree add "../ProSimNet-reactor" dev/reactor
```

### Preparar el .claude/ del nuevo worktree

```bash
# 1. Crear estructura
mkdir -p "../ProSimNet-<equipo>/.claude/rules"
mkdir -p "../ProSimNet-<equipo>/.claude/physics"
mkdir -p "../ProSimNet-<equipo>/.claude/commands"
mkdir -p "../ProSimNet-<equipo>/.claude/equipment"

# 2. Copiar ficheros comunes (desde ProSimNet/.claude/)
cp .claude/ARCHITECTURE.md     "../ProSimNet-<equipo>/.claude/"
cp .claude/functions.md        "../ProSimNet-<equipo>/.claude/"
cp .claude/howto.md            "../ProSimNet-<equipo>/.claude/"
cp .claude/rules/*.md          "../ProSimNet-<equipo>/.claude/rules/"
cp .claude/physics/*.md        "../ProSimNet-<equipo>/.claude/physics/"
cp .claude/commands/*.md       "../ProSimNet-<equipo>/.claude/commands/"
cp .claude/equipment/common.md "../ProSimNet-<equipo>/.claude/equipment/"

# 3. Crear .claude/equipment/<equipo>.md (scope y estado del equipo)
# 4. Actualizar CLAUDE.local.md con el contexto del equipo y worktree
```

### Estructura de directorios resultante

```
GITHUB/
├── ProSimNet-gasifier/   ← dev/gasifier  (o ProSimNet/ si no se renombró)
│   └── .claude/          ← reglas + equipment/gasifier.md
├── ProSimNet-reactor/    ← dev/reactor
│   └── .claude/          ← reglas + equipment/reactor.md  (sin gasifier.md)
└── ProSimNet-adsorber/   ← dev/adsorber  (futuro)
    └── .claude/          ← reglas + equipment/adsorber.md
```

### Cada .claude/ de equipo contiene

| Fichero | Presente en todos | Solo en el equipo propio |
|---------|:-----------------:|:------------------------:|
| `rules/*.md` | ✓ | |
| `physics/*.md` | ✓ | |
| `commands/*.md` | ✓ | |
| `ARCHITECTURE.md`, `functions.md`, `howto.md` | ✓ | |
| `equipment/common.md` | ✓ | |
| `equipment/<equipo>.md` | | ✓ |
| `equipment/gasifier_modes.md` | | ✓ solo gasifier |

### Sincronizar actualizaciones de reglas entre worktrees

Cuando se modifique un fichero de `.claude/` en el gasificador que deba propagarse:

```bash
# Desde ProSimNet-gasifier/ (o ProSimNet/)
cp .claude/rules/balance-rules.md  "../ProSimNet-reactor/.claude/rules/"
cp .claude/physics/energy-balances.md "../ProSimNet-reactor/.claude/physics/"
# etc. — solo los ficheros modificados
```

---

## Proyectos Claude Code por equipo

Cada equipo se trabaja en un proyecto Claude Code separado apuntando a su rama:

| Proyecto Claude Code | Rama git | Scope de la sesión |
|---------------------|----------|--------------------|
| ProSimNet (gasifier) | `dev/gasifier` | Solo ficheros de gasifier |
| ProSimNet (reactor)  | `dev/reactor`  | Solo ficheros de reactor  |
| ProSimNet (main)     | `main`          | Librerías comunes, CLAUDE.md |

Para cambiar de equipo en VS Code:
```bash
git checkout dev/<equipo>
# Abrir nuevo proyecto Claude Code o cambiar rama en el terminal
```

---

## Naming de commits por rama

```
feat(<equipo>): nueva funcionalidad específica del equipo
fix(<equipo>):  corrección de bug en el equipo
test(<equipo>): nuevo test o corrección de test
docs(<equipo>): documentación del equipo
refactor(<equipo>): refactorización sin cambio de comportamiento

feat(physics):  nueva función en librerías comunes  ← solo desde main
fix(physics):   corrección en librerías comunes      ← solo desde main
```

---

## Añadir `.claude/` a un nuevo proyecto de equipo

Cuando se crea un proyecto Claude Code para un nuevo equipo, copiar:
- `.claude/rules/` — todas las reglas
- `.claude/physics/` — reglas físicas
- `.claude/equipment/<equipo>.md` — solo el fichero del equipo
- `.claude/howto.md` — guía de operaciones del entorno
- `.claude/ARCHITECTURE.md`, `.claude/functions.md`

No copiar los ficheros de equipment de otros equipos.
