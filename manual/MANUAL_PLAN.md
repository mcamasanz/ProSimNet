# Manual de Usuario — Plan y Filosofía

> **Estado**: planificación. Este documento define la estructura y el criterio
> editorial del manual. El contenido real se redactará en documentos separados
> cuando el código de cada sección esté estabilizado.

---

## 1. Propósito del manual

El manual de usuario no es documentación técnica interna del código. Es una
referencia para un ingeniero de proceso que quiere utilizar la herramienta para
simular un equipo real: sabe termodinámica, transferencia de calor y adsorción,
pero no necesariamente conoce la implementación.

El manual debe responder tres preguntas:

1. **¿Qué hace esta función y cómo la llamo?** → Parte A: Referencia de funciones
2. **¿Qué ecuaciones está resolviendo el simulador?** → Parte B: Modelos físicos
3. **¿Cómo verifico que el resultado es correcto?** → Parte B: Tests y validación

---

## 2. Principios editoriales

- **Lenguaje**: español técnico. Las variables y nombres de función se mantienen
  en inglés (son nombres del código).
- **Nivel**: ingeniero con formación en procesos químicos o energéticos. No se
  explican conceptos básicos de termodinámica, pero sí las hipótesis específicas
  adoptadas en la implementación.
- **Formato**: Markdown con ecuaciones LaTeX inline (`$...$`) y en bloque (`$$...$$`).
  Cada sección debe poder leerse de forma independiente.
- **Actualización**: el manual se actualiza cada vez que se estabiliza un módulo
  nuevo. No se documenta código en estado borrador.
- **Sin duplicar código**: el manual no reproduce bloques de código extensos.
  Muestra llamadas de ejemplo mínimas y enlaza al notebook de test correspondiente.

---

## 3. Estructura del manual

```
manual/
├── MANUAL_PLAN.md           ← este archivo (índice y filosofía)
│
├── part_a_reference/        ← Parte A: Referencia de funciones
│   ├── A00_overview.md
│   ├── A01_config_functions.md
│   ├── A02_state_functions.md
│   ├── A03_physics_functions.md
│   ├── A04_discretization_functions.md
│   ├── A05_solver_functions.md
│   ├── A06_postprocessing_functions.md
│   └── A07_utils_functions.md
│
└── part_b_models/           ← Parte B: Modelos físicos y tests
    ├── B00_overview.md
    ├── B01_heater_model.md
    ├── B02_adsorber_model.md
    └── B03_common_physics.md
```

---

## 4. Parte A — Referencia de funciones

### Criterio de clasificación

Las funciones se clasifican por su **responsabilidad**, no por su ubicación en
el árbol de carpetas. Un usuario que busca "cómo construir las condiciones de
contorno del calentador" no debería necesitar saber en qué subdirectorio vive
`build_boundary_c_config`.

### Secciones previstas

#### A00 — Visión general
- Mapa conceptual: qué construye cada grupo de funciones y en qué orden se llaman.
- Diagrama de flujo de una simulación completa (config → estado inicial → run_step → postproceso).

#### A01 — Funciones de configuración (`build_*`)

Funciones que construyen los diccionarios de parámetros que consume el simulador.
Se agrupan por equipo y por tipo de configuración.

| Grupo | Funciones (a completar) |
|-------|-------------------------|
| Gas — propiedades puras | `build_gas_prop_config` |
| Gas — condiciones de contorno | `build_boundary_c_config` (heater), `build_boundary_c_config` (adsorber) |
| Térmico — condición de pared | `build_thermal_bc_config` |
| Transporte | `build_transport_config` |
| Pared dinámica (shell-tube) | `build_wall_config` |
| Adsorbente | `build_adsorbent_config` |
| Condiciones iniciales | `build_initial_c_config` |
| Propiedades de sólidos | `build_solid_prop_config` |

Para cada función: firma completa, tabla de parámetros, tabla de salidas, ejemplo
mínimo de llamada, restricciones y modos disponibles.

#### A02 — Funciones de estado (`pack_*`, `unpack_*`, `build_sv0_*`)

Funciones que gestionan el vector de estado del integrador ODE.

| Función | Equipo | Descripción |
|---------|--------|-------------|
| `pack_state_vector` | Heater | Empaqueta `C`, `Hg` [, `Tw`] |
| `unpack_state_vector` | Heater | Desempaqueta y recupera `Tg` por Newton |
| `build_sv0_from_results` | Heater | Genera sv0 de continuación |
| `pack_state_vector` | Adsorber | Empaqueta `C`, `q`, `Hg`, `Ts` [, `Tw`] |
| `unpack_state_vector` | Adsorber | Desempaqueta y recupera `Tg` por Newton |
| `build_sv0_from_results` | Adsorber | Genera sv0 de continuación |

Incluye tabla de layouts del vector de estado para cada equipo y modo.

#### A03 — Funciones de física (`src/physics/`)

Funciones de bajo nivel que calculan propiedades y flujos. Se documentan por
submódulo físico.

| Submódulo | Funciones (a completar) |
|-----------|-------------------------|
| `mixture_gas` | `compute_gas_mixture_properties` |
| `thermodynamics/enthalpy` | `calc_volumetric_enthalpy`, `recover_Tg_from_Hg` |
| `thermodynamics/solid_props` | `build_solid_prop_config`, `eval_solid_property` |
| `momentum/ergun` | `ergun_face_velocity` |
| `momentum/darcy_weisbach` | `continuity_face_velocity` |
| `transport/nusselt` | `h_wall_tube` |
| `thermal/wall_heat_flux` | `wall_heat_flux` |
| `thermal/wall_ode` | `wall_exterior_q`, `wall_axial_q`, `wall_ode_rhs` |

#### A04 — Funciones de discretización (`src/discretization/`)

Esquemas numéricos de flujo. Se documenta el convenio de signos, la orientación
de caras, y las condiciones de contorno disponibles.

| Función | Descripción |
|---------|-------------|
| `convective_flux` | Flujo convectivo upwind en caras |
| `diffusive_flux` | Flujo difusivo en caras |
| `gas_enthalpy_convective_flux` | Flujo convectivo de entalpía |
| `gas_diffusive_heat_flux` | Conducción axial del gas |
| `solid_diffusive_heat_flux` | Conducción axial del sólido |

#### A05 — Funciones de simulación (`run_step`)

Punto de entrada para el usuario. Se documenta en detalle.

| Función | Equipo | Descripción |
|---------|--------|-------------|
| `run_step` | Heater | Integra un intervalo temporal del calentador |
| `run_step` | Adsorber | Integra un paso del ciclo PSA/TSA/VSA |

Para cada una: parámetros, valor de retorno, opciones de solver, gestión del
caché, modo `show_progress`, continuación entre pasos.

#### A06 — Funciones de postproceso

| Función | Descripción |
|---------|-------------|
| `build_heater_results` | Extrae historia de resultados del heater |
| `build_adsorber_results` | Extrae historia de resultados del adsorber |
| `balance_report` | Calcula e imprime el informe de balances |
| `Graph_*` | Funciones de visualización estandarizadas |

Documenta los atributos del objeto `SimpleNamespace` devuelto (`_Tg_results`,
`_Tw_results`, `_P_results`, etc.) con sus unidades y shapes.

#### A07 — Utilidades (`src/utils/`)

| Función | Descripción |
|---------|-------------|
| `profiled` | Decorador de perfilado de tiempo |
| `fit_single_T` | Ajuste de isoterma a datos experimentales |
| `arrh` | Factor de Arrhenius para isotermas dependientes de T |

---

## 5. Parte B — Modelos físicos y tests

### Criterio de documentación

Cada equipo tiene su propio documento `B0X_<equipo>_model.md` con tres bloques:

1. **Hipótesis del modelo** — listado explícito de todas las simplificaciones adoptadas.
2. **Ecuaciones** — sistema de EDPs/EDOs en forma continua y discretizada, con
   definición de cada término, sus unidades y su signo físico.
3. **Tests** — para cada notebook de test: qué se prueba, qué se espera observar,
   y cómo interpretar el resultado.

### Secciones previstas

#### B00 — Visión general de los modelos
- Tabla comparativa de los equipos implementados: variables de estado, tipo de
  flujo, presencia de sólido adsorbente, modelo de pared.
- Convenciones globales: sistema de unidades SI, orientación del eje axial z,
  convenio de signos en los flujos de calor.

#### B01 — Modelo del calentador (Heater 1D)

**Hipótesis**
- Tubo vacío (sin lecho empaquetado)
- Gas ideal, flujo unidimensional axial
- Porosidad efectiva = 1 (todo el volumen es gas)
- Sin dispersión axial de especies (Pe_masa >> 1)
- Conducción axial del gas incluida
- Pared opcional: modelo shell-tube con ODE de temperatura de pared por celda

**Ecuaciones** (a desarrollar)
- Balance de especies: `dC_i/dt = ...`
- Balance de energía del gas: `dHg/dt = ...`
- ODE de pared (shell-tube): `rho_w * cp_w * A_w * dz * dTw/dt = Q_gw + Q_ext + Q_ax`
- Recuperación de Tg desde Hg: Newton sobre la ecuación de entalpía
- Velocidades: continuidad axial `v * C_tot = cte`
- Coeficiente h_wall: Dittus-Boelter con temperatura de película cuando Tw conocida

**Tests**

| Notebook | Tests | Qué verifica |
|----------|-------|--------------|
| `test_heater_00_single_run.ipynb` | — | A completar |
| `test_heater_01_mode.ipynb` | — | A completar |
| `test_heater_02_shell_tube.ipynb` | TEST 1–5 | A completar |

#### B02 — Modelo del adsorbedor (Adsorber 1D)

**Hipótesis**
- Lecho empaquetado, fase gas + fase sólida adsorbente
- Gas ideal, flujo unidimensional axial
- Cinética LDF para la adsorción
- Equilibrio descrito por isoterma multicomponente (DSL u otras)
- Velocidad calculada por Ergun (momento implícito)
- Pared opcional: modelo shell-tube independiente del sólido adsorbente (Tw ≠ Ts)

**Ecuaciones** (a desarrollar)
- Balance de especie en fase gas: `epsi * dC_i/dt = ...`
- Cinética LDF: `dq_i/dt = k_mtc,i * (q_eq,i - q_i)`
- Balance de energía del gas: `dHg/dt = ...`
- Balance de energía del sólido: `(1-epsi)*rho_s*Cp_s * dTs/dt = ...`
- ODE de pared (shell-tube): igual que heater
- Velocidades: Ergun → `v_face` en caras

**Tests**

| Notebook | Tests | Qué verifica |
|----------|-------|--------------|
| `test_psa_00_isoLibs_zeolites.ipynb` | — | A completar |
| `test_psa_01_steps_adsorption.ipynb` | — | A completar |
| `test_psa_02_shell_tube.ipynb` | TEST 1–5 | A completar |
| `test_psa_07_config_benchmark.ipynb` | — | A completar |
| `test_psa_08_nodes_benchmark.ipynb` | — | A completar |
| `test_psa_09_species_benchmark.ipynb` | — | A completar |

#### B03 — Física común a todos los equipos

Documenta los módulos físicos compartidos entre equipos:
- Propiedades de mezcla de gas (regla de Wilke, gas ideal)
- Entalpía volumétrica y recuperación de temperatura
- Propiedades de sólidos (base de datos `soliddb.txt`, modos constant/polynomial)
- Transferencia de calor en tubo: correlación Nusselt / Dittus-Boelter
- Modelo de pared dinámica (wall_ode): física común compartida

---

## 6. Criterios de escritura por sección

### Para las funciones (Parte A)

Cada entrada de función debe incluir:

```
### nombre_de_funcion

**Módulo**: `src/ruta/al/modulo.py`
**Propósito**: una frase.

**Parámetros**

| Nombre | Tipo | Unidad | Descripción |
|--------|------|--------|-------------|
| ...    | ...  | ...    | ...         |

**Devuelve**

| Clave / atributo | Tipo | Unidad | Descripción |
|-----------------|------|--------|-------------|
| ...             | ...  | ...    | ...         |

**Notas**
- Hipótesis relevantes para el usuario
- Modos disponibles y cuándo usar cada uno
- Errores que lanza y por qué

**Ejemplo mínimo**

```python
# código mínimo funcional
```

**Ver también**: enlace a notebook de test relevante
```

### Para las ecuaciones (Parte B)

Cada ecuación debe presentarse:
1. En forma continua (derivadas parciales, integrales si aplica)
2. En forma discretizada (diferencias finitas / volúmenes finitos)
3. Con tabla de símbolos: nombre, unidad SI, descripción

Los términos fuente deben diferenciarse de los flujos de transporte.
Los signos deben justificarse físicamente.

---

## 7. Lo que el manual NO es

- No es documentación automática generada desde docstrings (eso lo haría Sphinx).
- No es un tutorial paso a paso de Python.
- No reproduce la teoría general de adsorción o transferencia de calor.
- No documenta código en estado borrador o en desarrollo activo.
- No reemplaza los comentarios internos del código.

---

## 8. Orden de redacción recomendado

Cuando se retome la escritura del manual, el orden lógico es:

1. `B03_common_physics.md` — física compartida primero (base común)
2. `B01_heater_model.md` — equipo más simple (sin adsorción)
3. `A01_config_functions.md` — funciones de configuración del heater
4. `A05_solver_functions.md` — `run_step` del heater
5. `B02_adsorber_model.md` — equipo más complejo
6. Resto de secciones A en el orden del flujo de trabajo del usuario

---

*Documento creado: 2026-04-23 — pendiente de desarrollo.*
