# ARCHITECTURE — Índice de referencia del proyecto

> Este archivo es el punto de entrada. El contenido detallado está descompuesto
> en subcarpetas temáticas. Léelo de arriba a abajo la primera vez; después
> ve directamente a la sección que necesitas.

---

## Estructura de la documentación (.claude/)

```
.claude/
├── ARCHITECTURE.md               ← este archivo (índice)
├── GASIFIER_BOOTSTRAP_PROMPT.md  ← prompt de arranque para nuevas conversaciones
├── settings.json                 ← configuración Claude Code
│
├── commands/                ← slash commands
│   ├── new-equipment.md     → /new-equipment  plantilla completa equipo nuevo
│   ├── check-rhs.md         → /check-rhs      auditoría RHS checklist 12 pasos
│   └── physics-rules.md     → /physics-rules  referencia rápida reglas físicas
│
├── rules/                   ← reglas de escritura de código
│   ├── code-format.md            → naming, docstrings, decoradores, imports, caché
│   ├── units-shapes.md           → tabla SI, shapes de arrays, conversiones
│   ├── layer-separation.md       → qué va en cada capa (physics/, units/, rhs/...)
│   ├── balance-rules.md          → estándar obligatorio de check_balances para todos los equipos
│   ├── signals-and-control.md    → BC reconfigurables, resolve(), snap, controladores, optimización
│   └── validation-from-articles.md → metodología case_card: de artículo a simulación
│
├── equipment/               ← catálogo por equipo
│   ├── common.md            → funciones reutilizables por todos los equipos
│   ├── heater.md            → estado, archivos, config, diferencias
│   ├── adsorber.md          → estado, LDF, isotermas, pasos PSA
│   ├── gasifier.md          → estado, reacciones, modos, q_masstransfer
│   ├── valve.md             → ISA-75.01, interfaz auxiliares
│   └── future-auxiliaries.md → plantilla bombas, ventiladores, compresores
│
└── physics/                 ← conocimiento físico codificado
    ├── cross-phase.md       → transferencia sólido→gas, gas→adsorbido
    ├── shell-tube.md        → pared dinámica, 4 modos thermal_bc
    ├── energy-balances.md   → ecuaciones, signos, residual, balances
    ├── hydraulics.md        → Ergun, continuidad, válvula, v superficial
    └── reactions.md         → SCM, drying, pyrolysis, char, plantilla nueva rxn
```

---

## Estructura del proyecto (src/)

```
src/
├── units/<equipo>/           ← estado y config (específico del equipo)
│   ├── state.py              pack/unpack del vector de estado
│   ├── state_extraction.py   build_<equipo>_results → col
│   └── config/               build_gas_prop_config, build_boundary_c_config, ...
├── boundary_conditions/      ← contornos (específico del equipo)
│   ├── <equipo>_boundary.py  get_<equipo>_boundary()
│   └── valve.py              valve_superficial_velocity()
├── physics/                  ← física pura (REUTILIZABLE sin modificación)
│   ├── thermodynamics/       pure_gas, mixture, enthalpy, solid_props
│   ├── transport/            diffusion, transfer_coefficients, nusselt
│   ├── momentum/             ergun, darcy_weisbach
│   ├── thermal/              wall_heat_flux, wall_ode
│   ├── mixture_gas.py        compute_gas_mixture_properties
│   └── reactions/            drying, pyrolysis, char_conversion
├── discretization/           ← esquemas numéricos (REUTILIZABLE)
│   ├── fluxes.py             convective_flux, diffusive_flux, gas_enthalpy_convective_flux
│   └── face_reconstruction.py
├── solvers/
│   ├── rhs/rhs_<equipo>.py   core_rhs(t, sv, params) → dydt
│   └── runner_<equipo>.py    run_step(...) → (t_arr, y_hist, col)
├── postprocessing/
│   ├── variables_plot.py     Graph_P, Graph_Tg, ... (REUTILIZABLE)
│   └── <equipo>_balances.py  molar_balance, energy_balance
├── io/                       gasdb_reader, soliddb_reader, fuels_reader
├── utils/                    profiling, isotherm_models, isotherm_fitting
│   └── signals.py            resolve(signal, t, snap) — universal BC resolver
└── control/                  ← señales, controladores y optimización (pendiente)
    ├── signals.py            ramp, step, pulse, sine, piecewise → callable(t)
    ├── controllers.py        on_off, proportional, pid → callable(t, snap)
    └── optimization.py       parametric_sweep, optimize_bc, sensitivity_analysis
```

```
tools/                        ← herramientas ejecutables (notebooks + scripts)
├── kinetics/                 ajuste TGA/FBR → parámetros fuel YAML
├── validation/               ejecutar case_card, comparar con datos de artículo
├── benchmarks/               sensibilidad de malla, tolerancias, timing
└── campaigns/                barridos paramétricos y optimización de proceso
```

---

## Equipos implementados

| Equipo | Estado | sv (sin/con shell_tube) | Fases |
|--------|--------|-------------------------|-------|
| Heater | ✓ Validado | (nc+1)·N / (nc+2)·N | Gas, [Pared] |
| Adsorber | ✓ Validado | (2nc+2)·N / (2nc+3)·N | Gas, Adsorbida, Sólido |
| Gasifier | ✓ Implementado | 16·N / 17·N | Gas (9 esp.), Sólido (3 comp.), Acumuladores energéticos |

---

## Reglas de arquitectura (resumen ejecutivo)

1. `physics/` no importa de `solvers/`, `units/`, `boundary_conditions/`
2. El RHS es el único punto de ensamblado de la física
3. Sin validaciones dentro del RHS (van en el runner)
4. Cada equipo tiene su propio runner y su propio RHS
5. `params` dict se construye siempre con funciones `build_*`
6. BC acepta `float | callable(t) | callable(t, snap)` — resolución vía `resolve()` en el runner
7. `resolve()` nunca se llama dentro del RHS (hot path del Jacobiano BDF)
8. Detalle completo → `rules/layer-separation.md` y `rules/signals-and-control.md`

---

## Reglas físicas críticas

| Regla | Documento |
|-------|-----------|
| Cross-phase sólido→gas: `dHgdt += epsi·Σ src·h_i(Ts)` | `physics/cross-phase.md` |
| Shell-tube: ninguna combinación thermal_bc está prohibida | `physics/shell-tube.md` |
| Residual energético ≈ Q_rxn (no es error numérico) | `physics/energy-balances.md` |
| q_gs_vol tiene signo OPUESTO en dHgdt y dTsdt | `physics/energy-balances.md` |
| Tasas heterogéneas en m³_BED → dividir por epsi para m³_GAS | `rules/units-shapes.md` |

---

## Para añadir un equipo nuevo

```
1. Lee equipment/common.md → identifica qué reutilizas
2. Ejecuta /new-equipment → guía paso a paso
3. Audita el RHS con /check-rhs antes de simular
4. Verifica balances con <equipo>_balances.py
```
