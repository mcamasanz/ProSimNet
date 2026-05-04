# ProSimNet — Process Simulation Network

Herramienta de simulación 1D de equipos de proceso industrial basada en modelos
físico-matemáticos. Desarrollada en Python con integración ODE (scipy BDF/LSODA).

## Equipos implementados

| Equipo | Módulo | Estado |
|---|---|---|
| Calentador de gas 1D (shell-tube) | `src/solvers/runner_heater.py` | Validado experimentalmente |
| Adsorbedor 1D PSA/TSA/VSA | `src/solvers/runner_adsorption.py` | Validado |
| Gasificador de biomasa 1D | `src/solvers/runner_gasifier.py` | Implementado — pendiente validación experimental |

## Estructura del proyecto

```
src/
├── units/
│   ├── heater/          ← configuración, estado y extracción del calentador
│   └── adsorber/        ← configuración, estado y extracción del adsorbedor
├── physics/             ← termodinámica, transporte, momentum, térmica (agnóstico al equipo)
├── discretization/      ← flujos convectivos y difusivos (volúmenes finitos 1D)
├── boundary_conditions/ ← contornos por equipo
├── solvers/rhs/         ← RHS del ODE por equipo
└── utils/               ← profiling, modelos de isoterma
materials/
├── fluids/gasdb.txt     ← propiedades puras de gases (N2, O2, CO, CO2, H2, H2O, CH4, ...)
└── solids/soliddb.txt   ← propiedades de sólidos estructurales (SS316L, Al2O3, ...)
```

## Documentación

- `CLAUDE.md` — filosofía de trabajo, reglas de calidad y convenciones del proyecto
- `.claude/ARCHITECTURE.md` — patrones de implementación, layouts de estado, guía de extensión
- `.claude/functions.md` — catálogo completo de funciones públicas con firmas y unidades
- `manual/MANUAL_PLAN.md` — plan del manual de usuario (en desarrollo)

## Unidades

Sistema SI en todo el código. Temperatura en K, presión en bar (almacenamiento interno),
longitudes en m, energía en J, flujos en mol/m³ y W/m³.

## Física pendiente (gasificador)

- Convección sólida en balance de energía (`−div(vs·Cp_s·Ts)`) para modos updraft/conveyor
- Reacciones homogéneas en fase gas (WGS, tar cracking)
- Validación experimental contra `manual/modelo1dcinetico.pdf`
