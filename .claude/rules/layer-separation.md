# Separación de capas — qué va en cada módulo

## Regla fundamental

**Un módulo no importa de capas superiores a él.**

```
physics/ ← no importa de solvers/, units/, boundary_conditions/
discretization/ ← no importa de physics/ (solo geometría)
solvers/rhs/ ← importa de physics/, discretization/, boundary_conditions/
solvers/runner_*.py ← importa de solvers/rhs/, units/<equipo>/
postprocessing/ ← importa de physics/ (para reconstruir), NO de solvers/rhs/
```

---

## Tabla de responsabilidades por capa

| Capa | Path | Contiene | NO contiene |
|------|------|----------|-------------|
| **Física pura** | `src/physics/` | Ecuaciones físicas agnósticas del equipo | Llamadas a BC, estado ODE, plots |
| **Discretización** | `src/discretization/` | Esquemas de flujo, reconstrucción en caras | Física (µ, ρ, k), equipos específicos |
| **Condiciones de contorno** | `src/boundary_conditions/` | Contornos por paso/modo, lógica de válvulas | Integración ODE, física interna |
| **Config del equipo** | `src/units/<eq>/config/` | Construcción y validación de sub-dicts de params | Cálculo físico, integración |
| **Estado del equipo** | `src/units/<eq>/state.py` | Pack/unpack del vector de estado | Física, propiedades |
| **Extracción** | `src/units/<eq>/state_extraction.py` | Reconstrucción del objeto col desde y_hist | Integración ODE |
| **RHS** | `src/solvers/rhs/rhs_<eq>.py` | Física específica del proceso | Post-proceso, plots, validación |
| **Runner** | `src/solvers/runner_<eq>.py` | Validación de params + integrador ODE | Física directa, plots |
| **Post-proceso** | `src/postprocessing/` | Gráficas y balances de verificación | RHS, física interna |
| **IO** | `src/io/` | Lectura de bases de datos (gasdb, soliddb, fuels) | Física, integración |
| **Utils** | `src/utils/` | Herramientas agnósticas (profiling, isotermas) | Equipos específicos |

---

## Módulos genuinamente reutilizables (nunca modificar para un equipo nuevo)

```
src/physics/thermodynamics/pure_gas.py      build_pure_gas_properties, eval_species_property
src/physics/thermodynamics/mixture.py       wilke_mix_property, ideal_gas_density
src/physics/thermodynamics/enthalpy.py      calc_species_enthalpy, recover_Tg_from_Hg
src/physics/thermodynamics/solid_props.py   eval_solid_property
src/physics/transport/diffusion.py          binary_diffusivity, mixture_diffusivity
src/physics/transport/transfer_coefficients.py  compute_transfer_coefficients
src/physics/transport/nusselt.py            h_wall_tube, compute_Ra_Di
src/physics/momentum/ergun.py              ergun_face_velocity
src/physics/momentum/darcy_weisbach.py     continuity_face_velocity
src/physics/thermal/wall_heat_flux.py      wall_heat_flux   (4 modos)
src/physics/thermal/wall_ode.py            wall_ode_rhs, wall_exterior_q
src/physics/mixture_gas.py                 compute_gas_mixture_properties
src/discretization/fluxes.py              convective_flux, diffusive_flux, gas_enthalpy_convective_flux
src/discretization/face_reconstruction.py  face_property, upwind_reconstruction
src/boundary_conditions/valve.py           valve_superficial_velocity
src/io/gasdb_reader.py                     read_gasdb
src/io/soliddb_reader.py                   read_soliddb
src/postprocessing/variables_plot.py       Graph_P, Graph_Tg, Graph_v, ... (11 funciones)
src/utils/profiling.py                     @profiled, print_benchmark_functions
```

---

## Módulos que se crean por equipo (plantilla, no copiar-pegar)

```
src/units/<equipo>/state.py                pack_state_vector, unpack_state_vector
src/units/<equipo>/state_extraction.py     build_<equipo>_results
src/units/<equipo>/config/gas_props.py     build_gas_prop_config (re-export de pure_gas)
src/units/<equipo>/config/boundary_c.py   build_boundary_c_config  ← específico
src/units/<equipo>/config/initial_c.py    build_initial_conditions ← específico
src/units/<equipo>/config/thermal_bc.py   build_thermal_bc_config  (re-export)
src/units/<equipo>/config/transport.py    build_transport_config   ← adaptar
src/units/<equipo>/config/wall_c.py       build_wall_config        (re-export)
src/boundary_conditions/<equipo>_boundary.py  get_<equipo>_boundary  ← específico
src/solvers/rhs/rhs_<equipo>.py           core_rhs                 ← específico
src/solvers/runner_<equipo>.py            run_step                 ← específico
src/postprocessing/<equipo>_balances.py   molar_balance, energy_balance ← específico
```

---

## Antipatrones a evitar

```python
# ❌ INCORRECTO: physics importa de units
# src/physics/thermodynamics/enthalpy.py
from src.units.gasifier.state import unpack_state_vector  # NUNCA

# ❌ INCORRECTO: RHS importa de postprocessing
# src/solvers/rhs/rhs_gasifier.py
from src.postprocessing.gasifier_balances import energy_balance  # NUNCA

# ❌ INCORRECTO: config módulo hace cálculo físico
# src/units/gasifier/config/boundary_c.py
rho_g = ideal_gas_density(P, T, MW)  # NUNCA en config

# ✅ CORRECTO: RHS importa de physics y discretization
from src.physics.thermodynamics.enthalpy import calc_species_enthalpy
from src.discretization.fluxes import convective_flux
from src.boundary_conditions.gasifier_boundary import get_gasifier_boundary
```

---

## config/ re-exports vs funciones propias

Los módulos de config que "re-exportan" de physics son wrappers de conveniencia:

```python
# src/units/gasifier/config/gas_props.py — re-export puro
from src.physics.thermodynamics.pure_gas import build_pure_gas_properties

def build_gas_prop_config(species, mode="polynomial", ...):
    return build_pure_gas_properties(species=species, mode=mode, ...)
```

Los módulos con lógica propia definen las condiciones del proceso:

```python
# src/units/gasifier/config/boundary_c.py — lógica específica
def build_boundary_c_config(mode, Q_in, T_in, y_in, ...):
    # Aquí va la lógica de modos: batch, cstr, updraft, conveyor
    ...
```
