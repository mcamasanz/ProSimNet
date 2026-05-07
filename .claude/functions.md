# functions.md — Índice de funciones del proyecto

> El catálogo detallado está en `equipment/`. Este archivo es el índice de acceso rápido.
> Actualizar cuando se añada, modifique o elimine una función pública.

---

## Funciones comunes (todos los equipos)

Ver `equipment/common.md` para firmas completas, parámetros y retornos.

| Módulo | Funciones clave |
|--------|----------------|
| `physics/thermodynamics/pure_gas.py` | `build_pure_gas_properties`, `eval_species_property` |
| `physics/thermodynamics/mixture.py` | `wilke_mix_property`, `molar_mix_property`, `ideal_gas_density` |
| `physics/thermodynamics/enthalpy.py` | `calc_species_enthalpy`, `calc_volumetric_enthalpy`, `recover_Tg_from_Hg` |
| `physics/thermodynamics/solid_props.py` | `eval_solid_property` |
| `physics/mixture_gas.py` | `compute_gas_mixture_properties` |
| `physics/transport/transfer_coefficients.py` | `compute_transfer_coefficients` |
| `physics/transport/diffusion.py` | `binary_diffusivity`, `mixture_diffusivity`, `axial_dispersion` |
| `physics/transport/nusselt.py` | `h_wall_tube`, `compute_Ra_Di` |
| `physics/momentum/ergun.py` | `ergun_face_velocity` |
| `physics/momentum/darcy_weisbach.py` | `continuity_face_velocity` |
| `physics/thermal/wall_heat_flux.py` | `wall_heat_flux` |
| `physics/thermal/wall_ode.py` | `wall_ode_rhs`, `wall_exterior_q`, `wall_axial_q` |
| `discretization/fluxes.py` | `convective_flux`, `diffusive_flux`, `gas_enthalpy_convective_flux`, `solid_convective_flux` |
| `discretization/face_reconstruction.py` | `face_property`, `upwind_reconstruction` |
| `boundary_conditions/valve.py` | `valve_superficial_velocity` |
| `io/gasdb_reader.py` | `read_gasdb` |
| `io/soliddb_reader.py` | `read_soliddb` |
| `io/fuels_reader.py` | `read_fueldb` |
| `postprocessing/variables_plot.py` | `Graph_P`, `Graph_Tg`, `Graph_Ts`, `Graph_Tw`, `Graph_v`, `Graph_y`, `Graph_C`, `Graph_q`, `Graph_rho_solid`, `Graph_Hg`, `Graph_profiles` |
| `utils/profiling.py` | `@profiled`, `print_benchmark_functions` |
| `utils/isotherm_models.py` | `langmuir`, `DSL`, `DSLF` |
| `utils/isotherm_fitting.py` | `fit_single_T`, `fit_multi_T` |
| `utils/mixture_isotherm.py` | `iast`, `rast` |
| `utils/signals.py` | `resolve(signal, t, snap)` — resuelve `float \| callable(t) \| callable(t, snap)` a valor; `resolve_config_values(cfg, t, snap, keys)` — batch resolver para dicts de BC |
| `utils/optimization.py` | `parametric_sweep`, `sensitivity_analysis`, `optimize_bc` — ver `howto.md` §Análisis paramétrico para contrato completo |
| `control/signals.py` | `ramp`, `step`, `pulse`, `piecewise`, `sine`, `constant` → `callable(t)` |
| `control/controllers.py` | `proportional`, `onoff`, `feedforward` → `callable(t, snap)` |

---

## Funciones específicas por equipo

### Heater → `equipment/heater.md`

| Módulo | Funciones |
|--------|-----------|
| `units/heater/state.py` | `pack_state_vector`, `unpack_state_vector` |
| `units/heater/state_extraction.py` | `build_heater_results` |
| `units/heater/config/*.py` | `build_gas_prop_config`, `build_boundary_c_config`, `build_initial_conditions`, `build_thermal_bc_config`, `build_transport_config`, `build_wall_config` |
| `boundary_conditions/heater_boundary.py` | `get_heater_boundary` |
| `solvers/rhs/rhs_heater.py` | `core_rhs` |
| `solvers/runner_heater.py` | `run_step` |
| `postprocessing/heater_balances.py` | `molar_balance`, `energy_balance` |

### Adsorber → `equipment/adsorber.md`

| Módulo | Funciones |
|--------|-----------|
| `units/adsorber/state.py` | `pack_state_vector`, `unpack_state_vector` |
| `units/adsorber/state_extraction.py` | `build_adsorber_results` |
| `units/adsorber/velocity_history.py` | `compute_velocity_history` |
| `units/adsorber/config/*.py` | `build_gas_prop_config`, `build_adsorbent_config`, `build_boundary_c_config`, `build_initial_conditions`, `build_thermal_bc_config`, `build_transport_config`, `build_wall_config` |
| `boundary_conditions/adsorber_boundary.py` | `get_step_boundary` |
| `solvers/rhs/rhs_adsorption.py` | `core_rhs` |
| `solvers/runner_adsorption.py` | `run_step` |
| `postprocessing/adsorber_balances.py` | `molar_balance`, `energy_balance` |

### Gasifier → `equipment/gasifier.md`

| Módulo | Funciones |
|--------|-----------|
| `units/gasifier/state.py` | `pack_state_vector`, `unpack_state_vector`, `set_state`, `build_sv0_from_results` |
| `units/gasifier/state_extraction.py` | `build_gasifier_results` → col con `_Q_mt_acc_results`, `_Q_rxn_acc_results` |
| `units/gasifier/config/boundary_c.py` | `build_bc_config` (reemplaza `build_boundary_c_config`; sin parámetro `mode`) |
| `units/gasifier/config/*.py` | `build_gas_prop_config`, `build_solid_prop_config`, `build_initial_c_config`, `build_thermal_bc_config`, `build_transport_config`, `build_wall_config` |
| `boundary_conditions/gasifier_boundary.py` | `get_gasifier_boundary` (genérico, sin dispatch por mode) |
| `physics/reactions/drying.py` | `drying_rate`, `drying_gas_source`, `drying_enthalpy_sink` |
| `physics/reactions/pyrolysis.py` | `pyrolysis_rate`, `pyrolysis_sources`, `pyrolysis_enthalpy_sink`, `compute_pyrolysis_dH` |
| `physics/reactions/char_conversion.py` | `char_het_rates`, `char_gas_sources`, `char_reaction_heat`, `particle_diameter`, `specific_surface_area` |
| `solvers/rhs/rhs_gasifier.py` | `core_rhs` — sv 16·N o 17·N (con acumuladores) |
| `solvers/runner_gasifier.py` | `run_step` — parámetro `max_step` añadido |
| `postprocessing/gasifier_balances.py` | `check_balances(col, params, verbose)` → dict; `display_balances(bal)` → tabla formateada en Jupyter |
| `postprocessing/gasifier_plots.py` | Plots individuales: `plot_temperatures`, `plot_solid_evolution`, `plot_gas_composition`, `plot_pressure`, `plot_velocities`, `plot_summary` |
| `postprocessing/gasifier_plots.py` | **Plots de barrido** (input: `df` de `parametric_sweep` + `results`): `plot_sweep_profiles`, `plot_sweep_composition`, `plot_sweep_solid`, `plot_sweep_pressure`, `plot_sweep_metrics` — ver `howto.md` §Análisis paramétrico |

### Valve / Auxiliares → `equipment/valve.md`, `equipment/future-auxiliaries.md`

| Módulo | Funciones |
|--------|-----------|
| `boundary_conditions/valve.py` | `valve_superficial_velocity` |

---

## Dict params — claves obligatorias

### Comunes (todos los equipos)

```python
"n_comp", "N", "dz", "Ai", "Di", "Pi", "Po",
"prop_gas", "MW", "gas_T_ref",
"bc_config", "trans_config", "thermal_bc_config", "energy"
```

### Específicas por equipo

| Equipo | Claves adicionales |
|--------|-------------------|
| Heater | — (ninguna adicional; `wall_config` opcional) |
| Adsorber | `iso_fn`, `dH`, `epsi`, `rho_s`, `Cp_s`, `k_s`, `prop_lecho` |
| Gasifier | `epsi_r`, `dp0`, `rho_char0`, `fuel_config`, `solid_config`, `species` |

Ver `equipment/<equipo>.md` para la lista completa con tipos y unidades.

---

## Objeto col — atributos por equipo

| Atributo | Heater | Adsorber | Gasifier |
|----------|--------|----------|---------|
| `_t_results` | ✓ | ✓ | ✓ |
| `_z` | ✓ | ✓ | ✓ |
| `_species` | ✓ | ✓ | ✓ |
| `_P_results` | ✓ | ✓ | ✓ |
| `_Tg_results` | ✓ | ✓ | ✓ |
| `_Ts_results` | — | ✓ | ✓ |
| `_Tw_results` | ✓/None | ✓/None | ✓/None |
| `_Hg_results` | ✓ | ✓ | ✓ |
| `_y_results` | ✓ | ✓ | ✓ |
| `_C_results` | ✓ | ✓ | ✓ |
| `_q_results` | — | ✓ | — |
| `_rho_solid_results` | — | — | ✓ |
| `_v_results` | ✓ | ✓ | ✓ |
| `_v_in_results` | ✓ | ✓ | ✓ |
| `_v_out_results` | ✓ | ✓ | ✓ |
| `_C_in_results` | ✓ | ✓/NaN | ✓/NaN |
| `_T_in_results` | ✓ | ✓/NaN | ✓/NaN |
| `_Q_mt_acc_results` | — | — | ✓ (n_t, N) [J/m³_bed] |
| `_Q_rxn_acc_results` | — | — | ✓ (n_t, N) [J/m³_bed] |
