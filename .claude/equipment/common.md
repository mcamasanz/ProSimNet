# Funciones comunes — reutilizables por todos los equipos

Estos módulos no saben qué equipo los usa. Se importan directamente sin modificación.

---

## Contrato obligatorio de BC reconfigurables (todos los equipos)

Desde la implementación de referencia en el gasificador en adelante, **todos los equipos deben** cumplir este contrato en sus `build_bc_config` y runners:

### 1. Tipos aceptados en cada campo BC

```python
# Todo campo BC acepta los tres tipos — validar en build_bc_config, no en el RHS
float | ndarray          # constante
callable(t) → float      # perfil temporal
callable(t, snap) → float  # feedback de estado
```

### 2. Resolución en el runner (nunca en el RHS)

```python
from src.utils.signals import resolve

# En run_step, antes de pasar al integrador:
v_in_now = resolve(params["bc_config"]["v_gas_in"], t_current, snap)
T_in_now = resolve(params["bc_config"]["T_gas_in"], t_current, snap)
```

### 3. Construcción del snap al final de cada paso

```python
snap = {
    "t":         float,           # tiempo [s]
    "P_bar":     ndarray(N,),
    "Tg_K":      ndarray(N,),
    "Ts_K":      ndarray(N,),     # si el equipo tiene fase sólida
    "y_gas":     ndarray(nc, N),
    "rho_solid": ndarray(3, N),   # si el equipo tiene fase sólida
    "v_out":     float,
    "v_in":      float,
}
# El snap de equipos sin sólido omite Ts_K y rho_solid
```

### 4. Validación de tipo en build_bc_config

```python
def _validate_bc_field(name: str, value) -> None:
    if not (isinstance(value, (int, float, np.ndarray)) or callable(value)):
        raise ValueError(f"{name} debe ser float o callable, got {type(value)}")
```

**Referencia completa:** `.claude/rules/signals-and-control.md`
**Equipo de referencia:** gasificador — primera implementación del patrón

---

## Template de referencia para test_00

Al crear el primer test de un nuevo equipo (`test_<equipo>_00_config_survey.ipynb`),
usar como plantilla directa:

**`test/gasifier/test_gasifier_00_config_survey.ipynb`**

Ese notebook es la referencia canónica. Estructura, convenciones de variables,
formato de tablas, orden de TEST 1→5, validación sin integración — todo está
implementado de forma correcta y aprobada en ese fichero.

Ver reglas detalladas en `.claude/rules/test-methodology.md` §"test_<equipo>_00_config_survey".

---

## Termodinámica del gas

### `src/physics/thermodynamics/pure_gas.py`
```python
build_pure_gas_properties(species, mode, n_comp, N, db_path) → dict prop_gas
# mode: "polynomial" (gasdb polinomios) | "constant" (Cp constante)
# prop_gas contiene: MW(nc,), mu(T), k(T), Cp_molar(T), h_molar(T), Tref, Tmax

eval_species_property(entry, T_arr, n_comp) → ndarray(len(T), nc)
# entry: lista de callables o ndarray constante
# T_arr: array de temperaturas (tratado como eje espacial)
```

### `src/physics/thermodynamics/mixture.py`
```python
wilke_mix_property(x, prop_individual, MW) → ndarray(N,)
# x: (N, nc) — EXCEPCIÓN de shape (celdas primero)
# Regla de Wilke para mu, k de mezcla

molar_mix_property(x, prop_individual) → ndarray(N,)
# Promedio molar para Cp, etc.

ideal_gas_density(P_Pa, T, MW_mix) → ndarray(N,)  # [kg/m³]
```

### `src/physics/thermodynamics/enthalpy.py`
```python
calc_species_enthalpy(Tg, prop_gas, n_comp, gas_T_ref) → ndarray(nc, N)
# Devuelve h_i(T) [J/mol] para cada especie y celda
# ⚠️ shape (nc, N): especies primero, celdas segundo

calc_volumetric_enthalpy(C, Tg, prop_gas, n_comp, epsi, gas_T_ref) → ndarray(N,)
# Hg = epsi * Σ_i C_i * h_i(Tg)  [J/m³_bed]

recover_Tg_from_Hg(C, Hg, prop_gas, n_comp, epsi, Tg_guess, gas_T_ref) → ndarray(N,)
# Inversión por Newton: Hg = epsi * Σ C_i * h_i(Tg) → Tg
# Tg_guess: warm-start desde cache["Tg_last"]
```

### `src/physics/thermodynamics/solid_props.py`
```python
eval_solid_property(entry, T) → ndarray(N,)
# entry: float | callable | dict con segmentos
# Evalúa rho, Cp, k del sólido (pared, char, etc.) en función de T
```

---

## Propiedades de mezcla (punto de entrada completo)

### `src/physics/mixture_gas.py`
```python
compute_gas_mixture_properties(P_Pa, Tg, x, prop_gas, n_comp, N) → dict
# x: (N, nc)  — fracciones molares en formato Wilke
# Devuelve: rho_g(N,), mu_g(N,), k_g(N,), h_i(N,nc), Dim(N,nc)
# Punto de entrada único: llama a Wilke, ideal_gas, entalpía, difusión
```

---

## Transporte

### `src/physics/transport/transfer_coefficients.py`
```python
compute_transfer_coefficients(
    Tg, Ts, x, gas_props, u_rel, prop_gas, prop_lecho, Di, trans_config, n_comp, N,
    Tw=None, L=None
) → dict
# Modos: "constant" (usa h_bed, h_wall del trans_config)
#         "correlation" (Ranz-Marshall h_bed, Dittus-Boelter h_wall)
# Devuelve: h_bed(N,), h_wall(N,), D_disp(nc,N) o None, k_mtc(nc,N) o None
```

### `src/physics/transport/diffusion.py`
```python
binary_diffusivity(T, P_Pa, i, j, prop_gas) → float  # [m²/s]
mixture_diffusivity(T, P_Pa, x, prop_gas, n_comp) → ndarray(N, nc)  # [m²/s]
axial_dispersion(v, dp, D_mol) → ndarray(N,)  # correlación Wakao
```

### `src/physics/transport/nusselt.py`
```python
h_wall_tube(Re, Pr, Di, L, mu_bulk, mu_wall, k_film) → ndarray(N,)  # [W/m²/K]
compute_Ra_Di(Tg, Tw, Di, prop_gas_film) → ndarray(N,)  # Rayleigh para conv. natural
```

---

## Hidráulica

### `src/physics/momentum/ergun.py`
```python
ergun_face_velocity(P_Pa, rho_g, mu_g, epsi, dp, v_in, v_out, dz, N) → ndarray(N+1,)
# Para lechos empaquetados. P en Pa, dp en m.
```

### `src/physics/momentum/darcy_weisbach.py`
```python
continuity_face_velocity(rho_g, v_in, v_out, N) → ndarray(N+1,)
# Para tubo vacío (conservación de masa sin fricción interna)
```

---

## Pared térmica

### `src/physics/thermal/wall_heat_flux.py`
```python
wall_heat_flux(Tg, h_wall, thermal_bc_config, N, Ai, Pi, Po, dz) → tuple
# Devuelve: (qwall_vol(N,), Qwall_dot, qwall_area(N,))
# 4 modos: "adiabatic", "fixed_twall", "heatfluxwall", "ambient_htc"
# Solo para shell_tube=False. Con shell_tube usar h_wall*Pi/Ai*(Tw-Tg).
```

### `src/physics/thermal/wall_ode.py`
```python
wall_ode_rhs(Tg, Tw, h_wall, wall_config, thermal_bc_config, Pi, Po, dz) → ndarray(N,)
# dTw/dt para la ODE de pared dinámica (shell_tube=True)
wall_exterior_q(Tw, thermal_bc_cfg, k_w, Pi, Po, dz, N) → ndarray(N,)  # [W/celda]
wall_axial_q(Tw, k_w, A_w, dz, N) → ndarray(N,)  # conducción axial en pared
```

---

## Discretización

### `src/discretization/fluxes.py`
```python
convective_flux(phi_cell, v_face, phi_in, phi_out, bc_in, bc_out) → ndarray(N+1,)
# Upwind de primer orden. phi_cell: (N,). v_face: (N+1,).

diffusive_flux(phi_cell, Gamma, dz, phi_in, phi_out, bc_in, bc_out, face_method) → ndarray(N+1,)
# Gamma: difusividad (N,) o (nc,N). face_method: "arithmetic"|"harmonic"

gas_enthalpy_convective_flux(Tg_cell, C_cell, v_face, prop_gas, n_comp, gas_T_ref, T_in, C_in) → ndarray(N+1,)
# F_h[k] = v_face[k] * H_upwind[k]  donde H = Σ C_i * h_i(T)  [W/m²]
# ⚠️ H_cell NO incluye epsi (el factor epsi va en dHgdt del RHS)

gas_diffusive_heat_flux(Tg_cell, k_g, dz, T_in, T_out, bc_in, bc_out) → ndarray(N+1,)
# Conducción axial del gas [W/m²]

solid_convective_flux(rho_cell, vs_face, rho_solid_in) → ndarray(N+1,)
# Flujo convectivo de densidad sólida [kg/m²/s]
```

### `src/discretization/face_reconstruction.py`
```python
face_property(phi_cell, phi_in, phi_out, bc_in, bc_out, method) → ndarray(N+1,)
upwind_reconstruction(phi_cell, v_face, phi_in, phi_out) → ndarray(N+1,)
```

---

## Válvula (auxiliar)

### `src/boundary_conditions/valve.py`
```python
valve_superficial_velocity(Cv, P_up_bar, P_down_bar, T, MW_mix, epsi, Ai) → float
# Modelo ISA-75.01. Devuelve v_superficial [m/s]. Ver equipment/valve.md.
```

---

## IO y utilidades

### `src/io/gasdb_reader.py`
```python
read_gasdb(db_path, species) → dict prop_gas_raw
# Lee gasdb.txt: MW, mu(T), k(T), Cp(T), h(T), Tmin, Tmax, sigmaLJ, epsK
# Polinomios grado-7 en τ = ΔT/ΔT_range, válidos hasta 5000K
```

### `src/io/soliddb_reader.py`
```python
read_soliddb(db_path, material) → dict prop_solid_raw
# Materiales: SS316L, P265GH, Inconel625, Al2O3, SiC, SiO2, Cu
```

### `src/io/fuels_reader.py`
```python
read_fueldb(fuel_path) → dict fuel_config
# fuel_config contiene: heating_values, pyrolysis_yields, kinetics, char_composition, co_co2_ratio
```

### `src/utils/profiling.py`
```python
@profiled  # decorator — mide tiempo de cada llamada
print_benchmark_functions(n=10)  # imprime las N funciones más lentas
```

### `src/postprocessing/variables_plot.py`
```python
# 11 funciones gráficas, todas con firma: (col, params, **kwargs) → fig
Graph_P, Graph_Tg, Graph_Ts, Graph_Tw, Graph_v, Graph_y,
Graph_C, Graph_q, Graph_rho_solid, Graph_Hg, Graph_profiles
```
