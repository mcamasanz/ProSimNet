# Gasifier — Reactor de biomasa 1D

## Descripción física

Reactor de lecho empaquetado donde la biomasa sufre secado, pirólisis y reacciones
de gasificación del char. Produce un gas de síntesis (syngas) rico en CO, H₂, CH₄.
El sólido puede estar estático o moverse; el modo de operación se determina
implícitamente por las condiciones de contorno (no hay parámetro `mode` explícito).

**Casos de uso:** gasificación de biomasa, char, residuos sólidos.

---

## Vector de estado

```
Sin shell_tube:  sv = [C(9,N), rho_s(3,N), Hg(N), Ts(N), Q_mt_acc(N), Q_rxn_acc(N)]
                 tamaño = 16 · N

Con shell_tube:  sv = [C(9,N), rho_s(3,N), Hg(N), Ts(N), Tw(N), Q_mt_acc(N), Q_rxn_acc(N)]
                 tamaño = 17 · N

Variables primarias:
  C_i        [mol/m³_gas]  — 9 especies gas: CO, CO2, H2O, H2, O2, CH4, C2H4, tar, N2
  rho_bio    [kg/m³_bed]   — densidad bulk biomasa
  rho_char   [kg/m³_bed]   — densidad bulk char
  rho_moi    [kg/m³_bed]   — densidad bulk humedad
  Hg         [J/m³_bed]    — entalpía volumétrica del gas = ε·Σ C_i·h_i(Tg)
  Ts         [K]           — temperatura del sólido (integrada directamente)
  Tw         [K]           — temperatura de la pared (solo con shell_tube)

Acumuladores energéticos (siempre presentes, inician en 0):
  Q_mt_acc   [J/m³_bed]   — ∫q_masstransfer dt  (entalpía portada por masa sól→gas)
  Q_rxn_acc  [J/m³_bed]   — ∫Q_rxn_vol dt       (calor de reacciones en el sólido)

Por qué Q_mt_acc y Q_rxn_acc están en el sv:
  BDF los integra con el mismo control de error que el resto del sistema.
  Permiten cierres energéticos exactos sin re-evaluar el RHS.
  Sin ellos, Q_mt y Q_rxn solo son recuperables como residuales (aproximación).
```

---

## Archivos específicos

| Archivo | Función principal | Descripción |
|---------|-------------------|-------------|
| `src/units/gasifier/state.py` | `pack_state_vector`, `unpack_state_vector` | Layout sv, índices _IDX |
| `src/units/gasifier/state_extraction.py` | `build_gasifier_results` | Objeto col con `_Q_mt_acc_results`, `_Q_rxn_acc_results` |
| `src/units/gasifier/config/gas_props.py` | `build_gas_prop_config` | 9 especies (gasdb) |
| `src/units/gasifier/config/solid_props.py` | `build_solid_prop_config` | Biomasa, char, moisture Cp, k |
| `src/units/gasifier/config/boundary_c.py` | `build_bc_config` | BCs explícitas (v_gas_in, outlet, v_solid, direction) |
| `src/units/gasifier/config/initial_c.py` | `build_initial_c_config` | sv0 (con Tw_init si shell_tube) |
| `src/boundary_conditions/gasifier_boundary.py` | `get_gasifier_boundary` | Evalúa BCs en t sin dispatch por mode |
| `src/physics/reactions/drying.py` | `drying_rate`, `drying_gas_source`, `drying_enthalpy_sink` | Cinética de secado |
| `src/physics/reactions/pyrolysis.py` | `pyrolysis_rate`, `pyrolysis_sources`, `pyrolysis_enthalpy_sink`, `compute_pyrolysis_dH` | Pirólisis primaria |
| `src/physics/reactions/char_conversion.py` | `char_het_rates`, `char_gas_sources`, `char_reaction_heat`, `particle_diameter`, `specific_surface_area` | Reacciones heterogéneas del char |
| `src/solvers/rhs/rhs_gasifier.py` | `core_rhs` | RHS 12 pasos |
| `src/solvers/runner_gasifier.py` | `run_step` | Valida + integra |
| `src/postprocessing/gasifier_balances.py` | `check_balances`, `total_mass_balance`, `print_summary` | Balances con cierres exactos |

---

## Especies gaseosas (orden fijo)

```python
_IDX = {"CO": 0, "CO2": 1, "H2O": 2, "H2": 3, "O2": 4,
        "CH4": 5, "C2H4": 6, "tar": 7, "N2": 8}
# nc = 9 — FIJO. El orden determina el layout de C en el sv.
```

---

## Puntos de operación — determinados por las BC (no hay parámetro mode)

| Punto de operación | `v_gas_in` | `outlet` | `v_solid` | `direction` | `inlet_mode` |
|--------------------|-----------|----------|-----------|-------------|--------------|
| batch (cerrado)    | None | `"open"` | 0 | — | — |
| semibatch (venteo) | None | `"vent"` | 0 | — | — |
| CSTR / paso gas    | > 0 | `"open"` | 0 | — | — |
| updraft            | > 0 | `"open"` | > 0 | `"updraft"` | `"prescribed"` |
| downdraft          | > 0 | `"open"` | > 0 | `"downdraft"` | `"prescribed"` |
| conveyor           | > 0 | `"open"` | > 0 | `"updraft"/"downdraft"` | `"computed"` |

```python
# Ejemplos
bc = build_bc_config(n_comp=9)                                         # batch
bc = build_bc_config(n_comp=9, outlet="vent", v_vent_max=0.50)         # semibatch
bc = build_bc_config(n_comp=9, v_gas_in=0.05, T_gas_in=800., y_gas_in=y_air)  # CSTR
bc = build_bc_config(n_comp=9, v_gas_in=0.05, T_gas_in=800., y_gas_in=y_air,
                     v_solid=1.5e-4, direction="updraft",
                     rho_solid_in=rho_s, T_solid_in=300.)              # updraft
```

---

## Reacciones implementadas

| Reacción | Función | Retorno |
|----------|---------|---------|
| Secado: H₂O(l) → H₂O(g) | `drying_rate` | `r_dry` [kg/m³_bed/s] |
| Pirólisis: biomasa → gas + char | `pyrolysis_rate` | `r_pyr` [kg/m³_bed/s] |
| Char + O₂ → CO/CO₂ | `char_het_rates` (r_ox) | [kg_char/m³_bed/s] |
| Char + CO₂ → 2CO (Boudouard) | `char_het_rates` (r_CO2) | [kg_char/m³_bed/s] |
| Char + H₂O → CO + H₂ | `char_het_rates` (r_H2O) | [kg_char/m³_bed/s] |

**Pendiente:** WGS (CO + H₂O ⇌ CO₂ + H₂), tar cracking.

---

## Regla crítica: q_masstransfer en dHgdt

```python
# source_gas construido en paso 8 con las 3 fuentes (en mol/m³_gas/s):
source_gas[j] += src_dry_H2O   / epsi_safe   # solo H2O
source_gas[j] += src_pyr_gas[j]/ epsi_safe   # todas las especies
source_gas[j] += src_char_gas[j]/ epsi_safe  # CO, CO2, H2O

# En paso 10 — OBLIGATORIO:
h_i_Ts = calc_species_enthalpy(Ts_arr, prop_gas, nc, gas_T_ref)   # (nc, N)
q_masstransfer = epsi_r * np.sum(source_gas * h_i_Ts, axis=0)     # (N,)
dHgdt += q_masstransfer

# Acumulador (paso 10, junto con q_masstransfer):
dQ_mt_acc_dt  = q_masstransfer  # integrado por BDF → Q_mt exacto en post-proceso
dQ_rxn_acc_dt = Q_rxn_vol       # integrado por BDF → Q_rxn exacto en post-proceso
```

Ver `.claude/physics/cross-phase.md` para la explicación física completa.

---

## Clip defensivo en el RHS (líneas 172-185)

BDF perturba el estado para estimar el Jacobiano; esas perturbaciones pueden hacer
`C < 0` o `rho_solid < 0`, lo que produce NaN en `dp=(rho/rho0)^(1/3)` y `rho_g < 0`.

```python
# Al inicio del RHS, ANTES de cualquier cálculo físico:
C_mat     = np.maximum(state["C"],         0.0)
rho_solid = np.maximum(state["rho_solid"], 0.0)

# Recomputar desde C clipado para consistencia:
Ctot_arr = np.sum(C_mat, axis=0)
y_mat    = C_mat / np.maximum(Ctot_arr, 1e-300)[None, :]
P_bar    = np.maximum(Ctot_arr * R_GAS * Tg_arr / 1e5, 1e-6)
P_Pa     = P_bar * 1e5
```

**Umbral de sólido depleted** (`_EPS_RHO = 1e-6`): cuando `rho_total < 1e-6`, se
ponen `q_gs_vol = 0` y `dTsdt = 0`. Sin esto, `a_p → ∞` (SCM) produce `dTs/dt ~ 1e10`.

---

## SCM — Shrinking Core Model para char

```python
dp = particle_diameter(rho_char, rho_char0, dp0)   # (N,) [m]
a_p = specific_surface_area(dp, epsi_r)            # (N,) [m²/m³_bed]
# La resistencia a la transferencia de masa depende de dp → h_bed evoluciona
```

**Re y Sc en Ranz-Marshall se clipean a ≥ 0** para proteger las potencias fraccionarias
cuando rho_g se acerca a cero durante la estimación del Jacobiano.

---

## Claves requeridas en params

```python
_REQUIRED_COMMON = (
    "n_comp", "N", "dz", "Ai", "Di", "Pi", "Po",
    "prop_gas", "MW", "gas_T_ref",
    "bc_config", "trans_config", "thermal_bc_config", "energy",
)

_REQUIRED_GASIFIER = (
    "epsi_r",       # float — porosidad [-]
    "dp0",          # float — diámetro inicial partícula [m]
    "rho_char0",    # float — densidad ref. char para SCM [kg/m³_bed]
    "fuel_config",  # dict — output de read_fueldb()
    "solid_config", # dict — output de build_solid_prop_config()
    "species",      # list[str] — 9 especies en orden fijo
)
# wall_config: opcional → activa shell_tube (sv crece de 16N a 17N)
# dH_pyr: inyectado automáticamente en runner si no está en params
```

---

## Atributos del objeto gasifier (SimpleNamespace)

```python
gasifier._t_results              # (n_t,)
gasifier._z                      # (N,)
gasifier._species                # list[str] — 9 elementos
gasifier._P_results              # (n_t, N) [bar]
gasifier._Tg_results             # (n_t, N) [K]
gasifier._Ts_results             # (n_t, N) [K]
gasifier._Tw_results             # (n_t, N) o None
gasifier._Hg_results             # (n_t, N) [J/m³_bed]
gasifier._y_results              # (n_t, 9, N) [-]
gasifier._C_results              # (n_t, 9, N) [mol/m³_gas]
gasifier._rho_solid_results      # (n_t, 3, N) [kg/m³_bed] — [bio, char, moi]
gasifier._v_results              # (n_t, N) [m/s]
gasifier._v_in_results           # (n_t,) [m/s]
gasifier._v_out_results          # (n_t,) [m/s]
gasifier._C_in_results           # (n_t, 9) o NaN si batch
gasifier._T_in_results           # (n_t,) o NaN si batch
gasifier._Q_mt_acc_results       # (n_t, N) [J/m³_bed]  ∫q_mt dt
gasifier._Q_rxn_acc_results      # (n_t, N) [J/m³_bed]  ∫Q_rxn dt
```

---

## Balances disponibles

```python
from src.postprocessing.gasifier_balances import (
    check_balances,      # balance completo con cierres exactos (ver balance-rules.md)
    total_mass_balance,  # Δm_gas + Δm_solid = flux → cierre ★ numérico
    print_summary,       # alias que llama check_balances
)
```

**Cierres numéricos reales (★):**
- `total_mass_balance`: Δm_gas + Δm_solid − flux_masa ≈ 0 (siempre, con o sin rxn)
- `Cierre_Hg`: ΔHg − Fh_neto + Q_gs − Q_wall − Q_mt_exact ≈ 0 (exacto, acum. ODE)
- `Cierre global`: ΔHg + ΔHs_proxy − Fh_neto − Q_wall − Q_rxn_exact − Q_mt_exact ≈ pequeño
  (residual = error del proxy ΔHs por T_ref=0)
- `Cierre pared` (shell_tube): ΔHw − Q_gw ≈ 0

**Residuales físicos (no son errores):**
- `fuente_rxn_i`: moles producidos/consumidos de cada especie gaseosa
- `S_rxn_j`: masa transformada de cada componente sólido
- `Q_rxn_exact`: calor de reacciones en el sólido (exacto del acumulador)
- `Q_mt_exact`: entalpía portada por masa sól→gas (exacto del acumulador)
