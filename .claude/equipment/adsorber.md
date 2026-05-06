# Adsorber — Columna PSA/TSA/VSA

## Descripción física

Lecho empaquetado de sólido adsorbente por el que circula gas. Las moléculas de gas
se adsorben sobre el sólido según la cinética LDF (Linear Driving Force) y el
equilibrio descrito por la isoterma. El ciclo PSA alterna pasos (ads, purge,
blowdown, pressurization) controlados por `bc_config` y el argumento `step`.

**Casos de uso:** separación CO₂/N₂, producción H₂, separación O₂/N₂,
cualquier ciclo de adsorción cíclica.

---

## Vector de estado

```
Sin shell_tube:  sv = [C(nc,N),  q(nc,N),  Hg(N),  Ts(N)]           tamaño = (2·nc + 2) · N
Con shell_tube:  sv = [C(nc,N),  q(nc,N),  Hg(N),  Ts(N),  Tw(N)]   tamaño = (2·nc + 3) · N

Variables primarias:
  C_i  [mol/m³_gas]  — concentraciones molares gas
  q_i  [mol/kg]      — carga adsorbida por especie
  Hg   [J/m³_bed]    — entalpía volumétrica del gas
  Ts   [K]           — temperatura del sólido (adsorbente + adsorbato)

Variables secundarias:
  Tg, P, y  — recuperadas de C, Hg
```

---

## Archivos específicos

| Archivo | Función principal | Descripción |
|---------|-------------------|-------------|
| `src/units/adsorber/state.py` | `pack_state_vector`, `unpack_state_vector` | Layout sv |
| `src/units/adsorber/state_extraction.py` | `build_adsorber_results` | Objeto col |
| `src/units/adsorber/velocity_history.py` | `compute_velocity_history` | v_face post-proceso |
| `src/units/adsorber/config/adsorbent.py` | `build_adsorbent_config` | D_p, epsi_p, tau, rho_s, Cp_s |
| `src/units/adsorber/config/boundary_c.py` | `build_boundary_c_config` | Pasos PSA (ads, purge…) |
| `src/boundary_conditions/adsorber_boundary.py` | `get_step_boundary` | Lógica PSA por paso |
| `src/solvers/rhs/rhs_adsorption.py` | `core_rhs` | RHS 10 pasos con LDF |
| `src/solvers/runner_adsorption.py` | `run_step` | Valida + integra un paso PSA |
| `src/postprocessing/adsorber_balances.py` | `molar_balance`, `energy_balance` | Balances |
| `src/utils/isotherm_models.py` | `langmuir`, `DSL`, `DSLF`, ... | Modelos de isoterma puros |
| `src/utils/isotherm_fitting.py` | `fit_single_T`, `fit_multi_T` | Ajuste a datos exp. |
| `src/utils/mixture_isotherm.py` | `iast`, `rast` | Equilibrio multicomponente |

---

## Física específica — LDF (Linear Driving Force)

```python
# En el RHS paso 9 (balance de carga):
q_eq = iso_fn(P_partial, Ts)              # (nc, N) [mol/kg] — equilibrio
dqdt = k_mtc * (q_eq - q)                # (nc, N) [mol/kg/s]

# Clip: evitar q < 0
dqdt = np.where(q < 0, np.maximum(dqdt, 0), dqdt)

# En paso 8 (especies gaseosas) — sink de gas:
source_gas[i] -= rho_s * (1 - epsi) / epsi * dqdt[i]   # [mol/m³_gas/s]

# En paso 10 (energía sólido) — calor de adsorción:
Q_ads_vol = np.sum(-dH * rho_s * dqdt, axis=0)          # (N,) [W/m³_bed]
# dTs += Q_ads_vol / Cs_vol
```

**No hay q_masstransfer en dHgdt para adsorción**: las moléculas no "aparecen"
en el gas con temperatura Ts; el equilibrio gas-sólido modifica C y q directamente.
El acoplamiento térmico va por q_gs_vol (HT convectivo superficial).

---

## Pasos PSA disponibles

| Step | Descripción | Inlet | Outlet |
|------|-------------|-------|--------|
| `"ads"` | Adsorción (flujo positivo) | C_feed, T_feed, Q_feed | P_out fija |
| `"purge"` | Purga (flujo inverso) | C_purge, T_purge, Q_purge | P_out fija |
| `"blowdown"` | Despresurización (solo outlet) | Cerrado | P_out fija |
| `"pressurization"` | Represurización (solo inlet) | C_feed, Q_pr | Cerrado |
| `"pr_feed"` | Repres. con alimentación | C_feed, Q_pr | Cerrado |
| `"wait"` | Espera (sin flujo) | Cerrado | Cerrado |

---

## Isotermas disponibles

```python
# Modelos puros (una T, una especie)
langmuir(P, q_sat, b)
DSL(P, q_s1, b1, q_s2, b2)        # Dual-Site Langmuir
DSLF(P, q_s1, b1, n1, q_s2, b2, n2)  # DSL con Freundlich

# Parámetros en función de T (Arrhenius)
b(T) = b0 * exp(-dH / (R * T))

# Multicomponente
iast(P_partial_list, T, isotherm_fns)  → q_eq (nc,)
rast(P_partial_list, T, isotherm_fns, activity_coeff_fn) → q_eq (nc,)
```

---

## Claves requeridas en params

```python
_REQUIRED_COMMON = (
    "n_comp", "N", "dz", "Ai", "Di", "Pi", "Po",
    "prop_gas", "MW", "gas_T_ref",
    "bc_config", "trans_config", "thermal_bc_config", "energy",
)

_REQUIRED_ADSORPTION = (
    "iso_fn",    # callable: iso_fn(P_partial_list, Ts) → q_eq(nc, N)
    "dH",        # ndarray(nc, N) [J/mol] — entalpía de adsorción
    "epsi",      # float — porosidad del lecho [-]
    "rho_s",     # float — densidad bulk del sólido [kg/m³_bed]
    "Cp_s",      # float — Cp del adsorbente [J/kg/K]
    "k_s",       # float — conductividad del adsorbente [W/m/K]
    "prop_lecho",# dict — D_p, a_surf, tau, epsi_p para transporte
)
```

---

## Atributos del objeto col (adsorber)

```python
col._t_results          # (n_t,)
col._z                  # (N,)
col._species            # list[str]
col._P_results          # (n_t, N) [bar]
col._Tg_results         # (n_t, N) [K]
col._Ts_results         # (n_t, N) [K]
col._Tw_results         # (n_t, N) o None
col._Hg_results         # (n_t, N) [J/m³_bed]
col._y_results          # (n_t, nc, N) [-]
col._C_results          # (n_t, nc, N) [mol/m³_gas]
col._q_results          # (n_t, nc, N) [mol/kg]
col._v_results          # (n_t, N) [m/s]
col._v_in_results       # (n_t,) [m/s]
col._v_out_results      # (n_t,) [m/s]
col._C_in_results       # (n_t, nc) o NaN si sin inlet
col._T_in_results       # (n_t,) o NaN
```
