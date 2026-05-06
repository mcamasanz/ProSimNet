# Heater — Calentador 1D (tubo vacío)

## Descripción física

Tubo vacío por el que circula gas. Sin lecho, sin reacciones, sin adsorción.
La única física añadida respecto a la convección pura es el intercambio de calor
con la pared (4 modos en thermal_bc) y opcionalmente la dinámica de la propia pared.

**Casos de uso:** precalentador de gas, intercambiador de calor tubo-carcasa,
horno eléctrico, sección de calentamiento de una columna PSA.

---

## Vector de estado

```
Sin shell_tube:  sv = [C(nc, N),  Hg(N)]               tamaño = (nc + 1) · N
Con shell_tube:  sv = [C(nc, N),  Hg(N),  Tw(N)]       tamaño = (nc + 2) · N

Variables primarias:
  C_i  [mol/m³_gas]  — concentraciones molares por especie
  Hg   [J/m³_bed]    — entalpía volumétrica del gas (epsi_r · Σ C_i · h_i(Tg))

Variables secundarias (recuperadas):
  Tg   [K]   — temperatura gas (Newton de Hg y C)
  P    [bar]  — presión (gas ideal)
  y    [-]    — fracciones molares

No hay:
  Ts  — el heater no tiene fase sólida reactiva
  q   — no hay adsorción
```

---

## Archivos específicos

| Archivo | Función principal | Descripción |
|---------|-------------------|-------------|
| `src/units/heater/state.py` | `pack_state_vector`, `unpack_state_vector` | Layout sv |
| `src/units/heater/state_extraction.py` | `build_heater_results` | Objeto col |
| `src/units/heater/config/gas_props.py` | `build_gas_prop_config` | Re-export de pure_gas |
| `src/units/heater/config/boundary_c.py` | `build_boundary_c_config` | Flujo continuo: Q_in, T_in, y_in, P_out |
| `src/units/heater/config/initial_c.py` | `build_initial_conditions` | sv0: T_gas, y_gas, P_gas |
| `src/units/heater/config/thermal_bc.py` | `build_thermal_bc_config` | 4 modos de pared |
| `src/units/heater/config/transport.py` | `build_transport_config` | h_wall, sin k_mtc |
| `src/units/heater/config/wall_c.py` | `build_wall_config` | Material, A_w, Di, Do |
| `src/boundary_conditions/heater_boundary.py` | `get_heater_boundary` | Flujo continuo |
| `src/solvers/rhs/rhs_heater.py` | `core_rhs` | RHS 10 pasos (sin reacciones) |
| `src/solvers/runner_heater.py` | `run_step` | Valida + integra |
| `src/postprocessing/heater_balances.py` | `molar_balance`, `energy_balance` | Balances |

---

## Diferencias respecto a otros equipos

| Aspecto | Heater | Adsorber | Gasifier |
|---------|--------|----------|---------|
| Hidráulica | `continuity_face_velocity` (tubo vacío) | `ergun_face_velocity` (lecho) | `ergun_face_velocity` (lecho) |
| Fase sólida | No | Sólido adsorbente (Ts, q) | Sólido reactivo (Ts, rho_s) |
| Reacciones | No | LDF adsorción | Drying, pyrolysis, char |
| q_masstransfer | No | No (LDF no transfiere al gas con h) | **Sí — obligatorio** |
| Dispersión axial | No (plug-flow) | Sí (D_disp) | Opcional |
| Ecuaciones por celda | nc + 1 (+1 si Tw) | 2·nc + 2 (+1 si Tw) | 13 (+1 si Tw) |

---

## Modos de bc_config

```python
bc_config = build_boundary_c_config(
    mode        = "continuous",   # único modo del heater
    Q_in        = 1e-3,           # [m³/s] caudal volumétrico entrada
    T_in        = 573.15,         # [K] temperatura entrada
    y_in        = {"N2": 1.0},    # fracciones molares entrada
    P_out       = 1.01325e5,      # [Pa] presión salida
)
```

---

## Atributos del objeto col (heater)

```python
col._t_results          # (n_t,) [s]
col._z                  # (N,)   [m]
col._species            # list[str]
col._P_results          # (n_t, N) [bar]
col._Tg_results         # (n_t, N) [K]
col._Tw_results         # (n_t, N) [K] o None si no shell_tube
col._Hg_results         # (n_t, N) [J/m³_bed]
col._y_results          # (n_t, nc, N) [-]
col._C_results          # (n_t, nc, N) [mol/m³_gas]
col._v_results          # (n_t, N) [m/s] velocidad media celda
col._v_in_results       # (n_t,) [m/s]
col._v_out_results      # (n_t,) [m/s]
col._C_in_results       # (n_t, nc) [mol/m³_gas]
col._T_in_results       # (n_t,) [K]
```

---

## Claves requeridas en params

```python
_REQUIRED_COMMON = (
    "n_comp", "N", "dz", "Ai", "Di", "Pi", "Po",
    "prop_gas", "MW", "gas_T_ref",
    "bc_config", "trans_config", "thermal_bc_config", "energy",
)
# El heater no tiene _REQUIRED específicas adicionales.
# wall_config es opcional (activa shell_tube).
```
