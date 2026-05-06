# /new-equipment — Plantilla para modelar un nuevo equipo

Cuando se invoque este comando, guía al usuario paso a paso para añadir un nuevo
equipo al simulador siguiendo la arquitectura establecida.

## Paso 1 — Inventario de fases y variables de estado

Antes de escribir ningún archivo, responde estas preguntas con el usuario:

1. **¿Cuántas fases existen?**
   - Fase gas (siempre): nc especies → C_i [mol/m³_gas]
   - ¿Fase adsorbida?: q_i [mol/kg] por especie
   - ¿Fase sólida reactiva?: rho_s_j [kg/m³_bed] por componente (biomasa, char, etc.)
   - ¿Temperatura del sólido separada de la del gas?: Ts [K]
   - ¿Pared dinámica?: Tw [K] (solo si wall_config activo)

2. **¿Qué transferencia de masa ocurre entre fases?**
   - Gas ↔ adsorbido (LDF): k_mtc * (q_eq - q)
   - Sólido → gas (reacciones heterogéneas): drying, pyrolysis, char combustion/gasification
   - REGLA CRÍTICA: toda transferencia de masa entre fases requiere un término fuente
     en las ecuaciones de especie Y un término de entalpía en el balance de energía
     de la fase receptora (ver Sección FÍSICA CROSS-PHASE más abajo).

3. **¿Qué reacciones hay?**
   - Homogéneas (gas-gas): WGS, tar cracking, combustión homogénea
   - Heterogéneas (gas-sólido): drying, pyrolysis, char combustion, Boudouard, steam gasification
   - ¿Son endotérmicas o exotérmicas? → impacto en balance de energía del sólido

4. **¿Flujo de sólido?**
   - Batch/CSTR: sólido estático (vs = 0)
   - Updraft/Downdraft: sólido se mueve → término convectivo ∂(vs·ρs)/∂z en balance de masa sólida
   - Conveyor: sólido entra y sale → condición de contorno de sólido + balance global de masa

5. **¿Shell-tube activo?** → ver Sección SHELL-TUBE más abajo

---

## Paso 2 — Estructura de archivos a crear

Para un equipo llamado `<equipo>`, crear exactamente:

```
src/units/<equipo>/
├── __init__.py
├── state.py               ← pack_state_vector() + unpack_state_vector()
├── state_extraction.py    ← build_<equipo>_results() → SimpleNamespace col
└── config/
    ├── __init__.py
    ├── gas_props.py        ← build_gas_prop_config()       [reutilizar de otro equipo]
    ├── boundary_c.py       ← build_boundary_c_config()     [específico del equipo]
    ├── initial_c.py        ← build_initial_conditions()    [específico del equipo]
    ├── thermal_bc.py       ← build_thermal_bc_config()     [reutilizar de otro equipo]
    ├── transport.py        ← build_transport_config()      [reutilizar de otro equipo]
    ├── wall_c.py           ← build_wall_config()           [reutilizar de otro equipo]
    └── [solid_props.py]    ← build_solid_prop_config()     [solo si hay sólido reactivo]

src/boundary_conditions/
└── <equipo>_boundary.py   ← get_<equipo>_boundary()

src/physics/reactions/     (solo si hay reacciones nuevas)
├── drying.py
├── pyrolysis.py
└── char_conversion.py     (o equivalente para el combustible nuevo)

src/solvers/rhs/
└── rhs_<equipo>.py        ← core_rhs(t, sv, params) → dydt

src/solvers/
└── runner_<equipo>.py     ← run_step(...) → (t_arr, y_hist, col)

src/postprocessing/
└── <equipo>_balances.py   ← molar_balance() + energy_balance()
```

---

## Paso 3 — Layout del vector de estado

Define el layout ANTES de escribir pack/unpack. Documentarlo en el docstring de `state.py`.

**Regla de orden:** gas primero, luego sólido, luego energía, luego temperaturas, luego pared.

```python
# Ejemplo genérico (ajustar según fases del equipo):
# sv = [C(nc,N), [q(nc,N)], [rho_s(n_s,N)], Hg(N), [Ts(N)], [Tw(N)]]
# tamaño = (nc + [nc] + [n_s] + 1 + [1] + [1]) * N

# Ejemplo heater (nc especies, sin sólido reactivo):
# sv = [C(nc,N), Hg(N), [Tw(N)]]      tamaño = (nc+1[+1]) * N

# Ejemplo adsorbedor (nc especies, adsorción + sólido térmico):
# sv = [C(nc,N), q(nc,N), Hg(N), Ts(N), [Tw(N)]]  tamaño = (2nc+2[+1]) * N

# Ejemplo gasificador (9 especies, 3 sólidos reactivos):
# sv = [C(9,N), rho_s(3,N), Hg(N), Ts(N), [Tw(N)]]  tamaño = (14[+1]) * N
```

---

## Paso 4 — RHS: los 10 pasos obligatorios

El RHS de cualquier equipo sigue este orden sin excepciones:

```python
def core_rhs(t: float, sv: np.ndarray, params: dict) -> np.ndarray:
    """
    RHS del ODE para <equipo>.

    Estado: sv = [C(nc,N), ..., Hg(N), Ts(N)[, Tw(N)]]    # documentar layout exacto
    Retorna: dydt mismo shape que sv
    """
    # ── 1. Lectura de params ───────────────────────────────────────────────────
    nc, nn, dz, epsi, Ai, Di, Pi, Po = ...
    mode = params["bc_config"]["mode"]
    energy = bool(params.get("energy", True))

    # ── 2. Desempaquetado del vector de estado ────────────────────────────────
    # Usar unpack_state_vector(sv, params) de src/units/<equipo>/state.py
    # Recuperar Tg de Hg por Newton: recover_Tg_from_Hg(C, Hg, prop_gas, ...)

    # ── 3. Condiciones de contorno ────────────────────────────────────────────
    # Llamar a get_<equipo>_boundary(t, P_cell, ..., bc_config) → dict bc
    # Extraer: v_in, v_out, C_in, T_in, [vs_face, rho_s_inlet]

    # ── 4. Propiedades de mezcla del gas ──────────────────────────────────────
    # compute_gas_mixture_properties(P_Pa, Tg, x, prop_gas) → rho_g, mu_g, k_g

    # ── 5. Velocidades y coef. de transporte ──────────────────────────────────
    # ergun_face_velocity(...) → v_face para lechos empaquetados
    # compute_transfer_coefficients(...) → h_bed, h_wall, D_disp

    # ── 6. Cinética / Tasas de reacción ───────────────────────────────────────
    # Calcular r_i(C, Ts, ...) → [mol/m³_bed/s] o [kg/m³_bed/s]
    # REGLA: todas las tasas en m³_BED, dividir por epsi para pasar a m³_GAS en paso 8

    # ── 7. [Conveyor/móvil] Cálculo de rho_solid_in si procede ──────────────
    # Solo si el sólido tiene flujo convectivo y las condiciones de contorno
    # del sólido se calculan desde balance global (modo conveyor)

    # ── 8. Balance de especies gaseosas ───────────────────────────────────────
    # source_gas[i] += r_i / epsi_safe    ← unidades: mol/m³_GAS/s
    # dCdt[i] = -(F_conv[1:] - F_conv[:-1]) / dz + source_gas[i]

    # ── 9. [Si hay sólido] Balance de densidades / carga ──────────────────────
    # dq_i/dt  = k_mtc * (q_eq - q)              ← adsorción (LDF)
    # drho_s/dt = -(F_conv_s[1:]-F_conv_s[:-1])/dz + src_solid  ← sólido reactivo

    # ── 10. Balances de energía ────────────────────────────────────────────────
    # SIEMPRE incluir (bajo flag `energy`):
    #   Gas:    dHgdt = -epsi*div_h_conv - div_diff - q_gs + q_wall + q_masstransfer
    #   Sólido: dTsdt = (Q_rxn + q_gs) / Cs_vol
    #   Pared:  dTwdt = wall_ode_rhs(...)   ← solo si shell_tube
    # REGLA: q_masstransfer = epsi * Σᵢ source_gas[i] * h_i(Ts) SIEMPRE que
    #        exista transferencia de masa sólido→gas (ver FÍSICA CROSS-PHASE)

    # ── 11. [Shell-tube] ODE de pared ─────────────────────────────────────────
    # Solo si params.get("wall_config") is not None
    # dTwdt = wall_ode_rhs(Tg, Tw, h_wall, wall_config, thermal_bc_config, ...)

    # ── 12. Empaquetado del RHS ────────────────────────────────────────────────
    # dydt = np.concatenate([dCdt.ravel(), [dqdt.ravel(),] drho_s.ravel(),] dHgdt, dTsdt [,dTwdt]])
    return dydt
```

---

## Paso 5 — Balance de energía: checklist de completitud

Antes de considerar el balance de energía correcto, verificar:

- [ ] **Gas (Hg):** tiene término convectivo `epsi * div_h_conv`
- [ ] **Gas (Hg):** tiene término de difusión térmica `div_qg_diff`
- [ ] **Gas (Hg):** tiene intercambio gas-sólido `q_gs_vol` (signo: positivo cuando gas calienta sólido)
- [ ] **Gas (Hg):** tiene flujo de pared `q_wall_vol`
- [ ] **Gas (Hg):** tiene `q_masstransfer` si hay transferencia de masa sólido→gas
- [ ] **Sólido (Ts):** tiene calor de reacción `Q_rxn_vol` (suma de todas las reacciones)
- [ ] **Sólido (Ts):** tiene intercambio gas-sólido `q_gs_vol` (signo OPUESTO al gas)
- [ ] **Pared (Tw):** si `shell_tube=True`, tiene su ODE propia (`wall_ode_rhs`)
- [ ] **Pared (Tw):** si `shell_tube=False`, `q_wall` va directamente al gas con `wall_heat_flux`

---

## SECCIÓN: FÍSICA CROSS-PHASE (transferencia de masa entre fases)

### Regla universal: toda masa que cruza una frontera de fase lleva su entalpía

**Caso 1 — Sólido → Gas** (drying, pyrolysis, char reactions):

```python
# En las ecuaciones de especie (paso 8):
source_gas[i] += masa_tasa_i / MW_i / epsi_r   # [mol/m³_gas/s]

# En la energía del gas (paso 10) — OBLIGATORIO:
h_i_Ts = calc_species_enthalpy(Ts_arr, prop_gas, nc, gas_T_ref)  # (nc, N)
q_masstransfer = epsi_r * np.sum(source_gas * h_i_Ts, axis=0)    # (N,) [J/m³_bed/s]
dHgdt += q_masstransfer

# En la energía del sólido (paso 10):
# Los calores de reacción (Q_dry, Q_pyr, Q_char) ya capturan la energía de transformación.
# El "calor sensible de salida de masa" está implícito en la variación de Cs_vol.
```

**Caso 2 — Gas → Adsorbido** (adsorción LDF):

```python
# En la carga adsorbida (paso 9):
dqdt[i] = k_mtc[i] * (q_eq_i - q[i])   # [mol/kg/s]

# En las especies gaseosas (paso 8) — OBLIGATORIO:
dCdt[i] -= rho_s * (1 - epsi) / epsi * dqdt[i]   # [mol/m³_gas/s]
# (o equivalente en mol/m³_bed: -rho_s * dqdt[i])

# En la energía del gas (paso 10):
# El calor de adsorción va al SÓLIDO, no al gas:
# q_ads_vol = -dH[i] * rho_s * k_mtc[i] * (q_eq_i - q[i])  [W/m³_bed]
# dTsdt += q_ads_vol / Cs_vol   (positivo = calor al sólido cuando adsorbe)
# El gas NO recibe el calor directamente; el q_gs_vol ya lo equilibra más tarde.
```

**Regla mnemotécnica:** Las moléculas aparecen a Ts (temperatura del donante), no a Tg.
Usar `h_i(Ts)`, no `h_i(Tg)`, para el término de entalpía del cross-transfer.

---

## SECCIÓN: SHELL-TUBE — Implicaciones de activar wall_config

Cuando `params.get("wall_config") is not None` → `shell_tube = True`:

1. **Vector de estado crece en N**: añadir `Tw(N)` al final del sv
2. **pack/unpack**: actualizar en `state.py` para gestionar el caso shell_tube
3. **RHS paso 11**: añadir ODE `dTwdt = wall_ode_rhs(...)` y concatenar al dydt
4. **RHS paso 10**: cambiar `q_wall_vol` en gas:
   - Con shell_tube: `qwall_vol = h_wall * Pi/Ai * (Tw - Tg)` (usa Tw dinámico)
   - Sin shell_tube: `qwall_vol, _, _ = wall_heat_flux(Tg, h_wall, thermal_bc_config, ...)`
5. **thermal_bc_config**: todos los 4 modos son compatibles con shell_tube=True:
   - `adiabatic`: Q_ext = 0 (Tw evoluciona libre)
   - `fixed_twall`: T_wall prescribe la temperatura EXTERIOR (To), NO la interior Tw
   - `heatfluxwall`: flujo prescrito hacia el exterior de la pared
   - `ambient_htc`: resistencias en serie (convección interna + conducción + ext)
6. **state_extraction.py**: incluir `_Tw_results` en el objeto col (None si no shell_tube)
7. **energy_balance()**: incluir `Q_wall` con el caso shell_tube (ver `gasifier_balances.py`)
8. **runner**: `_validate_params` NO debe prohibir ningún thermal_bc_mode con wall_config

---

## SECCIÓN: CHECKLIST RUNNER

El runner de cualquier equipo debe:
- [ ] Validar todas las claves de `_REQUIRED_COMMON` + `_REQUIRED_<EQUIPO>`
- [ ] Inyectar valores derivados si no están precomputados (e.g., `dH_pyr`)
- [ ] Reiniciar caché al inicio de cada paso (salvo "Tg_last" que es warm-start)
- [ ] Construir `t_arr` con `n_sec` puntos por segundo + el punto final exacto
- [ ] Usar `solve_ivp(BDF)` por defecto, `odeint(LSODA)` como alternativa
- [ ] Llamar `build_<equipo>_results()` para construir el objeto `col`
- [ ] Decorar con `@profiled`
