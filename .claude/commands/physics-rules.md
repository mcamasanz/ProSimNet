# /physics-rules — Referencia de reglas físicas del simulador

Cuando se invoque este comando, mostrar este documento como referencia antes de
implementar o revisar cualquier balance físico. Si el usuario pregunta sobre un
tipo específico de fenómeno, responde PRIMERO con la regla de esta referencia
y LUEGO con la implementación.

---

## REGLA 1 — Transferencia de masa sólido → gas

**Aplicable a:** drying (H₂O sólido → H₂O vapor), pyrolysis (biomasa → CO, CO₂, H₂, CH₄, tar, H₂O, char), char reactions (char + O₂/CO₂/H₂O → CO, CO₂, H₂)

**Lo que se requiere en el código:**

```
1. Ecuación de especie gaseosa (dC_i/dt):
   source_gas[i] += tasa_mol_bed[i] / epsi_r   [mol/m³_gas/s]

2. Energía del gas (dHg/dt) — OBLIGATORIO, a menudo olvidado:
   h_i_Ts = calc_species_enthalpy(Ts, prop_gas, nc, gas_T_ref)  # (nc, N) J/mol
   q_masstransfer = epsi_r * np.sum(source_gas * h_i_Ts, axis=0) # (N,) J/m³_bed/s
   dHgdt += q_masstransfer

3. Energía del sólido (dTs/dt):
   Los calores de reacción (Q_dry, Q_pyr, Q_char) capturan la transformación química.
   El "calor sensible de salida de masa" es implícito en la variación de Cs_vol.
```

**Por qué h_i(Ts) y no h_i(Tg):**
Las moléculas se producen en la interfaz sólido-gas a temperatura Ts. Si usáramos
h_i(Tg), estaríamos afirmando que aparecen a la temperatura del gas, lo cual es
físicamente incorrecto para reacciones heterogéneas.

**Qué pasa si se omite q_masstransfer:**
- dC_i crece (más moles en gas) pero Hg no crece con ellos
- Al recuperar Tg de Hg = epsi·ΣC_i·h_i(Tg), el solver baja Tg artificialmente
- El gas se enfría cuando el sólido produce gas → error físico

---

## REGLA 2 — Transferencia de masa gas → fase adsorbida

**Aplicable a:** adsorción PSA/TSA/VSA (gas se adsorbe sobre sólido poroso)

```
1. Carga adsorbida (dq_i/dt):
   dqdt[i] = k_mtc[i] * (q_eq[i] - q[i])   [mol/kg/s]

2. Ecuación de especie gaseosa (dC_i/dt):
   dCdt[i] -= rho_s * (1-epsi)/epsi * dqdt[i]   [mol/m³_gas/s]
   (equivalente: source_gas_bed[i] = -rho_s * dqdt[i]  [mol/m³_bed/s])

3. Energía del sólido (dTs/dt):
   Q_ads_vol[i] = -dH[i] * rho_s * k_mtc[i] * (q_eq[i] - q[i])  [W/m³_bed]
   (positivo = calor al sólido cuando se adsorbe, negativo = calor del sólido cuando desorbe)
   dTsdt += ΣᵢQ_ads_vol[i] / Cs_vol

4. Energía del gas (dHg/dt):
   NO hay q_masstransfer directo porque las moléculas no "desaparecen" al gas con
   entalpía; el equilibrio gas-sólido ya está capturado en la evolución de C y q.
   El acoplamiento térmico va por q_gs_vol (HT convectivo superficial).
```

---

## REGLA 3 — Modelo shell-tube (pared dinámica)

**Condición de activación:** `params.get("wall_config") is not None`

**Efectos en el código:**

| Elemento | Sin shell_tube | Con shell_tube |
|----------|---------------|----------------|
| sv size | nc·N + ... + N (Ts) | ... + N (Tw) |
| q_wall en gas | `wall_heat_flux(Tg, h_wall, thermal_bc, ...)` | `h_wall·Pi/Ai·(Tw-Tg)` |
| ODE pared | NO existe | `dTw/dt = wall_ode_rhs(...)` |
| thermal_bc "fixed_twall" | Tg→T_wall prescrito | T_wall prescribe To (exterior), NO Tw |
| thermal_bc "adiabatic" | Tg no pierde calor | Tw evoluciona libre (adiabática exterior) |
| `_Tw_results` en col | `None` | `ndarray(n_t, N)` |

**Los 4 modos de thermal_bc son todos compatibles con shell_tube=True.**
- `adiabatic`: Q_ext = 0 en la ODE de pared
- `fixed_twall`: T_wall = temperatura de la cara EXTERIOR de la pared (To)
- `heatfluxwall`: flujo prescrito Qwall [W] hacia el exterior
- `ambient_htc`: resistencias en serie (h_int + R_cond + h_ext)

**Validación en runner:** NUNCA prohibir una combinación (thermal_bc_mode, shell_tube).
Cualquier combinación es físicamente válida; las restricciones que existían eran incorrectas.

---

## REGLA 4 — Estructura del balance de energía (nomenclatura y signos)

```
dHg/dt [J/m³_bed/s] = 
    - epsi · div_h_conv     ← transporte convectivo de entalpía absoluta (salida es positivo)
    - div_qg_diff           ← difusión térmica axial del gas
    - q_gs_vol              ← HT gas→sólido (+cuando gas caliente, sólido frío)
    + q_wall_vol            ← HT pared→gas (+cuando pared caliente, gas frío)
    + q_masstransfer        ← entalpía de nuevas moléculas desde sólido (+siempre, si fuente > 0)

dTs/dt [K/s] = 
    (Q_rxn_vol + q_gs_vol) / Cs_vol
    
    donde:
    Q_rxn_vol = Q_dry + Q_pyr + Q_char   [W/m³_bed]
                (signos: positivo = calor AL sólido)
    q_gs_vol  = h_bed · a_p · (Tg - Ts)  [W/m³_bed]
                (positivo = gas caliente → sólido frío: GAS CEDE CALOR AL SÓLIDO)

dTw/dt [K/s] = 
    (Q_gw_cell - Q_ext_cell) / (rho_w · cp_w · A_w · dz)
    
    donde:
    Q_gw_cell  = h_wall · Pi · dz · (Tg - Tw)   [W/celda] positivo: gas→pared
    Q_ext_cell = f(thermal_bc_mode)               [W/celda] positivo: pared→exterior
```

**Consistencia gas + sólido:**
El término `q_gs_vol` aparece con signo POSITIVO en la ecuación de Ts y con signo NEGATIVO
en la de Hg. Esto garantiza que la energía se conserva internamente: lo que el gas pierde
el sólido lo gana.

---

## REGLA 5 — Unidades internas y conversiones obligatorias

| Variable | Unidad interna | Nota |
|----------|----------------|------|
| Presión | Pa | Contornos e isotermas reciben bar → convertir antes de Ergun |
| Temperatura | K | Siempre |
| Concentración | mol/m³_gas | Dividir fuentes bed→gas por epsi |
| Carga adsorbida | mol/kg | - |
| Densidad sólida | kg/m³_bed | Incluye epsi implícitamente |
| Velocidad | m/s superficial | v_face = v_superficial, v_intersticial = v/epsi |
| Entalpía molar | J/mol | calc_species_enthalpy → (nc, N) |
| Entalpía volumétrica | J/m³_bed | Hg = epsi · Σ C_i · h_i(Tg) |
| Flujo entálpico | W/m²_section | F_h = v · H_cell (H_cell sin epsi) |
| Tasas heterogéneas | kg/m³_bed/s o mol/m³_bed/s | /epsi para pasar a mol/m³_gas |
| Tasas homogéneas | mol/m³_gas/s | Directo a source_gas |
| Calores volumétricos | W/m³_bed | Para dHg/dt y dTs/dt |

---

## REGLA 6 — Shapes de arrays (convención obligatoria)

```python
C          : (nc, N)    # especies primero, celdas segundo
q          : (nc, N)    # misma convención que C
rho_solid  : (n_s, N)   # componentes sólidos primero, celdas segundo
h_i        : (nc, N)    # calc_species_enthalpy devuelve (nc, N)
x          : (N, nc)    # fracción molar para wilke_mix: celdas primero (EXCEPCIÓN)
Tg, Ts, Tw : (N,)       # siempre 1D
v_face     : (N+1,)     # caras (N+1 puntos para N celdas)
F_conv     : (N+1,)     # flujo en caras
div_F      : (N,)       # divergencia en celdas = (F[1:] - F[:-1]) / dz
```

---

## REGLA 7 — Patrones de naming en el codebase

```python
# Tasas de reacción (prefijo r_):
r_dry    # [kg_moisture/m³_bed/s]
r_pyr    # [kg_biomass/m³_bed/s]
r_ox     # [kg_char/m³_bed/s]  — combustión char-O2
r_CO2    # [kg_char/m³_bed/s]  — Boudouard
r_H2O    # [kg_char/m³_bed/s]  — steam gasification

# Fuentes en fase gas (prefijo src_):
src_dry_H2O  # [mol/m³_bed/s]
src_pyr_gas  # (nc, N) [mol/m³_bed/s]
src_char_gas # (nc, N) [mol/m³_bed/s]
source_gas   # (nc, N) [mol/m³_GAS/s]  ← ya dividido por epsi

# Calores en sólido (prefijo Q_):
Q_dry    # [W/m³_bed] — calor de secado (sink en sólido)
Q_pyr    # [W/m³_bed] — calor de pirólisis (sink en sólido)
Q_char   # [W/m³_bed] — calor de reacciones char (source en sólido)
Q_rxn_vol # suma de todos los calores sólidos [W/m³_bed]

# Calores volumétricos de intercambio (prefijo q_):
q_gs_vol        # [W/m³_bed] gas↔sólido
q_wall_vol      # [W/m³_bed] pared→gas
q_masstransfer  # [W/m³_bed] entalpía de masa cross-phase sólido→gas

# Coeficientes de transporte:
h_bed_arr  # (N,) [W/m²/K] — gas-sólido (bed HTC)
h_wall_arr # (N,) [W/m²/K] — gas-pared (wall HTC)
D_disp_mat # (nc, N) [m²/s] — dispersión axial (puede ser None en plug-flow)
k_mtc_arr  # (nc, N) [1/s]  — coef. transferencia masa LDF
```

---

## REGLA 8 — Caché del RHS

```python
# Al inicio del RHS (o del runner antes de cada paso):
params.setdefault("_cache", {})

# Propiedades que se cachean:
cache["gas_props"]   = dict con rho, mu, k, h_i
cache["trans_props"] = dict con h_bed, h_wall, D_disp
cache["Tg_last"]     = Tg_arr.copy()  # warm-start Newton — NO limpiar entre pasos

# Propiedades que se reinician al inicio de cada run_step:
cache.pop("gas_props", None)
cache.pop("trans_props", None)
cache.pop("rho_solid_in_conveyor", None)

# Modos de actualización:
# "frozen"  → usar caché si existe, calcular una vez
# "always"  → recalcular en cada llamada
# "frozen" es útil para simulaciones rápidas exploratorias
```

---

## REGLA 9 — Modelo de viscosidad y conductividad (regla de Wilke)

```python
# x_mat tiene shape (N, nc) — EXCEPCIÓN al convenio (nc, N)
x_mat = y.T  # y es (nc, N) → transponer para Wilke

# compute_gas_mixture_properties retorna:
gas_props = {
    "rho_g"  : ndarray(N,),    # [kg/m³]
    "mu_g"   : ndarray(N,),    # [Pa·s]
    "k_g"    : ndarray(N,),    # [W/m/K]
    "h_i"    : ndarray(N, nc), # [J/mol]  ← ojo: shape (N, nc), no (nc, N)
    "Dim"    : ndarray(N, nc), # [m²/s]
}
# Pero calc_species_enthalpy retorna (nc, N) — distintos!
```

---

## REGLA 10 — Balances de verificación post-simulación

Implementar en `<equipo>_balances.py` con estas 3 funciones:

```python
def molar_balance(col, params):
    # Cierre: ΔC_total = ∫(v_in·C_in - v_out·C_out) dt + ΣΔq (si hay adsorción)
    # imbalance_rel = |ΔC - flux_net| / max(|ΔC|, |flux_net|)

def solid_conversion_summary(col, params):
    # Solo si hay sólido reactivo
    # X_biomass, X_moisture, X_char; rho_s inicial vs final

def energy_balance(col, params):
    # ΔHg + ΔHs ≈ flux_enthalpy_net + Q_wall + Q_rxn
    # residual ≈ Q_rxn (no rastreado; debe ser ~0 sin reacciones)
    # Retorna dict con delta_Hg, delta_Hs_approx, flux_enthalpy_net,
    #          Q_wall_total, h_wall_approx, residual, residual_rel, note
```
