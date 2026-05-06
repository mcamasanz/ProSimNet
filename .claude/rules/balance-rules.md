# Reglas de balance — estándar obligatorio para todos los equipos

## Principio

Todo equipo implementado en ProSimNet **DEBE** tener una función `check_balances`
en `src/postprocessing/<equipo>_balances.py` que:

1. Muestre la ecuación ODE de referencia de cada subsistema.
2. Liste cada término con valor numérico, unidades y origen.
3. Calcule el residual e interprete su significado físico.
4. Indique si el residual debe ser ≈ 0 (★ numérico) o ≠ 0 (físico esperado).

Sin esta función, el equipo **no está validado**.

---

## Balances obligatorios por tipo de equipo

| Balance | Heater | Adsorber | Gasifier | Reactor (futuro) |
|---------|--------|----------|---------|-----------------|
| Masa total gas+sólido | — | — | ✓ | ✓ |
| Especies gas (por especie) | ✓ | ✓ | ✓ | ✓ |
| Masa adsorbida (por especie) | — | ✓ | — | — |
| Masa sólida (por componente) | — | — | ✓ | ✓ |
| Energía gas (Hg) ★ | ✓ | ✓ | ✓ | ✓ |
| Energía sólido (Ts) ★ | — | ✓ | ✓ | ✓ |
| Energía pared (Tw) ★ | ✓ (si ST) | ✓ (si ST) | ✓ (si ST) | ✓ (si ST) |

---

## Convención de integración

Todos los términos se integran sobre el **volumen del lecho** y la **duración**:

```
∫₀ᴸ ∫₀ᵀ término_volumétrico [W/m³_bed] dz dt / L  →  [J/m³_bed]
∫₀ᴸ ∫₀ᵀ término_volumétrico [kg/m³_bed/s] dz dt / L →  [kg/m³_bed]
∫₀ᴸ ∫₀ᵀ término_volumétrico [mol/m³_gas/s] ε dz dt / L → [mol/m³_bed]
```

La integral espacial sobre N celdas: `Σ_k valor_k · dz`. División por L = dz·N para
obtener el valor normalizado por unidad de volumen de lecho. La integral temporal
se calcula con la regla trapezoidal sobre los puntos guardados.

Unidades de presentación: **[kg/m³_bed]**, **[mol/m³_bed]**, **[J/m³_bed]**.
Independientes del diámetro del reactor. Comparables entre reactores de distinto tamaño.

---

## Convención de signos

| Cantidad | Signo positivo significa |
|----------|--------------------------|
| Acumulación `Δ(·)` | Ganancia neta en el volumen de control |
| Flujo convectivo neto | Entra más de lo que sale |
| Q_gs (gas↔sólido) | Gas **pierde** calor (cede al sólido) |
| Q_wall (pared→gas) | Gas **gana** calor de la pared |
| Q_gw (gas→pared) | Pared **gana** calor del gas |
| fuente_rxn_i | Especie i **producida** por reacciones |
| S_rxn_j (sólido) | Componente j **producido** (>0) o consumido (<0) |
| Q_mt | Gas **gana** entalpía de las moléculas producidas a Ts |
| Q_rxn | Calor generado (+) o consumido (−) por reacciones en el sólido |

> Balance de gas: `ΔHg = Fh_conv_neto − Q_gs + Q_wall + Q_mt`
>
> Balance sólido: `ΔHs = Q_gs + Q_rxn`
>
> Balance global: `ΔHg + ΔHs = Fh_neto + Q_wall + Q_rxn_exact + Q_mt_exact`

---

## Clasificación de residuales

Cada residual **debe clasificarse explícitamente** en una de estas dos categorías:

### ★ Residual numérico (debe ser ≈ 0)

Indica error de integración o inconsistencia del modelo. Umbral: **1%** relativo.

| Balance | Residual ★ | Condición |
|---------|-----------|-----------|
| Masa total | `Δm_gas + Δm_solid − flux_masa ≈ 0` | siempre |
| Energía gas | `ΔHg − Fh_neto + Q_gs − Q_wall − Q_mt_exact ≈ 0` | requiere acumulador Q_mt |
| Energía sólido | `ΔHs − Q_gs − Q_rxn_exact ≈ 0` | requiere H_j integral + acumulador Q_rxn |
| Energía pared | `ΔHw − Q_gw − Q_ext − Q_ax ≈ 0` | shell_tube |
| Global | `(ΔHg + ΔHs) − (Fh + Q_wall + Q_rxn + Q_mt) ≈ 0` | ambos acumuladores |

### Residual físico (se espera ≠ 0, es información útil)

| Balance | Residual físico | Qué representa |
|---------|----------------|----------------|
| Especies gas | `fuente_rxn_i` | Moles producidos/consumidos por reacciones |
| Masa sólida | `S_rxn_j` | Masa de sólido transformada por reacciones |

---

## ⚠ REGLA CRÍTICA — Entalpía sólida con Cp(T) variable

### El error del proxy Cp(T)·T

Si `Cp_j(T)` es temperatura-dependiente (polinomio en T), el proxy
`Hs_proxy = Cs_vol·Ts = Σ_j ρ_j·Cp_j(Ts)·Ts` **NO es la entalpía verdadera**:

```
H_j_true(Ts) = ∫_{T_ref}^{Ts} Cp_j(T) dT        ← integral exacta
Cp_j(Ts)·Ts                                        ← proxy, solo exacto si Cp es constante
```

Para `Cp_j(T) = a + b·T`:
- True:  `∫ Cp dT = a·Ts + b·Ts²/2`
- Proxy: `Cp(Ts)·Ts = a·Ts + b·Ts²`
- Error: `b·Ts²/2`  → a T=1200K con b≈2 J/kg/K² → error ~1.4 MJ/kg

**Consecuencia en el balance:** el residual del balance sólido es
`ΔHs_proxy − Q_gs − Q_rxn ≈ ∫(Ts·dCp/dTs·dTs/dt)·dt`, que puede ser
**20-30% de ΔHs** para materiales con Cp polinomial.

### La corrección térmica correcta

La identidad `d(Σ_j ρ_j·H_j(Ts))/dt = Q_rxn + q_gs` requiere:

```
thermal_mass_correction = −Σ_j H_j(Ts) · src_s[j]     ← CORRECTO (integral de Cp)
thermal_mass_correction = −Ts · Σ_j Cp_j(Ts) · src_s[j] ← INCORRECTO para Cp(T) variable
```

**La versión con `Cp_j(Ts)·Ts` solo es exacta si `Cp_j = constante`.**

### Implementación obligatoria

**Paso 1** — En el lector de propiedades del combustible (p.ej. `fuels_reader.py`),
construir `h_fn` junto con `Cp_fn` desde los mismos coeficientes polinomiales:

```python
def _make_h_fn(coeffs, T_clip_min=273.0, T_clip_max=1700.0):
    """H(T) = ∫_{T_clip_min}^T Cp(T') dT'  (H(T_clip_min) = 0)"""
    arr = np.asarray(coeffs, float)

    def _poly_int(T_arr):
        res = np.zeros_like(np.asarray(T_arr, float))
        for i, a in enumerate(arr):
            res += a / (i + 1) * T_arr ** (i + 1)
        return res

    Cp_min = float(sum(arr[i] * T_clip_min**i for i in range(len(arr))))
    Cp_max = float(sum(arr[i] * T_clip_max**i for i in range(len(arr))))
    P_min  = float(_poly_int(np.array([T_clip_min]))[0])
    P_max  = float(_poly_int(np.array([T_clip_max]))[0])
    H_max  = P_max - P_min

    def h_fn(T):
        T_arr  = np.atleast_1d(np.asarray(T, float))
        result = np.empty_like(T_arr)
        in_rng = (T_arr >= T_clip_min) & (T_arr <= T_clip_max)
        result[in_rng] = _poly_int(T_arr[in_rng]) - P_min
        result[T_arr < T_clip_min] = Cp_min * (T_arr[T_arr < T_clip_min] - T_clip_min)
        result[T_arr > T_clip_max] = H_max + Cp_max * (T_arr[T_arr > T_clip_max] - T_clip_max)
        return result
    return h_fn
```

**Paso 2** — En el config del equipo (p.ej. `solid_props.py`), exponer `h_fns`:
```python
return {"Cp_fns": Cp_fns, "h_fns": h_fns, ...}
```

**Paso 3** — En el RHS, usar `H_j(Ts)` en la corrección térmica:
```python
H0 = np.asarray(h_fns[0](Ts_arr), float)   # (N,) [J/kg]
H1 = np.asarray(h_fns[1](Ts_arr), float)
H2 = np.asarray(h_fns[2](Ts_arr), float)
thermal_mass_correction = -(H0*(src_s[0]+conv_s[0])
                           + H1*(src_s[1]+conv_s[1])
                           + H2*(src_s[2]+conv_s[2]))  # (N,) [J/m³_bed/s]
```

**Paso 4** — En `check_balances`, usar `H_j(Ts)` para ΔHs:
```python
def _Hs_true(rs, Ts_):
    res = np.zeros(len(Ts_), dtype=float)
    for j in range(3):
        res += rs[j] * np.asarray(h_fns[j](Ts_), float)
    return res  # (N,) [J/m³_bed]

dHs = (float(np.sum(_Hs_true(rho_s[-1], Ts[-1]) * dz))
       - float(np.sum(_Hs_true(rho_s[0],  Ts[0])  * dz)))
```

---

## Patrón de acumuladores ODE para cierres energéticos exactos

Para obtener cierres energéticos **★ numéricos reales** sin re-evaluar el RHS,
los términos no recuperables desde los resultados almacenados se añaden al
**vector de estado como ODEs triviales** (acumuladores):

```python
# ── En el RHS (al final de la sección de energía) ─────────────────
dQ_mt_acc_dt  = q_masstransfer   # (N,) [J/m³_bed/s]  ∫q_mt dt
dQ_rxn_acc_dt = Q_rxn_vol        # (N,) [J/m³_bed/s]  ∫Q_rxn_vol dt — SIN thermal_correction

# Si energy=False:
dQ_mt_acc_dt  = np.zeros(nn)
dQ_rxn_acc_dt = np.zeros(nn)

# ── En pack_state_vector (siempre al final, después de Tw) ─────────
# sv = [C, rho_solid, Hg, Ts, Tw?, Q_mt_acc, Q_rxn_acc]

# ── En check_balances ──────────────────────────────────────────────
Q_mt_exact  = float(np.sum(gasifier._Q_mt_acc_results[-1]  * dz))
Q_rxn_exact = float(np.sum(gasifier._Q_rxn_acc_results[-1] * dz))
closure_gas = dHg - Fh_conv_net + Q_gs - Q_wall - Q_mt_exact  # ≈ 0 ★
closure_sol = dHs - Q_gs - Q_rxn_exact                         # ≈ 0 ★ (con H_j integral)
```

### ⚠ Q_rxn_acc NO debe incluir la thermal_correction

```python
# INCORRECTO — rompe el cierre del sólido:
dQ_rxn_acc_dt = Q_rxn_vol + thermal_mass_correction

# CORRECTO — solo el calor de reacciones:
dQ_rxn_acc_dt = Q_rxn_vol
```

**Por qué:** `thermal_mass_correction` es un término **interno al ODE** que hace
`d(Σ ρⱼ·Hⱼ)/dt = Q_rxn + q_gs` exacto. No es un flujo energético externo.
Si se incluye en el acumulador, el cierre del sólido produce:
`ΔHs − Q_gs − Q_rxn_acc = −∫thermal_correction dt ≠ 0`.

### Reglas del patrón

1. Los acumuladores siempre van **al final** del sv, después de Tw.
2. Se inicializan a **cero** en cada `run_step` (la integral acumula desde t=0).
3. `unpack_state_vector` no los extrae (no se usan en la física del RHS).
4. Solo añadir acumuladores para términos que no son recuperables post-hoc.
   No añadir para Q_gs, Q_wall, Fh_conv (recuperables desde el estado).

---

## Términos exactos vs. aproximados (gasificador)

| Término | Exacto desde resultados | Razón si aproximado |
|---------|------------------------|---------------------|
| ΔHg | Sí | — |
| **ΔHs** | **Sí** (con h_fns ∫Cp dT) | — |
| ΔTw | Sí | — |
| Fh_conv_neto | Sí | — |
| Flujo_másico_neto | Sí | — |
| Q_gs (constant transport) | Sí (con a_p dinámico del SCM) | — |
| Q_gs (correlation transport) | No — aprox. | h_bed no almacenado; usar h_bed medio |
| Q_wall (adiabático, heatfluxwall) | Sí | — |
| Q_cond_gas (N=1) | Sí = 0 | No hay gradiente axial en 0D |
| **Q_mt** | **Sí** (acumulador ODE) | — |
| **Q_rxn_sólido** | **Sí** (acumulador ODE) | — |

**Nota:** ΔHs deja de ser "aproximado" en cuanto se usa `H_j(Ts)` en lugar del
proxy `Cp_j(Ts)·Ts`. El error del proxy puede ser 20-30% con Cp polinomial.

---

## Estructura mínima de `check_balances`

```python
def check_balances(gasifier, params: dict, verbose: bool = True) -> dict:
    """
    Balance completo término a término.

    Secciones:
        1. [si hay sólido]  Masa total gas+sólido (★)
        2.                  Especies gas (residual físico = fuente_rxn_i)
        3. [si hay sólido]  Masa sólida por componente (residual físico = S_rxn_j)
        4.                  Energía gas — Hg (★ con Q_mt exacto del acumulador)
        5. [si hay sólido]  Energía sólido — Ts (★ con H_j integral + Q_rxn acumulador)
        6. [si shell_tube]  Energía pared — Tw (★)
        7.                  Balance global energía (★ con Q_mt_exact + Q_rxn_exact)

    Todos los valores en [J/m³_bed] (normalizado por L = dz·N).

    Retorna dict con claves:
        mass_total, species_gas, solid, energy_gas, energy_solid, energy_wall, report
    """
```

### Formato de salida obligatorio

```
─── MASA [kg/m³]  ·  ★ = cierre numérico ────────────────────────────
  m_gas              Δm      0.0      Δm      —  (→ reacciones)
  m_sólido           Δm      0.0      Δm      —  (reacciones →)
★ m_total            Δm_tot  flux     ≈0   0.00%

─── ENERGÍA GAS [J/m³] ──────────────────────────────────────────────
  ΔHg                        = ±X.XXe+XX  [acumulación gas]
  Fh_conv_neto               = ±X.XXe+XX  [exacto]
  Q_gs (gas→sólido)          = ±X.XXe+XX  [h_bed × a_p × ΔT]
  Q_wall (pared→gas)         = ±X.XXe+XX  [thermal_bc]
  Q_mt (sól→gas, exacto)     = ±X.XXe+XX  [acum. ODE]
  ─────────────────────────────────────────────────────
★ Cierre_Hg                  = ≈0         rel=0.00%  ✓ OK

─── ENERGÍA SÓLIDO [J/m³] ───────────────────────────────────────────
  ΔHs [∫Cp dT, exacto]       = ±X.XXe+XX  [Σ ρⱼ·H_j(Ts), integral de Cp]
  Q_gs_al_sólido             = ±X.XXe+XX  [h_bed × a_p × ΔT]
  Q_rxn (exacto)             = ±X.XXe+XX  [acum. ODE: ∫Q_rxn_vol dt]
  ─────────────────────────────────────────────────────
★ Cierre_Hs (ΔHs−Q_gs−Q_rxn)= ≈0         rel=0.00%  ✓ OK
```

---

## Lista de verificación para que todos los balances cierren

Al implementar un equipo reactivo con fase sólida, verificar en orden:

### Balance de masa total (★)
- [ ] `Δm_gas = Σ_i ΔC_i · MW_i · epsi_r · dz` — usar MW en kg/mol
- [ ] `Δm_solid = Σ_j Δrho_j · dz`
- [ ] `flux_masa` con `epsi_r` en flujos de entrada/salida

### Balance de energía del gas (★)
- [ ] `Hg = epsi_r · Σ C_i · h_i(Tg)` como variable de estado (no Tg)
- [ ] Acumulador `Q_mt_acc` en el sv (al final, antes de Q_rxn_acc)
- [ ] `Q_mt = epsi_r · Σ source_gas_i · h_i(Ts)` — evaluar en **Ts**, no Tg

### Balance de energía del sólido (★) — los tres requisitos son necesarios
- [ ] **Corrección térmica con integral**: `thermal_correction = −Σ_j H_j(Ts) · src_j`
      donde `H_j(Ts) = ∫_{T_ref}^{Ts} Cp_j dT` (NO `Cp_j(Ts)·Ts`)
- [ ] **Q_rxn_acc = ∫Q_rxn_vol dt** (SIN thermal_correction en el acumulador)
- [ ] **ΔHs en check_balances = Σ_j ρ_j·H_j(Ts)** (NO `Cs_vol·Ts`)
      — Si falta cualquiera de los tres, el balance del sólido falla.

### Balance global (★)
- [ ] Cierra automáticamente si los balances de gas y sólido cierran por separado
- [ ] `(ΔHg + ΔHs) − (Fh + Q_wall + Q_rxn_exact + Q_mt_exact) ≈ 0`

---

## Qué NO debe hacer `check_balances`

- No debe re-ejecutar el RHS (solo opera sobre los resultados almacenados).
- No debe hacer plots (eso va en `postprocessing/<equipo>_plots.py`).
- No debe lanzar excepciones si un balance no cierra — solo reportar.
- No debe mezclar equipos: cada `check_balances` es específico de su equipo.

---

## Para añadir un nuevo equipo

1. Identificar todas las ODEs del RHS y sus términos con signos.
2. Para sólidos con Cp(T): construir `h_fns` (∫Cp dT) junto con `Cp_fns`.
3. Decidir qué términos NO son recuperables post-hoc → añadir como acumuladores al sv.
4. Definir qué balances son aplicables (ver tabla de obligatorios).
5. Clasificar cada residual: ★ numérico o físico.
6. Implementar `check_balances` en `src/postprocessing/<equipo>_balances.py`.
7. Llamarla en el primer notebook de validación del equipo.

El gasificador (`gasifier_balances.py::check_balances`) es la implementación
de referencia para equipos reactivos con fase sólida y acumuladores energéticos.
El heater (`heater_balances.py`) es la referencia para equipos de flujo puro.
