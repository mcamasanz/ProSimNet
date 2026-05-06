# Balances de energía — estructura y convenciones

## Ecuaciones del modelo

### Gas (variable de estado: Hg)

```
dHg/dt = − epsi · div_h_conv       ← transporte convectivo de entalpía absoluta
         − div_qg_diff              ← conducción axial del gas
         − q_gs_vol                 ← HT gas→sólido (positivo cuando Tg > Ts)
         + q_wall_vol               ← HT pared→gas  (positivo cuando T_pared > Tg)
         + q_masstransfer           ← entalpía de nuevas moléculas desde sólido
                                      (solo si hay transferencia sólido→gas)
```

Donde:
```python
Hg         = epsi_r * np.sum(C * h_i(Tg), axis=0)    # [J/m³_bed]
div_h_conv = (Fh[1:] - Fh[:-1]) / dz                 # [J/m³_bed/s]
q_gs_vol   = h_bed * a_p * (Tg - Ts)                  # [W/m³_bed] positivo: gas cede
q_masstransfer = epsi_r * np.sum(source_gas * h_i(Ts), axis=0)  # [W/m³_bed]
```

---

### ⚠ REGLA CRÍTICA — ODE del sólido con corrección de masa

La ecuación **incorrecta** que produce desbalances globales (~30%) en lechos reactivos:
```
Cs_vol · dTs/dt = Q_rxn + q_gs          ← INCORRECTO cuando el sólido se consume
```

La ecuación **correcta** para un sólido con masa variable:
```
d(Cs_vol · Ts)/dt = Q_rxn + q_gs

expandiendo la derivada del producto:
Cs_vol · dTs/dt = Q_rxn + q_gs − Ts · dCs_vol/dt
                                 ─────────────────
                                 TÉRMINO OBLIGATORIO
```

El término `−Ts · dCs_vol/dt` representa el **calor sensible que abandona el sólido
con la masa consumida**. Al convertirse biomasa/char/humedad en gas:
- La masa sale llevándose su energía sensible `Cp_j(Ts) · Ts`
- Sin este término, el balance global energético falla ~30% en simulaciones reactivas
- Con este término, el balance global cierra a < 1% (solo error de integración BDF)

**Implementación correcta en el RHS:**
```python
Cp0 = np.asarray(Cp_fns[0](Ts_arr), float)   # (N,) [J/kg/K]
Cp1 = np.asarray(Cp_fns[1](Ts_arr), float)
Cp2 = np.asarray(Cp_fns[2](Ts_arr), float)
Cs_vol = rho_bio*Cp0 + rho_char*Cp1 + rho_moi*Cp2
Cs_vol = np.maximum(Cs_vol, 1e-6)

# Corrección de masa: usar H_j(Ts) = ∫_{T_ref}^{Ts} Cp_j dT (integral exacta de Cp)
# NO usar Cp_j(Ts)·Ts: solo es exacto si Cp es constante. Con Cp(T) variable,
# el error en el balance global puede ser 20-30%.
H0 = np.asarray(h_fns[0](Ts_arr), float)   # (N,) [J/kg]
H1 = np.asarray(h_fns[1](Ts_arr), float)
H2 = np.asarray(h_fns[2](Ts_arr), float)
thermal_correction = -(H0*(src_s[0]+conv_s[0])
                      + H1*(src_s[1]+conv_s[1])
                      + H2*(src_s[2]+conv_s[2]))   # (N,) [J/m³_bed/s]

dTsdt = np.where(solid_present,
                 (Q_rxn_vol + q_gs_vol + thermal_correction) / Cs_vol,
                 0.0)
```

**Acumulador Q_rxn:** `dQ_rxn_acc_dt = Q_rxn_vol` (SIN thermal_correction — es interna al ODE).

**ΔHs en balances:** usar `Σ_j ρ_j · H_j(Ts)` (no `Cs_vol·Ts`). Los `h_fns` deben
construirse junto con los `Cp_fns` en `fuels_reader.py` como `∫_{T_clip_min}^T Cp_j dT`.

---

### Pared (solo si shell_tube, variable de estado: Tw)

```
rho_w · Cp_w · A_w · dTw/dt = Q_gw_cell − Q_ext_cell + Q_ax_cell
```

---

## Convención de signos

| Término | Signo positivo significa |
|---------|--------------------------|
| `q_gs_vol` en dHgdt | Gas **cede** calor al sólido (Tg > Ts) — el gas se enfría |
| `q_gs_vol` en dTsdt | Gas **calienta** el sólido (Tg > Ts) — el sólido se calienta |
| `q_wall_vol` en dHgdt | La pared **calienta** el gas |
| `Q_rxn_vol` en dTsdt | Las reacciones **calientan** el sólido (exotérmico) |
| `q_masstransfer` en dHgdt | Gas gana entalpía de las moléculas producidas a Ts |
| `thermal_correction` en dTsdt | Positivo cuando sólido se consume: Ts sube más rápido |

**Verificación:** `q_gs_vol` aparece con el mismo signo en dTsdt y con signo
OPUESTO en dHgdt. Si es igual en ambos, hay un bug de signo.

---

## ⚠ REGLA CRÍTICA — Clip defensivo en el RHS (protección Jacobiano BDF)

BDF perturba el estado para estimar el Jacobiano; esas perturbaciones producen
`C < 0` y `rho_solid < 0`, lo que causa:
- `Ctot < 0` → `P < 0` → `rho_g < 0` → `Re < 0` → `Re**0.6 = NaN` → fallo LU
- `rho_char < 0` → `dp = (rho/rho0)^(1/3) = NaN` → NaN en toda la cinética

**Regla: clip siempre al inicio del RHS, antes de cualquier cálculo físico:**
```python
C_mat     = np.maximum(state["C"],         0.0)   # (nc, N)
rho_solid = np.maximum(state["rho_solid"], 0.0)   # (3, N)

# Recomputar variables derivadas desde C_mat clipado:
Ctot_arr = np.sum(C_mat, axis=0)
y_mat    = C_mat / np.maximum(Ctot_arr, 1e-300)[None, :]
P_bar    = np.maximum(Ctot_arr * R_GAS * Tg_arr / 1e5, 1e-6)
P_Pa     = P_bar * 1e5
```

**También:** proteger Re y Sc en Ranz-Marshall de potencias fraccionarias negativas:
```python
Re = np.maximum(rho_g * v_int * dp / np.maximum(mu_g, 1e-15), 0.0)
Sc = np.maximum(mu_g / np.maximum(rho_g * D_im, 1e-20), 0.0)
```

---

## ⚠ REGLA CRÍTICA — Umbral de clip de densidades sólidas

El umbral del clip debe ser `_EPS_RHO = 1e-6` (no `1e-12`).

Con `1e-12`: a `t = 7200 s`, el solver necesita `Δt < eps × t ≈ 1.6e-12 s`
para localizar el cruce por cero → **fallo BDF** ("Required step size is less than
spacing between numbers").

Con `1e-6` (1 μg/m³): el clip activa antes del límite de precisión → BDF estable.

```python
_EPS_RHO = 1.0e-6   # kg/m³_bed — "agotado" físicamente

d_rho_s[j] = np.where(rho_s[j] < _EPS_RHO, np.maximum(d_rho_s[j], 0.0), d_rho_s[j])
```

---

## ⚠ REGLA CRÍTICA — Máscara de sólido agotado (solid_present)

Cuando el sólido se consume completamente:
- SCM: `dp → 0` → `a_p = 6(1−ε)/dp → ∞` (aunque dp se clipea, puede ser 1e-6 m)
- `Cs_vol → 0` (clipeado a 1e-6)
- → `dTs/dt = q_gs / Cs_vol → 1e10 K/s` → rigidez extrema → **fallo BDF**

Solución: desactivar intercambio y ODE del sólido cuando no hay masa sólida:
```python
solid_present = (rho_biomass + rho_char + rho_moisture) > _EPS_RHO   # (N,)

q_gs_vol = h_bed * a_p * (Tg - Ts)
q_gs_vol = np.where(solid_present, q_gs_vol, 0.0)   # sin superficie = sin HT

dTsdt = np.where(solid_present,
                 (Q_rxn + q_gs + thermal_correction) / Cs_vol,
                 0.0)
```

---

## Patrón de acumuladores ODE para cierres energéticos exactos

Para cierres ★ numéricos reales sin re-evaluar el RHS, añadir acumuladores al sv:

```python
# En el RHS (junto con q_masstransfer y Q_rxn_vol), SIEMPRE al final del sv:
dQ_mt_acc_dt  = q_masstransfer   # (N,) [J/m³_bed/s]
dQ_rxn_acc_dt = Q_rxn_vol        # (N,) [J/m³_bed/s]  — SIN thermal_correction
# ⚠ thermal_correction es interna al ODE (hace d(Cs·Ts)/dt = Q_rxn + q_gs exacto).
#   Incluirla en el acumulador produce: ΔHs − Q_gs − Q_rxn_acc = −∫thermal_correction ≠ 0

# Si energy=False:
dQ_mt_acc_dt  = np.zeros(nn, float)
dQ_rxn_acc_dt = np.zeros(nn, float)
```

En post-proceso, todos los cierres deben ser ≈ 0:
```python
Q_mt  = np.sum(gasifier._Q_mt_acc_results[-1]  * dz)  # [J/m²] exacto
Q_rxn = np.sum(gasifier._Q_rxn_acc_results[-1] * dz)  # [J/m²] exacto

closure_Hg     = dHg - Fh_neto + Q_gs - Q_wall - Q_mt           # ≈ 0 ★
closure_Hs     = dHs_proxy - Q_gs - Q_rxn                        # ≈ 0 ★
closure_global = (dHg + dHs_proxy) - (Fh_neto + Q_wall + Q_rxn + Q_mt)  # ≈ 0 ★
```

**Qué acumular:** solo términos que provienen del RHS y NO son recuperables
desde el estado almacenado (`source_gas`, `Q_rxn_vol`).
NO acumular: Q_gs, Q_wall (recuperables desde h_bed × ΔT y thermal_bc).

---

## Balances de verificación: qué cierra y qué no

| Balance | Residual | Tipo |
|---------|---------|------|
| Masa total | ≈ 0 siempre | ★ Numérico |
| Energía gas (con Q_mt acumulador) | ≈ 0 | ★ Numérico |
| Energía sólido (con ODE corregida) | ≈ 0 | ★ Numérico |
| Balance global (con ambas correcciones) | ≈ 0 | ★ Numérico |
| Especies gas `ΔC_i − flux_i` | = fuente_rxn_i ≠ 0 | Físico esperado |
| Masa sólida `Δm_s_j` | = S_rxn_j ≠ 0 | Físico esperado |
| Pared ΔHw − Q_gw | ≈ 0 (shell_tube) | ★ Numérico |
