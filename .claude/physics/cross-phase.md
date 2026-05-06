# Transferencia de masa entre fases

## Regla universal

**Toda masa que cruza una frontera de fase lleva consigo su entalpía.**

En el ODE, esto se traduce en que cuando una especie aparece (o desaparece) en una fase,
hay que añadir (o restar) su contribución energética en el balance de entalpía de esa fase.

---

## Caso 1 — Sólido → Gas (drying, pyrolysis, char reactions)

### Qué ocurre físicamente
La biomasa o el char reaccionan y liberan moléculas gaseosas a temperatura Ts.
Esas moléculas entran al espacio intersticial del lecho a temperatura Ts, no a Tg.

### Términos obligatorios en el RHS

```python
# ── Paso 8. Balance de especies gaseosas ──────────────────────────────────
# Fuentes de masa (mol/m³_BED/s → mol/m³_GAS/s):
src_dry_H2O     = drying_gas_source(r_dry, MW_H2O)      # (N,) mol/m³_bed/s
src_pyr_gas, _  = pyrolysis_sources(r_pyr, yields, MW, species)  # (nc,N) mol/m³_bed/s
src_char_gas    = char_gas_sources(r_ox, r_CO2, r_H2O, ...)       # (nc,N) mol/m³_bed/s

epsi_safe = max(float(epsi_r), 1e-10)
source_gas = np.zeros((nc, nn))
for j, sp in enumerate(species):
    if sp == "H2O":
        source_gas[j] += src_dry_H2O / epsi_safe
    source_gas[j] += src_pyr_gas[j] / epsi_safe
    source_gas[j] += src_char_gas[j] / epsi_safe

dCdt[i] = -(F[1:] - F[:-1]) / dz + source_gas[i]

# ── Paso 10. Energía del gas (Hg) — OBLIGATORIO ───────────────────────────
# Las moléculas aparecen a Ts (temperatura del sólido donante), no a Tg.
h_i_Ts = calc_species_enthalpy(Ts_arr, prop_gas, nc, gas_T_ref)  # (nc, N) J/mol
q_masstransfer = epsi_r * np.sum(source_gas * h_i_Ts, axis=0)    # (N,) J/m³_bed/s

dHgdt = (-epsi_r * div_h_conv
         - div_qg_diff
         - q_gs_vol
         + qwall_vol
         + q_masstransfer)          # ← término que a menudo se olvida
```

### Qué ocurre si se omite q_masstransfer

El estado `Hg = epsi_r · Σ Cᵢ · hᵢ(Tg)` crece al aumentar `Cᵢ` (nuevos moles),
pero el ODE de `Hg` no lo sabe. Al recuperar `Tg` de `Hg` mediante Newton,
el solver tiene que bajar `Tg` para que `epsi_r · Σ Cᵢ · hᵢ(Tg)` cuadre con el `Hg` bajo.
**Resultado: el gas se enfría artificialmente cuando el sólido produce gas.**

### Temperatura de referencia: Ts, no Tg

```python
# ✅ Correcto: las moléculas salen del sólido a Ts
h_i_Ts = calc_species_enthalpy(Ts_arr, prop_gas, nc, gas_T_ref)

# ❌ Incorrecto: asume que aparecen a la temperatura del gas
h_i_Tg = calc_species_enthalpy(Tg_arr, prop_gas, nc, gas_T_ref)
```

### Energía del sólido — no hay doble contabilidad

El sólido ya tiene sus propios sumideros de reacción:
```python
Q_dry  = -drying_enthalpy_sink(r_dry)       # calor latente de vaporización
Q_pyr  = -pyrolysis_enthalpy_sink(r_pyr, dH_pyr)  # entalpía de pirólisis
Q_char = char_reaction_heat(r_ox, r_CO2, r_H2O, ...)  # calor de combustión/gasif.
Q_rxn_vol = Q_dry + Q_pyr + Q_char          # W/m³_bed

dTsdt = (Q_rxn_vol + q_gs_vol) / Cs_vol
```
Estos términos capturan la energía de la **transformación química**.
El q_masstransfer en Hg captura el **calor sensible de las nuevas moléculas**.
Son físicamente distintos; no hay solapamiento.

---

## Caso 2 — Gas → Fase adsorbida (LDF)

### Qué ocurre físicamente
Las moléculas de gas se adsorben sobre el sólido. No "desaparecen" con temperatura,
sino que cambian de fase. El calor de adsorción va al sólido.

### Términos obligatorios en el RHS

```python
# ── Paso 9. Carga adsorbida ───────────────────────────────────────────────
q_eq = iso_fn(P_partial, Ts)               # (nc, N) [mol/kg]
dqdt = k_mtc * (q_eq - q)                 # (nc, N) [mol/kg/s]
dqdt = np.where(q < 0, np.maximum(dqdt, 0), dqdt)  # clip

# ── Paso 8. Especies gaseosas — sink ─────────────────────────────────────
# Las moléculas dejan el gas:
source_gas[i] -= rho_s * (1 - epsi) / epsi * dqdt[i]   # [mol/m³_gas/s]
# (equivalente: source_bed[i] = -rho_s * dqdt[i]  [mol/m³_bed/s])

# ── Paso 10. Energía del sólido — calor de adsorción ────────────────────
Q_ads_vol = np.sum(-dH * rho_s * dqdt, axis=0)   # (N,) [W/m³_bed]
# dH > 0 para adsorción exotérmica (convención habitual)
# Q_ads_vol > 0 cuando q < q_eq (adsorción) → calor al sólido
dTsdt += Q_ads_vol / Cs_vol

# ── Paso 10. Energía del gas — NO hay q_masstransfer ────────────────────
# Las moléculas no "reaparecen" en el gas con temperatura.
# El acoplamiento térmico gas-sólido ya va por q_gs_vol.
```

---

## Caso 3 — Comparación lado a lado

| Aspecto | Sólido → Gas | Gas → Adsorbido |
|---------|-------------|-----------------|
| Dirección de masa | Sólido libera → gas recibe | Gas entrega → sólido recibe |
| source_gas | + (fuente) | − (sumidero) |
| q_masstransfer en dHgdt | **Sí** — `epsi·Σ src·h_i(Ts)` | **No** |
| Calor en sólido | Q_rxn (latente + reacción) | Q_ads = −dH · rho_s · dqdt |
| Temperatura de referencia | Ts del sólido | No aplica |
| Ejemplos | Drying, pyrolysis, char rxns | PSA, TSA, VSA |

---

## Checklist al implementar una nueva transferencia de masa

```
□ ¿Qué fase pierde masa?       → Fuente negativa en esa fase
□ ¿Qué fase gana masa?         → Fuente positiva en esa fase
□ ¿Cuántos moles y de qué especie?  → source_gas[i] += ... / epsi
□ ¿A qué temperatura aparecen?  → Ts (sólido donante) para h_i
□ ¿Se añade q_masstransfer al receptor?  → Sí si gas recibe masa sólida
□ ¿El sólido tiene su propio balance de energía?  → Sí si Ts es variable del estado
□ ¿Hay doble contabilidad?      → No: Q_rxn en sólido ≠ q_masstransfer en gas
```
