# Modelos de reacción — patrones y física

## Clasificación de reacciones

| Tipo | Descripción | Implementación | Fase productora |
|------|-------------|----------------|-----------------|
| **Heterogéneas** | Gas + sólido → productos gas | `src/physics/reactions/` | Sólido → gas |
| **Homogéneas** | Gas → gas | `src/physics/reactions/` (pendiente) | Gas → gas |
| **LDF (adsorción)** | Gas → fase adsorbida | En RHS adsorbedor | Gas → adsorbido |

---

## Reacciones heterogéneas — patrón común

Toda reacción heterogénea (sólido-gas) sigue este patrón en el RHS:

```python
# 1. Calcular tasa cinética [kg/m³_bed/s] o [mol/m³_bed/s]
r = rate_function(C_mat, Ts_arr, rho_solid, ...)   # (N,)

# 2. Fuente en especie gaseosa [mol/m³_GAS/s]:
source_gas[i] += stoich * r / MW_i / epsi_r

# 3. Sumidero en sólido [kg/m³_bed/s]:
d_rho_solid[j] -= r

# 4. Calor al sólido [W/m³_bed]:
Q_rxn_j = -dH_rxn * r / MW_reactivo   # negativo = endotérmica

# 5. q_masstransfer para el gas — VER physics/cross-phase.md
```

---

## Reacciones implementadas en el gasificador

### Secado (drying.py)

```python
r_dry = drying_rate(rho_moisture, Ts, kinetics["drying"])
# Arrhenius: r = rho_moisture · A · exp(−E/RT)  [kg/m³_bed/s]
# Produce: H₂O(g) desde H₂O(l) en el sólido

src_H2O = drying_gas_source(r_dry, MW_H2O)  # [mol/m³_bed/s]
Q_dry   = drying_enthalpy_sink(r_dry)        # [W/m³_bed] = r_dry · H_vap
# Nota: Q_dry solo captura el calor latente. El calor sensible del vapor
# que entra al gas va por q_masstransfer (ver cross-phase.md).
```

### Pirólisis (pyrolysis.py)

```python
r_pyr = pyrolysis_rate(rho_biomass, Ts, kinetics["pyrolysis"])
# Arrhenius: r = rho_biomass · A · exp(−E/RT)  [kg/m³_bed/s]
# Produce: gases según yields (CO, CO2, H2, CH4, tar, H2O, char)

src_pyr_gas, src_char = pyrolysis_sources(r_pyr, yields, MW_gas, species)
# src_pyr_gas: (nc, N) [mol/m³_bed/s]
# src_char:    (N,) [kg/m³_bed/s] — va al d_rho_char positivo

Q_pyr = pyrolysis_enthalpy_sink(r_pyr, dH_pyr)
# dH_pyr > 0 (endotérmica) → sumidero en Ts
# dH_pyr se calcula con compute_pyrolysis_dH(heating_values, yields)
```

### Reacciones del char (char_conversion.py)

```python
r_ox, r_CO2, r_H2O = char_het_rates(
    rho_char, C_gas, Ts, v_cell, rho_g, mu_g, Tg, P_Pa,
    prop_gas, fuel_config, params_rxn, epsi_r, species,
)
# r_ox:  C + O₂ → CO/CO₂    (combustión)  [kg_char/m³_bed/s]
# r_CO2: C + CO₂ → 2CO      (Boudouard)   [kg_char/m³_bed/s]
# r_H2O: C + H₂O → CO + H₂  (steam gasif) [kg_char/m³_bed/s]
# Modelo SCM (Shrinking Core): resistencia difusión externa + cinética
# co_co2_ratio determina el reparto CO/CO₂ en combustión

src_char_gas = char_gas_sources(r_ox, r_CO2, r_H2O, Ts, char_comp, co_co2, MW, species)
# Devuelve fuentes de CO, CO2, H2O, H2 en mol/m³_bed/s

Q_char = char_reaction_heat(r_ox, r_CO2, r_H2O, Ts, heating_values, co_co2, char_comp)
# Calor de reacción al sólido [W/m³_bed]
# Positivo: exotérmico (combustión del char)
# Negativo: endotérmico (Boudouard, steam gasification)
```

---

## Modelo SCM — Shrinking Core Model

```python
# Diámetro de partícula dinámico:
dp = particle_diameter(rho_char, rho_char0, dp0)
# dp = dp0 * (rho_char / rho_char0)^(1/3)    [m]

# Superficie específica:
a_p = specific_surface_area(dp, epsi_r)
# a_p = 6 * (1 - epsi_r) / dp               [m²/m³_bed]

# La resistencia a la transferencia de masa depende de dp:
# h_bed y k_mtc (si aplica) deben recalcularse con dp actualizado
# → usar compute_transfer_coefficients con prop_lecho_dyn
prop_lecho_dyn = {**prop_lecho, "D_p": dp, "a_surf": a_p}
```

---

## Reacciones homogéneas (pendiente de implementar)

```
WGS (Water-Gas Shift):
  CO + H₂O ⇌ CO₂ + H₂    ΔH = −41.2 kJ/mol (exotérmica)
  Cinética: k = A·exp(−E/RT)·[CO]·[H₂O] − k_r·[CO₂]·[H₂]

Tar cracking:
  tar → α·CO + β·H₂ + γ·CH₄ + char
  Cinética: 1er orden en [tar]

Implementación futura en: src/physics/reactions/homogeneous_reactions.py
Firma prevista:
  r_hom = homogeneous_rates(C_mat, Tg_arr, species) → ndarray(n_rxn, N) [mol/m³_gas/s]
  source_gas[i] += Σⱼ stoich[i,j] * r_hom[j]   # directo, sin /epsi (ya en m³_gas)
  # Calor homogéneo va directamente al gas (sin q_masstransfer adicional):
  q_rxn_gas = Σⱼ (-dH_rxn[j]) * r_hom[j]   [W/m³_gas]
  dHgdt += q_rxn_gas * epsi_r               [W/m³_bed]
```

---

## Isotermas (adsorción)

```python
# Modelos puros (src/utils/isotherm_models.py):
q_eq = langmuir(P_partial, q_sat, b)              # [mol/kg]
q_eq = DSL(P_partial, q_s1, b1, q_s2, b2)        # Dual-Site Langmuir
q_eq = DSLF(P_partial, q_s1, b1, n1, q_s2, b2, n2)  # DSL con Freundlich

# Parámetros dependientes de T (Arrhenius):
b_T = b0 * np.exp(dH_ads / (R_GAS * T))           # b aumenta al bajar T

# Multicomponente (src/utils/mixture_isotherm.py):
q_eq_mix = iast(P_partial_list, T, iso_fns_list)   # (nc,) o (nc, N)
```

---

## Checklist para añadir una nueva reacción

```
□ ¿Es heterogénea (sólido-gas) u homogénea (gas-gas)?
□ Definir reactivos y productos con estequiometría exacta
□ Definir ΔH_rxn [J/mol_reactivo] (positivo = endotérmica)
□ Elegir modelo cinético (Arrhenius, SCM, LH, etc.)
□ Crear función pura en src/physics/reactions/<reaccion>.py
□ Añadir source_gas[i] en paso 8 del RHS (con /epsi si es bed)
□ Añadir sumidero en sólido en paso 9 si consume masa sólida
□ Añadir Q_rxn en sólido en paso 10 si la reacción calienta el sólido
□ Si heterogénea: añadir q_masstransfer en Hg (ver cross-phase.md)
□ Si homogénea: añadir q_rxn_gas = -dH * r directamente a dHgdt
□ Documentar en equipment/gasifier.md (o del equipo correspondiente)
```
