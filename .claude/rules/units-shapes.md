# Unidades y shapes de arrays

## Sistema de unidades — SI puro internamente

| Variable | Unidad interna | Nota |
|----------|----------------|------|
| Presión | **Pa** | Contornos e isotermas reciben **bar** → convertir: `P_Pa = P_bar * 1e5` |
| Temperatura | **K** | Siempre. Nunca °C. |
| Longitud | **m** | `dz`, `Di`, `L`, `dp` |
| Tiempo | **s** | `t`, `t_max`, `dt` |
| Concentración molar gas | **mol/m³_gas** | `C_i`, `C_in` |
| Carga adsorbida | **mol/kg** | `q_i` |
| Densidad sólida bulk | **kg/m³_bed** | `rho_biomass`, `rho_char`, `rho_s` |
| Velocidad superficial | **m/s** | `v_face`, `v_in`, `v_out` |
| Velocidad intersticial | **m/s** | `v_interstitial = v_superficial / epsi` |
| Entalpía molar | **J/mol** | `h_i`, retorno de `calc_species_enthalpy` |
| Entalpía volumétrica | **J/m³_bed** | `Hg = epsi · Σ C_i · h_i(Tg)` |
| Flujo entálpico | **W/m²_sección** | `F_h = v · H_cell` donde `H_cell = Σ C_i · h_i` (sin epsi) |
| Masa molar | **kg/mol** | `MW` array (nc,) |
| Calor volumétrico | **W/m³_bed** | `Q_rxn_vol`, `q_gs_vol`, `q_wall_vol` |
| Coef. HTC | **W/m²/K** | `h_bed`, `h_wall` |
| Coef. transferencia masa | **1/s** | `k_mtc` (LDF) |
| Difusividad | **m²/s** | `D_disp`, `D_bin`, `D_eff` |
| Densidad gas | **kg/m³** | `rho_g` |
| Viscosidad dinámica | **Pa·s** | `mu_g` |
| Conductividad térmica | **W/m/K** | `k_g`, `k_wall` |
| Tasa de reacción | **kg/m³_bed/s** o **mol/m³_bed/s** | Siempre en bed; dividir por epsi para gas |
| Área transversal | **m²** | `Ai` (interna), `A_w` (pared) |
| Perímetro | **m** | `Pi` (interno), `Po` (externo) |
| Porosidad | **—** | `epsi`, `epsi_r` ∈ (0, 1) |

---

## Conversiones frecuentes

```python
# bar → Pa
P_Pa = P_bar * 1e5

# mol/m³_bed → mol/m³_gas
source_gas = source_bed / epsi_r

# kg/m³_bed/s → mol/m³_bed/s
src_mol = src_kg / MW_i

# J/m³_bed para Hg
Hg = epsi_r * np.sum(C * h_i, axis=0)   # C (nc,N), h_i (nc,N) → Hg (N,)
```

---

## Shapes de arrays — convención estricta

| Array | Shape | Descripción |
|-------|-------|-------------|
| `C` | **(nc, N)** | Concentraciones: especies primero, celdas segundo |
| `q` | **(nc, N)** | Carga adsorbida: misma convención que C |
| `rho_solid` | **(n_s, N)** | Densidades sólidas: componentes primero |
| `h_i` | **(nc, N)** | Entalpía molar: retorno de `calc_species_enthalpy` |
| `source_gas` | **(nc, N)** | Fuentes en m³_gas |
| `y` (fracciones molares) | **(nc, N)** | Misma que C |
| `x` (para Wilke) | **(N, nc)** | ⚠️ **EXCEPCIÓN**: `x = y.T` antes de pasar a Wilke |
| `Tg`, `Ts`, `Tw` | **(N,)** | Temperaturas: siempre 1D |
| `P` | **(N,)** | Presión en celdas [bar] o [Pa] según contexto |
| `v_face` | **(N+1,)** | Velocidad en caras |
| `F_conv`, `F_diff` | **(N+1,)** | Flujos en caras |
| `div_F` | **(N,)** | Divergencia: `(F[1:] - F[:-1]) / dz` |
| `D_disp` | **(nc, N)** o `None` | Dispersión axial; None en plug-flow |
| `k_mtc` | **(nc, N)** | Coef. LDF por especie y celda |
| `h_bed`, `h_wall` | **(N,)** | HTC: siempre 1D por celda |
| `MW` | **(nc,)** | Masas molares: solo por especie |
| `Cp_fns` | lista de callables | `Cp_fns[j](T)` → float o ndarray |

---

## Operaciones vectoriales frecuentes

```python
# Entalpía volumétrica del gas
Hg = epsi_r * np.sum(C * h_i, axis=0)         # (N,)

# q_masstransfer (cross-phase sólido→gas)
h_i_Ts = calc_species_enthalpy(Ts, prop_gas, nc, gas_T_ref)   # (nc, N)
q_mt   = epsi_r * np.sum(source_gas * h_i_Ts, axis=0)         # (N,)

# Presión desde concentraciones (gas ideal)
Ctot = np.sum(C, axis=0)                       # (N,)
P_bar = Ctot * R_GAS * Tg / 1e5               # (N,) [bar]

# Fracciones molares
y = C / np.maximum(Ctot, 1e-300)              # (nc, N)
x = y.T                                         # (N, nc) para Wilke

# Divergencia del flujo
div_F = (F[1:] - F[:-1]) / dz                 # (N,) [unidad_F / m]

# Capacidad calorífica sólida volumétrica
Cs_vol = (rho_bio * Cp_fns[0](Ts)
        + rho_char * Cp_fns[1](Ts)
        + rho_moi  * Cp_fns[2](Ts))           # (N,) [J/m³_bed/K]
```

---

## Regla de conversión de tasas (bed → gas)

```python
epsi_safe = max(float(epsi_r), 1e-10)   # evitar /0

# Una tasa en mol/m³_BED/s → mol/m³_GAS/s
source_gas[i] += src_bed[i] / epsi_safe

# Una entalpía en mol/m³_BED/s × J/mol = J/m³_BED/s (ya está en bed)
q_mt = epsi_r * np.sum(source_gas * h_i, axis=0)   # ← epsi_r porque source_gas está en m³_gas
```

---

## Comprobación rápida de consistencia

Antes de añadir un nuevo término al RHS, responder:
1. ¿En qué volumen de referencia está la tasa? (bed, gas, pared)
2. ¿En qué unidades? (mol, kg, J, W)
3. ¿El receptor del término está en las mismas unidades?
4. ¿El shape es correcto? (N,) para vectores de celda, (nc,N) para matrices especie-celda
