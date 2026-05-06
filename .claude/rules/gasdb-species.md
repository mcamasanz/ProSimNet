# Cómo añadir una especie gaseosa a gasdb.txt

## Formato del archivo

`materials/fluids/gasdb.txt` es NDJSON (una línea JSON por especie).
Cada registro sigue esta estructura:

```json
{
  "formula": "XX",
  "name_es": "nombre en español",
  "inchikey": "XXXX-UHFFFAOYSA-N",
  "cas": "XXXXXXX-XX-X",
  "molar_mass": {"value": MW_kg_mol, "units": "kg/mol"},
  "transport": {"lj": {"sigma_A": sigma, "epsilon_over_k_K": eps_k}},
  "reference": {"Tref_K": 298},
  "limits": {"Tmax_K": 5000},
  "cap_at_tmax": {
    "T_K": 5000.0,
    "mu_Pa_s": ..., "cp_J_per_kgK": ..., "k_W_per_mK": ...,
    "u_J_per_mol": ..., "h_J_per_mol": ..., "s_J_per_molK": ...
  },
  "polynomials": {
    "basis": "deltaT",
    "degree_max": 7,
    "properties": {
      "mu": {"units": "Pa*s",    "a0_to_a7": [...]},
      "cp": {"units": "J/kg/K",  "a0_to_a7": [...]},
      "k":  {"units": "W/m/K",   "a0_to_a7": [...]},
      "u":  {"units": "J/mol",   "a0_to_a7": [...]},
      "h":  {"units": "J/mol",   "a0_to_a7": [...]},
      "s":  {"units": "J/mol/K", "a0_to_a7": [...]}
    }
  }
}
```

Todos los polinomios tienen la forma:
`f(T) = Σᵢ aᵢ·(T − Tref)ⁱ`  con T clipado a `[Tref, Tmax]`.

---

## Paso 1 — Obtener Cp de los datos de Fluent (o NIST Shomate)

### Opción A: datos de Fluent (piecewise polynomial)

Fluent entrega Cp en dos rangos (bajo y alto) con hasta 5 coeficientes en T:
```python
coef_low  = [a0, a1, a2, a3, a4]  # Cp[J/kg/K] = a0 + a1*T + a2*T² + a3*T³ + a4*T⁴  (298–1000 K)
coef_high = [a0, a1, a2, a3, a4]  # (1000–5000 K)
```

### Opción B: NIST Shomate

```python
# Cp [J/mol/K] = A + B*t + C*t² + D*t³ + E/t²    t = T[K]/1000
# Rango 1: 298–1400 K
sho1 = dict(A=..., B=..., C=..., D=..., E=...)
# Rango 2: 1400–6000 K
sho2 = dict(A=..., B=..., C=..., D=..., E=...)

def cp_mass(T):   # J/kg/K
    t = T / 1000.0
    cp_mol = ...  # evaluar Shomate por rango
    return cp_mol / MW_kg_mol
```

### Ajuste grado 7 en base deltaT

Usando `_gen_themoLibs_cp_coeff.py` como referencia:

```python
import numpy as np

Tref   = 298.0
degree = 7
T_data = np.concatenate([np.linspace(298, 1000, 400), np.linspace(1000, 5000, 800)])
theta  = T_data - Tref
y_cp   = cp_mass(T_data)          # J/kg/K

c_desc = np.polyfit(theta, y_cp, degree)   # orden decreciente
c_cp   = c_desc[::-1]                      # a0, a1, ..., a7

# Verificar error relativo < 1%
err_rel = 100 * np.abs(np.polyval(c_desc, theta) - y_cp) / y_cp
print(f"rel_max = {np.max(err_rel):.4f}%")
```

---

## Paso 2 — Viscosidad µ: Chapman-Enskog + Neufeld (1972)

Requiere parámetros de Lennard-Jones: σ [Å] y ε/k_B [K].

**Fuentes LJ recomendadas:** Reid, Prausnitz & Poling (App. B);
Poling, Prausnitz & O'Connell (App. A).

### Integral de colisión Ω^(2,2) de Neufeld (1972)

```python
# Constantes de Neufeld (1972)
A_N=1.16145; B_N=0.14874; C_N=0.52487; D_N=0.7732
E_N=2.16178; F_N=2.43787; G_N=-6.435e-4; H_N=7.27371

def omega22(T, eps_k):
    Tstar = T / eps_k
    return (A_N/Tstar**B_N + C_N/np.exp(D_N*Tstar)
            + E_N/np.exp(F_N*Tstar) + G_N/np.exp(H_N*Tstar))
```

### Fórmula Chapman-Enskog

```python
def mu_ce(T_arr, MW_kg_mol, sigma_A, eps_k):
    """µ [Pa·s] — Chapman-Enskog con Ω^(2,2) de Neufeld."""
    T  = np.asarray(T_arr, float)
    M  = MW_kg_mol * 1000.0      # g/mol
    # 26.693e-7: factor (µP·s/Å²·(g/mol·K)^0.5) → 1 µP = 1e-7 Pa·s
    return 26.693e-7 * np.sqrt(M * T) / (sigma_A**2 * omega22(T, eps_k))
```

**Nota:** el factor correcto es `26.693e-7` (no `26.693e-6`).
El resultado de la fórmula está en µPoise (CGS); multiplicar por 1e-7 da Pa·s.

Ajustar grado 7 en base deltaT con el mismo procedimiento que Cp.

---

## Paso 3 — Conductividad k: Eucken modificado

```python
R_GAS = 8.31446261815324   # J/mol/K

def k_eucken(T_arr, MW_kg_mol, sigma_A, eps_k):
    """k [W/m/K] — relación de Eucken modificada."""
    mu   = mu_ce(T_arr, MW_kg_mol, sigma_A, eps_k)
    cp_m = cp_mass(T_arr)               # J/kg/K — misma función del Paso 1
    return mu * (cp_m + 1.25 * R_GAS / MW_kg_mol)
```

Ajustar grado 7 en base deltaT.

**Limitaciones conocidas:** el Eucken modificado sobreestima k en moléculas
polares (NH3, H2O) en ~10-20%. Es suficientemente preciso para los balances
energéticos del modelo, pero no debe usarse como referencia de cálculo de
fenómenos de transporte precisos.

---

## Paso 4 — Entalpía h, energía interna u, entropía s

### Entalpía h [J/mol]

```python
# h(Tref) ≈ Cp_molar(Tref) * Tref   [aprox. integral lineal desde 0 K]
Cp_Tref = cp_mass(Tref) * MW_kg_mol  # J/mol/K
h_Tref  = Cp_Tref * Tref             # J/mol

# Para T > Tref: h(T) = h(Tref) + ∫_{Tref}^T Cp_molar dT
# usando la integral analítica del Shomate (Opción B) o integración numérica
```

Opción B (Shomate):
```python
def h_shomate_int(T_arr, sho, T_lo):
    """∫_{T_lo}^T Cp dT [J/mol] usando Shomate."""
    def _I(t):
        return 1000*(sho['A']*t + sho['B']*t**2/2 + sho['C']*t**3/3
                     + sho['D']*t**4/4 - sho['E']/t)
    t  = np.asarray(T_arr, float) / 1000.0
    t0 = T_lo / 1000.0
    return _I(t) - _I(t0)
```

Ajustar grado 7 en base deltaT.

### Energía interna u [J/mol]

```python
# u = h - RT  (gas ideal)
u_Tref = h_Tref - R_GAS * Tref
# Para el polinomio:
u_arr = h_arr - R_GAS * T_data     # generar datos y ajustar
```

Equivalentemente: `a0(u) = a0(h) - R*Tref`, `a1(u) = a1(h) - R`, `aₙ(u)=aₙ(h)` para n≥2.

### Entropía s [J/mol/K]

```python
# s(Tref) = S°(298.15 K) de NIST-JANAF  [J/mol/K]
# s(T) = S°(Tref) + ∫_{Tref}^T (Cp/T) dT

def ds_shomate(T_arr, sho, T_lo):
    def _Is(t):
        return (sho['A']*np.log(t) + sho['B']*t + sho['C']*t**2/2
                + sho['D']*t**3/3 - sho['E']/(2*t**2))
    t  = np.asarray(T_arr, float) / 1000.0
    t0 = T_lo / 1000.0
    return _Is(t) - _Is(t0)
```

---

## Paso 5 — Cap at Tmax

Evaluar todas las propiedades a `T = Tmax` usando las funciones originales
(no el polinomio ajustado) y almacenarlas en `cap_at_tmax`.
Estas se usan cuando el código extrapola por encima de Tmax (clipping de propiedades).

---

## Paso 6 — Añadir línea a gasdb.txt

Generar la línea JSON compacta (sin saltos de línea) y añadirla al final del archivo.
Verificar con `read_gasdb` que no hay error de parse ni fórmula duplicada.

```python
from src.io.gasdb_reader import read_gasdb
db = read_gasdb("materials/fluids/gasdb.txt")
assert "NUEVA_ESPECIE" in db
```

---

## Checklist antes de añadir

- [ ] Parámetros LJ (σ, ε/k_B) obtenidos de fuente bibliográfica documentada
- [ ] Cp validado contra NIST o Fluent en ≥3 temperaturas
- [ ] µ validado contra NIST o literatura en ≥2 temperaturas
- [ ] k validado contra NIST o literatura en ≥2 temperaturas
- [ ] Error relativo del ajuste Cp < 1% en todo el rango
- [ ] h(Tref) = a0(h) coherente con integral de Cp desde 0 K
- [ ] s(Tref) = a0(s) coincide con S°(Tref) de NIST-JANAF
- [ ] u(Tref) = h(Tref) - R*Tref  verificado
- [ ] `read_gasdb` no lanza excepciones tras añadir la línea

---

## Especies en la base de datos

| Formula | Tref [K] | σ [Å]  | ε/k_B [K] | Fuente LJ          |
|---------|----------|--------|-----------|-------------------|
| N2      | 298      | 3.798  | 71.4      | Reid et al.       |
| O2      | 298      | 3.467  | 106.7     | Reid et al.       |
| H2      | 298      | 2.8227 | 59.7      | Reid et al.       |
| CH4     | 298      | 3.822  | 137.0     | Reid et al.       |
| C2H6    | 298      | 4.418  | 230.0     | Reid et al.       |
| CO2     | 298      | 3.941  | 195.2     | Reid et al.       |
| CO      | 298      | 3.758  | 148.6     | Reid et al.       |
| C2H4    | 298      | 4.163  | 224.7     | Reid et al.       |
| H2O     | 383      | 2.641  | 809.1     | Reid et al.       |
| NH3     | 298      | 2.900  | 558.3     | Reid et al.       |
