# Gas Property Database — `gasdb.txt`

Base de datos de propiedades termofísicas puras de gases en formato **NDJSON** (un objeto JSON por línea). Indexada por fórmula química.

---

## Estructura de un registro

```jsonc
{
  "formula":    "N2",                         // clave primaria — fórmula química
  "name_es":   "nitrogeno",
  "inchikey":  "IJGRMHOSHXDMSA-UHFFFAOYSA-N",
  "cas":       "7727-37-9",
  "molar_mass": { "value": 0.02801384, "units": "kg/mol" },
  "transport": {
    "lj": {
      "sigma_A":          3.798,              // diámetro colisión Lennard-Jones [Å]
      "epsilon_over_k_K": 71.4               // energía LJ / k_B [K]
    }
  },
  "reference": { "Tref_K": 298 },            // temperatura de referencia
  "limits":    { "Tmax_K": 1500 },           // temperatura máxima de validez
  "cap_at_tmax": {                           // valor del polinomio evaluado en Tmax
    "T_K":            1500,
    "mu_Pa_s":        5.407e-5,
    "cp_J_per_kgK":   1243.84,
    "k_W_per_mK":     0.0880,
    "u_J_per_mol":    34605.0,
    "h_J_per_mol":    47080.0,
    "s_J_per_molK":   241.77
  },
  "polynomials": {
    "basis":      "deltaT",                  // f(T) = Σ aᵢ·(T−Tref)ⁱ
    "degree_max": 7,
    "properties": {
      "mu": { "units": "Pa*s",    "a0_to_a7": [...] },  // viscosidad dinámica
      "cp": { "units": "J/kg/K",  "a0_to_a7": [...] },  // calor específico másico
      "k":  { "units": "W/m/K",   "a0_to_a7": [...] },  // conductividad térmica
      "u":  { "units": "J/mol",   "a0_to_a7": [...] },  // energía interna molar
      "h":  { "units": "J/mol",   "a0_to_a7": [...] },  // entalpía molar
      "s":  { "units": "J/mol/K", "a0_to_a7": [...] }   // entropía molar
    }
  }
}
```

**Reglas del polinomio:**
- Base `deltaT`: `f(T) = a0 + a1·(T−Tref) + a2·(T−Tref)² + … + a7·(T−Tref)⁷`
- `a0` es el valor exacto de la propiedad en `T = Tref`
- `T` se recorta a `[Tref, Tmax]` antes de evaluar (sin extrapolación)
- `cap_at_tmax` almacena el valor del polinomio en `Tmax`; se devuelve para cualquier `T ≥ Tmax`
- Usar ceros (`0.0`) para los coeficientes de grado no necesario

---

## Gases disponibles

| Fórmula | Nombre         | Tref [K] | Tmax [K] | MW [g/mol] |
|---------|----------------|----------|----------|------------|
| N2      | Nitrógeno      | 298      | 1500     | 28.01      |
| O2      | Oxígeno        | 298      | 1800     | 32.00      |
| H2      | Hidrógeno      | 298      | 1200     | 2.016      |
| CH4     | Metano         | 298      | 800      | 16.04      |
| C2H6    | Etano          | 298      | 800      | 30.07      |
| CO2     | Dióxido de CO  | 298      | 1800     | 44.01      |
| CO      | Monóxido de CO | 298      | 600      | 28.01      |
| H2O     | Agua (vapor)   | 383      | 1400     | 18.02      |

---

## Cómo añadir un nuevo gas

### Paso 1 — Preparar los datos termofísicos

Obtener datos tabulados de **NIST WebBook**, **DIPPR** o **Perry's** para las 6 propiedades en un rango de temperatura suficientemente amplio:

| Propiedad | Unidades | Fuente recomendada |
|-----------|----------|-------------------|
| μ viscosidad dinámica | Pa·s    | NIST, Chapman-Enskog |
| cp calor específico    | J/kg·K  | NIST, JANAF |
| k conductividad térmica| W/m·K   | NIST |
| u energía interna molar| J/mol   | NIST, JANAF (integrar Cv) |
| h entalpía molar       | J/mol   | NIST, JANAF |
| s entropía molar       | J/mol·K | NIST, JANAF |

Elegir `Tref` como la temperatura de referencia base de los datos (habitualmente 298 K; para gases condensables puede ser mayor). Elegir `Tmax` como el límite superior de validez del ajuste.

### Paso 2 — Ajustar los coeficientes polinómicos

Abrir `themoLibs.xlsx`, copiar la hoja **Template** y renombrarla con la fórmula química.

Introducir los datos medidos/tabulados en las columnas azules de la rejilla de temperatura. Ajustar los coeficientes `a0..a7` con una de estas opciones:

**Python (recomendado):**
```python
import numpy as np

T_data  = np.array([298, 400, 600, 800, 1000, 1200])   # K
mu_data = np.array([1.78e-5, 2.10e-5, 2.60e-5, ...])   # Pa·s

Tref = 298.0
dT   = T_data - Tref
coeffs = np.polyfit(dT, mu_data, deg=7)[::-1]           # a0 primero
print(coeffs)
```

**Excel:**
```
=LINEST(mu_data, dT^{1,2,3,4,5,6,7}, TRUE, FALSE)
```

Introducir los coeficientes en la tabla de la hoja Excel y verificar que el error `%` en las columnas rojas es aceptable (objetivo: < 1 % en todo el rango).

### Paso 3 — Calcular los parámetros de Lennard-Jones

Si no se dispone de valores tabulados, estimar mediante la regla de correspondencia de estados:

```
σ [Å]    ≈ 2.44 · (Tc/Pc)^(1/3)    [Tc en K, Pc en atm]
ε/kB [K] ≈ 0.77 · Tc
```

Fuente preferente: Apéndice A de *Transport Phenomena* (Bird, Stewart, Lightfoot).

### Paso 4 — Calcular los valores en `cap_at_tmax`

```python
def poly_eval(a, Tref, T):
    dT = T - Tref
    return sum(ai * dT**i for i, ai in enumerate(a))

cap_mu = poly_eval(a_mu, Tref, Tmax)
cap_cp = poly_eval(a_cp, Tref, Tmax)
# ... etc.
```

### Paso 5 — Añadir la línea a `gasdb.txt`

Añadir una nueva línea al final del archivo, **un JSON compacto por línea**, siguiendo exactamente el esquema del resto de entradas:

```jsonc
{"formula":"XX","name_es":"nombre","inchikey":"...","cas":"...","molar_mass":{"value":0.0,"units":"kg/mol"},"transport":{"lj":{"sigma_A":0.0,"epsilon_over_k_K":0.0}},"reference":{"Tref_K":298},"limits":{"Tmax_K":1000},"cap_at_tmax":{...},"polynomials":{"basis":"deltaT","degree_max":7,"properties":{"mu":{"units":"Pa*s","a0_to_a7":[...]},"cp":{"units":"J/kg/K","a0_to_a7":[...]},"k":{"units":"W/m/K","a0_to_a7":[...]},"u":{"units":"J/mol","a0_to_a7":[...]},"h":{"units":"J/mol","a0_to_a7":[...]},"s":{"units":"J/mol/K","a0_to_a7":[...]}}}}
```

> **Importante:** No dejar líneas en blanco en medio del archivo. Verificar que el JSON es válido con `python -c "import json; [json.loads(l) for l in open('gasdb.txt') if l.strip()]"`.

### Paso 6 — Regenerar `themoLibs.xlsx`

```bash
C:\ProgramData\anaconda3\python.exe materials/fluids/_gen_themoLibs.py
```

### Paso 7 — Verificar la integración con el modelo

```python
from src.physics.thermodynamics.pure_gas import build_pure_gas_properties

props = build_pure_gas_properties(["XX"], mode="polynomial")
print(props["mu"])   # debe devolver un callable
print(props["MW"])
```

---

## Cómo usar la base de datos en el modelo

```python
from src.physics.thermodynamics.pure_gas import build_pure_gas_properties

# Propiedades constantes evaluadas a 600 K
props = build_pure_gas_properties(
    species  = ["N2", "CO2"],
    mode     = "constant",
    Temp     = 600.0,
    db_path  = "materials/fluids/gasdb.txt",
)
# props["mu"]  → np.ndarray (2,)  [Pa·s]
# props["k"]   → np.ndarray (2,)  [W/m/K]
# props["MW"]  → np.ndarray (2,)  [kg/mol]

# Propiedades como función de T (para integración dinámica)
props = build_pure_gas_properties(["N2", "CO2"], mode="polynomial")
# props["mu"]  → list[callable]  — cada f(T) acepta escalar o ndarray
```

---

## Notas

- El archivo debe estar en **UTF-8 sin BOM**.
- La clave `formula` es case-sensitive: `"N2"` ≠ `"n2"`.
- Si `Tref` de un gas es distinto de 298 K (caso H2O: 383 K), los coeficientes `a0` de `u`, `h` y `s` son los valores en esa temperatura de referencia propia, no en 298 K.
- Los parámetros LJ de transporte (`sigma_A`, `epsilon_over_k_K`) son independientes de los polinomios de propiedades y son usados exclusivamente por el módulo de difusión Chapman-Enskog.
