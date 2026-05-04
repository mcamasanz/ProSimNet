# Solid Material Property Database — `soliddb.txt`

Base de datos de propiedades termofísicas de materiales sólidos en formato **NDJSON** (un objeto JSON por línea). Indexada por identificador de material (`id`).

---

## Estructura de un registro

```jsonc
{
  "id":       "SS316L",                      // clave primaria — identificador único
  "name_es":  "acero inoxidable 316L",
  "name_en":  "stainless steel 316L",
  "category": "metal",                       // "metal" | "ceramic" | "polymer" | ...
  "reference": { "Tref_K": 298 },           // temperatura de referencia
  "limits":    { "Tmax_K": 1273 },          // temperatura máxima de validez
  "cap_at_tmax": {                          // valor del polinomio evaluado en Tmax
    "T_K":           1273,
    "rho_kg_per_m3": 7577.5,
    "cp_J_per_kgK":  619.9,
    "k_W_per_mK":    24.43
  },
  "polynomials": {
    "basis":      "deltaT",                  // f(T) = Σ aᵢ·(T−Tref)ⁱ
    "degree_max": 7,
    "properties": {
      "rho": { "units": "kg/m3",  "a0_to_a7": [...] },  // densidad
      "cp":  { "units": "J/kg/K", "a0_to_a7": [...] },  // calor específico másico
      "k":   { "units": "W/m/K",  "a0_to_a7": [...] }   // conductividad térmica
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

## Materiales disponibles

| id         | Nombre                    | Categoría | Tref [K] | Tmax [K] | ρ [kg/m³] | cp [J/kg·K] | k [W/m·K] |
|------------|---------------------------|-----------|----------|----------|-----------|-------------|-----------|
| SS316L     | Acero inoxidable 316L     | metal     | 298      | 1273     | 7950      | 500         | 14.4      |
| P265GH     | Acero al carbono P265GH   | metal     | 298      | 873      | 7850      | 490         | 51.0      |
| Inconel625 | Inconel 625 (UNS N06625)  | metal     | 298      | 1273     | 8440      | 410         | 9.8       |
| Al2O3      | Alúmina densa 99%         | ceramic   | 298      | 1800     | 3960      | 777         | 30.0      |
| SiC        | Carburo de silicio denso  | ceramic   | 298      | 1773     | 3210      | 750         | 120.0     |
| SiO2_fused | Sílice fundida (cuarzo)   | ceramic   | 298      | 1473     | 2200      | 740         | 1.38      |
| Cu         | Cobre puro OFHC           | metal     | 298      | 1273     | 8960      | 385         | 385.0     |

> **Nota de precisión:** los ajustes polinómicos para metales son lineales con error < 2 %. Para cerámicas con cp fuertemente no lineal (Al2O3, SiC), el error puede ser del 10-15 % en el extremo superior del rango, debido a la saturación tipo Debye. Usar `themoLibs.xlsx` para refinar los ajustes con datos experimentales.

---

## Cómo añadir un nuevo material

### Paso 1 — Preparar los datos termofísicos

Obtener datos tabulados para las 3 propiedades en el rango de temperatura de interés:

| Propiedad            | Unidades | Fuentes recomendadas                         |
|----------------------|----------|----------------------------------------------|
| ρ densidad           | kg/m³    | Ficha técnica fabricante, ASM Handbook        |
| cp calor específico  | J/kg·K   | NIST, ASM Handbook, Incropera & DeWitt        |
| k conductividad      | W/m·K    | NIST, ASM Handbook, Incropera & DeWitt        |

Elegir `Tref = 298 K` salvo que los datos de referencia usen otra temperatura base. Elegir `Tmax` como el límite práctico de la aplicación (no superar el rango de validez de los datos).

### Paso 2 — Calcular el coeficiente de expansión volumétrica (para ρ)

La densidad disminuye con la temperatura según la expansión volumétrica:

```
ρ(T) ≈ ρ₀ · (1 − β_vol · (T − Tref))
→ a0 = ρ₀,   a1 = −ρ₀ · β_vol
```

donde `β_vol = 3 · α_lineal` [1/K]. Valores típicos de `β_vol`:

| Material          | β_vol [1/K] |
|-------------------|-------------|
| Aceros inoxidables| 48 × 10⁻⁶  |
| Aceros al carbono | 36 × 10⁻⁶  |
| Inconel           | 38 × 10⁻⁶  |
| Al2O3             | 24 × 10⁻⁶  |
| SiC               | 12 × 10⁻⁶  |
| Sílice fundida    | 1.6 × 10⁻⁶ |
| Cobre             | 51 × 10⁻⁶  |

### Paso 3 — Ajustar los coeficientes polinómicos

Abrir `themoLibs.xlsx`, copiar la hoja **Template** y renombrarla con el nuevo `id`.

Introducir los datos tabulados en las columnas azules. Ajustar con Python:

```python
import numpy as np

T_data  = np.array([298, 400, 600, 800, 1000, 1200])   # K
cp_data = np.array([500, 515, 540, 560, 585, 610])      # J/kg·K

Tref   = 298.0
dT     = T_data - Tref
degree = 2   # para metales suele bastar grado 1-2; cerámicas pueden necesitar 3-4

coeffs = np.polyfit(dT, cp_data, deg=degree)[::-1]     # a0 primero
# Rellenar con ceros hasta 8 coeficientes
a = list(coeffs) + [0.0] * (8 - len(coeffs))
print(a)
```

Verificar en las columnas rojas de `themoLibs.xlsx` que el error es aceptable:
- Objetivo metales: < 2 % en todo el rango
- Objetivo cerámicas: < 10 % (aceptar mayor error si el rango de T es muy amplio)

Si el error es alto, reducir `Tmax` al rango donde el polinomio sea preciso, o aumentar el grado del ajuste.

### Paso 4 — Calcular los valores en `cap_at_tmax`

```python
def poly_eval(a, Tref, T):
    dT = T - Tref
    return sum(ai * dT**i for i, ai in enumerate(a))

cap_rho = poly_eval(a_rho, Tref, Tmax)
cap_cp  = poly_eval(a_cp,  Tref, Tmax)
cap_k   = poly_eval(a_k,   Tref, Tmax)
```

### Paso 5 — Añadir la línea a `soliddb.txt`

Añadir una nueva línea al final del archivo, **un JSON compacto por línea**:

```jsonc
{"id":"NUEVO_ID","name_es":"nombre en español","name_en":"name in english","category":"metal","reference":{"Tref_K":298},"limits":{"Tmax_K":1000},"cap_at_tmax":{"T_K":1000,"rho_kg_per_m3":0.0,"cp_J_per_kgK":0.0,"k_W_per_mK":0.0},"polynomials":{"basis":"deltaT","degree_max":7,"properties":{"rho":{"units":"kg/m3","a0_to_a7":[...]},"cp":{"units":"J/kg/K","a0_to_a7":[...]},"k":{"units":"W/m/K","a0_to_a7":[...]}}}}
```

> **Importante:** No dejar líneas en blanco en medio del archivo. Verificar que el JSON es válido:
> ```bash
> python -c "import json; [json.loads(l) for l in open('soliddb.txt') if l.strip()]"
> ```

### Paso 6 — Regenerar `themoLibs.xlsx`

```bash
C:\ProgramData\anaconda3\python.exe materials/solids/_gen_themoLibs.py
```

### Paso 7 — Verificar la integración con el modelo

```python
from src.physics.thermodynamics.solid_props import build_solid_prop_config, eval_solid_property

# Modo constante (propiedades en Tref)
cfg = build_solid_prop_config("NUEVO_ID", mode="constant")
print(cfg["rho"], cfg["cp"], cfg["k"])

# Modo polinómico (función de T)
cfg = build_solid_prop_config("NUEVO_ID", mode="polynomial")
T_test = 800.0   # K
print(eval_solid_property(cfg["cp"], T_test))
```

---

## Cómo usar la base de datos en el modelo

```python
from src.physics.thermodynamics.solid_props import build_solid_prop_config, eval_solid_property
import numpy as np

# ── Modo constante (propiedades fijas en Tref) ────────────────────────────────
cfg = build_solid_prop_config("SS316L", mode="constant")
rho = cfg["rho"]    # float [kg/m³]
cp  = cfg["cp"]     # float [J/kg·K]
k   = cfg["k"]      # float [W/m·K]

# ── Modo polinómico (propiedades como f(T)) ───────────────────────────────────
cfg = build_solid_prop_config("SS316L", mode="polynomial")
T_wall = np.linspace(400, 900, 10)          # K, array de nodos
rho_T  = eval_solid_property(cfg["rho"], T_wall)   # array [kg/m³]
cp_T   = eval_solid_property(cfg["cp"],  T_wall)   # array [J/kg·K]
k_T    = eval_solid_property(cfg["k"],   T_wall)   # array [W/m·K]

# ── Modo fixed (valores directos, sin BD) ────────────────────────────────────
cfg = build_solid_prop_config(
    "cualquier_id",
    mode      = "fixed",
    rho_fixed = 7800.0,
    cp_fixed  = 510.0,
    k_fixed   = 45.0,
)
```

---

## Notas

- El archivo debe estar en **UTF-8 sin BOM**.
- La clave `id` es case-sensitive: `"SS316L"` ≠ `"ss316l"`.
- La densidad `rho` se incluye como polinomio para capturar la expansión térmica. Para aplicaciones donde la variación de ρ con T es irrelevante, basta con `a0 = ρ₀` y el resto ceros.
- El modo `"fixed"` de `build_solid_prop_config` permite usar valores directos sin acceder a la BD, útil para parámetros de ajuste o materiales no catalogados.
- Los valores de `cap_at_tmax` deben ser exactamente lo que devuelve el polinomio en `Tmax` (no valores de literatura), para garantizar continuidad en la evaluación.
