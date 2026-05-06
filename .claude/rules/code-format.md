# Reglas de formato de código

## Idioma

| Elemento | Idioma |
|----------|--------|
| Nombres de variables, funciones, clases, atributos | **Inglés técnico** |
| Comentarios que aporten valor (decisión no obvia, hipótesis, convención) | **Español** |
| Docstrings de funciones públicas | **Inglés** (Parameters / Returns / Notes) |
| Comentarios triviales que repiten lo obvio | **No escribir** |

---

## Funciones

### Firma obligatoria

```python
def function_name(
    arg1: np.ndarray,   # (shape) [unidad] — descripción
    arg2: float,        # [unidad]
    params: dict,       # ver .claude/equipment/common.md §params
) -> np.ndarray:        # (shape) [unidad]
```

### Docstring mínimo para funciones públicas

```python
def compute_transfer_coefficients(
    Tg: np.ndarray,
    Ts: np.ndarray,
    ...
) -> dict:
    """
    Compute gas-solid and gas-wall heat/mass transfer coefficients.

    Parameters
    ----------
    Tg : ndarray (N,)   gas temperature [K]
    Ts : ndarray (N,)   solid temperature [K]

    Returns
    -------
    dict with keys:
        h_bed  : ndarray (N,)   [W/m²/K]
        h_wall : ndarray (N,)   [W/m²/K]
        D_disp : ndarray (nc, N) or None  [m²/s]
    """
```

### Funciones internas del RHS (hot path)

- **Sin docstring**. El RHS se llama millones de veces; no añadir overhead de texto.
- Un comentario de sección (`# ── Paso 8. Balance de especies ──`) es suficiente.
- Sin validaciones (`assert`, `if param not in ...`) dentro del RHS. Van en el runner.

---

## Clases

Solo se usan clases para:
- `SimpleNamespace` — objeto de resultados `col` (inmutable post-construcción)
- No hay clases de equipo; la lógica va en funciones puras + `params` dict

```python
# Correcto: SimpleNamespace como contenedor de resultados
result = types.SimpleNamespace(
    _t_results   = t_arr,
    _Tg_results  = Tg_hist,
    _Hg_results  = Hg_hist,
)

# Incorrecto: clase con estado mutable y lógica interna
class Gasifier:
    def step(self): ...  # NO — esto mezcla estado y física
```

---

## Variables

### Naming por tipo

```python
# Tasas de reacción (prefijo r_) — kg/m³_bed/s o mol/m³_bed/s
r_dry       # drying rate
r_pyr       # pyrolysis rate
r_ox        # char oxidation rate

# Fuentes en bed (prefijo src_) — antes de /epsi
src_dry_H2O      # (N,) mol/m³_bed/s
src_pyr_gas      # (nc, N) mol/m³_bed/s
src_char_gas     # (nc, N) mol/m³_bed/s

# Fuentes en gas (prefijo source_) — ya dividido por epsi
source_gas       # (nc, N) mol/m³_gas/s

# Calores de reacción en sólido (prefijo Q_) — W/m³_bed
Q_dry       # calor del secado
Q_pyr       # calor de pirólisis
Q_char      # calor de char reactions
Q_rxn_vol   # suma total

# Flujos de calor volumétricos (prefijo q_) — W/m³_bed
q_gs_vol        # gas↔sólido
q_wall_vol      # pared→gas
q_masstransfer  # entalpía de masa cross-phase

# Coeficientes de transporte (prefijo h_ o k_)
h_bed_arr    # (N,) W/m²/K
h_wall_arr   # (N,) W/m²/K
k_mtc_arr    # (nc, N) 1/s
D_disp_mat   # (nc, N) m²/s  (puede ser None)

# Acumuladores del RHS (prefijo d...dt)
dCdt_mat     # (nc, N)
dHgdt_arr    # (N,)
dTsdt_arr    # (N,)
dTwdt_arr    # (N,)  solo si shell_tube
```

### Constantes del módulo (nivel módulo, no dentro de funciones)

```python
R_GAS = 8.31446261815324   # [J/mol/K]
_IDX  = {"CO": 0, "CO2": 1, "H2O": 2, "H2": 3, "O2": 4,
          "CH4": 5, "C2H4": 6, "tar": 7, "N2": 8}
```

### Nombres a evitar

```
a, b, tmp, var1, kk, value2  ← nunca en código de producción
x, y, z  ← solo si son coordenadas espaciales y el contexto es claro
T  ← usar Tg, Ts, Tw, T_in, T_wall según la fase
```

---

## Caché dentro del RHS

```python
# Acceso siempre con setdefault + get/pop
cache = params.setdefault("_cache", {})

# Leer
gas_props = cache.get("gas_props")
Tg_guess  = cache.get("Tg_last", np.full(nn, 700.0))

# Escribir
cache["gas_props"]   = gas_props_new
cache["Tg_last"]     = Tg_arr.copy()   # warm-start Newton

# NO borrar "Tg_last" dentro del RHS (warm-start entre pasos)
# SÍ borrar en runner.run_step al inicio de cada llamada:
cache.pop("gas_props", None)
cache.pop("trans_props", None)
```

---

## Decoradores obligatorios

```python
from src.utils.profiling import profiled

@profiled
def core_rhs(t: float, sv: np.ndarray, params: dict) -> np.ndarray:
    ...

@profiled
def run_step(...):
    ...
```

---

## Imports

```python
# Orden: stdlib → terceros → src/ (alphabético dentro de cada bloque)
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from src.discretization.fluxes import convective_flux, diffusive_flux
from src.physics.thermodynamics.enthalpy import calc_species_enthalpy
from src.utils.profiling import profiled
```

Sin imports dentro de funciones salvo casos excepcionales documentados.

---

## Comentarios de sección en el RHS

```python
# ── 1. Lectura de params ───────────────────────────────────────────────────
# ── 2. Desempaquetado del vector de estado ────────────────────────────────
# ── 3. Condiciones de contorno ────────────────────────────────────────────
# ── 4. Propiedades de mezcla del gas ──────────────────────────────────────
# ── 5. Velocidades y coef. de transporte ──────────────────────────────────
# ── 6. Cinética / Tasas de reacción ───────────────────────────────────────
# ── 7. [Conveyor] Cálculo de rho_solid_in ─────────────────────────────────
# ── 8. Balance de especies gaseosas ───────────────────────────────────────
# ── 9. [Sólido] Balance de densidades / carga ─────────────────────────────
# ── 10. Balances de energía ────────────────────────────────────────────────
# ── 11. [Shell-tube] ODE de pared ─────────────────────────────────────────
# ── 12. Empaquetado del RHS ────────────────────────────────────────────────
```
