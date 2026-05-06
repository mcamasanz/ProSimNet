# Integración de señales y control en un equipo ProSimNet

> **Referencia de arquitectura:** `.claude/rules/signals-and-control.md`
> **Implementación de referencia:** gasificador — `dev/gasifier`, commit `efa137b`
> **Ámbito:** guía práctica con código exacto para replicar en cualquier equipo nuevo

---

## Qué se consigue con este patrón

Cualquier parámetro de BC (velocidad de entrada, temperatura, composición, T_wall, Qwall, Cv…) puede ser:

| Tipo | Ejemplo | Cuándo usar |
|------|---------|-------------|
| `float` | `T_wall=1073.15` | Operación nominal fija |
| `callable(t) → float` | `T_wall=ramp(0, 1.0, 700.0, 1200.0)` | Perfil temporal programado |
| `callable(t, snap) → float` | `T_wall=proportional(...)` | Control por retroalimentación de estado |

El RHS y el runner son **agnósticos al tipo**: nunca comprueban si un BC es constante o callable. Solo llaman a `resolve()`.

---

## Módulos involucrados

| Módulo | Ruta | Propósito |
|--------|------|-----------|
| `resolve()` | `src/utils/signals.py` | Evaluador universal de señales |
| `resolve_config_values()` | `src/utils/signals.py` | Resolver un dict de config de una vez |
| Primitivas | `src/control/signals.py` | `ramp`, `step`, `pulse`, `piecewise`, `sine` |
| Controladores | `src/control/controllers.py` | `onoff`, `proportional`, `feedforward` |
| Snap del equipo | `src/solvers/rhs/rhs_<equipo>.py` | Construcción del estado observable |

---

## Paso a paso para un equipo nuevo

### 1. Config builder — aceptar callables

En `src/units/<equipo>/config/boundary_c.py` y `thermal_bc.py`, usar `_opt_val()` en lugar de `_opt_float()` para los parámetros que deben admitir señales:

```python
def _opt_val(val):
    """Pasa callables sin conversión; convierte floats a float."""
    if val is None:
        return None
    if callable(val):
        return val          # se resolverá en el RHS via resolve()
    return float(val)

def _check_pos(val, name):
    if val is None or callable(val):
        return              # callables: sin validación estática
    if not np.isfinite(val) or val <= 0.0:
        raise ValueError(f"{name} must be finite > 0")
```

Para `Qwall` (puede ser scalar, array, o callable):

```python
if Qwall is None:
    Qwall_val = None
elif callable(Qwall):
    Qwall_val = Qwall       # callable(t) o callable(t, snap)
else:
    _q = np.asarray(Qwall, dtype=float)
    Qwall_val = float(_q) if _q.ndim == 0 else _q
```

**Parámetros que típicamente se hacen callables:**

| Config | Parámetros | Razón |
|--------|-----------|-------|
| `boundary_c` | `v_gas_in`, `T_gas_in`, `y_gas_in` | Perfil de operación temporal |
| `boundary_c` | `v_out`, `Cv` | Control de salida / válvula programada |
| `thermal_bc` | `T_wall`, `Qwall` | Control de calentamiento externo |
| `thermal_bc` | `T_ambi`, `h_ambi` | Condiciones externas variables |

Los parámetros fijos de geometría (`Di`, `Do`, `k_wall`, `rho_wall`, `Cp_wall`) son siempre `float` — no necesitan ser callables.

---

### 2. Evaluador BC — añadir `snap` y usar `resolve()`

En `src/boundary_conditions/<equipo>_boundary.py`:

```python
from src.utils.signals import resolve as _resolve

def get_<equipo>_boundary(
    t: float,
    ...,
    snap: dict | None = None,   # ← nuevo parámetro
    ...
) -> dict:
    _snap = snap if snap is not None else {}

    # Gas inlet: resolver con resolve() en lugar del patrón manual callable(t)/value
    v_in = float(_resolve(bc_config["v_gas_in"], t, _snap))
    T_in = float(_resolve(bc_config["T_gas_in"], t, _snap))
    y_raw = _resolve(bc_config["y_gas_in"], t, _snap)
    y_in  = np.asarray(y_raw, dtype=float).reshape(-1)

    # v_out callable (p.ej. perfil temporal de apertura de válvula):
    v_out_cfg     = bc_config.get("v_out")
    v_out_resolved = float(_resolve(v_out_cfg, t, _snap)) if v_out_cfg is not None else ...

    # Cv callable (Cv controlado por estado):
    Cv_cfg     = bc_config.get("Cv")
    Cv_resolved = float(_resolve(Cv_cfg, t, _snap)) if Cv_cfg is not None else None
    ...
```

La función `_eval_gas_inlet` del gasificador es el ejemplo canónico:

```python
def _eval_gas_inlet(bc_config: dict, t: float, n_comp: int, snap: dict):
    v_in = float(_resolve(bc_config["v_gas_in"], t, snap))
    T_in = float(_resolve(bc_config["T_gas_in"], t, snap))
    y_raw = _resolve(bc_config["y_gas_in"], t, snap)
    y_in  = np.asarray(y_raw, dtype=float).reshape(-1)
    if len(y_in) != n_comp:
        raise ValueError(...)
    return v_in, T_in, y_in
```

---

### 3. RHS — construcción del snap y resolución del thermal BC

En `src/solvers/rhs/rhs_<equipo>.py`, tres cambios en este orden:

#### 3a. Import al inicio del módulo

```python
from src.utils.signals import resolve_config_values as _resolve_cfg
```

#### 3b. Al final del paso 2 (después de desempaquetar el estado)

Construir `_snap_rhs` desde el estado actual y calcular el snap efectivo.
La prioridad de los tres orígenes es:

```python
# Snap del equipo (para señales de retroalimentación de estado)
_snap_rhs = {
    "t":         t,
    "Tg":        Tg_arr,           # (N,) [K]
    "Ts":        Ts_arr,           # (N,) [K]  — si el equipo tiene sólido
    "Tg_out":    float(Tg_arr[-1]),
    "Ts_mean":   float(np.mean(Ts_arr)),
    "P_bar":     P_bar,            # (N,) [bar]
    "P_out_bar": float(P_bar[-1]),
    "C":         C_mat,            # (nc, N)
    "rho_solid": rho_solid,        # (3, N) — si aplica
    "rho_bio":   rho_biomass,
    "rho_char":  rho_char,
    "rho_moi":   rho_moisture,
}
# Prioridad: coordinador de planta > runner > RHS interno
snap = params.get("_snap_external") or params.get("_snap_runner") or _snap_rhs
```

Los campos del snap son los que tiene sentido exponer como variables observables del equipo. Para un equipo sin sólido (heater, reactor de gas), omitir `Ts`, `rho_solid`, etc.

#### 3c. En el paso 3 (contornos) — pasar snap al evaluador BC

```python
bc = get_<equipo>_boundary(
    t=t, ...,
    snap=snap,          # ← añadir
    ...
)
```

#### 3d. Antes del bloque de energía — resolver thermal BC una vez

```python
# Resolver callables en thermal_bc_cfg ANTES del bloque de energía y del paso de pared.
# T_wall, Qwall, T_ambi, h_ambi pueden ser señales temporales o de control.
_tbc = _resolve_cfg(
    thermal_bc_cfg, t, snap,
    keys=["T_wall", "Qwall", "T_ambi", "h_ambi"],
)
```

Luego usar `_tbc` en lugar de `thermal_bc_cfg` en todas las llamadas:

```python
# wall_heat_flux (paso de energía del gas):
qwall_vol, _, _ = wall_heat_flux(
    Tg=Tg_arr, h_wall=h_wall_arr,
    thermal_bc_config=_tbc,    # ← _tbc, no thermal_bc_cfg
    N=nn, Ai=Ai, Pi=Pi, Po=Po, dz=dz,
)

# wall_exterior_q (paso 11, ODE de pared shell-tube):
Q_ext_cell = wall_exterior_q(
    Tw_arr=Tw_arr, thermal_bc_cfg=_tbc,    # ← _tbc
    k_w_arr=k_w_arr, Pi=Pi, Po=Po, dz=dz, N=nn,
)
```

---

## Por qué `_tbc` se calcula FUERA del bloque `if not energy`

Si `_tbc` se calculara dentro del `else: # energy=True`, no estaría disponible para el paso 11 (ODE de pared), que corre independientemente del flag `energy`. Calcularlo una vez antes del bloque de energía garantiza que tanto la energía del gas como la ODE de pared usen exactamente la misma resolución de la señal para ese instante `t`.

---

## Por qué los controladores son stateless

Los controladores de `src/control/controllers.py` (`onoff`, `proportional`) son funciones puras de `(t, snap)` sin ninguna variable de estado interna. Esto es obligatorio porque:

1. El solucionador BDF llama al RHS múltiples veces con el **mismo t** durante la estimación del Jacobiano (perturbaciones de estado). Un controlador con integrador acumularía error espuriamente.
2. La integral del PID debe vivir fuera del ODE, en el runner (entre n_sec sub-intervalos).

Para PID en el runner:

```python
# En runner_<equipo>.run_step — nivel del runner, NO dentro del RHS:
_integral = [0.0]
_t_prev   = [0.0]

def pid_ctrl(t, snap):
    error  = setpoint - sensor_fn(snap)
    dt     = t - _t_prev[0]
    _integral[0] += error * dt
    _t_prev[0]    = t
    return bias + Kp*error + Ki*_integral[0]

params["bc_config"]["T_wall"] = pid_ctrl   # callable(t, snap)
```

---

## Los tres orígenes del snap — cuándo usar cada uno

| Origen | Cómo activar | Cuándo usar |
|--------|-------------|-------------|
| `_snap_rhs` (por defecto) | Siempre disponible | Equipo standalone; control por retroalimentación dentro del mismo paso del solver |
| `params["_snap_runner"]` | El runner lo inyecta antes de llamar al integrador | Cuando el usuario quiere que el controlador opere a la frecuencia del runner, no del Jacobiano |
| `params["_snap_external"]` | El coordinador de planta lo inyecta | Redes multi-equipo; el snap contiene variables de otros equipos |

El RHS no sabe cuál está activo — solo hace `snap = params.get("_snap_external") or params.get("_snap_runner") or _snap_rhs`. El código del equipo no cambia al escalar de equipo standalone a red de equipos.

---

## Uso desde un notebook (ejemplos de los 3 tipos)

```python
from src.control.signals import ramp, step, pulse, piecewise
from src.control.controllers import proportional, onoff
from src.units.gasifier.config.thermal_bc import build_thermal_bc_config

# ── Tipo 1: constante (sin cambios) ───────────────────────────────────────────
tbc = build_thermal_bc_config(mode="fixed_twall", ..., T_wall=1073.15)

# ── Tipo 2: perfil temporal ────────────────────────────────────────────────────
T_wall_ramp = ramp(t_start=0.0, slope=1.0, value_init=700.0, value_max=1073.15)
tbc = build_thermal_bc_config(mode="fixed_twall", ..., T_wall=T_wall_ramp)

# Rampa escalon:
T_wall_step = step(t_step=600.0, value_before=700.0, value_after=1073.15)

# Perfil segmentado:
T_wall_pw = piecewise([0, 300, 600, 3600], [700, 900, 1073.15, 1073.15])

# ── Tipo 3: retroalimentación de estado ───────────────────────────────────────
# Controlador P: mantener Ts_mean a 900 K ajustando T_wall
T_wall_ctrl = proportional(
    setpoint=900.0,
    gain=5.0,
    channel_in=lambda snap: snap.get("Ts_mean", 900.0),
    output_bias=1073.15,
    output_min=700.0,
    output_max=1300.0,
)
tbc = build_thermal_bc_config(mode="fixed_twall", ..., T_wall=T_wall_ctrl)

# Controlador on/off: caudal de aire según temperatura del sólido
from src.units.gasifier.config.boundary_c import build_bc_config
v_ctrl = onoff(
    setpoint=850.0, band=50.0,
    channel_in=lambda snap: snap.get("Ts_mean", 800.0),
    output_on=0.05, output_off=0.0,
)
bc = build_bc_config(n_comp=9, v_gas_in=v_ctrl, ...)
```

---

## Checklist de implementación para un equipo nuevo

- [ ] `boundary_c.py` / `thermal_bc.py`: cambiar `_opt_float` → `_opt_val`; actualizar `_check_pos` para saltarse callables
- [ ] `boundary_c.py` / `thermal_bc.py`: `Qwall` y otros arrays aceptan `callable` antes del `np.asarray()`
- [ ] `<equipo>_boundary.py`: añadir parámetro `snap: dict | None = None`; usar `_resolve()` en todos los BC de inlet/outlet
- [ ] `rhs_<equipo>.py`: import `resolve_config_values as _resolve_cfg`
- [ ] `rhs_<equipo>.py` paso 2: construir `_snap_rhs` con los campos observables del equipo
- [ ] `rhs_<equipo>.py` paso 2: calcular `snap = _snap_external or _snap_runner or _snap_rhs`
- [ ] `rhs_<equipo>.py` paso 3: pasar `snap=snap` a `get_<equipo>_boundary()`
- [ ] `rhs_<equipo>.py` antes del bloque de energía: `_tbc = _resolve_cfg(thermal_bc_cfg, t, snap, keys=[...])`
- [ ] `rhs_<equipo>.py`: usar `_tbc` en `wall_heat_flux()` y `wall_exterior_q()` (no `thermal_bc_cfg`)
- [ ] Test básico: verificar que `resolve(T_wall_ramp, t=300.0)` devuelve el valor correcto
- [ ] Test con equipo: ejecutar una simulación con `T_wall=ramp(...)` y comprobar que los balances siguen cerrando

**Fichero de referencia canónico:** `src/solvers/rhs/rhs_gasifier.py` — buscar el comentario `# Snap del equipo` en paso 2 y el comentario `# Resolver callables en thermal_bc_cfg` en paso 10.
