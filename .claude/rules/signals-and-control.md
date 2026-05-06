# Señales de contorno, control y optimización — especificación del framework

> **Estado:** especificación aprobada — pendiente de implementación en `dev/gasifier`
> **Equipo de referencia:** gasificador — primera implementación; patrón obligatorio para todos los equipos futuros
> **Scope del documento:** arquitectura, contratos de interfaz y restricciones de diseño

---

## 1. Visión del sistema

ProSimNet evoluciona en tres fases:

| Fase | Estado | Descripción |
|------|--------|-------------|
| **Equipos** | En curso | Desarrollo individual: gasificador, heater, adsorbedor, reactor... |
| **Señales y control** | Especificado | BC reconfigurables → perfiles temporales → feedback de estado → controladores |
| **Redes y optimización** | Futuro | Varios equipos conectados, optimización multivariable, análisis de sensibilidad |

Este documento cubre las fases 2 y 3 como especificación técnica vinculante.

---

## 2. Jerarquía de condiciones de contorno (BC)

Cualquier BC de cualquier equipo puede ser de uno de tres tipos:

| Tipo | Firma | Ejemplo de uso |
|------|-------|----------------|
| **Constante** | `float` (o `ndarray`) | Operación nominal fija: `T_wall = 1073.15` |
| **Perfil temporal** | `callable(t) → float` | Rampa de temperatura, pulso de caudal |
| **Feedback de estado** | `callable(t, snap) → float` | Control on/off, proporcional, supervisorio |

### 2.1 Función `resolve` — mecanismo universal

```python
import inspect

def resolve(signal, t: float, snap: dict | None = None):
    """
    Resolve a BC signal to its current value.

    Parameters
    ----------
    signal : float | callable(t) | callable(t, snap)
        The BC definition.
    t : float
        Current time [s].
    snap : dict | None
        State snapshot from the previous runner step (only passed to
        callables with 2 parameters).

    Returns
    -------
    float | ndarray
        Resolved value in SI units.
    """
    if callable(signal):
        sig = inspect.signature(signal)
        if len(sig.parameters) == 1:
            return signal(t)          # perfil temporal
        else:
            return signal(t, snap)    # feedback de estado
    return signal                     # constante
```

**Ubicación:** `src/utils/signals.py` — módulo de utils común, no de control.
Todos los `build_bc_config` de todos los equipos llaman a `resolve` en el runner, nunca dentro del RHS.

### 2.2 Validación en `build_bc_config`

```python
# Correcto: aceptar float | callable, rechazar cualquier otra cosa
if not (isinstance(v_gas_in, (int, float, np.ndarray)) or callable(v_gas_in)):
    raise ValueError(f"v_gas_in must be float or callable, got {type(v_gas_in)}")
```

La validación ocurre en el `build_*_config`, nunca en el RHS.
Tipos inválidos fallan rápido, antes de la integración.

---

## 3. BC reconfigurables por equipo (scope inicial: gasificador)

### Gas inlet (ya soportan `callable(t)` — patrón establecido)

| BC | Tipo actual | Tipo objetivo |
|----|------------|---------------|
| `v_gas_in` | `float | callable(t)` | ✓ ya implementado |
| `T_gas_in` | `float | callable(t)` | ✓ ya implementado |
| `y_gas_in` | `ndarray | callable(t)` | ✓ ya implementado |

### Outlet (a implementar en gasificador)

| BC | Tipo objetivo | Descripción |
|----|--------------|-------------|
| `v_out` | `float | callable(t) | callable(t, snap)` | Control de caudal de salida |
| `Cv` | `float | callable(t, snap)` | Coeficiente válvula ISA; exclusivo con `v_out > 0` |

### Thermal BC (a implementar en gasificador)

| BC | Modo | Tipo objetivo |
|----|------|---------------|
| `T_wall` | `fixed_twall` | `float | callable(t) | callable(t, snap)` |
| `Q_wall` | `heatfluxwall` | `float | callable(t)` |
| `h_amb` | `ambient_htc` | `float | callable(t)` |

---

## 4. El snapshot de estado (`snap`)

Cuando un BC usa feedback de estado, el runner construye un snapshot con las variables observables del equipo al **final del paso anterior** (lag de un paso — evita stiffness en el Jacobiano del BDF).

```python
snap = {
    "t":         float,           # tiempo [s]
    "P_bar":     ndarray(N,),     # presión en celdas [bar]
    "Tg_K":      ndarray(N,),     # temperatura gas [K]
    "Ts_K":      ndarray(N,),     # temperatura sólido [K]
    "y_gas":     ndarray(nc, N),  # fracciones molares [-]
    "rho_solid": ndarray(3, N),   # densidades sólidas [kg/m³_bed]
    "v_out":     float,           # velocidad de salida actual [m/s]
    "v_in":      float,           # velocidad de entrada actual [m/s]
}
```

**Reglas del snap:**
- Se construye **una vez por paso del runner**, nunca dentro del Jacobiano del BDF
- Solo lo reciben callables con firma `(t, snap)` — los `callable(t)` no lo reciben
- En redes multi-equipo, el snap se extiende con variables de otros equipos — fuera de scope en la fase actual
- El runner lo pasa a `resolve(signal, t, snap)` y a los controladores; el RHS no lo ve nunca

---

## 5. Señales primitivas — `src/control/signals.py`

Todas devuelven `callable(t) → float` listas para usar como BC.

```python
def ramp(t0: float, t1: float, v0: float, v1: float) -> callable:
    """Rampa lineal de v0 a v1 entre t0 y t1. Constante fuera del rango."""

def step(t_step: float, v_before: float, v_after: float) -> callable:
    """Escalón en t_step."""

def pulse(t_start: float, t_end: float, v_on: float, v_off: float = 0.0) -> callable:
    """Pulso rectangular encendido en [t_start, t_end]."""

def sine(amplitude: float, frequency: float, offset: float = 0.0) -> callable:
    """Señal sinusoidal: offset + amplitude * sin(2π * frequency * t)."""

def piecewise(points: list[tuple[float, float]]) -> callable:
    """Perfil general a tramos: [(t0, v0), (t1, v1), ...]. Interpolación lineal."""
```

**Convenio de composición:**

```python
# Las señales son componibles porque devuelven callables:
from src.control.signals import ramp, step

# Rampa de arranque seguida de operación nominal
v_in = lambda t: ramp(0, 300, 0, 0.05)(t) + step(300, 0, 0)(t)

# O simplemente piecewise:
v_in = piecewise([(0, 0.0), (300, 0.05), (3600, 0.05)])
```

---

## 6. Controladores — `src/control/controllers.py`

Todos devuelven `callable(t, snap) → float`.

```python
def on_off(
    sensor_fn: callable,      # callable(snap) → float — qué mide
    setpoint: float,
    band: float,
    v_on: float,
    v_off: float,
) -> callable:
    """
    Controlador on/off con banda muerta.
    Activa v_on cuando sensor < setpoint - band/2.
    Activa v_off cuando sensor > setpoint + band/2.
    """

def proportional(
    sensor_fn: callable,
    setpoint: float,
    gain: float,
    v_bias: float = 0.0,
    v_min: float = 0.0,
    v_max: float = float("inf"),
) -> callable:
    """
    Controlador proporcional: v = v_bias + gain * (setpoint - sensor).
    Saturado en [v_min, v_max].
    """

def pid(
    sensor_fn: callable,
    setpoint: float,
    Kp: float, Ki: float, Kd: float,
    v_min: float = 0.0,
    v_max: float = float("inf"),
) -> callable:
    """
    Controlador PID.
    IMPORTANTE: el integrador (∫error dt) lo mantiene el runner vía closure,
    NO como variable de estado del ODE. Ver sección 7.
    """
```

### Convención `sensor_fn`

```python
# Sensor: temperatura del gas en la celda de salida (z = L)
sensor_Tg_out = lambda snap: snap["Tg_K"][-1]

# Sensor: presión máxima del lecho
sensor_P_max  = lambda snap: float(np.max(snap["P_bar"]))

# Sensor: conversión total de char (densidad media)
sensor_char   = lambda snap: float(np.mean(snap["rho_solid"][1]))

# Uso con controlador proporcional:
T_wall_ctrl = proportional(
    sensor_fn = sensor_Tg_out,
    setpoint  = 1073.15,    # [K]
    gain      = 50.0,       # [K_wall / K_error]
    v_bias    = 1073.15,
    v_min     = 400.0,
    v_max     = 1500.0,
)
# T_wall_ctrl es callable(t, snap) → float
```

---

## 7. Estado de controladores y el runner

Los controladores **no tienen estado propio persistente** entre pasos del solver.
Si un controlador necesita memoria (integral del PID, estado del filtro), el runner la mantiene externamente mediante closure:

```python
# El runner crea el estado del integrador en su scope
_integral_error = [0.0]     # lista mutable — accesible por closure
_t_prev         = [0.0]

def pid_with_state(t: float, snap: dict) -> float:
    error = setpoint - sensor_fn(snap)
    dt    = t - _t_prev[0]
    _integral_error[0] += error * dt
    _t_prev[0] = t
    return v_bias + Kp * error + Ki * _integral_error[0]

# pid_with_state es callable(t, snap) — se pasa como BC normal
params["bc_config"]["T_wall"] = pid_with_state
```

Esta estrategia evita añadir variables de control al vector de estado del ODE, que complicaría el Jacobiano del BDF.

---

## 8. Optimización y análisis de sensibilidad — `src/control/optimization.py`

Interfaz entre `scipy.optimize` y `run_step`. Permite:
- **Barridos paramétricos:** barrer ER, T_wall, caudal de sólido → obtener LHV, CGE, T_max
- **Optimización de setpoints:** maximizar LHV, minimizar tar, minimizar consumo energético
- **Análisis de sensibilidad:** cuantificar impacto de parámetros (±10% en Ea → ?% en composición)

```python
def parametric_sweep(
    base_params: dict,
    sweep_vars: dict[str, list],   # {"v_gas_in": [0.05, 0.08, 0.10], "T_wall": [900, 1000]}
    objective_fn: callable,        # callable(col) → dict  — extrae métricas del resultado
    solver_config: dict,
    n_jobs: int = 1,               # paralelismo
) -> pd.DataFrame:
    """Run cartesian product of sweep_vars and return metrics DataFrame."""

def optimize_bc(
    base_params: dict,
    decision_vars: dict[str, tuple[float, float]],  # BC → (min, max)
    objective_fn: callable,        # callable(col) → float (a minimizar)
    solver_config: dict,
    method: str = "L-BFGS-B",
) -> OptimizeResult:
    """Optimize BC values to minimize/maximize a scalar objective."""

def sensitivity_analysis(
    base_params: dict,
    param_path: str,               # e.g. "fuel_config.kinetics.pyrolysis.E"
    delta_pct: float = 0.10,       # ±10% perturbación
    objective_fn: callable,
    solver_config: dict,
) -> dict:
    """One-at-a-time sensitivity. Returns % change in objective per % change in param."""
```

---

## 9. Herramientas de prueba — `tools/`

```
tools/
├── kinetics/              ← ajuste de curvas TGA/FBR → parámetros fuel YAML
│   ├── tga_fitter.py      fit isoconversional (KAS, OFW, Starink) → Ea, A
│   ├── fbr_fitter.py      fit datos FBR → yields de pirólisis
│   └── notebooks/
├── validation/            ← ejecutar case_card y comparar contra datos de artículo
│   ├── case_runner.py     carga caso, simula, devuelve col
│   ├── comparator.py      T(z), y_i(z) vs. datos experimentales, RMSE, R²
│   └── notebooks/
│       └── val_A0_gasifier.ipynb
├── benchmarks/            ← rendimiento numérico y sensibilidad de malla
│   ├── grid_sensitivity.py
│   ├── tolerance_sweep.py
│   └── notebooks/
└── campaigns/             ← barridos paramétricos y optimización
    └── notebooks/
```

**Relación con `src/control/`:**
- `tools/` contiene notebooks ejecutables y scripts de alto nivel
- `src/control/` contiene las funciones importables (signals, controllers, optimization)
- `tools/campaigns/` usa `src/control/optimization.parametric_sweep` — no reescribe lógica

---

## 10. Redes de equipos — arquitectura futura (no implementar aún)

En la fase de redes, ProSimNet conecta varios equipos:

```
Gasifier → [syngas] → HeatExchanger → [gas caliente] → Boiler
                ↘ [char] → Combustor
```

El mecanismo de señales se extiende naturalmente:
- El snap de cada equipo incluye variables de otros equipos del mismo paso
- Los controladores pueden actuar sobre BCs de equipos aguas abajo
- La orquestación de la red vive en `src/networks/` (futuro)

**Restricción:** No anticipar esta arquitectura en el código actual de equipos individuales. La interfaz del snap es suficiente como punto de extensión futuro.

---

## 11. Restricciones de diseño — resumen obligatorio

| Restricción | Razón |
|-------------|-------|
| `resolve()` solo se llama en el runner, nunca en el RHS | Evita overhead en el Jacobiano del BDF (millones de llamadas) |
| Snap evaluado una vez por paso del runner | Lag de un paso evita stiffness artificial por feedback instantáneo |
| Controladores sin estado propio — el runner mantiene el integrador | El vector de estado del ODE no se mezcla con lógica de control |
| Validación de tipos en `build_bc_config`, no en el RHS | Fallo rápido y claro; el RHS es hot path |
| Los callables devuelven siempre valores en SI | Sin conversiones de unidades ocultas dentro de señales |
| `src/control/` es librería pura — no importa de `src/units/<equipo>/` | Capa de control por encima de los equipos, no acoplada |

---

## 12. Rol del gasificador como equipo de referencia

El gasificador (`dev/gasifier`) es el **primer equipo en implementar** la arquitectura de señales reconfigurables. Al hacerlo, establece:

1. El patrón de llamada a `resolve()` en el runner
2. La construcción del snap al final de cada paso
3. La validación de BC en `build_bc_config`
4. Los tests de señales en `test/gasifier/`

Todos los equipos futuros seguirán este patrón. La documentación de referencia para implementadores es este fichero + el código del gasificador una vez completado.
