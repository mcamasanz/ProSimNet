# Equipos auxiliares futuros — Bombas, ventiladores, compresores

## Principio de diseño

Los equipos auxiliares NO tienen vector de estado propio (no se integran en el ODE).
Son funciones puras que devuelven una velocidad superficial o un caudal dado
el estado termodinámico en sus caras y sus parámetros de diseño.

Se ubican en `src/boundary_conditions/` porque su rol es prescribir condiciones de contorno.

---

## Plantilla genérica de auxiliar de flujo

```python
# src/boundary_conditions/<auxiliar>.py

def <auxiliar>_superficial_velocity(
    operating_point: dict,   # parámetros del equipo (curva bomba, caudal nominal, etc.)
    fluid_state_in:  dict,   # {"P_bar": float, "T_K": float, "rho": float, "MW": float}
    fluid_state_out: dict,   # ídem en la cara de salida
    Ai:              float,  # área interna [m²]
) -> float:
    """
    Calcula la velocidad superficial [m/s] impuesta por el auxiliar.

    Para bombas/ventiladores: v = Q_vol / Ai  donde Q_vol viene de la curva del equipo.
    Para compresores: depende del caudal másico y la densidad en la cara de descarga.
    """
    ...
    return v_superficial   # [m/s]  positivo = flujo en dirección positiva z
```

---

## Ventilador / soplante (fan, blower)

```
Modelo simplificado: caudal volumétrico prescrito (operación en punto fijo)
o curva característica P-Q lineal.

Parámetros en operating_point:
  "Q_nom"     : float   [m³/s]  caudal nominal a condiciones de referencia
  "P_ref"     : float   [Pa]    presión de referencia
  "T_ref"     : float   [K]     temperatura de referencia
  "curve"     : callable o None  P_static(Q) → Pa  (curva característica)

Física:
  Q_actual = Q_nom * (rho_ref / rho_actual)  # corrección por densidad
  v = Q_actual / Ai
```

---

## Bomba (líquido o gas comprimido)

```
Modelo: curva H-Q parabólica o caudal prescrito.

Parámetros en operating_point:
  "Q_nom"     : float   [m³/s]
  "H_nom"     : float   [m]     altura nominal (líquido) o ΔP [Pa]
  "eta"       : float   [-]     eficiencia hidráulica
  "curve"     : callable o None  H(Q)

Física para líquido incompresible:
  ΔP = rho * g * H_nom * eta
  v  = Q_nom / Ai
```

---

## Compresor

```
Modelo isentrópico o politrópico con eficiencia.

Parámetros en operating_point:
  "P_ratio"   : float   [-]    relación de compresión P_out / P_in
  "eta_is"    : float   [-]    eficiencia isentrópica
  "Q_in_nom"  : float   [m³/s] caudal de aspiración nominal

Física (gas ideal, proceso isentrópico):
  T_out_is = T_in * P_ratio^((gamma-1)/gamma)
  T_out    = T_in + (T_out_is - T_in) / eta_is
  Q_out    = Q_in * (P_in / P_out) * (T_out / T_in)
  v        = Q_out / Ai
```

---

## Integración en bc_config

Cualquier auxiliar se integra en `bc_config` como fuente de condición de contorno:

```python
bc_config = {
    "mode": "continuous",
    "inlet": {
        "type": "fan",                      # tipo de auxiliar
        "operating_point": {...},           # parámetros del equipo
    },
    "outlet": {"type": "pressure", "P_bar": 1.01325},
}

# En get_<equipo>_boundary():
if bc["inlet"]["type"] == "fan":
    v_in = fan_superficial_velocity(
        operating_point = bc["inlet"]["operating_point"],
        fluid_state_in  = {"P_bar": P_cell[0], "T_K": Tg_cell[0], "rho": rho_g[0]},
        fluid_state_out = {...},
        Ai              = params["Ai"],
    )
```

---

## Checklist para añadir un auxiliar nuevo

```
□ Función pura en src/boundary_conditions/<auxiliar>.py
□ Firma: (operating_point, fluid_state_in, fluid_state_out, Ai) → float
□ Devuelve v_superficial [m/s], positivo en dirección +z
□ Sin estado interno (no modifica params, no tiene caché)
□ Documentar el modelo físico usado (norma, correlación, curva)
□ Añadir a equipment/future-auxiliaries.md
□ Añadir entrada en equipment/common.md si es genéricamente reutilizable
```
