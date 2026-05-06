# Valve — Válvula de control (ISA-75.01)

## Descripción física

Modelo de válvula de control lineal / porcentual según norma ISA-75.01.
Calcula la velocidad superficial del fluido que pasa a través de la válvula
en función de la caída de presión y las condiciones del fluido.

**Casos de uso:** control de flujo en pasos PSA (pressurization, blowdown),
válvulas de purga, reguladores de presión en redes de columnas.

---

## Función principal

```python
# src/boundary_conditions/valve.py
valve_superficial_velocity(
    Cv:       float,       # coeficiente de caudal de la válvula [m³/s/√(Pa)]
    P_up:     float,       # presión aguas arriba [bar]
    P_down:   float,       # presión aguas abajo [bar]
    T:        float,       # temperatura del fluido [K]
    MW_mix:   float,       # masa molar de la mezcla [kg/mol]
    epsi:     float,       # porosidad del lecho (para v superficial)
    Ai:       float,       # área interna del tubo [m²]
) -> float                 # velocidad superficial [m/s]
```

---

## Modelo ISA-75.01

```
Para gas compresible:

  ΔP_cr = 0.5 · P_up   (caída de presión crítica — choke flow)

  Si ΔP < ΔP_cr:  (flujo subcrítico)
    Q_vol = Cv · √(ΔP / (ρ_ref · P_up))   [m³/s a condiciones de referencia]

  Si ΔP ≥ ΔP_cr:  (flujo crítico / choked)
    Q_vol = Cv · √(ΔP_cr / (ρ_ref · P_up))

  v_superficial = Q_vol_actual / Ai
```

---

## Uso en bc_config del adsorbedor

```python
# En get_step_boundary para pasos con válvula:
v_in = valve_superficial_velocity(
    Cv    = bc_config["Cv_feed"],
    P_up  = P_feed_bar,
    P_down = P_cell[0],      # presión en la primera celda
    T     = T_feed,
    MW_mix = MW_mix_feed,
    epsi  = params["epsi"],
    Ai    = params["Ai"],
)
```

---

## Extensión a equipos futuros: bombas y ventiladores

Ver `.claude/equipment/future-auxiliaries.md` para la plantilla de implementación.

La interfaz recomendada para cualquier auxiliar de flujo es:

```python
def <auxiliary>_superficial_velocity(
    operating_point: dict,   # parámetros del punto de operación
    fluid_state:     dict,   # P, T, MW, rho en la cara
    geometry:        dict,   # Ai, Di
) -> float                   # v_superficial [m/s]
```
