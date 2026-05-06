# Hidráulica — modelos de velocidad en 1D

> Para la teoría completa de condiciones de contorno de salida (isobaro, válvula ISA,
> venteo proporcional, sellado) ver **[outlet-bc.md](outlet-bc.md)**.

## Cuándo usar cada modelo

| Modelo | Función | Cuándo usar |
|--------|---------|-------------|
| Ergun | `ergun_face_velocity` | Lecho empaquetado (adsorbedor, gasificador) |
| Continuidad | `continuity_face_velocity` | Tubo vacío (heater) |
| Darcy-Weisbach | `darcy_weisbach_face_velocity` | Diagnóstico de caída de presión en tubo |
| Válvula | `valve_superficial_velocity` | Flujo controlado por válvula ISA-75.01 |

---

## Ergun — lecho empaquetado

```python
# src/physics/momentum/ergun.py
v_face = ergun_face_velocity(
    P_Pa:   ndarray(N,),    # presión en centros de celda [Pa]
    rho_g:  ndarray(N,),    # densidad gas [kg/m³]
    mu_g:   ndarray(N,),    # viscosidad dinámica [Pa·s]
    epsi:   float,          # porosidad del lecho [-]
    dp:     ndarray(N,),    # diámetro de partícula por celda [m]  ← puede variar (SCM)
    v_in:   float,          # velocidad superficial en cara 0 [m/s]
    v_out:  float,          # velocidad superficial en cara N [m/s]
    dz:     float,          # longitud de celda [m]
    N:      int,
) -> ndarray(N+1,)          # velocidades superficiales en caras [m/s]
```

**Ecuación de Ergun:**
```
−∂P/∂z = (150·µ·(1−ε)²)/(ε³·dp²) · v  +  (1.75·ρ·(1−ε))/(ε³·dp) · v|v|
```

**Importante para gasificador:** `dp` puede variar celda a celda cuando el char
se consume por SCM. Usar `particle_diameter(rho_char, rho_char0, dp0)` para
obtener `dp` dinámico antes de llamar a Ergun.

---

## Continuidad — tubo vacío

```python
# src/physics/momentum/darcy_weisbach.py
v_face = continuity_face_velocity(
    rho_g:  ndarray(N,),    # densidad gas [kg/m³]
    v_in:   float,          # velocidad en cara 0 [m/s]
    v_out:  float,          # velocidad en cara N [m/s]
    N:      int,
) → ndarray(N+1,)
```

Conserva el flujo másico `ρ·v = const` en cada cara. Apropiado cuando no hay
pérdida de presión por fricción con partículas (heater, zona vacía de reactor).

---

## Velocidad intersticial vs superficial

```python
# Superficial (usada en el modelo): v_face = v_superficial
# Intersticial (velocidad real del fluido):
v_interstitial = v_superficial / epsi

# El modelo trabaja siempre con v_superficial.
# Los flujos convectivos también usan v_superficial:
F_conv = v_face * phi_upwind   # [mol/m²/s] o [J/m²/s]
```

---

## Convección sólida (gasificador modo conveyor)

```python
# src/discretization/fluxes.py
F_solid = solid_convective_flux(
    rho_cell:     ndarray(N,),   # densidad sólida componente j [kg/m³_bed]
    vs_face:      ndarray(N+1,), # velocidad sólida en caras [m/s]  (puede ser cte)
    rho_solid_in: float or None, # condición de contorno aguas arriba
) → ndarray(N+1,)                # flujo convectivo sólido [kg/m²/s]

d_rho_s[j] += -(F_solid[1:] - F_solid[:-1]) / dz   # (N,) [kg/m³_bed/s]
```

**Dirección de vs_signed:** positivo = sólido sube (updraft), negativo = sólido baja.

---

## Presión en el modelo

```python
# Presión recuperada en cada celda (gas ideal):
Ctot  = np.sum(C, axis=0)              # (N,) [mol/m³_gas]
P_bar = Ctot * R_GAS * Tg / 1e5       # (N,) [bar]
P_Pa  = P_bar * 1e5                    # (N,) [Pa]

# La presión NO es una variable del estado.
# Se recupera de C y Tg en cada paso del RHS.
# Esto implica que el modelo es isobaro a escala de celda
# (la presión de contorno se impone vía bc_config).
```

---

## Caída de presión total en el lecho

```python
# Post-proceso (no en el RHS):
dP_total = P_cell[0] - P_cell[-1]   # [bar]  (positivo: más presión en entrada)

# Para gasificador, la caída de presión varía durante la reacción
# porque dp disminuye al consumirse el char (epsi aumenta).
```
