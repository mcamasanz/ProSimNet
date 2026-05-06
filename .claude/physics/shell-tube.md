# Modelo de pared dinámica (shell-tube)

## Cuándo se activa

```python
shell_tube = params.get("wall_config") is not None
```

Si `wall_config` está presente → `shell_tube = True` → la pared tiene su propia ODE.

---

## Impacto en el vector de estado

```
Sin shell_tube:  sv = [..., Hg(N), Ts(N)]              tamaño base
Con shell_tube:  sv = [..., Hg(N), Ts(N), Tw(N)]       tamaño base + N
```

`Tw` se añade **siempre al final** del vector de estado.

---

## Impacto en pack/unpack (state.py)

```python
def unpack_state_vector(sv, params):
    shell_tube = params.get("wall_config") is not None
    # ... desempaquetar C, rho_s, Hg, Ts como siempre ...
    if shell_tube:
        Tw = sv[idx: idx + nn]
    else:
        Tw = None
    return {"C": C, "Hg": Hg, "Ts": Ts, "Tw": Tw, ...}
```

---

## Impacto en el RHS paso 10 — q_wall en el gas

```python
if shell_tube:
    # Tw es una variable del estado → q_wall depende del Tw dinámico
    qwall_vol = h_wall_arr * (Pi / Ai) * (Tw_arr - Tg_arr)   # (N,) [W/m³_bed]
else:
    # Tw no existe → q_wall calculado desde thermal_bc_config
    qwall_vol, _, _ = wall_heat_flux(
        Tg=Tg_arr, h_wall=h_wall_arr,
        thermal_bc_config=thermal_bc_cfg,
        N=nn, Ai=Ai, Pi=Pi, Po=Po, dz=dz,
    )
```

---

## Impacto en el RHS paso 11 — ODE de pared

```python
if shell_tube:
    A_w      = float(wall_config["A_w"])
    mat      = wall_config["material"]
    rho_w    = eval_solid_property(mat["rho"], Tw_arr)   # (N,)
    cp_w     = eval_solid_property(mat["cp"],  Tw_arr)   # (N,)
    k_w      = eval_solid_property(mat["k"],   Tw_arr)   # (N,)

    Q_gw_cell  = h_wall_arr * Pi * dz * (Tg_arr - Tw_arr)  # [W/celda] gas→pared
    Q_ext_cell = wall_exterior_q(Tw_arr, thermal_bc_cfg, k_w, Pi, Po, dz, nn)
    Q_ax_cell  = wall_axial_q(Tw_arr, k_w, A_w, dz, nn)

    dTwdt = (Q_gw_cell - Q_ext_cell + Q_ax_cell) / (rho_w * cp_w * A_w * dz)
```

---

## Compatibilidad con thermal_bc_config

**NINGUNA combinación está prohibida.** Todas son físicamente válidas:

| thermal_bc mode | Sin shell_tube | Con shell_tube |
|-----------------|---------------|----------------|
| `"adiabatic"` | Tg no pierde calor a pared | Tw evoluciona libre (Q_ext=0) |
| `"fixed_twall"` | T_wall prescribe temperatura de la cara **interior** del tubo | T_wall prescribe temperatura de la cara **exterior** (To); Tw interior es dinámica |
| `"heatfluxwall"` | Flujo prescrito Q_wall [W] entra directo al gas | Flujo prescrito sale por la cara exterior de la pared |
| `"ambient_htc"` | Resistencias en serie (h_int + R_cond + h_ext) aplicadas a Tg | Resistencias aplicadas entre Tw y T_ambiente |

**Regla de validación:** en `_validate_<equipo>_params()`, el bloque de wall_config
solo debe verificar que las claves obligatorias están presentes (`material`, `A_w`,
`Di`, `Do`, `T_w_init`). Nunca cruzar con thermal_bc_mode.

---

## Impacto en state_extraction.py

```python
shell_tube = params.get("wall_config") is not None

Tw_hist = np.zeros((n_t, nn)) if shell_tube else None

# En el loop de reconstrucción:
if shell_tube:
    Tw_hist[k] = sv[idx_Tw: idx_Tw + nn]

result = SimpleNamespace(
    ...
    _Tw_results = Tw_hist,   # ndarray(n_t, N) o None
)
```

---

## Impacto en energy_balance (postprocessing)

```python
if shell_tube:
    # Q_wall al gas desde la cara interior de la pared (usa h_wall y Tw)
    dT_gw     = Tw - Tg                                          # (n_t, N)
    Qwall_dot = h_wall_val * Pi * dz * np.sum(dT_gw, axis=1)    # (n_t,) [W]
    Q_wall_total = np.trapz(Qwall_dot / Ai, t)                  # [J/m²]
else:
    # Q_wall calculado desde wall_heat_flux para cada instante
    ...
```

---

## wall_config — estructura requerida

```python
wall_config = build_wall_config(
    material  = "SS316L",      # nombre en soliddb.txt
    Di        = 0.05,          # diámetro interior de la pared [m]
    Do        = 0.06,          # diámetro exterior de la pared [m]
    T_w_init  = 300.0,         # temperatura inicial de la pared [K]
    # A_w calculado internamente: π/4 * (Do² - Di²)
)
# wall_config contiene: material(dict con rho,cp,k), A_w, Di, Do, T_w_init
```

---

## Checklist shell_tube

```
□ state.py: unpack extrae Tw si shell_tube; pack incluye Tw al final
□ RHS paso 10: qwall_vol usa h_wall*(Pi/Ai)*(Tw-Tg) en lugar de wall_heat_flux
□ RHS paso 11: ODE dTwdt presente y concatenada al dydt final
□ state_extraction.py: _Tw_results es ndarray o None según shell_tube
□ runner: _validate_params solo verifica claves, nunca combina con thermal_bc_mode
□ energy_balance: rama shell_tube usa Tw histórico para Q_wall
□ test notebook: verifica las 4 combinaciones thermal_bc + shell_tube=True
```
