# Condiciones de contorno de salida — teoría, derivación e implementación

> Documento generado tras el desarrollo y depuración del modo isobaro del gasificador.
> Aplica a cualquier equipo ProSimNet con salida de gas variable.

---

## 1. Marco de 4 modos de outlet

La salida de gas se controla mediante la clave `v_out` y `Cv` en `bc_config`:

| Modo | `v_out` | `Cv` | Física | P resultado |
|------|---------|------|--------|-------------|
| Sellado | `0.0` | `None` | Sin salida. Gas acumulado. | P libre (sube) |
| Venteo proporcional | `> 0` | `None` | `v_out = max(0, (P−P_out)/P_out) · v_max` | Parcial |
| Isobaro exacto | `None` | `None` | Balance exacto: v_out mantiene P = P_out | P ≈ P_out |
| Válvula ISA-75.01 | `None` | `> 0` | `Q ∝ Cv·√(ΔP·P_up/(T_up·Sg))` | P ≈ P_out |

`v_out` y `Cv` son **mutuamente excluyentes para venteo**. `P_out_bar` siempre es un parámetro fijo requerido.

---

## 2. Modo isobaro exacto — derivación rigurosa

### 2.1 Condición de partida

Gas ideal: `P = Ctot · R · Tg`. Para `dP/dt = 0`:

```
Ctot · dTg/dt + Tg · dCtot/dt = 0
→ dCtot/dt = −(Ctot/Tg) · dTg/dt          ... (*)
```

### 2.2 Balance de especie (para Ctot)

En la celda de salida (0D o cara N en 1D):

```
dCtot/dt = (v_in·Ctot_in − v_out·Ctot) / dz  +  Σ source_gas_i
```

Igualando con (*):

```
v_out·Ctot/dz = v_in·Ctot_in/dz  +  Σ source_gas_i  +  Ctot/Tg · dTg/dt

v_out = F_in/Ctot  +  F_rxn/(ε·Ctot)  +  dz·(dTg/dt)/Tg          ... (★)
```

donde:
- `F_in = v_in · Ctot_in`  [mol/m²/s]
- `F_rxn = ε · dz · Σ source_gas_i`  [mol/m²/s]

### 2.3 Cálculo de dTg/dt — el término que cancela el outlet

Expandiendo `d(Hg)/dt = d(ε · Σ C_i · h_i)/dt`:

```
dHg/dt = ε · Σ dC_i/dt · h_i(Tg)  +  ε · Ctot · Cp_mix · dTg/dt
```

Sustituyendo `dC_i/dt` del balance de especie:

```
ε · Σ dC_i/dt · h_i(Tg) = −ε · (v_out · H_vol − Fh_in) / dz
                          + ε · Σ source_gas_i · h_i(Tg)
```

Al sustituir en la ecuación de energía del RHS:

```
dHg/dt = −ε·(v_out·H_vol − Fh_in)/dz  −  q_gs  +  q_wall  +  q_mt
```

**Los términos convectivos de outlet SE CANCELAN.** Queda solo:

```
ε · Ctot · Cp_mix · dTg/dt = q_wall − q_gs + ε · Σ src_gas · (h_i(Ts) − h_i(Tg))
                            = q_wall − q_gs + q_mt_diff                   ... (**)
```

### 2.4 Fórmula explícita final

Sustituyendo (**) en (★):

```
v_out = F_in / Ctot_target
      + F_rxn / (ε · Ctot_target)
      + dz · (q_wall − q_gs + q_mt_diff) / (ε · Tg · Ctot_target · Cp_mix)
```

donde `Ctot_target = P_out · 1e5 / (R · Tg)` (**NO** Ctot_actual).

### 2.5 Por qué Ctot_target y no Ctot_actual

Usar `Ctot_actual` en el denominador hace que la fórmula calcule v_out para **mantener la presión que hay ahora**, no para llevarla a P_out. Si P > P_out:
- `Ctot_actual > Ctot_target`
- `F_rxn / Ctot_actual < F_rxn / Ctot_target`
- v_out resulta menor → sale menos gas → P sube más → bucle divergente

`Ctot_target` introduce el **efecto restaurador**: si P > P_out, la fórmula prescribe un v_out mayor que el que tendría a P_out → el exceso de gas sale → P baja hacia P_out.

### 2.6 Por qué aparece ε en el denominador

Hg [J/m³_bed] y C [mol/m³_gas] usan volúmenes de referencia distintos. Al derivar v_out del sistema acoplado energía-especie, la conversión entre ambas referencias introduce un factor ε en el denominador de los términos reactivo y térmico.

Verificación numérica: sin ε → v_out calculado = v_out_correcto × ε → P_max sube.

El término de inlet `F_in / Ctot` **no lleva ε** porque el flujo convectivo de entrada ya es superficial y usa la misma referencia que el balance de especie.

---

## 3. Error de derivación frecuente — dTg/dt desde dHgdt

Es tentador usar `dTg/dt ≈ dHgdt / (ε · Ctot · Cp_mix)`. Esta aproximación es **incorrecta** porque:

```
dHgdt = ε · Σ dC_i/dt · h_i(Tg)  +  ε · Ctot · Cp_mix · dTg/dt
```

El primer término incluye el flujo convectivo de salida, que **depende de v_out**. Usarlo introduce una dependencia circular y un término β espurio en la fórmula de v_out. La derivación correcta (§2.3) muestra que ese término cancela con el outlet de la ecuación de energía, dejando solo los términos de transferencia de calor.

---

## 4. Implementación en el RHS (gasificador)

La corrección se aplica **tras el paso 10** (después de calcular dHgdt con v_out_bc provisional):

```python
# ── Tras paso 10 ──────────────────────────────────────────────────────
_Ctot_target = P_out_bar * 1e5 / (R_GAS * Tg_out)   # ← Ctot_target, nunca Ctot_actual

# Cp_mix por diferencia finita de entalpías molares (1 K)
_h_Tg    = calc_species_enthalpy(Tg_arr,       prop_gas, nc, gas_T_ref)[:, -1]
_h_Tg_p1 = calc_species_enthalpy(Tg_arr + 1.0, prop_gas, nc, gas_T_ref)[:, -1]
_Cp_mix  = np.dot(y_out, _h_Tg_p1 - _h_Tg)           # [J/mol/K]

# q_mt_diff = ε · Σ src_gas · (h_i(Ts) − h_i(Tg))    [J/m³_bed/s]
_q_mt_diff = epsi_r * np.dot(source_gas[:, -1], h_i_Ts[:, -1] - _h_Tg)

# dTg/dt solo del calor de pared, gs y cruce de fase (outlet cancelado)
_dTg_num   = qwall_vol[-1] - q_gs_vol[-1] + _q_mt_diff
_v_thermal = dz * _dTg_num / (epsi_r * Tg_out * Ctot_target * Cp_mix)

# v_out explícito — sin β, sin dependencia circular
_v_out_exact = max(0.0,
    F_in_mol / _Ctot_target          # inlet
    + F_rxn  / (epsi_r * _Ctot_target)  # reacciones  ← ε en denominador
    + _v_thermal                         # térmica     ← ε en denominador
)

# Correcciones diferenciales (usan estado actual — flujos reales en la cara)
_delta_v = _v_out_exact - v_out_bc
dCdt_mat[:, -1] -= _delta_v * C_mat[:, -1] / dz
dHgdt_arr[-1]   -= _delta_v * (epsi_r * Ctot_actual * H_mol_out) / dz
```

### Corrección a dHgdt

```
Δ(dHgdt[-1]) = −ε · Δv_out · Ctot_actual · H_mol(Tg) / dz
             = −Δv_out · Hg[-1] / dz
```

Las correcciones usan `Ctot_actual` y `Hg_actual` porque representan el flujo real que sale por la cara, no el flujo objetivo.

---

## 5. Warm-start del BC provisional (paso 3)

El RHS llama a `get_gasifier_boundary` en el **paso 3** antes de calcular source_gas. Esto da `v_out_bc` provisional (lagged). Su calidad solo afecta al tamaño de `delta_v`, no a la exactitud final.

```python
# Caché actualizado tras la corrección exacta (solo en avance nominal, no Jacobiano)
if (t - t_prev) > 1e-6:
    cache["thermal_expansion_flux_last"] = max(0.0, _v_thermal)
    cache["source_total_flux_last"]       = sum(source_gas) * epsi_r * dz
```

La guarda `_dt > 1e-6` evita sobreescribir el caché durante la estimación del Jacobiano (misma t), lo que causaría `_dt = 0 → dTg/dt → ∞`.

---

## 6. Válvula ISA-75.01 como proxy isobaro práctico

La válvula ISA da un control de presión muy efectivo en la práctica:

| Parámetro | Isobaro exacto (`v_out=None`) | ISA grande (`Cv >> 1`) |
|-----------|------------------------------|------------------------|
| P_max / P_out | 1.000 (por diseño) | ≈ 1.001–1.005 |
| Stiffness | Ninguno (v_out explícito) | Bajo (raíz cuadrada suaviza) |
| Complejidad impl. | Alta | Baja |
| Fisicidad | Control ideal | Modelo de válvula real |

**Regla práctica:** para simular operación isobara con un modelo físico de válvula, usar `Cv` suficientemente grande (10–50 para geometrías típicas de reactor de laboratorio).

La válvula ISA no introduce stiffness significativo porque `∂v_out/∂P ∝ 1/√(ΔP)` — finito y acotado cuando ΔP > 0.

---

## 7. Reconstrucción de v_out en post-proceso

El v_out real del modo isobaro **no forma parte del vector de estado**. `state_extraction.py` lo reconstruye en un segundo pase tras el bucle principal:

```python
# Término térmico: de la derivada numérica del perfil Tg almacenado
dTg_dt    = np.gradient(Tg_hist[:, -1], t_arr)
v_thermal = dz * np.maximum(dTg_dt, 0.0) / np.maximum(Tg_hist[:, -1], 1.0)

# Término reactivo: pérdida de masa sólida → gas producido
rho_solid_total = np.sum(rho_s_hist[:, :, -1], axis=1)
drho_dt         = np.gradient(rho_solid_total, t_arr)   # ≤ 0 durante reacciones
src_mass        = np.maximum(0.0, -drho_dt)              # [kg/m³_bed/s]
MW_mean         = np.sum(y_hist_3d[:, :, -1] * MW_arr, axis=1)  # [kg/mol]
Ctot_target     = P_out_Pa / (R_GAS * np.maximum(Tg_hist[:, -1], 1.0))
v_rxn           = src_mass * dz / (MW_mean * epsi_r * Ctot_target)

v_out_hist = np.maximum(0.0, v_in_hist + v_thermal + v_rxn)
```

**Por qué funciona:** toda la masa sólida que se pierde se convierte en gas (no hay otra fase). La derivada numérica captura tanto secado como pirólisis como char oxidation.

---

## 8. Coherencia física de v_out_C ≈ v_out_D

Es físicamente correcto que el modo isobaro exacto (C) y una válvula ISA bien dimensionada (D) den perfiles de v_out prácticamente iguales. La cantidad de gas que necesita salir la determina la física interna (expansión térmica + reacciones), no el mecanismo de salida. Si ambos controlan P ≈ P_out, el flujo molar evacuado debe ser el mismo.

La diferencia de presión residual (caso D: ΔP ≈ 0.006 bar) es la "presión motriz" necesaria para abrir la válvula. No indica un error sino el modelo físico de la válvula.

---

## 9. Stiffness y modo isobaro

El modo isobaro **no introduce stiffness artificial** con la implementación correcta porque:
- v_out_exact se calcula como fórmula explícita (no feedback)
- El Jacobiano ve `∂v_out/∂state` que refleja la física real (no una ganancia artificial)
- `Ctot_target` no depende del estado actual → `∂(F_rxn/Ctot_target)/∂C ≈ 0` (solo via kinetics)

El proportional feedback `v_out = (Ctot_actual - Ctot_target)/Ctot_target · v_relax` SÍ introduce stiffness: eigenvalue λ ≈ −v_relax · Ctot / dz (puede ser −300 s⁻¹), que obliga al solver a pasos muy pequeños.

---

## 10. Checklist para implementar outlet isobaro en un nuevo equipo

- [ ] Usar `Ctot_target = P_out / (R · Tg)` en el denominador, nunca `Ctot_actual`
- [ ] Incluir ε en denominador de término reactivo: `F_rxn / (ε · Ctot_target)`
- [ ] Incluir ε en denominador de término térmico: `dz · Q_net / (ε · Tg · Ctot_target · Cp_mix)`
- [ ] Término inlet sin ε: `F_in / Ctot_target`
- [ ] Calcular Q_net = q_wall − q_gs + q_mt_diff **tras el balance de energía** (paso 10)
- [ ] No usar dHgdt directamente para dTg/dt (ver §3)
- [ ] Correcciones diferenciales con Ctot_actual y Hg_actual (flujos reales en la cara)
- [ ] Guarda `_dt > 1e-6` para actualizar caché (evitar singularidad en Jacobiano)
- [ ] Reconstruir v_out en state_extraction con segundo pase (no es variable de estado)
