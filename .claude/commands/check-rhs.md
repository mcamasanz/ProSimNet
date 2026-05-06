# /check-rhs — Auditoría del RHS de un equipo

Cuando se invoque este comando, leer el archivo RHS indicado (o el RHS del equipo
activo si no se especifica) y verificar cada punto de la checklist. Reportar con
✓ (correcto), ✗ (falta/error), ⚠ (presente pero revisar).

## Instrucciones

1. Lee el archivo RHS a auditar (el usuario lo indicará o busca en `src/solvers/rhs/`)
2. Para cada sección de la checklist, indica el número de línea del código relevante
3. Si detectas un problema, explica qué falta y cómo corregirlo con código concreto
4. Al final, genera un resumen de hallazgos críticos vs. avisos menores

---

## Checklist — 10 pasos estructurales

### Paso 1 — Lectura de params
- [ ] `nc`, `nn`, `dz`, `epsi`/`epsi_r`, `Ai`, `Di`, `Pi`, `Po` extraídos al inicio
- [ ] `mode` leído de `bc_config` (batch, CSTR, updraft, conveyor, ads, purge, etc.)
- [ ] `energy` leído como `bool(params.get("energy", True))`
- [ ] Sin cálculo físico en este paso (solo asignaciones)

### Paso 2 — Desempaquetado del vector de estado
- [ ] Usa `unpack_state_vector(sv, params)` o descompone manualmente con índices documentados
- [ ] `shell_tube` detectado: `shell_tube = params.get("wall_config") is not None`
- [ ] Si shell_tube → extrae `Tw` del sv
- [ ] `Tg` recuperado por Newton: `recover_Tg_from_Hg(C, Hg, prop_gas, nc, epsi, Tg_guess, gas_T_ref)`
- [ ] `Tg_guess` usa warm-start: `cache.get("Tg_last", np.full(nn, 700.0))`
- [ ] `cache["Tg_last"] = Tg.copy()` al final del desempaquetado
- [ ] **CRÍTICO — Clip defensivo ante perturbaciones del Jacobiano BDF:**
      ```python
      C_mat     = np.maximum(state["C"],         0.0)
      rho_solid = np.maximum(state["rho_solid"], 0.0)
      # Recomputar CTOT, Y, P desde C_mat clipado (no usar los del state):
      Ctot_arr = np.sum(C_mat, axis=0)
      y_mat    = C_mat / np.maximum(Ctot_arr, 1e-300)[None, :]
      P_bar    = np.maximum(Ctot_arr * R_GAS * Tg_arr / 1e5, 1e-6)
      ```
      Sin este clip: BDF perturba C→negativo → Ctot<0 → P<0 → rho_g<0 → Re^0.6=NaN → fallo.
- [ ] Presión calculada: `P = Ctot * R_GAS * Tg / 1e5` [bar] (desde C clipado)
- [ ] Fracciones molares: `y = C / np.maximum(Ctot, 1e-300)` shape (nc, N)

### Paso 3 — Condiciones de contorno
- [ ] Llama a `get_<equipo>_boundary(t, P_cell, ..., bc_config)` → dict bc
- [ ] Extrae `v_in`, `v_out` como floats [m/s]
- [ ] Extrae `C_in`, `T_in` (pueden ser None en batch/Neumann)
- [ ] `has_inlet = (C_in is not None and T_in is not None)`
- [ ] Si hay sólido móvil: extrae `vs_signed`, `vs_face`, `rho_s_inlet`

### Paso 4 — Propiedades de mezcla
- [ ] Usa `compute_gas_mixture_properties(P_Pa, Tg, x, prop_gas, ...)` → `rho_g`, `mu_g`, `k_g`
- [ ] x_mat shape (N, nc) como requiere `wilke_mix_property`
- [ ] Modo de actualización respetado: `"frozen"` reutiliza caché, `"always"` recalcula

### Paso 5 — Velocidades y transporte
- [ ] Ergun: `ergun_face_velocity(P_Pa, rho_g, mu_g, epsi, dp, v_in, v_out, dz, N)`
- [ ] Tubo vacío: `continuity_face_velocity(rho_g, v_in, v_out, ...)` o Darcy-Weisbach
- [ ] `v_cell = 0.5 * (v_face[:-1] + v_face[1:])` (N,) [m/s]
- [ ] `compute_transfer_coefficients(...)` → `h_bed`, `h_wall`, `D_disp`
- [ ] Para gasificador: `dp` actualizado con SCM (`particle_diameter`), `a_p` actualizado

### Paso 6 — Cinética / Tasas de reacción
- [ ] Todas las tasas en [mol/m³_BED/s] o [kg/m³_BED/s] — NO en m³_gas
- [ ] Tasas clipadas: `np.maximum(r, 0.0)` cuando procede (tasas irreversibles)
- [ ] Si hay SCM: `dp` dinámico calculado a partir de `rho_char / rho_char0`
- [ ] Si hay LDF: `q_eq = iso_fn(P_partial, Ts)` shape (nc, N)

### Paso 7 — [Conveyor] Cálculo de rho_solid_in
- [ ] Solo presente si `mode == "conveyor"` (condicional explícito)
- [ ] Balance global de masa sólida: `rho_in = rho_out + m_s2g_int / vs_abs`
- [ ] Resultado guardado en `cache["rho_solid_in_conveyor"]`

### Paso 8 — Balance de especies gaseosas
- [ ] Loop sobre especies con índice correcto
- [ ] Fuentes de sólido→gas dividen por `epsi_safe`: `source_gas[i] += src_bed[i] / epsi_safe`
- [ ] `convective_flux(phi_cell=C_mat[i], v_face=v_face, phi_in=C_in_i, ...)` shape (N+1,)
- [ ] `diffusive_flux(...)` solo si `D_disp_mat is not None`
- [ ] `dCdt[i] = -(F_tot[1:] - F_tot[:-1]) / dz + source_gas[i]`
- [ ] Clip en cero si `C[i] < 0` y `dCdt[i] < 0` (evitar valores negativos de concentración)

### Paso 9 — [Sólido] Balance de carga / densidades
**Si hay adsorción (LDF):**
- [ ] `dqdt[i] = k_mtc[i] * (q_eq[i] - q[i])` shape (N,)
- [ ] Clip: `dqdt[i] = np.where(q[i] < 0, np.maximum(dqdt[i], 0), dqdt[i])`

**Si hay sólido reactivo:**
- [ ] `d_rho_s[j] = conv_s[j] + src_s[j]` para cada componente sólido
- [ ] **Clip con umbral `_EPS_RHO = 1e-6` (NO 1e-12):**
      `np.where(rho_s[j] < _EPS_RHO, np.maximum(d_rho_s[j], 0), d_rho_s[j])`
      Con 1e-12: a t>1200s BDF necesita Δt<eps×t≈1e-12 s → fallo "step size too small".
- [ ] Transporte convectivo sólido solo si `vs_signed != 0 and nn > 1`

### Paso 10 — Balances de energía

**⚠ Este paso tiene más fuentes de error que todos los demás juntos.**

**Gas (Hg):**
- [ ] `gas_enthalpy_convective_flux(Tg, C, v_face, prop_gas, nc, gas_T_ref, T_in, C_in)` → (N+1,)
- [ ] `gas_diffusive_heat_flux(Tg, k_g, dz, T_in, T_out=None, bc_in, bc_out="neumann")` → (N+1,)
- [ ] `div_h_conv = (Fh[1:] - Fh[:-1]) / dz` shape (N,)
- [ ] `dHgdt = -epsi * div_h_conv - div_diff - q_gs + q_wall`
- [ ] **CRÍTICO:** Si hay transferencia sólido→gas:
      `h_i_Ts = calc_species_enthalpy(Ts_arr, prop_gas, nc, gas_T_ref)` shape (nc, N)
      `q_masstransfer = epsi * np.sum(source_gas * h_i_Ts, axis=0)` shape (N,)
      `dHgdt += q_masstransfer`
- [ ] Sign check: `q_gs_vol = h_bed * a_p * (Tg - Ts)` → positivo cuando gas calienta sólido
- [ ] q_wall con shell_tube: `h_wall * Pi/Ai * (Tw - Tg)` | sin shell_tube: `wall_heat_flux(...)`

**Sólido (Ts) — si existe:**
- [ ] `Cs_vol = Σⱼ rho_s_j * Cp_fns[j](Ts)` → clipeado en mínimo (evitar /0)
- [ ] **CRÍTICO — Corrección de masa con integral de Cp (sólido reactivo):**
      La identidad exacta es `d(Σ_j ρⱼ·H_j(Ts))/dt = Q_rxn + q_gs`
      donde `H_j(Ts) = ∫_{T_ref}^{Ts} Cp_j dT` (integral, NO `Cp_j(Ts)·Ts`).
      Usar `Cp_j(Ts)·Ts` en lugar de `H_j(Ts)` produce un error del 20-30%
      en el balance del sólido cuando Cp es polinomial en T.
      ```python
      H0 = np.asarray(h_fns[0](Ts_arr), float)   # ∫Cp0 dT desde T_ref  [J/kg]
      H1 = np.asarray(h_fns[1](Ts_arr), float)
      H2 = np.asarray(h_fns[2](Ts_arr), float)
      thermal_correction = -(H0*(src_s[0]+conv_s[0])
                            + H1*(src_s[1]+conv_s[1])
                            + H2*(src_s[2]+conv_s[2]))   # (N,) [J/m³_bed/s]
      dTsdt = (Q_rxn_vol + q_gs_vol + thermal_correction) / Cs_vol
      ```
      Prerequisito: `h_fns` deben construirse junto con `Cp_fns` en el lector
      de combustible (ver `fuels_reader.py::_make_h_fn`).
- [ ] **CRÍTICO — Máscara `solid_present`:**
      Cuando el sólido se agota (`rho_total < _EPS_RHO`), `a_p → ∞` y `Cs_vol → 0`
      → `dTs/dt ~ 1e10 K/s` → rigidez extrema → fallo BDF.
      ```python
      solid_present = (rho_biomass + rho_char + rho_moisture) > _EPS_RHO
      q_gs_vol  = np.where(solid_present, q_gs_vol,  0.0)
      dTsdt_arr = np.where(solid_present,
                           (Q_rxn_vol + q_gs_vol + thermal_correction) / Cs_vol,
                           0.0)
      ```
- [ ] Q_rxn_vol sign: positivo = calor al sólido (exotérmico en el sólido)
- [ ] q_gs_vol sign: positivo cuando gas > sólido (gas cede calor al sólido)

**Pared (Tw) — si shell_tube:**
- [ ] ODE completa: `dTwdt = wall_ode_rhs(Tg, Tw, h_wall, wall_config, thermal_bc_config, ...)`
- [ ] Incluida en dydt final

**Acumuladores energéticos (SIEMPRE, para cierres exactos):**
- [ ] `dQ_mt_acc_dt = q_masstransfer` (o `np.zeros(nn)` si `energy=False`)
- [ ] **`dQ_rxn_acc_dt = Q_rxn_vol`  — SIN thermal_mass_correction** (o `np.zeros(nn)` si `energy=False`)
      La corrección térmica es INTERNA al ODE (hace `d(Cs·Ts)/dt = Q_rxn + q_gs`);
      incluirla en el acumulador duplicaría el término y rompería el cierre del sólido:
      `dHs − Q_gs − Q_rxn_acc = −∫thermal_correction dt ≠ 0`
- [ ] Con estos acumuladores: `dHs − Q_gs − Q_rxn_acc = 0 ★`
      y el global `(dHg + dHs) − (Fh_neto + Q_wall + Q_rxn + Q_mt) ≈ 0 ★`

### Paso 11 — [Shell-tube] ODE de pared
- [ ] Presente solo si `shell_tube = True`
- [ ] `wall_ode_rhs` importado de `src.physics.thermal.wall_ode`
- [ ] `dTwdt` shape (N,)

### Paso 12 — Empaquetado del RHS
- [ ] Orden de concatenación IDÉNTICO al layout del sv documentado en `state.py`
- [ ] Shapes correctos antes de concatenar (ravel para matrices, mantener (N,) para vectores)
- [ ] Layout obligatorio:
      `[dCdt(nc·N), d_rho_s(n_s·N), dHgdt(N), dTsdt(N) [, dTwdt(N)], dQ_mt_acc(N), dQ_rxn_acc(N)]`
- [ ] **Acumuladores SIEMPRE AL FINAL, después de Tw** — si se insertan antes, el desempaquetado
      de `state.py` extrae Ts donde esperaba Tw, etc. → fallo silencioso.
- [ ] Shape final = shape del sv de entrada (verificar con `assert dydt.shape == sv.shape`)

---

## Checklist — Convenciones globales

- [ ] Sin validaciones dentro del RHS (deben estar en el runner)
- [ ] Sin imports dentro de la función (todos al nivel de módulo)
- [ ] Sin print/logging dentro del RHS (bajo flag debug únicamente)
- [ ] Todas las unidades en SI (Pa, K, m, m/s, mol/m³, J/m³, kg/m³)
- [ ] `R_GAS = 8.31446261815324` definido como constante del módulo
- [ ] `_IDX` dict de índices de especies definido como constante del módulo
- [ ] Caché: solo leer, nunca crear estructuras nuevas sin `setdefault`
- [ ] `@profiled` decorator en `core_rhs`

---

## Resumen de hallazgos

Al terminar la auditoría, generar tabla:

| # | Severidad | Paso | Descripción | Línea | Corrección |
|---|-----------|------|-------------|-------|-----------|
| 1 | CRÍTICO | 10 | Falta q_masstransfer en dHgdt | 475 | Añadir epsi*Σ src*h_i(Ts) |
| 2 | AVISO | 8 | Clip de concentraciones no implementado | 394 | ... |

Categorías: CRÍTICO (error físico), AVISO (inconsistencia potencial), MEJORA (optimización)
