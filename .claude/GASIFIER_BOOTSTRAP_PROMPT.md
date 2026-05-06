# Prompt de arranque — Proyecto ProSimNet (2025)

> **Uso:** Copia el bloque entre `===` y pégalo al inicio de una nueva conversación
> para dar contexto completo desde el primer mensaje.

---

```
================================================================================
CONTEXTO DE ARRANQUE — PROYECTO ProSimNet
================================================================================

## 1. Equipo y forma de trabajar

Soy ingeniero de proceso desarrollando una herramienta de simulación modular
de equipos industriales (calentadores, columnas de adsorción, gasificadores).
La forma de trabajar es:

1. Primero análisis, luego código. Identificar objetivo, diagnosticar causa,
   proponer estrategia, explicar impacto → implementar.
2. ARCHITECTURE.md (.claude/) y functions.md (.claude/) son las referencias
   obligatorias de arquitectura. Leerlos antes de proponer cualquier cambio.
3. Sin código zombie: sin huérfanos, sin duplicados, sin versiones a medias.
4. Código SI puro, nombres en inglés técnico, comentarios en español cuando
   aporten valor (no triviales).
5. Respuestas directas y técnicas. Sin explicar conceptos básicos de Python
   o de ingeniería química.

---

## 2. Estado actual del proyecto — 3 equipos implementados

### Equipos disponibles:
- **Heater** (calentador 1D, tubo vacío): `rhs_heater.py` + `runner_heater.py`
  sv = [C(nc,N), Hg(N), [Tw(N)]]
  
- **Adsorber** (columna PSA/TSA/VSA): `rhs_adsorption.py` + `runner_adsorption.py`
  sv = [C(nc,N), q(nc,N), Hg(N), Ts(N), [Tw(N)]]
  
- **Gasifier** (reactor biomasa 1D): `rhs_gasifier.py` + `runner_gasifier.py`
  sv = [C(9,N), rho_s(3,N), Hg(N), Ts(N), [Tw(N)]]
  Modos: batch, cstr, updraft, conveyor
  Reacciones: drying, pyrolysis, char combustion/Boudouard/steam gasification

### Física implementada en gasificador:
- 9 especies gas: CO, CO2, H2O, H2, O2, CH4, C2H4, tar, N2
- 3 sólidos: biomasa (ρ_bio), char (ρ_char), humedad (ρ_moi)
- Modelo SCM para char con diámetro de partícula dinámico
- Propiedades del gas: polinomios grado-7 en τ = ΔT/ΔT_range hasta 5000K
- Shell-tube opcional: pared dinámica Tw con ODE propia

### Pendiente de implementar:
- Convección del sólido en balance de energía (término div(vs·Cp_s·Ts))
- Reacciones homogéneas en fase gas (WGS: CO+H2O⇌CO2+H2; tar cracking)
- Validación contra modelo1dcinetico.pdf (referencia experimental)

---

## 3. Reglas físicas críticas (descubiertas en desarrollo)

### Transferencia sólido→gas (CRÍTICA):
Cuando el sólido produce especies gaseosas (drying, pyrolysis, char reactions),
el balance de energía del gas debe incluir el término de entalpía de esas moléculas:
```python
h_i_Ts = calc_species_enthalpy(Ts, prop_gas, nc, gas_T_ref)  # (nc, N)
q_masstransfer = epsi_r * np.sum(source_gas * h_i_Ts, axis=0) # (N,)
dHgdt += q_masstransfer
```
Sin este término, el gas se enfría artificialmente cuando el sólido produce gas.
Ver .claude/commands/physics-rules.md §REGLA 1 para explicación completa.

### Shell-tube:
Ninguna combinación (thermal_bc_mode, shell_tube=True/False) está prohibida.
Con fixed_twall + shell_tube=True: T_wall prescribe temperatura EXTERIOR del tubo.

---

## 4. Comandos disponibles

- `/new-equipment` → plantilla completa para modelar un equipo nuevo
- `/check-rhs` → auditoría de RHS (checklist de 10 pasos + física cross-phase)
- `/physics-rules` → referencia de reglas físicas (cross-phase, shell-tube, signos, unidades)

---

## 5. Archivos de referencia que debes leer al empezar

1. `.claude/ARCHITECTURE.md` — §1-14 (patrones base) + §15-19 (gasificador, 2025)
2. `.claude/functions.md` — catálogo de funciones (actualizar si añades funciones)
3. `.claude/commands/physics-rules.md` — reglas físicas obligatorias
4. `src/solvers/rhs/rhs_gasifier.py` — RHS de referencia más completo del proyecto
5. `src/postprocessing/gasifier_balances.py` — patrón de balances incluyendo energy_balance

================================================================================
```
