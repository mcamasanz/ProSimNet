# Próximos pasos — al cerrar VS Code

## 1. Renombrar la carpeta del proyecto (PENDIENTE)

La carpeta de memoria de Claude ya está renombrada a `ProSimNet`.
Solo falta renombrar la carpeta del disco, que no se pudo hacer con VS Code abierto.

**Opción A — Windows Explorer:**
1. Cierra VS Code completamente
2. Navega a: `C:\Users\MiguelCamaraSanz\OneDrive - Fundacion CIRCE\GITHUB\`
3. Renombra `GasifierSimNet` → `ProSimNet`
4. Reabre VS Code y abre la carpeta `ProSimNet`

**Opción B — PowerShell (fuera de VS Code):**
```powershell
Rename-Item "C:\Users\MiguelCamaraSanz\OneDrive - Fundacion CIRCE\GITHUB\GasifierSimNet" "ProSimNet"
```

---

## 2. Tras el renombrado — continuar con el gasificador

### Física pendiente de implementar (por orden de prioridad):

**2.1 Convección sólida en balance de energía**
- Archivo: `src/solvers/rhs/rhs_gasifier.py` — `core_rhs()` paso 10
- Falta: término `−div(vs · Cp_s · Ts)` en `dTsdt` cuando `vs ≠ 0`
- Aplica a modos: `updraft` y `conveyor`
- Referencia: `.claude/physics/energy-balances.md`

**2.2 Reacciones homogéneas en fase gas**
- Archivo nuevo a crear: `src/physics/reactions/homogeneous_reactions.py`
- Reacciones: WGS (`CO + H₂O ⇌ CO₂ + H₂`), tar cracking
- Referencia: `.claude/physics/reactions.md` (sección "Reacciones homogéneas pendiente")

**2.3 Notebooks de simulación**
- `test/test_gasifier_01_batch.ipynb` — modo batch: drying + pyrolysis
- `test/test_gasifier_02_cstr.ipynb` — modo CSTR con flujo de gas

**2.4 Deprecar modo `fixed` en `build_gas_prop_config`**
- Archivo: `src/units/gasifier/config/gas_props.py`
- El modo `"fixed"` (propiedades suministradas por el usuario) es redundante con
  `"constant"` y genera mantenimiento innecesario. Eliminarlo cuando no queden
  referencias activas en tests o notebooks.
- Acción: buscar `mode="fixed"` en codebase antes de eliminar.

**2.5 Validación experimental**
- Referencia: `manual/modelo1dcinetico.pdf`
- Objetivo: comparar perfiles Tg, Ts, composición del syngas con datos del paper

---

## 3. Visión a largo plazo — ProSimNet

Este repositorio está modelando el gasificador como primer equipo reactivo.
La herramienta final **ProSimNet (Process Simulation Network)** incluirá:

| Equipo | Estado |
|--------|--------|
| Calentador 1D (Heater) | ✅ Validado |
| Adsorbedor 1D (PSA/TSA/VSA) | ✅ Validado |
| Gasificador 1D | 🔄 En desarrollo |
| Reactor tubular 1D (futuro) | ⬜ Pendiente |
| Intercambiador de calor (futuro) | ⬜ Pendiente |
| Red de equipos conectados (futuro) | ⬜ Pendiente |

---

## 4. Comandos útiles al retomar el trabajo

```
/new-equipment   → plantilla para modelar un equipo nuevo
/check-rhs       → auditoría de cualquier RHS (checklist 12 pasos)
/physics-rules   → referencia rápida de reglas físicas
```

Documentación en `.claude/`:
- `equipment/gasifier.md` — estado del gasificador, archivos, claves params
- `physics/cross-phase.md` — regla q_masstransfer (no olvidar en nuevas rxns)
- `physics/reactions.md` — plantilla para añadir una nueva reacción
