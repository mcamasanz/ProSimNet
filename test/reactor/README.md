# Tests — Reactor Tubular 1D

| Archivo | Tipo | Descripción |
|---------|------|-------------|
| `test_reactor_00_config_survey.ipynb` | Tutorial | Verifica todos los builders de config sin integrar. Caso de referencia: NH₃ (N₂+H₂, lecho catalítico de hierro, nc=3, N=10). |
| `test_reactor_01_cstr_nh3.ipynb` | Tutorial | Primera integración completa: CSTR (N=1) catalítico adiabático a 10 bar, síntesis NH₃, verificación de balances con `check_balances`. |

## Notas

- Equipo genérico: tubo vacío o lecho catalítico, reacciones configurables vía `reactions_config`.
- Primer caso de uso: síntesis de NH₃ con calentamiento por inducción electromagnética.
- Referencia de arquitectura: `.claude/equipment/reactor.md`.
