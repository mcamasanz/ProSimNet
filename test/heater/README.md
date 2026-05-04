# Tests — Heater

Tests del calentador de tubo 1D con flujo de gas inerte (sin reacciones).
Cubren desde un caso simple de un solo run hasta la matriz completa de modos BC y comparaciones 0D/1D.

| Archivo | Tipo | Descripción |
|---------|------|-------------|
| `test_heater_00_single_run.ipynb` | Tutorial | Simulación transitoria de 300 s de N₂ en tubo de 51 celdas con 5 W de calefacción en pared; incluye verificación de balances y visualización de perfiles axiales y evolución temporal. |
| `test_heater_01_mode.ipynb` | Técnico | Matriz completa de combinaciones modo BC térmico (adiabático / heatfluxwall / fixed_twall / ambient_htc) × con/sin modelo dinámico de pared para el heater 1D. |
| `test_heater_02_0d_1d_cstr_prf.ipynb` | Tutorial | Cuatro modos canónicos del heater (0D/1D × batch/CSTR-PRF): valida comportamiento isocórico, régimen estacionario y cierre de balances energéticos en cada configuración. |
