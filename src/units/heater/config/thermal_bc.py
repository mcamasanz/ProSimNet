"""
@module heater.config.thermal_bc
@brief Validación del bloque de condición de contorno térmica exterior del heater.

@details
Re-exporta build_thermal_bc_config desde el módulo del adsorbedor.
Los cuatro modos de contorno térmico (adiabatic, ambient_htc, fixed_twall,
heatfluxwall) son idénticos para ambos equipos: la función no depende de
ningún parámetro específico de lecho empaquetado ni de adsorción.

Cuando se centralice este builder en un módulo src/config/ compartido,
este fichero deberá actualizarse para importar desde allí.
"""

from src.units.adsorber.config.thermal_bc import build_thermal_bc_config  # noqa: F401
