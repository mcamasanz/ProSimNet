"""
@module postprocessing.velocity_reconstruction
@brief Re-exportación de la herramienta de reconstrucción de velocidad.

@details
La implementación reside en `src.units.adsorber.velocity_history` junto al
resto del post-proceso específico del adsorbedor.

Este módulo expone `compute_velocity_history` como herramienta de
postproceso standalone para casos en que se disponga de `y_hist` y `params`
pero no del objeto `col` devuelto por `run_step`.

Para el flujo habitual (run_step → col) la velocidad ya está en
`col._v_results` y no es necesario llamar a esta función.
"""

from src.units.adsorber.velocity_history import compute_velocity_history  # noqa: F401

__all__ = ["compute_velocity_history"]
