"""
@module valve
@brief Velocidad superficial del gas a través de una válvula de control.

@details
Implementa la expresión semi-empírica de válvula que relaciona el coeficiente Cv
(International valve sizing coefficient) con el caudal volumétrico de gas comprimible
y, a partir de él, con la velocidad superficial equivalente en la sección transversal
del lecho.

Fórmula de caudal normalizado:
    Q_n [Nm³/h] = 414.97 · Cv · sqrt(ΔP_bar · P_up_bar / (T_up · Sg))

donde:
    ΔP_bar = (P_up - P_down) / 1e5        [bar]
    P_up_bar = P_up / 1e5                  [bar]
    Sg = MW_mix / MW_air                   [-]   densidad relativa al aire
    MW_air = 28.96e-3 kg/mol

Conversión a caudal real a condiciones locales:
    Q_real [m³/s] = Q_n [Nm³/s] · (T_up / T_n) · (P_n / P_up)
    T_n = 273.15 K,  P_n = 1.01325e5 Pa

Velocidad superficial:
    v [m/s] = Q_real / (epsi · Ai)

La función es positiva cuando P_up > P_down (flujo directo). Si ΔP ≤ 0 o Cv = 0
devuelve 0.

Referencia: ISA-75.01.01 — "Flow Equations for Sizing Control Valves".

Unidades entrada: SI (P en Pa, T en K, MW en kg/mol).
"""

from __future__ import annotations

import numpy as np

# Constantes de referencia fijas
_MW_AIR = 28.96e-3     # masa molar del aire de referencia [kg/mol]
_T_NTP  = 273.15       # temperatura normal [K]
_P_NTP  = 1.01325e5   # presión normal [Pa]

from src.utils.profiling import profiled


@profiled
def valve_superficial_velocity(
    Cv: float,
    P_up_Pa: float,
    P_down_Pa: float,
    T_up: float,
    MW_mix: float,
    epsi: float,
    Ai: float,
) -> float:
    """
    @brief
    Calcula la velocidad superficial del gas a través de una válvula de control.

    @details
    La velocidad resultante es siempre >= 0: representa el módulo del caudal en
    la dirección positiva (P_up → P_down). El signo de dirección lo asigna el
    llamador según el contexto (entrada positiva, salida positiva o negativa).

    Si ΔP = P_up - P_down ≤ 0 o Cv = 0, la función devuelve 0 (sin flujo).

    Parameters
    ----------
    Cv       : float — coeficiente de válvula [-]. Debe ser >= 0.
    P_up_Pa  : float — presión aguas arriba [Pa]. Debe ser > 0.
    P_down_Pa: float — presión aguas abajo [Pa]. Debe ser >= 0.
    T_up     : float — temperatura del gas aguas arriba [K]. Debe ser > 0.
    MW_mix   : float — masa molar media del gas aguas arriba [kg/mol]. Debe ser > 0.
    epsi     : float — porosidad del lecho [-]. Debe estar en (0, 1).
    Ai       : float — área interna de la sección transversal del lecho [m²]. Debe ser > 0.

    Returns
    -------
    v : float — velocidad superficial equivalente [m/s]. Siempre >= 0.

    Notes
    -----
    - La fórmula 414.97 proviene de la ISA-75.01 para gas en condiciones normales.
    - Para Cv muy pequeño o ΔP muy pequeño el caudal puede ser numéricamente cero.
    - No incluye corrección de flujo crítico (choked flow) — válida para
      P_down > P_up / 2 aproximadamente.
    """
    Cv_val       = float(Cv)
    P_up_val     = float(P_up_Pa)
    P_down_val   = float(P_down_Pa)
    T_up_val     = float(T_up)
    MW_mix_val   = float(MW_mix)
    epsi_val     = float(epsi)
    Ai_val       = float(Ai)

    delta_P_Pa = P_up_val - P_down_val
    if delta_P_Pa <= 0.0 or Cv_val == 0.0:
        return 0.0

    # Densidad relativa del gas respecto al aire [-]
    Sg = MW_mix_val / _MW_AIR

    # Presiones en bar
    delta_P_bar = delta_P_Pa / 1.0e5
    P_up_bar    = P_up_val   / 1.0e5

    # Caudal normalizado [Nm³/h] → [Nm³/s]
    Qn_h  = 414.97 * Cv_val * np.sqrt(
        (delta_P_bar * P_up_bar) / max(T_up_val * Sg, 1.0e-30)
    )
    Qn_s  = Qn_h / 3600.0

    # Caudal real a condiciones locales [m³/s]
    Q_real = Qn_s * (T_up_val / _T_NTP) * (_P_NTP / max(P_up_val, 1.0e-30))

    # Velocidad superficial [m/s]
    v = Q_real / max(epsi_val * Ai_val, 1.0e-30)
    return float(v)
