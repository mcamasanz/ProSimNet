"""
@module nusselt
@brief Correlaciones de Nusselt para convección forzada, natural y mixta en tubo.

@details
Funciones puras que reciben números adimensionales ya calculados y devuelven
el coeficiente de transferencia de calor. No dependen de ningún equipo
específico y pueden usarse en adsorbers, heaters, reactores tubulares, etc.

Correlaciones implementadas:
    h_wall_tube   — convección forzada (Dittus-Boelter / Nu laminar) combinada
                    con convección natural (Churchill-Chu 1975) mediante la
                    regla de potencias de convección mixta (Churchill 1977).
                    Si Ra=None → solo convección forzada (comportamiento previo).

    compute_Ra_Di — calcula el número de Rayleigh basado en el diámetro interno,
                    a partir de las propiedades del gas a temperatura de película.

Convención de signo de Ra:
    Ra ≥ 0. Se calcula con |ΔT| = |Tw − Tg|. Cuando ΔT = 0, Ra = 0 y Nu_NC = 0.

Unidades: SI.
    Re     [-]
    Pr     [-]
    Ra     [-]
    D      [m]
    k_film [W/m/K]
    h_wall [W/m²/K]
"""

from __future__ import annotations

import numpy as np


_G_GRAV = 9.80665   # m/s² — aceleración de la gravedad estándar


def compute_Ra_Di(
    Tg: np.ndarray,
    Tw: np.ndarray,
    rho_Tg: np.ndarray,
    mu_film: np.ndarray,
    k_film: np.ndarray,
    Cp_film: np.ndarray,
    Di: float,
) -> np.ndarray:
    """
    Número de Rayleigh basado en el diámetro interno Di.

    Usa la aproximación de Boussinesq con β = 1/T_film (gas ideal) y
    propiedades de película para μ, k, cp. La densidad se corrige desde
    la densidad bulk (a Tg) a la temperatura de película: ρ_film = ρ_Tg·Tg/T_film.

    Parameters
    ----------
    Tg      : ndarray(N,) — temperatura del gas [K]
    Tw      : ndarray(N,) — temperatura de la pared [K]
    rho_Tg  : ndarray(N,) — densidad del gas a temperatura Tg [kg/m³]
    mu_film : ndarray(N,) — viscosidad dinámica a T_film [Pa·s]
    k_film  : ndarray(N,) — conductividad térmica a T_film [W/m/K]
    Cp_film : ndarray(N,) — calor específico másico a T_film [J/kg/K]
    Di      : float       — diámetro interno del tubo [m]

    Returns
    -------
    Ra : ndarray(N,) — número de Rayleigh [-], Ra ≥ 0
    """
    Tg_a  = np.asarray(Tg,      dtype=float).reshape(-1)
    Tw_a  = np.asarray(Tw,      dtype=float).reshape(-1)
    rho_a = np.asarray(rho_Tg,  dtype=float).reshape(-1)
    mu_a  = np.asarray(mu_film,  dtype=float).reshape(-1)
    k_a   = np.asarray(k_film,   dtype=float).reshape(-1)
    Cp_a  = np.asarray(Cp_film,  dtype=float).reshape(-1)

    T_film   = 0.5 * (Tg_a + Tw_a)
    dT       = np.abs(Tw_a - Tg_a)

    # Densidad a temperatura de película (gas ideal: ρ ∝ 1/T a presión constante)
    rho_film = rho_a * Tg_a / np.maximum(T_film, 1.0e-3)

    # ν = μ/ρ [m²/s],  α = k/(ρ·cp) [m²/s],  β = 1/T_film [K⁻¹] para gas ideal
    nu    = mu_a / np.maximum(rho_film, 1.0e-30)
    alpha = k_a  / np.maximum(rho_film * Cp_a, 1.0e-30)
    beta  = 1.0  / np.maximum(T_film, 1.0e-3)

    return _G_GRAV * beta * dT * float(Di)**3 / np.maximum(nu * alpha, 1.0e-60)


def h_wall_tube(
    Re: np.ndarray,
    Pr: np.ndarray,
    D: float,
    k_film: np.ndarray,
    Ra: np.ndarray | None = None,
) -> np.ndarray:
    """
    Coeficiente de transferencia de calor gas-pared para flujo en tubo.

    Convección forzada:
        Nu_F = 3.66           (laminar, Re < 2300)
        Nu_F = 0.023 Re^0.8 Pr^0.4  (Dittus-Boelter, turbulento)

    Convección natural (solo si Ra se suministra):
        Correlación de Churchill-Chu (1975), válida para todos los Ra:

            Nu_NC = [0.825 + 0.387·Ra^(1/6) / f(Pr)]²
            f(Pr) = [1 + (0.492/Pr)^(9/16)]^(8/27)

    Convección mixta (Churchill 1977), n = 3:
        Nu = (Nu_F³ + Nu_NC³)^(1/3)

    Si Ra=None se devuelve Nu_F (comportamiento anterior — compatible con código existente).

    Parameters
    ----------
    Re     : ndarray(N,)        — Reynolds basado en diámetro de tubo [-]
    Pr     : ndarray(N,)        — Prandtl de la mezcla gaseosa [-]
    D      : float              — diámetro interno del tubo [m]
    k_film : ndarray(N,)        — conductividad térmica a T_film [W/m/K]
    Ra     : ndarray(N,) | None — Rayleigh basado en D [-]; None → solo forzada

    Returns
    -------
    h_wall : ndarray(N,) — HTC gas-pared [W/m²/K]

    Notes
    -----
    - Re, Pr y k_film deben evaluarse a T_film = 0.5*(Tg+Tw) cuando Tw está disponible.
    - Ra debe calcularse con `compute_Ra_Di` para garantizar consistencia de propiedades.
    - La transición laminar-turbulento (Re_crit = 2300) no afecta a Nu_NC.
    - Ra = 0 (ΔT = 0) → Nu_NC = 0; el resultado es Nu = Nu_F.
    """
    Re_arr = np.asarray(Re,     dtype=float).reshape(-1)
    Pr_arr = np.asarray(Pr,     dtype=float).reshape(-1)
    k_arr  = np.asarray(k_film, dtype=float).reshape(-1)

    # Convección forzada
    Nu_F = np.where(Re_arr < 2300.0,
                    3.66,
                    0.023 * Re_arr**0.8 * Pr_arr**0.4)

    if Ra is None:
        Nu = Nu_F
    else:
        Ra_arr = np.asarray(Ra, dtype=float).reshape(-1)
        Ra_pos = np.maximum(Ra_arr, 0.0)

        # Churchill-Chu (1975) — convección natural, superficie vertical, todos los Ra
        _f_Pr  = (1.0 + (0.492 / np.maximum(Pr_arr, 1.0e-10))**(9.0 / 16.0))**(8.0 / 27.0)
        Nu_NC  = (0.825 + 0.387 * Ra_pos**(1.0 / 6.0) / np.maximum(_f_Pr, 1.0e-30))**2

        # Convección mixta: regla de potencias n=3 (Churchill 1977)
        Nu = (Nu_F**3 + Nu_NC**3)**(1.0 / 3.0)

    return Nu * k_arr / float(D)
