"""
@module fluxes
@brief Flujos convectivos y difusivos en mallas 1D de volúmenes finitos para
       la columna PSA.

@details
Este módulo implementa los tipos de flujo numérico que aparecen en los
balances de la columna:

  1. convective_flux              : F_conv = v_face × phi_face   (upwind)
  2. diffusive_flux               : F_diff = -Γ_face × dφ/dz    (centrado)
  3. solid_convective_flux        : wrapper upwind para fase sólida con flujo
                                    en dirección opuesta o igual al gas
  4. gas_enthalpy_convective_flux : F_h = v_face × H_cell  (entalpía del gas)
  5. gas_diffusive_heat_flux      : wrapper de diffusive_flux para Tg y k_g
  6. solid_diffusive_heat_flux    : wrapper de diffusive_flux para Ts y k_s

Convención de signo:
  El flujo positivo en la cara k representa transporte de izquierda a derecha
  (en dirección del eje z, de inlet a outlet).

La contribución neta de los flujos al balance de una celda j es:

    d(phi_j)/dt * dz = F[j] - F[j+1]   + fuentes

  donde F[j] es el flujo en la cara izquierda y F[j+1] en la derecha.
  El signo del término difusivo cumple que F_diff > 0 cuando phi crece
  de derecha a izquierda (gradiente negativo en z).

Todas las funciones son puras: no tienen estado, no modifican los arrays
de entrada.

Unidades: SI. Los flujos tienen las unidades de [phi] × [m/s] para los
convectivos y [phi/m] × [W/m/K] para los difusivos.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from src.discretization.face_reconstruction import upwind_reconstruction, face_property
from src.physics.thermodynamics.enthalpy import calc_species_enthalpy
from src.utils.profiling import profiled


# =========================================================
# 1. Flujo convectivo genérico
# =========================================================

@profiled
def convective_flux(
    phi_cell: np.ndarray,
    v_face: np.ndarray,
    phi_in: Optional[float] = None,
    phi_out: Optional[float] = None,
) -> np.ndarray:
    """
    @brief
    Calcula el flujo convectivo de una propiedad escalar en todas las caras:

        F_conv[k] = v_face[k] × phi_face[k]

    donde phi_face se obtiene por upwind de primer orden.

    @details
    Esta función es el núcleo de la discretización convectiva. Se llama
    para cada componente de la concentración, para la fracción molar y para
    la densidad de entalpía del gas (a través de gas_enthalpy_convective_flux).

    El producto v × phi_upwind garantiza consistencia con el esquema upwind:
    si v > 0, F = v × phi_celda_izquierda; si v < 0, F = v × phi_celda_derecha.

    Parameters
    ----------
    phi_cell : np.ndarray, shape (nn,)
        Propiedad escalar en centros de celda [SI].
    v_face : np.ndarray, shape (nn+1,)
        Velocidad superficial en caras [m/s].
    phi_in : float, opcional
        Valor de phi en el contorno de entrada. Se usa cuando v_face[0] > 0.
        Si None y v_face[0] > 0, se usa phi_cell[0] (extrapolación).
    phi_out : float, opcional
        Valor de phi en el contorno de salida. Se usa cuando v_face[-1] < 0.
        Si None y v_face[-1] < 0, se usa phi_cell[-1] (extrapolación).

    Returns
    -------
    F_conv : np.ndarray, shape (nn+1,)
        Flujo convectivo en caras [SI × m/s].

    Notes
    -----
    - phi_in=None representa un contorno cerrado o de flujo libre saliente;
      en ese caso se extrapola con el primer nodo interior.
    - La función delega la reconstrucción en upwind_reconstruction. El producto
      v × phi se realiza aquí de forma vectorizada.
    """
    phi_cell_arr = np.asarray(phi_cell, dtype=float).reshape(-1)
    v_face_arr   = np.asarray(v_face,   dtype=float).reshape(-1)

    # Resolver valores de contorno efectivos
    phi_in_eff  = phi_in  if phi_in  is not None else phi_cell_arr[0]
    phi_out_eff = phi_out if phi_out is not None else phi_cell_arr[-1]

    phi_face = upwind_reconstruction(
        phi_cell=phi_cell_arr,
        v_face=v_face_arr,
        phi_in=phi_in_eff,
        phi_out=phi_out_eff,
    )
    return v_face_arr * phi_face


# =========================================================
# 2. Flujo difusivo genérico
# =========================================================

@profiled
def diffusive_flux(
    phi_cell: np.ndarray,
    Gamma: np.ndarray,
    dz: float,
    phi_in: Optional[float] = None,
    phi_out: Optional[float] = None,
    bc_in: Literal["dirichlet", "neumann"] = "neumann",
    bc_out: Literal["dirichlet", "neumann"] = "neumann",
    face_method: Literal["arithmetic", "harmonic"] = "arithmetic",
) -> np.ndarray:
    """
    @brief
    Calcula el flujo difusivo de una propiedad escalar en todas las caras:

        F_diff[k] = -Γ_face[k] × (phi[k] - phi[k-1]) / dz

    con condiciones de contorno Dirichlet o Neumann en las caras extremas.

    @details
    Esta es la discretización estándar centrada de orden 2 del término de
    Fourier / Fick. Para una malla uniforme de paso dz:

    Caras interiores (k = 1..nn-1):
        F_diff[k] = -Γ_face[k] × (phi_cell[k] - phi_cell[k-1]) / dz
        Γ_face[k] se interpola de las celdas adyacentes (aritmética o armónica).

    Cara de entrada (cara 0):
        Dirichlet: F_diff[0] = -Γ[0] × (phi_cell[0] - phi_in) / (0.5 dz)
                   El factor 0.5 dz es la distancia entre la cara y el centro
                   de la primera celda.
        Neumann:   F_diff[0] = 0   (flujo nulo en el contorno)

    Cara de salida (cara nn):
        Dirichlet: F_diff[-1] = -Γ[-1] × (phi_out - phi_cell[-1]) / (0.5 dz)
        Neumann:   F_diff[-1] = 0

    Parameters
    ----------
    phi_cell : np.ndarray, shape (nn,)
        Propiedad escalar en centros de celda [SI].
    Gamma : np.ndarray shape (nn,) o escalar
        Coeficiente difusivo por celda [SI]. Debe ser >= 0.
        Si es escalar, se expande a array uniforme.
    dz : float
        Tamaño de celda axial [m]. Malla uniforme.
    phi_in : float, opcional
        Valor impuesto en el contorno de entrada. Requerido si bc_in='dirichlet'.
    phi_out : float, opcional
        Valor impuesto en el contorno de salida. Requerido si bc_out='dirichlet'.
    bc_in : {"dirichlet", "neumann"}
        Condición de contorno en la cara de entrada.
    bc_out : {"dirichlet", "neumann"}
        Condición de contorno en la cara de salida.
    face_method : {"arithmetic", "harmonic"}
        Método de interpolación de Gamma en las caras interiores.

    Returns
    -------
    F_diff : np.ndarray, shape (nn+1,)
        Flujo difusivo en caras [Γ × SI / m]. Para calor: [W/m²].

    Notes
    -----
    - El signo del flujo sigue la convención de Fourier: F = -Γ dφ/dz.
      Un perfil decreciente en z produce F > 0 (flujo de izquierda a derecha).
    - La condición Neumann con flujo nulo es la condición de simetría o pared
      adiabática más común en los contornos de la columna PSA.
    - La interpolación armónica de Gamma en caras es más precisa cuando Gamma
      tiene saltos bruscos (ej. transición entre zonas porosas y libres).

    Side Effects
    ------------
    Ninguno. Función pura.
    """
    phi_cell_arr = np.asarray(phi_cell, dtype=float).reshape(-1)
    nn = phi_cell_arr.size
    dz_val = float(dz)

    if np.isscalar(Gamma):
        Gamma_arr = float(Gamma) * np.ones(nn, dtype=float)
    else:
        Gamma_arr = np.asarray(Gamma, dtype=float).reshape(-1)

    bc_in  = str(bc_in).strip().lower()
    bc_out = str(bc_out).strip().lower()
    face_method = str(face_method).strip().lower()

    F_diff = np.zeros(nn + 1, dtype=float)

    # Cara de entrada
    if bc_in == "dirichlet":
        F_diff[0] = -Gamma_arr[0] * (phi_cell_arr[0] - float(phi_in)) / (0.5 * dz_val)
    # else: F_diff[0] = 0.0  (Neumann, ya es 0 por np.zeros)

    # Caras interiores — vectorizado
    G_L = Gamma_arr[:-1]
    G_R = Gamma_arr[1:]

    if face_method == "harmonic":
        denom = np.maximum(G_L + G_R, 1.0e-300)
        Gamma_face = 2.0 * G_L * G_R / denom
    else:
        Gamma_face = 0.5 * (G_L + G_R)

    dphi = phi_cell_arr[1:] - phi_cell_arr[:-1]     # phi_R - phi_L para cada cara interior
    F_diff[1:nn] = -Gamma_face * dphi / dz_val

    # Cara de salida
    if bc_out == "dirichlet":
        F_diff[-1] = -Gamma_arr[-1] * (float(phi_out) - phi_cell_arr[-1]) / (0.5 * dz_val)
    # else: F_diff[-1] = 0.0  (Neumann)

    return F_diff


# =========================================================
# 3. Flujo convectivo de la fase sólida
# =========================================================

@profiled
def solid_convective_flux(
    rho_cell: np.ndarray,
    vs_face: np.ndarray,
    rho_solid_in: Optional[float],
) -> np.ndarray:
    """
    Upwind convective flux for a solid bulk-density species.

    Unlike gas flow (always positive velocity in the z-direction), the solid
    can flow counter-current to the gas. The Dirichlet BC is applied at the
    correct face based on the sign of vs_face:

        vs > 0  (downdraft / co-current):
            solid inlet at face 0 (left, z=0)  →  phi_in  = rho_solid_in
            solid outlet at face N (right, z=L) →  Neumann (free outflow)

        vs < 0  (updraft / counter-current):
            solid inlet at face N (right, z=L)  →  phi_out = rho_solid_in
            solid outlet at face 0 (left, z=0)  →  Neumann (free outflow)

        vs = 0  (batch / CSTR):
            no solid transport → all face fluxes are zero

    The function delegates upwind reconstruction to convective_flux, which in
    turn calls upwind_reconstruction. That function already handles negative
    velocities correctly: for vs < 0, rho_solid_in is passed as phi_out and
    applied as a Dirichlet condition at face N.

    Parameters
    ----------
    rho_cell     : ndarray(N,)    solid bulk density per cell [kg/m³_bed]
    vs_face      : ndarray(N+1,)  signed solid superficial velocity at faces [m/s]
                   Positive = solid flows in +z direction (downdraft).
                   Negative = solid flows in −z direction (updraft).
    rho_solid_in : float or None  bulk density at the solid inlet face [kg/m³_bed].
                   None → no solid inlet (batch / CSTR); all fluxes zero.

    Returns
    -------
    F_solid : ndarray(N+1,)  solid convective face flux [kg/m³_bed · m/s]

    Notes
    -----
    - For a constant vs (Option 1, first implementation), vs_face is uniform.
    - Future Option 2 (variable vs from solid continuity) will compute vs_face
      cell-by-cell; this function handles it without change.
    - The divergence of F_solid in the RHS:
          d_rho_s_i/dt = -(F_solid[1:] - F_solid[:-1]) / dz + S_s_i
    """
    vs_arr = np.asarray(vs_face, dtype=float).reshape(-1)

    # No solid convection in batch/CSTR or when no solid feed is defined
    if rho_solid_in is None or float(vs_arr[0]) == 0.0:
        return np.zeros(len(vs_arr), dtype=float)

    if vs_arr[0] > 0.0:
        # Downdraft / co-current: inlet at left face (z=0)
        return convective_flux(rho_cell, vs_face,
                               phi_in=float(rho_solid_in), phi_out=None)
    else:
        # Updraft / counter-current: inlet at right face (z=L)
        return convective_flux(rho_cell, vs_face,
                               phi_in=None, phi_out=float(rho_solid_in))


# =========================================================
# 4. Flujo convectivo de entalpía del gas
# =========================================================

@profiled
def gas_enthalpy_convective_flux(
    Tg_cell: np.ndarray,
    C_cell: np.ndarray,
    v_face: np.ndarray,
    prop_gas: dict,
    n_comp: int,
    gas_T_ref: float = 298.15,
    T_in: Optional[float] = None,
    C_in: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    @brief
    Calcula el flujo convectivo de entalpía del gas en todas las caras:

        F_h[k] = v_face[k] × H_cell_face[k]

    donde H_cell = Σ_i C_i(z) × h_i(T_g(z))  [J/m³], reconstruido por upwind.

    @details
    La densidad de entalpía del gas H_cell no incluye la porosidad eps porque
    el factor eps forma parte del balance de la celda, no del flujo en la cara.
    La entalpía del fluido que cruza la cara k es la que corresponde al nodo
    donor (aguas arriba), que sí tiene la información de T y C en esa posición.

    En la cara de entrada (cara 0), si v > 0 y se proporcionan T_in y C_in,
    H_in = Σ_i C_in_i × h_i(T_in) se usa como valor de contorno.

    Parameters
    ----------
    Tg_cell : np.ndarray, shape (nn,)
        Temperatura del gas en centros de celda [K].
    C_cell : np.ndarray, shape (nc, nn)
        Concentración molar de cada especie en centros de celda [mol/m³].
    v_face : np.ndarray, shape (nn+1,)
        Velocidad superficial en caras [m/s].
    prop_gas : dict
        Diccionario de propiedades puras del gas (salida de build_pure_gas_properties).
    n_comp : int
        Número de componentes (nc).
    gas_T_ref : float
        Temperatura de referencia fallback [K].
    T_in : float, opcional
        Temperatura en el contorno de entrada [K]. Requerido si v_face[0] > 0.
    C_in : np.ndarray, shape (nc,), opcional
        Concentración en el contorno de entrada [mol/m³]. Requerido si T_in no es None.

    Returns
    -------
    F_h : np.ndarray, shape (nn+1,)
        Flujo convectivo de entalpía del gas en caras [W/m²].

    Notes
    -----
    - H_cell NO incluye eps. El factor eps aparece en el balance de energía
      del gas (en core_rhs), no en la definición del flujo en la cara.
    - El flujo se calcula como F_h = v × H_upwind. La contribución de
      dispersión axial de calor va por separado en gas_diffusive_heat_flux.
    - T_in y C_in deben suministrarse cuando existe flujo entrante por la
      cara de entrada. Si se omiten y v[0] > 0, se extrapola con H_cell[0]
      (menos preciso).
    """
    Tg_arr = np.asarray(Tg_cell, dtype=float).reshape(-1)
    C_mat  = np.asarray(C_cell,  dtype=float)              # (nc, nn)
    v_face_arr = np.asarray(v_face, dtype=float).reshape(-1)

    # H_cell = Σ_i C_i × h_i(T_g)  shape (nn,)
    h_i = calc_species_enthalpy(Tg_arr, prop_gas, n_comp, gas_T_ref)  # (nc, nn)
    H_cell = np.sum(C_mat * h_i, axis=0)                               # (nn,)

    # H en el contorno de entrada
    if T_in is not None and C_in is not None:
        T_in_val = float(T_in)
        C_in_arr = np.asarray(C_in, dtype=float).reshape(-1)
        h_i_in   = calc_species_enthalpy(
            np.array([T_in_val]), prop_gas, n_comp, gas_T_ref
        )                                                               # (nc, 1)
        H_in = float(np.sum(C_in_arr * h_i_in[:, 0]))
    else:
        H_in = None

    return convective_flux(
        phi_cell=H_cell,
        v_face=v_face_arr,
        phi_in=H_in,
        phi_out=None,
    )


# =========================================================
# 4 & 5. Flujos difusivos de calor (wrappers nombrados)
# =========================================================

@profiled
def gas_diffusive_heat_flux(
    Tg_cell: np.ndarray,
    k_g: np.ndarray,
    dz: float,
    T_in: Optional[float] = None,
    T_out: Optional[float] = None,
    bc_in: Literal["dirichlet", "neumann"] = "neumann",
    bc_out: Literal["dirichlet", "neumann"] = "neumann",
    face_method: Literal["arithmetic", "harmonic"] = "arithmetic",
) -> np.ndarray:
    """
    @brief
    Calcula el flujo difusivo de calor en la fase gaseosa:

        q_diff_g[k] = -k_g_face[k] × dTg/dz[k]   [W/m²]

    @details
    Wrapper semántico de diffusive_flux para el gas. Representa la conducción
    axial de calor en el gas, que aparece en el balance de energía del gas
    cuando la dispersión térmica axial es significativa.

    En muchos modelos PSA este término se desprecia (bc_in=bc_out='neumann',
    flujo nulo en contornos). Se incluye por completitud del balance.

    Parameters
    ----------
    Tg_cell : np.ndarray, shape (nn,)
        Temperatura del gas en centros de celda [K].
    k_g : np.ndarray shape (nn,) o escalar
        Conductividad térmica efectiva del gas por celda [W/m/K].
    dz : float
        Tamaño de celda axial [m].
    T_in, T_out : float, opcional
        Temperaturas impuestas en los contornos [K]. Requeridas si bc='dirichlet'.
    bc_in, bc_out : {"dirichlet", "neumann"}
        Condiciones de contorno.
    face_method : {"arithmetic", "harmonic"}
        Interpolación de k_g en caras interiores.

    Returns
    -------
    q_diff_g : np.ndarray, shape (nn+1,)
        Flujo difusivo de calor del gas en caras [W/m²].
    """
    return diffusive_flux(
        phi_cell=Tg_cell,
        Gamma=k_g,
        dz=dz,
        phi_in=T_in,
        phi_out=T_out,
        bc_in=bc_in,
        bc_out=bc_out,
        face_method=face_method,
    )


@profiled
def solid_diffusive_heat_flux(
    Ts_cell: np.ndarray,
    k_s: np.ndarray,
    dz: float,
    T_in: Optional[float] = None,
    T_out: Optional[float] = None,
    bc_in: Literal["dirichlet", "neumann"] = "neumann",
    bc_out: Literal["dirichlet", "neumann"] = "neumann",
    face_method: Literal["arithmetic", "harmonic"] = "arithmetic",
) -> np.ndarray:
    """
    @brief
    Calcula el flujo difusivo de calor en la fase sólida del adsorbente:

        q_diff_s[k] = -k_s_face[k] × dTs/dz[k]   [W/m²]

    @details
    Wrapper semántico de diffusive_flux para el sólido. Representa la
    conducción axial en el lecho de adsorbente. La conductividad k_s es la
    efectiva del lecho sólido (no de la partícula individual).

    Para lechos empacados, k_s se estima habitualmente como:
        k_s_eff ≈ (1 - eps) * k_partícula

    Parameters
    ----------
    Ts_cell : np.ndarray, shape (nn,)
        Temperatura del sólido en centros de celda [K].
    k_s : np.ndarray shape (nn,) o escalar
        Conductividad térmica efectiva del sólido por celda [W/m/K].
    dz : float
        Tamaño de celda axial [m].
    T_in, T_out : float, opcional
        Temperaturas impuestas en los contornos [K].
    bc_in, bc_out : {"dirichlet", "neumann"}
        Condiciones de contorno.
    face_method : {"arithmetic", "harmonic"}
        Interpolación de k_s en caras interiores.

    Returns
    -------
    q_diff_s : np.ndarray, shape (nn+1,)
        Flujo difusivo de calor del sólido en caras [W/m²].
    """
    return diffusive_flux(
        phi_cell=Ts_cell,
        Gamma=k_s,
        dz=dz,
        phi_in=T_in,
        phi_out=T_out,
        bc_in=bc_in,
        bc_out=bc_out,
        face_method=face_method,
    )
