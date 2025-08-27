# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 14:17:30 2025

@author: MiguelCamaraSanz
"""

import numpy as np

R=8.314

# =============================================================================
# 
# =============================================================================

def __pick_T_key__(unit):
    # Columnas usan 'Tg'; tanques suelen usar 'T'
    return "Tg" if hasattr(unit, "_get_StateVar_") and "Tg" in unit._state_cell_vars else "T"

def __port_to_node__(unit, port: str, nodo: int | None = None) -> int:
    """
    bottom -> 0
    top    -> n-1
    side   -> requiere 'nodo'
    Para equipos de 1 nodo (tanques típicos), retorna 0.
    """
    n = getattr(unit, "_nodos", 1) or 1
    port = (port or "").lower()
    if n == 1:
        return 0
    if port == "bottom":
        return 0
    elif port == "top":
        return n - 1
    elif port == "side":
        if nodo is None:
            raise ValueError("Si port='side', debes proporcionar índice 'nodo'.")
        return int(nodo)
    else:
        raise ValueError(f"Port no reconocido: {port!r}")

def __collect_valve_sources__(unit):
    """
    Devuelve S_mol [nodos], S_sp [nodos,nc], S_eng [nodos] a partir de TODAS
    las válvulas conectadas al equipo 'unit' en este instante.
    Convierte Qn (Nm3/h) -> mol/s (T=273.15 K, P=1 atm).
    """
    nodos = getattr(unit, "_nodos", 1) or 1
    nc    = unit._ncomp
    R     = unit._R
    Tref  = 298.15
    Nm3_2_mol = 1.01325e5 / (R * 273.15)  # (Nm3) -> (mol) a CN; luego /3600 -> mol/s
    T_key = __pick_T_key__(unit)

    S_mol = np.zeros(nodos, dtype=float)
    S_sp  = np.zeros((nodos, nc), dtype=float)
    S_eng = np.zeros(nodos, dtype=float)

    t = unit._state_cell_vars['t']

    # ---------- Inlet simple ----------
    Valve_in = unit._conex.get("inlet", None)
    if Valve_in is not None:
        where_in = unit._conex.get("where_inlet", "bottom")
        j_in = __port_to_node__(unit, where_in)
        endTime = Valve_in.logic_params.get("start", 0) + Valve_in.logic_params.get("duration", 0)

        MW_in = float(np.sum(unit._MW * unit._xin))
        Pi = float(unit._get_StateVar_("P", nodo=j_in))
        Qn_in = Valve_in._get_Qn_(t, endTime, unit._Pin, unit._Tin, Pi, MW_in)  # Nm3/h
        Qin_mol_s = Qn_in * Nm3_2_mol / 3600.0
        Valve_in._Qn_log.append(Qn_in); Valve_in._t_log.append(t)

        Cp_in = float(np.dot(unit._xin, unit._cpg))
        S_mol[j_in]      += Qin_mol_s
        S_sp[j_in, :]    += Qin_mol_s * unit._xin
        S_eng[j_in]      += Qin_mol_s * Cp_in * (unit._Tin - Tref)

    # ---------- Outlet simple ----------
    Valve_out = unit._conex.get("outlet", None)
    if Valve_out is not None:
        where_out = unit._conex.get("where_outlet", "top")
        j_out = __port_to_node__(unit, where_out)
        endTime = Valve_out.logic_params.get("start", 0) + Valve_out.logic_params.get("duration", 0)

        Pi = float(unit._get_StateVar_("P", nodo=j_out))
        Ti = float(unit._get_StateVar_(T_key, nodo=j_out))
        xi = np.asarray(unit._get_StateVar_("x", nodo=j_out)).ravel()
        MW_loc = float(np.sum(unit._MW * xi))
        Qn_out = Valve_out._get_Qn_(t, endTime, Pi, Ti, unit._Pout, MW_loc)  # Nm3/h
        Qout_mol_s = Qn_out * Nm3_2_mol / 3600.0
        Valve_out._Qn_log.append(Qn_out); Valve_out._t_log.append(t)

        Cp_loc = float(np.dot(xi, unit._cpg))
        S_mol[j_out]   -= Qout_mol_s
        S_sp[j_out, :] -= Qout_mol_s * xi
        S_eng[j_out]   -= Qout_mol_s * Cp_loc * (Ti - Tref)

    # ---------- Válvulas inter-unidad (top/bottom/side) ----------
    valves_inter = []
    for key in ("valves_top", "valves_bottom", "valves_side"):
        vv = unit._conex.get(key, [])
        if vv: valves_inter.extend(vv)

    for valve in valves_inter:
        unit_A = valve._conex.get("unit_A", None)
        unit_B = valve._conex.get("unit_B", None)
        port_A = valve._conex.get("port_A", None)
        port_B = valve._conex.get("port_B", None)
        nodo_A = valve._conex.get("nodo_A", None)
        nodo_B = valve._conex.get("nodo_B", None)
        endTime = valve.logic_params.get("start", 0) + valve.logic_params.get("duration", 0)

        # Identificar “este” lado y el “otro”
        if unit_A._name == unit._name:
            j_self  = __port_to_node__(unit, port_A, nodo_A)
            P_self  = float(unit._get_StateVar_("P",  nodo=j_self))
            T_self  = float(unit._get_StateVar_(T_key, nodo=j_self))
            x_self  = np.asarray(unit._get_StateVar_("x", nodo=j_self)).ravel()
            MW_self = float(np.sum(unit._MW * x_self))

            # Otro extremo (columna o tanque)
            if hasattr(unit_B, "_get_StateVar_"):
                T_key_B = __pick_T_key__(unit_B)
                j_other = __port_to_node__(unit_B, port_B, nodo_B) if hasattr(unit_B, "_nodos") else None
                P_other = float(unit_B._get_StateVar_("P",  nodo=j_other) if j_other is not None else unit_B._get_StateVar_("P", port=port_B))
                T_other = float(unit_B._get_StateVar_(T_key_B, nodo=j_other) if j_other is not None else unit_B._get_StateVar_(T_key_B, port=port_B))
                x_other = np.asarray(unit_B._get_StateVar_("x",  nodo=j_other) if j_other is not None else unit_B._get_StateVar_("x", port=port_B)).ravel()
                MW_other = float(np.sum(unit_B._MW * x_other))
            else:
                raise ValueError(f"Válvula '{valve._name}': unit_B no expone _get_StateVar_.")
        elif unit_B._name == unit._name:
            j_self  = __port_to_node__(unit, port_B, nodo_B)
            P_self  = float(unit._get_StateVar_("P",  nodo=j_self))
            T_self  = float(unit._get_StateVar_(T_key, nodo=j_self))
            x_self  = np.asarray(unit._get_StateVar_("x", nodo=j_self)).ravel()
            MW_self = float(np.sum(unit._MW * x_self))

            if hasattr(unit_A, "_get_StateVar_"):
                T_key_A = __pick_T_key__(unit_A)
                j_other = __port_to_node__(unit_A, port_A, nodo_A) if hasattr(unit_A, "_nodos") else None
                P_other = float(unit_A._get_StateVar_("P",  nodo=j_other) if j_other is not None else unit_A._get_StateVar_("P", port=port_A))
                T_other = float(unit_A._get_StateVar_(T_key_A, nodo=j_other) if j_other is not None else unit_A._get_StateVar_(T_key_A, port=port_A))
                x_other = np.asarray(unit_A._get_StateVar_("x",  nodo=j_other) if j_other is not None else unit_A._get_StateVar_("x", port=port_A)).ravel()
                MW_other = float(np.sum(unit_A._MW * x_other))
            else:
                raise ValueError(f"Válvula '{valve._name}': unit_A no expone _get_StateVar_.")
        else:
            raise ValueError(f"Válvula interunit '{valve._name}': ninguna punta coincide con '{unit._name}'.")

        # Sentido por ΔP y aportes
        if P_self > P_other:
            # Sale de ESTE nodo
            Qn = valve._get_Qn_(t, endTime, P_self, T_self, P_other, MW_self)  # Nm3/h
            Qin_mol_s = Qn * Nm3_2_mol / 3600.0
            valve._Qn_log.append(Qn); valve._t_log.append(t)
            Cp_self = float(np.dot(x_self, unit._cpg))
            S_mol[j_self]   -= Qin_mol_s
            S_sp[j_self, :] -= Qin_mol_s * x_self
            S_eng[j_self]   -= Qin_mol_s * Cp_self * (T_self - Tref)
        elif P_self < P_other:
            # Entra al ESTE nodo
            Qn = valve._get_Qn_(t, endTime, P_other, T_other, P_self, MW_other)
            Qin_mol_s = Qn * Nm3_2_mol / 3600.0
            valve._Qn_log.append(Qn); valve._t_log.append(t)
            Cp_other = float(np.dot(x_other, unit._cpg))
            S_mol[j_self]   += Qin_mol_s
            S_sp[j_self, :] += Qin_mol_s * x_other
            S_eng[j_self]   += Qin_mol_s * Cp_other * (T_other - Tref)
        else:
            valve._Qn_log.append(0.0); valve._t_log.append(t)

    return S_mol, S_sp, S_eng


# =============================================================================
# 
# =============================================================================
def __wilke_phi_matrix__(prop, MW):
    """
    Construye φ_ij para Wilke/Wassiljewa/Mason–Saxena.
    prop: vector por especie (Pa·s si es μ, W/m/K si es k)
    MW  : vector por especie (kg/mol)
    return: matriz [ncomp, ncomp] con φ_ij
    """
    prop = np.asarray(prop, float)  # μ_i o k_i
    M    = np.asarray(MW,   float)

    ratio_prop = np.sqrt(prop[:, None] / prop[None, :])      # [i,j]
    ratio_M    = (M[None, :] / M[:, None])**0.25             # [i,j]
    denom      = np.sqrt(8.0) * np.sqrt(1.0 + (M[:, None] / M[None, :]))
    phi = ((1.0 + ratio_prop * ratio_M)**2) / denom
    np.fill_diagonal(phi, 1.0)
    return phi

def __wilke_mix__(x, prop, MW=None, phi=None, eps=1e-30):
    """
    Mezcla de Wilke genérica.
    x   : fracciones molares [nodos, ncomp]
    prop: propiedad pura por especie (μ_i o k_i) [ncomp]
    MW  : pesos moleculares [ncomp] (solo si no pasas phi)
    phi : matriz φ_ij opcional [ncomp, ncomp] (si ya la tienes cacheada)
    return: propiedad de mezcla por nodo [nodos]
    """
    x    = np.asarray(x,   float)  # [nodos, ncomp]
    prop = np.asarray(prop,float)  # [ncomp]
    if phi is None:
        if MW is None:
            raise ValueError("wilke_mix: pasa MW o una matriz phi precomputada.")
        phi = __wilke_phi_matrix__(prop, MW)  # [ncomp, ncomp]

    # Φ_i = sum_j x_j φ_ij  → [nodos, ncomp]
    Phi = x @ phi.T

    # μ_mix o k_mix = (sum_i x_i prop_i) / (sum_i x_i Φ_i)
    num = (x * prop[None, :]).sum(axis=1)    # [nodos]
    den = (x * Phi).sum(axis=1)              # [nodos]
    return num / np.maximum(den, eps)

def _mwMix_(x, MW):
    """ MW medio por nodo. x[nodos,ncomp], MW[ncomp] → [nodos] """
    return np.dot(x,MW)

def _rhoMix_(P, T, MWmix,):
    return (P * MWmix) / (R * T)

def _muMix_(x,mu,MW):
    return __wilke_mix__(x, mu, MW)

def _cpMix_(x,Cp):
    return np.dot(x,Cp)

def _kMix_(x,k,MW):
    return __wilke_mix__(x, k, MW)

def _omega_D_(T):
    """
    Integral de colisión para difusión. Tstar = T / eps_ij (eps en K).
    Ajuste clásico (Neufeld/van Pelt) para Ω_D.
    """
    Tstar = np.asarray(T, float)
    return (
        1.06036 / (Tstar**0.15610)
        + 0.19300 * np.exp(-0.47635 * Tstar)
        + 1.03587 * np.exp(-1.52996 * Tstar)
        + 1.76474 * np.exp(-3.89411 * Tstar)
    )

def _DijAll_(T, P, MW, sigma_A, epskB_A):
    """
    T [n], P [n, Pa] -> P_bar, MW [nc, kg/mol], sigma_A [nc, Å], epskB_A [nc, K]
    Devuelve Dij[n, nc, nc] con diag = inf.
    Fórmula: D_ij[m2/s] = 1e-4 * 0.001858*T^1.5 / (P_bar*sigma_ij^2*Omega_D) * sqrt(1/Mi + 1/Mj)
    """
    T   = np.asarray(T, float).reshape(-1)
    P   = np.asarray(P, float).reshape(-1) / 1e5  # bar
    MW  = np.asarray(MW, float).reshape(-1)
    sig = np.asarray(sigma_A, float).reshape(-1)  # Å
    eps = np.asarray(epskB_A, float).reshape(-1)  # K

    n  = T.size
    nc = MW.size
    Dij = np.empty((n, nc, nc), dtype=float)

    # precomputos por pares
    invM = 1.0 / MW
    sqrt_invM_sum = np.sqrt(invM[:, None] + invM[None, :])     # [i,j]
    sigma_ij = 0.5 * (sig[:, None] + sig[None, :])            # Å
    eps_ij   = np.sqrt(eps[:, None] * eps[None, :])           # K

    for k in range(n):
        Tk    = T[k]
        Pbar  = max(P[k], 1e-9)  # evita división por cero
        Tstar = Tk / np.maximum(eps_ij, 1e-12)
        Omega = _omega_D_(Tstar)
        # D_ij en cm2/s → m2/s con *1e-4
        Dij_k = 1e-4 * (0.001858 * Tk**1.5) / (Pbar * (sigma_ij**2) * Omega) * sqrt_invM_sum
        np.fill_diagonal(Dij_k, np.inf)
        Dij[k, :, :] = Dij_k
    return Dij  # [n, nc, nc]

def _DimMix_(x, Dij):
    """
    x [n, nc], Dij [n, nc, nc] -> Dim [n, nc]
    1/Dim_i = sum_{j!=i} x_j / D_ij
    """
    x  = np.asarray(x, float)
    Dij = np.asarray(Dij, float)
    n, nc = x.shape
    Dim = np.empty((n, nc), dtype=float)
    for k in range(n):
        # evitar /inf en diagonal ya puesta
        invD = 1.0 / Dij[k]            # [nc,nc] con 0 en diagonal
        invD[np.eye(nc, dtype=bool)] = 0.0
        Dim[k, :] = 1.0 / np.maximum((x[k, None, :] @ invD.T).ravel(), 1e-30)
    return Dim  # [n, nc]

def _DknMix_(T, MW, r_p,):
    """
    D_Kn,i = (2/3) r_p sqrt(8 R T / (pi M_i))
    T [n], MW [nc], r_p [m] escalar o [n]
    -> [n, nc]
    """
    T  = np.asarray(T, float).reshape(-1)     # [n]
    MW = np.asarray(MW, float).reshape(-1)    # [nc]
    n  = T.size
    nc = MW.size
    rp = np.asarray(r_p, float)
    if rp.ndim == 0: rp = np.full(n, float(rp))
    out = np.empty((n, nc), dtype=float)
    coef = (2.0/3.0) * rp[:, None]
    out[:, :] = coef * np.sqrt(8.0 * R * T[:, None] / (np.pi * MW[None, :]))
    return out

def _DporMix_(Dim, Dkn):
    """
    1/Dpor = 1/Dim + 1/Dkn  → Dpor
    Dim, Dkn: [n, nc]
    """
    Dim = np.asarray(Dim, float)
    Dkn = np.asarray(Dkn, float)
    return 1.0 / np.maximum(1.0/np.maximum(Dim, 1e-30) + 1.0/np.maximum(Dkn, 1e-30), 1e-30)

def _DeffMix_(Dpor, eps_s, tau):
    """
    Deff = (eps_s / tau) * Dpor
    eps_s, tau escalar(es) o [n]
    """
    Dpor = np.asarray(Dpor, float)
    n,_  = Dpor.shape
    eps_s = np.asarray(eps_s, float)
    tau   = np.asarray(tau,   float)
    if eps_s.ndim == 0: eps_s = np.full(n, float(eps_s))
    if tau.ndim   == 0: tau   = np.full(n, float(tau))
    return (eps_s[:, None] / np.maximum(tau[:, None], 1e-30)) * Dpor

def _avg_face_arith_(a):
    """Media aritmética a caras: a[n] -> af[n-1]."""
    a = np.asarray(a, float).ravel()
    return 0.5*(a[:-1] + a[1:])

def _avg_face_harm_(L, R, eps=1e-30):
    """Media armónica para magnitudes difusivas."""
    L = np.asarray(L, float).ravel()
    R = np.asarray(R, float).ravel()
    num = 2.0*L[:-1]*R[1:]
    den = np.maximum(L[:-1] + R[1:], eps)
    return num/den

def _avg_face_harm_matrix_(A, eps=1e-30):
    A = np.asarray(A, float)
    if A.ndim != 2:
        raise ValueError("A debe ser 2D (nodos, especies).")
    invL = 1.0/np.maximum(A[:-1, :], eps)
    invR = 1.0/np.maximum(A[ 1:, :], eps)
    return 1.0/np.maximum(0.5*(invL + invR), eps)

def _ergun_velocity_faces_(P, rho_f, mu_f, eps, d_p, xcenters, g=9.81, eps_min=1e-30):
    """
    Calcula u en caras con Ergun a partir del gradiente de P entre centros.
    Unidades SI: P[Pa], rho[kg/m3], mu[Pa·s], d_p[m], xcenters[m], g[m/s2]
    eps = porosidad del lecho (adimensional). Velocidad superficial (basada
    en sección total).
    
    Ecuación invertida (cara j+1/2):
      -dP/dz |_face  - rho_face * g  =  b * u  + a * |u| u
      a = 1.75 * (1-ε)/(ε^3 d_p) * rho_face
      b = 150 * (1-ε)^2/(ε^3 d_p^2) * mv_face

    Devuelve:
      v_faces      [n-1]
      dPdz_faces   [n-1]  (sin término gravitatorio)
      flow_dir     [n-1]  (+1 si u>=0; -1 si u<0)
    """
    
    P     = np.asarray(P,   float).ravel()
    xc    = np.asarray(xcenters, float).ravel()

    # Distancia entre centros → una por cara
    dz_faces    = np.maximum(np.diff(xc), eps_min)
    dPdz_faces  = (P[1:] - P[:-1]) / dz_faces     # Pa/m

    # Término motor: gP = -(dP/dz) - rho*g (puede ser ±)
    gdrive = -dPdz_faces - rho_f * g   # Pa/m

    # Coeficientes Ergun
    eps3 = max(eps**3, eps_min)
    a = 1.75 * (1.0 - eps) / (eps3 * d_p)     * rho_f             # [kg/m4] → con u^2 da Pa/m
    b = 150.0 * (1.0 - eps)**2 / (eps3 * d_p**2) * mu_f           # [Pa·s/m2] → con u da Pa/m

    # Resolver a*|u|u + b*u = gdrive  suponiendo sign(u) = sign(gdrive)
    s    = np.sign(gdrive)
    Gabs = np.abs(gdrive)
    # Magnitud positiva:
    v = (-b + np.sqrt(b*b + 4.0*a*Gabs)) / (2.0 * np.maximum(a, eps_min))
    # En el límite a→0 (régimen muy viscoso), v ≈ Gabs/b
    v = np.where(a < 1e-30, Gabs/np.maximum(b, eps_min), v)

    v_faces  = s * v
    flow_dir = np.where(v_faces >= 0.0, 1, -1).astype(int)
    return v_faces, dPdz_faces, flow_dir

def _vfaces_to_vcells_(v_faces):
    """
    v_faces: [n-1] → u_cells: [n]
    Borde: copia cara adyacente. Interior: media aritmética.
    """
    v_faces = np.asarray(v_faces, float).ravel()
    nfaces  = v_faces.size
    if nfaces < 1:
        raise ValueError("v_faces debe tener al menos 1 cara.")
    v_cells = np.empty(nfaces + 1, dtype=float)
    v_cells[0]      = v_faces[0]
    v_cells[-1]     = v_faces[-1]
    if nfaces > 1:
        v_cells[1:-1] = 0.5*(v_faces[:-1] + v_faces[1:])
    return v_cells

def _peclet_faces_(v_faces, L, D, eps=1e-30):
    """

    """
    u = np.abs(np.asarray(v_faces, float).ravel())

    # --- L a caras ---
    Lf = np.asarray(L, float)
    
    # --- D a caras (armónico si nodal) ---
    Df = np.asarray(D, float)
    
    # --- Pe = |u| * L / D ---
    U  = u[:, None]        # [n-1,1]
    Lf = Lf[:, None]       # [n-1,1]
    Pe = U * Lf / np.maximum(Df, eps)
    return Pe 


