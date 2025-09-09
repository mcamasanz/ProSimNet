# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 14:17:30 2025

@author: MiguelCamaraSanz
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from thermo import Chemical, Mixture

R=8.314
SIGMA=5.670374419e-8  # Stefan–Boltzmann (W/m²/K⁴)

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
    x = np.asarray(x)
    prop = np.asarray(prop)
    
    n_nodos, n_comp = x.shape

    if prop.ndim == 1:
        # propiedad constante: replicamos por nodo
        prop = np.tile(prop, (n_nodos, 1))

    assert prop.shape == (n_nodos, n_comp), \
        f"Propiedades deben ser (n_nodos, n_comp), tienes {prop.shape}"

    if phi is None:
        if MW is None:
            raise ValueError("Necesitas MW o phi para Wilke")
        # Usamos valores medios para calcular φ_ij (vale si no cambia demasiado)
        prop_avg = np.nanmean(prop, axis=0)
        phi = __wilke_phi_matrix__(prop_avg, MW)

    # Calculamos Phi_i por nodo: [n_nodos, n_comp]
    Phi = x @ phi.T

    # Numerador: sum_i x_i * prop_i  → [n_nodos]
    num = np.sum(x * prop, axis=1)

    # Denominador: sum_i x_i * Φ_i   → [n_nodos]
    den = np.sum(x * Phi, axis=1)

    return num / np.maximum(den, eps)


# =============================================================================
# 
# =============================================================================

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

def _propGas_(P, T, x, species):
    nn = np.shape(x)[0] # número de nodos
    nc = np.shape(x)[1] # número de componentes    
    # Peso molecular promedio (kg/mol)
    MW = np.array([Chemical(sp).MW for sp in species])
    mwMix = np.dot(x, MW) / 1000 # kg/mol
    # Inicialización de propiedades
    rhoMix = np.zeros(nn)
    muMix = np.zeros(nn)
    cpMix = np.zeros(nn)
    kMix = np.zeros(nn)
    
    # Bucle por nodo
    for i in range(nn):
        mix = Mixture(IDs=species, zs=x[i], T=T[i], P=P[i])
        
        # Propiedades individuales
        rhoMix[i] = mix.rho # kg/m3
        muMix[i] = mix.mug # Pa.s
        cpMix[i] = mix.Cpg # J/kg/K
        kMix[i] = mix.kg # W/m/K
     
    return mwMix, rhoMix, muMix, cpMix, kMix

def _set_propertyTable_(species,
                        T_range,
                        P_range, 
                        threshold=0.05,
                        fill_nan=True):

    
    properties = ['rho','mu', 'Cp', 'k']
    prop_table = {}
    summary = {}


    for sp in species:
        prop_table[sp] = {}
        for prop in properties:
            values = []
            for T in T_range:
                row = []
                for P in P_range:
                    try:
                        chem = Chemical(sp, T=T, P=P)
                        val = getattr(chem, prop, None)
                        row.append(val if val is not None else np.nan)
                    except:
                        row.append(np.nan)
                values.append(row)

            arr = np.array(values)
            minv = np.nanmin(arr)
            maxv = np.nanmax(arr)
            rel_diff = (maxv - minv) / maxv if maxv else 0.0

            behavior = 'Constant' if rel_diff < threshold else 'NoConstant'
            const_val = np.nanmean(arr) if behavior == 'Constant' else None

            interp_func = None
            if behavior == 'NoConstant':
                interp_func = RegularGridInterpolator((T_range, P_range), arr, bounds_error=False, fill_value=None)

            prop_table[sp][prop] = {
                'behavior': behavior,
                'const_val': const_val,
                'table': arr,
                'interp_func': interp_func,
                'T_range': T_range,
                'P_range': P_range
            }
            
            summary[(sp, prop)] = {'is_constant': behavior == 'Constant','mean': const_val}

            # Si se desea rellenar los NaN tras la generación
            if fill_nan and behavior == 'NoConstant':
                table = prop_table[sp][prop]['table']
                for i in range(table.shape[0]):
                    for j in range(table.shape[1]):
                        if np.isnan(table[i, j]):
                            # entorno local 3x3
                            i_min = max(0, i - 1)
                            i_max = min(table.shape[0], i + 2)
                            j_min = max(0, j - 1)
                            j_max = min(table.shape[1], j + 2)
                            
                            T_sub = T_range[i_min:i_max]
                            P_sub = P_range[j_min:j_max]
                            Z_sub = table[i_min:i_max, j_min:j_max]

                            T_coords, P_coords = np.meshgrid(T_sub, P_sub, indexing='ij')
                            mask_valid = ~np.isnan(Z_sub)

                            if np.sum(mask_valid) >= 4:
                                X = np.stack([
                                    np.ones(np.sum(mask_valid)),
                                    T_coords[mask_valid],
                                    P_coords[mask_valid],
                                    T_coords[mask_valid]**2,
                                    T_coords[mask_valid]*P_coords[mask_valid],
                                    P_coords[mask_valid]**2
                                ], axis=1)
                                y = Z_sub[mask_valid]
                                coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
                                t_val = T_range[i]
                                p_val = P_range[j]
                                tX = np.array([1, t_val, p_val, t_val**2, t_val*p_val, p_val**2])
                                table[i, j] = tX @ coeffs

    return prop_table, summary

def _get_propertyTable_(data, summary, species, propertie, T, P, T_range, P_range):
    info = summary[(species, propertie)]
    if info["is_constant"]:
        return info["mean"]
    else:
        interpolator = RegularGridInterpolator((T_range, P_range), data[species][propertie]["table"])
        return interpolator([[T, P]])[0]

def _get_propertieMixture_(data, summary, species, propertie, T, P, x, T_range, P_range):
    
    nn = len(T)
    nc = len(species)
    x = np.asarray(x)
    vals = np.zeros((nn, nc))  # vals[nodo, componente]

    # Obtener propiedad pura para cada especie en cada nodo
    for j, sp in enumerate(species):
        for i in range(nn):
            vals[i, j] = _get_propertyTable_(data, summary, sp, propertie, T[i], P[i], T_range, P_range)

    # Mezcla según tipo de propiedad
    if propertie.lower() in ['rho']:
        MW = np.array([Chemical(sp).MW for sp in species])  # kg/mol
        # fracción másica de cada componente
        w = (x * MW[None, :])
        w /= np.sum(w, axis=1, keepdims=True)
    
        # densidad de mezcla por nodo
        return 1.0 / np.sum(w / vals, axis=1)
    
    elif propertie.lower() in ['mu', 'k']:  # Wilke
        MW = np.array([Chemical(sp).MW for sp in species])
        prop_pure = np.nanmean(vals, axis=0)  # media por componente [ncomp]
        phi = __wilke_phi_matrix__(prop_pure, MW)
        return __wilke_mix__(x, vals, phi=phi)

    elif propertie.lower() in ['cp']:  # ponderación molar directa
        return np.sum(x * vals, axis=1)

    else:
        raise ValueError(f"Propiedad '{propertie}' no reconocida o no soportada.")


# =============================================================================
# 
# =============================================================================

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
    P   = np.asarray(P, float).reshape(-1) / 1.01325e5 # atm
    MW  = np.asarray(MW, float).reshape(-1) * 1E3 # kg/mol -> g/mol
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

def _Dz_(Dim,u,dp,Pe):
    Dz = (u[:,None] * dp) / np.maximum(Pe, 1e-30)
    return Dz

def _lamz_(rho_f, cp_mass_f, alpha_f, uStar, dp, Pe_e):
    alpha_disp = (uStar * dp) / np.maximum(Pe_e, 1e-30)
    alpha_eff  = np.maximum(alpha_f, alpha_disp)
    return rho_f * cp_mass_f * alpha_eff


# =============================================================================
# 
# =============================================================================
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

def _darcy_ergun_velocity_faces_(
    P, rho_f, mu_f,
    eps, d_p,
    xcenters, mask,
    k_clear,
    g=9.81, eps_min=1e-30
):

    P   = np.asarray(P, float).ravel()
    xc  = np.asarray(xcenters, float).ravel()
    msk = np.asarray(mask, bool).ravel()

    # Distancia entre centros → una por cara
    dz_faces    = np.maximum(np.diff(xc), eps_min)
    dPdz_faces  = (P[1:] - P[:-1]) / dz_faces     # Pa/m

    # Término motor (puede ser ±):
    gdrive = -dPdz_faces - rho_f * g              # Pa/m

    # ¿La cara toca lecho? (si cualquiera de las dos celdas adyacentes está en el lecho)
    in_bed = msk[:-1] | msk[1:]

    # ---- Coeficientes (vector) a, b por cara ----
    # Ergun
    eps3 = max(eps**3, eps_min)
    a_E = 1.75 * (1.0 - eps) / (eps3 * max(d_p, eps_min)) * rho_f
    b_E = 150.0 * (1.0 - eps)**2 / (eps3 * max(d_p**2, eps_min)) * mu_f
    # Darcy fuera del lecho
    b_D = mu_f / max(k_clear, eps_min)
    a_D = 0.0

    # Mezcla según la cara
    a = np.where(in_bed, a_E, a_D)
    b = np.where(in_bed, b_E, b_D)

    # ---- Resolver a*|u|u + b*u = gdrive ----
    s    = np.sign(gdrive)              # signo esperado de u
    Gabs = np.abs(gdrive)

    # Solución positiva de |u| para el cuadrático:
    # |u| = ( -b + sqrt(b^2 + 4 a Gabs) ) / (2 a)   ; si a→0 → Gabs/b
    quad_term = b*b + 4.0*a*Gabs
    sqrt_term = np.sqrt(np.maximum(quad_term, 0.0))

    u_abs_quad = ( -b + sqrt_term ) / ( 2.0 * np.maximum(a, eps_min) )
    u_abs_lin  = Gabs / np.maximum(b, eps_min)

    # Usar la vía lineal donde a≈0 (caras fuera del lecho)
    use_linear = (a <= 1e-30)
    u_abs = np.where(use_linear, u_abs_lin, u_abs_quad)

    v_faces  = s * u_abs
    flow_dir = np.where(v_faces >= 0.0, 1, -1).astype(int)

    return v_faces, dPdz_faces, flow_dir, in_bed


# =============================================================================
# 
# =============================================================================
def _peclet_faces_(v_faces, L, D, eps=1e-30):
    u = np.abs(np.asarray(v_faces, float).ravel())   # [n-1]
    Lf = np.asarray(L, float).ravel()                # [n-1]
    Df = np.asarray(D, float)
    if Df.ndim == 1:  # escalar por cara → resultado [n-1]
        return (u * Lf) / np.maximum(Df, eps)
    elif Df.ndim == 2:  # una por especie → [n-1, nc]
        return (u[:, None] * Lf[:, None]) / np.maximum(Df, eps)
    else:
        raise ValueError("D debe ser 1D o 2D (caras[, especies]).")

def _vFaces_to_vCells_(v_faces):
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

def _phiFaces_to_phiCells_(F):

    F = np.asarray(F, float)
    if F.ndim == 1:
        n = F.size
        out = np.empty(n+1, dtype=float)
        out[0] = F[0]
        out[-1] = F[-1]
        if n > 1:
            out[1:-1] = 0.5*(F[:-1] + F[1:])
        return out

    if F.ndim == 2:
        n, m = F.shape
        out = np.empty((n+1, m), dtype=float)
        out[0, :] = F[0, :]
        out[-1, :] = F[-1, :]
        if n > 1:
            out[1:-1, :] = 0.5*(F[:-1, :] + F[1:, :])
        return out

    raise ValueError("F debe ser 1D o 2D")

def _Re_(u, rho, mu, L):
    return rho * np.abs(u) * L / np.maximum(mu,1E-30)

def _Pr_(mu, cp_mass, k):
    return mu * cp_mass / np.maximum(k,1E-30)

def _Sc_(mu, rho, Dim):
    """
    Sc_i = mu / (rho * D_im,i)
    """
    return mu[:, None] / np.maximum(rho[:, None] * Dim,1E-30)

def _Nu_(Re, Pr,cor="wakao_funazkri"):
    if cor=="wakao_funazkri":
        """Nu = 2 + 1.1 Re^0.6 Pr^(1/3)"""
        return 2.0 + 1.1 * (Re**0.6) * (Pr)**(1.0/3.0)

def _Sh_(Re, Sc,cor="wakao_funazkri"):
    if cor=="wakao_funazkri":
        """Sh = 2 + 1.1 Re^0.6 Sc^(1/3); Sc: [n, nc]"""
        return 2.0 + 1.1 * (Re[:,None]**0.6) * (Sc)**(1.0/3.0)
    
def _hc_(Nu,k,dp):
    return Nu * k / dp 

def _kc_(Sh,Dim,dp):
    return Sh * Dim / dp

def _kldf_(Deff, dp, geom="sphere"):
    # Factores típicos LDF por geometría (≈1er modo del problema de difusión)
    betas = {"sphere": 15.0, "cylinder": 12.0, "slab": 8.0}
    beta  = betas.get(geom, 15.0)

    # r_p = dp/2  → k_LDF = beta * Deff / r_p^2
    rp2 = (0.5 * dp)**2  # [n]
    return beta * Deff / np.maximum(rp2, 1E-30)  # [n,nc]

def _Da_ext_(kc, a_s, L, eps, u_cells):
    denom = np.maximum(eps * u_cells[:, None], 1e-30)  # [n,1]
    return (kc * a_s * L) / denom                        # [n, ncomp]

def _Da_ldf_(kldf, L, u_cells, eps):
    u_int = np.maximum(u_cells, 1e-30) / max(eps, 1e-30)   # [n]
    return (kldf * L[:, None]) / u_int[:, None]            # [n, ncomp]

def _Bi_g_(hc, dp, ks):
    return (hc * (0.5 * dp)) / np.maximum(ks, 1e-30)         # [n]

def _Bi_c_(kc, dp, Deff):
    return (kc * (0.5 * dp)) / np.maximum(Deff, 1e-30)       # [n, ncomp]

def _U_global_(h_in, e, k_w, h_out, adi):
    """
    U global pared interna->externo (CELDAS):
      1/U = 1/h_in + e/k_w + 1/h_out   (si adi -> U=0)
    h_in puede ser escalar o [n].
    """
    if adi:
        return 0.0
    h_in = np.asarray(h_in, float)
    Rin  = 1.0 / np.maximum(h_in, 1e-30)
    Rw   = e / np.maximum(k_w if k_w is not None else 1e12, 1e-30)
    Rout = 0.0 if (h_out is None or h_out <= 0.0) else 1.0 / np.maximum(h_out, 1e-30)
    return 1.0 / (Rin + Rw + Rout)     # escalar o [n] según h_in

def _hrw_(Tg, Tw=None, eps_w=0.8):
    """
    hr interno [n] (W/m²/K).  eps_w = emisividad interna de la pared.
    - Si Tw es None ->  hr = 4*eps_w*sigma*Tg^3
    - Si Tw se da   ->  hr = eps_w*sigma*(Tg^2+Tw^2)*(Tg+Tw)
    """
    Tg = np.asarray(Tg, float).reshape(-1)
    if Tw is None:
        return 4.0 * eps_w * SIGMA * Tg**3
    Tw = np.asarray(Tw, float).reshape(-1)
    if Tw.size == 1:
        Tw = np.full_like(Tg, float(Tw))
    return eps_w * SIGMA * ((Tg**2 + Tw**2) * (Tg + Tw))


