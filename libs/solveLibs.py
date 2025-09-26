import numpy as np
from commonLibs import __collect_valve_sources__
from commonLibs import _flux_faces_phi_

def solveTanks(tank):
    """
    ¿Qué hago?
    ----------
    Resuelvo las ecuaciones diferenciales ordinarias (ODEs) de un tanque de almacenamiento de gases multicomponente. Calculo la derivada temporal del número de moles, composición y temperatura del tanque, teniendo en cuenta todos los balances de entrada y salida de masa, energía y especies, así como las conexiones de válvulas (entrada, salida, e inter-unidades).  
    La función implementa la lógica de un nodo tipo "tanque" en un sistema dinámico de simulación de redes de procesos (por ejemplo, para integración con `solve_ivp`).

    ¿Por qué/para qué/quién lo hago?
    --------------------------------
    - Para simular la dinámica de llenado/vaciado y mezcla de un tanque, considerando los efectos de la red de válvulas y conexiones en la planta.
    - Permite predecir, en cada instante, cómo evolucionan los moles, la composición y la temperatura, clave para control, seguridad y optimización.
    - Este método es invocado **en cada paso temporal** de la integración ODE global de la red de procesos.
    - Se utiliza internamente en el método `_rhs_()` de la clase `Tank`.

    Ejemplo de uso
    --------------
    >>> dydt = solveTanks(tank)
    # Se usa automáticamente desde tank._rhs_() durante la integración

    Avisos
    ------
    - El objeto `tank` debe tener inicializadas todas sus variables de estado (`_state_cell_vars`) y configuradas las conexiones (`_conex`), así como propiedades termodinámicas y físicas.
    - Las entradas y salidas están definidas por las válvulas conectadas: si alguna falta o su configuración es incorrecta, los balances pueden ser erróneos.
    - El método considera válvulas de entrada, salida e inter-unidad, y asigna el sentido de flujo según la diferencia de presiones.
    - Si el tanque es adiabático (`_adi=True`), no se considera pérdida térmica.
    - El sistema supone una única mezcla perfectamente agitada por nodo; si hay más nodos, todos se actualizan igual.
    - Las variables que retornan de aquí (`dydt`) son utilizadas directamente por el solver ODE, **no actualizan el estado del tanque directamente**.

    """
    nodos = tank._nodos
    ncomp = tank._ncomp

    t    = tank._state_cell_vars['t']
    N    = tank._state_cell_vars['N']        # escalar o array (esperado 1 nodo)
    T    = tank._state_cell_vars['T']
    P    = tank._state_cell_vars['P']
    x    = tank._state_cell_vars['x'].reshape(nodos, ncomp)

    # LOGGING
    tank._t_log.append(t)
    tank._N_log.append(N)
    tank._x_log.append(x)
    tank._T_log.append(T)
    tank._P_log.append(P)

    # === términos fuente por válvulas (universales) ===
    S_mol, S_sp, S_eng = __collect_valve_sources__(tank)
    sN   = float(S_mol.sum())          # mol/s
    sx   = S_sp.sum(axis=0)            # mol/s por especie
    sEng = float(S_eng.sum())          # J/s

    # === Pérdidas térmicas (igual que antes) ===
    Tref = 298.15
    if getattr(tank, "_adi", False):
        Q_loss = 0.0
    else:
        if all(hasattr(tank, a) for a in ("_hint","_hext","_kw","_Tamb")):
            Rtot = (1/tank._hint + tank._e/tank._kw + 1/tank._hext)
            Q_loss = (T - tank._Tamb) * tank._Aint / Rtot
        else:
            Q_loss = 0.0
    if hasattr(tank, "_Qloss_log"):
        tank._Qloss_log.append(Q_loss)

    # === ODEs (idéntica estructura, pero usando sN/sx/sEng) ===
    Cp_mix = float(np.dot(x.ravel(), tank._cpg)) if nodos == 1 else float(np.dot(x[0], tank._cpg))  # 1 nodo esperado
    dNdt = sN
    if N > 0:
        dxdt  = (sx - x * dNdt) / N       # x es (1,ncomp) -> broadcasting OK
        dTdt  = (sEng - Q_loss) / (N * Cp_mix) - ((T - Tref) / N) * dNdt
    else:
        dxdt = np.zeros_like(x)
        dTdt = 0.0
        


    dydt = np.concatenate((np.array([dNdt]).flatten(),
                           dxdt[:, :ncomp-1].flatten(),
                           np.array([dTdt]).flatten()))
    return dydt




def solveAdsColumn(adsColumn):
    nodos=adsColumn._nodos
    ncomp=adsColumn._ncomp
    dz = adsColumn._Lx
    
    t    = adsColumn._state_cell_vars['t']
    N    = adsColumn._state_cell_vars['N']
    Tg   = adsColumn._state_cell_vars['Tg']
    Ts   = adsColumn._state_cell_vars['Ts']
    x    = adsColumn._state_cell_vars['x'].reshape(nodos, ncomp)    
    q    = adsColumn._state_cell_vars['q'].reshape(nodos, ncomp)  
    P    = adsColumn._state_cell_vars['P'] 
    
    dNdt  =  np.zeros_like(N)
    dqdt  = np.zeros_like(q)
    dxdt  = np.zeros_like(x)    
    dTgdt = np.zeros_like(Tg)
    dTsdt = np.zeros_like(Ts)

    
    # === términos fuente por válvulas () ===
    S_mol, S_sp, S_eng = __collect_valve_sources__(adsColumn)
    # sN   = float(S_mol.sum())          # mol/s
    # sx   = S_sp.sum(axis=0)            # mol/s por especie
    # sEng = float(S_eng.sum())          # J/s
    
    v_faces   = adsColumn._state_face_vars["v"]
    v_cell   = adsColumn._state_cell_vars["v"]  
    flow_dir  = adsColumn._state_face_vars["flow_dir"]
    
    # --- RESOLVER dn/dt ---
    flux_faces = np.zeros(nodos-1)
    flux_species = np.zeros((nodos-1, ncomp))
    tol = 1e-10
    
    for i in range(nodos-1):
        vnet = v_faces[i] * flow_dir[i]
        if vnet >= tol:
            flux_faces[i] = vnet * N[i]
            flux_species[i, :] = vnet * N[i] * x[i, :]
        elif vnet <= -tol:
            flux_faces[i] = vnet * N[i+1]
            flux_species[i, :] = vnet * N[i+1] * x[i+1, :]
        else:
            flux_faces[i] = 0.0
            flux_species[i, :] = 0.0
    
    
    rhoCat = adsColumn._rho_s
    VolCat = adsColumn._Volsx
    matCat = rhoCat * VolCat
    kldf = adsColumn._state_cell_properties['kldf']/1e8
    qeq = adsColumn._qeq
    
    dqdt = kldf * (qeq  - q)          #[nodos,ncomp]
    dNadt = dqdt * matCat[:,None]  #[nodos,ncomp]
    
    
    # Balance total de moles (ya lo tienes)
    dNdt[0] = S_mol[0] - flux_faces[0]/dz[0]
    for j in range(1, nodos-1):
        dNdt[j] = (flux_faces[j-1] - flux_faces[j])/dz[j] + S_mol[j]
    dNdt[-1] = S_mol[-1] + flux_faces[-1]/dz[-1]
    
    dNdt-= np.sum(dNadt,axis=1) 
    
    # === Balance de especies ===
    dxdt = np.zeros_like(x)
    for k in range(ncomp-1):  # Solo ncomp-1 porque el último es por diferencia
        # Nodo entrada
        dxdt[0, k] = (S_sp[0, k]
                      - flux_species[0, k]/dz[0]
                      - dNadt[0, k]
                      - x[0, k]*dNdt[0]) / max(N[0],1e-20)
        # Nodos internos
        for j in range(1, nodos-1):
            adv = (flux_species[j-1, k] - flux_species[j, k])/dz[j]
            dxdt[j, k] = (adv
                          + S_sp[j, k]
                          - dNadt[j, k]
                          - x[j, k]*dNdt[j]) / max(N[j],1e-20)
        # Nodo salida
        dxdt[-1, k] = (S_sp[-1, k]
                       + flux_species[-1, k]/dz[-1]
                       - dNadt[-1, k]
                       - x[-1, k]*dNdt[-1]) / max(N[-1],1e-20)

    adsColumn._derivates['dNdt'] = dNdt.flatten()
    adsColumn._derivates['dxdt'] = dxdt.flatten()
    adsColumn._derivates['dqdt'] = dqdt.flatten()

    # RESOLVER dn/dt
    
    Q_loss = np.zeros_like(N)
    # === Guardar logs ===
    adsColumn._t_log.append(t)
    adsColumn._N_log.append(N)
    adsColumn._v_log.append(v_cell)
    adsColumn._x_log.append(x)
    adsColumn._q_log.append(q)
    adsColumn._Tg_log.append(Tg)
    adsColumn._Ts_log.append(Ts)
    adsColumn._P_log.append(P)
    adsColumn._Qloss_log.append(Q_loss)
    

    dydt = np.concatenate((dNdt.flatten(),
                           dxdt[:, :ncomp-1].flatten(),
                           dqdt.flatten(),
                           dTgdt.flatten(),
                           dTsdt.flatten(),
                           ))
    
    return dydt


    
    
    