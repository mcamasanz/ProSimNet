import numpy as np

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
    - El objeto `tank` debe tener inicializadas todas sus variables de estado (`_state_vars`) y configuradas las conexiones (`_conex`), así como propiedades termodinámicas y físicas.
    - Las entradas y salidas están definidas por las válvulas conectadas: si alguna falta o su configuración es incorrecta, los balances pueden ser erróneos.
    - El método considera válvulas de entrada, salida e inter-unidad, y asigna el sentido de flujo según la diferencia de presiones.
    - Si el tanque es adiabático (`_adi=True`), no se considera pérdida térmica.
    - El sistema supone una única mezcla perfectamente agitada por nodo; si hay más nodos, todos se actualizan igual.
    - Las variables que retornan de aquí (`dydt`) son utilizadas directamente por el solver ODE, **no actualizan el estado del tanque directamente**.

    """
    nodos = tank._nodos
    ncomp = tank._ncomp

    t    = tank._state_vars['t']
    N    = tank._state_vars['N']
    T    = tank._state_vars['T']
    P    = tank._state_vars['P']
    x_flat = tank._state_vars['x']
    x = x_flat.reshape(nodos, ncomp)

    # LOGGING (para plot y balances)
    tank._t_log.append(t)
    tank._N_log.append(N)
    tank._x_log.append(x)
    tank._T_log.append(T)
    tank._P_log.append(P)

    Nm3_2_mol = 1.01325e5 / (8.314 * 273.15)
    Tref = 298.15

    entradas = []
    salidas = []
    time_valve = t

    # --- Válvula de entrada (si existe) ---
    Valve_inlet = tank._conex.get("inlet", None)
    if Valve_inlet is not None:
        endTime_valve_inlet = (Valve_inlet.logic_params.get("start", 0) +
                               Valve_inlet.logic_params.get("duration", 0))
        MW_gas = np.sum(tank._MW * tank._xin)
        Pi = tank._get_StateVar_("P", port=tank._conex["where_inlet"])
        Qn_in = Valve_inlet._get_Qn_(time_valve, endTime_valve_inlet, tank._Pin, tank._Tin, Pi, MW_gas)
        Qin_mol_s = Qn_in * Nm3_2_mol / 3600.0
        Valve_inlet._Qn_log.append(Qn_in)
        Valve_inlet._t_log.append(t)
        Cp_in = np.dot(tank._xin, tank._cpg)
        entradas.append({"Q": Qin_mol_s, "x": tank._xin, "T": tank._Tin, "Cp": Cp_in})

    # --- Válvula de salida (si existe) ---
    Valve_outlet = tank._conex.get("outlet", None)
    if Valve_outlet is not None:
        endTime_valve_outlet = (Valve_outlet.logic_params.get("start", 0) +
                                Valve_outlet.logic_params.get("duration", 0))
        Pi = tank._get_StateVar_("P", port=tank._conex["where_outlet"])
        Ti = tank._get_StateVar_("T", port=tank._conex["where_outlet"])
        xi = tank._get_StateVar_("x", port=tank._conex["where_outlet"])
        MW_gas = np.sum(tank._MW * xi)
        Qn_out = Valve_outlet._get_Qn_(time_valve, endTime_valve_outlet, Pi, Ti, tank._Pout, MW_gas)
        Qout_mol_s = Qn_out * Nm3_2_mol / 3600.0
        Valve_outlet._Qn_log.append(Qn_out)
        Valve_outlet._t_log.append(t)
        Cp_mix = np.dot(xi, tank._cpg)
        salidas.append({"Q": Qout_mol_s, "x": xi, "T": Ti, "Cp": Cp_mix})

    # --- Válvulas interconectadas (interunit) ---
    valves_inter = []
    for key in ["valves_top", "valves_bottom", "valves_side"]:
        valves = tank._conex.get(key, [])
        if valves is not None:
            valves_inter.extend(valves)

    for valve in valves_inter:
        unit_A = valve._conex.get("unit_A", None)
        unit_B = valve._conex.get("unit_B", None)
        port_A = valve._conex.get("port_A", None)
        port_B = valve._conex.get("port_B", None)
     
        # Determina sentido (self y other)
        if unit_A._name == tank._name:
            # Este tanque es unit_A
            P_self   = tank._get_StateVar_("P", port=port_A)
            T_self   = tank._get_StateVar_("T", port=port_A)
            x_self   = tank._get_StateVar_("x", port=port_A)
            MW_self  = np.sum(tank._MW * x_self)
            other    = unit_B
            P_other  = other._get_StateVar_("P", port=port_B)
            T_other  = other._get_StateVar_("T", port=port_B)
            x_other  = other._get_StateVar_("x", port=port_B)
            MW_other = np.sum(other._MW * x_other)
        elif unit_B._name == tank._name:
            # Este tanque es unit_B
            P_self   = tank._get_StateVar_("P", port=port_B)
            T_self   = tank._get_StateVar_("T", port=port_B)
            x_self   = tank._get_StateVar_("x", port=port_B)
            MW_self  = np.sum(tank._MW * x_self)
            other    = unit_A
            P_other  = other._get_StateVar_("P", port=port_A)
            T_other  = other._get_StateVar_("T", port=port_A)
            x_other  = other._get_StateVar_("x", port=port_A)
            MW_other = np.sum(other._MW * x_other)
        else:
            raise ValueError(f"Válvula interunit '{valve._name}': ninguna unidad coincide con el tanque '{tank._name}'.")

        endTime_valve = (valve.logic_params.get("start", 0) +
                          valve.logic_params.get("duration", 0))

        # Sentido del flujo
        if P_self > P_other:
            # Sale de este tanque hacia el otro
            Qn = valve._get_Qn_(t, endTime_valve, P_self, T_self, P_other, MW_self)
            Q_mol_s = Qn * Nm3_2_mol / 3600.0
            valve._Qn_log.append(Qn)
            valve._t_log.append(t)
            salidas.append({"Q": Q_mol_s, "x": x_self, "T": T_self, "Cp": np.dot(x_self, tank._cpg)})
        
        elif P_self < P_other:
            # Entra desde el otro tanque hacia este
            Qn = valve._get_Qn_(t, endTime_valve, P_other, T_other, P_self, MW_other)
            Q_mol_s = Qn * Nm3_2_mol / 3600.0
            valve._Qn_log.append(Qn)
            valve._t_log.append(t)
            entradas.append({"Q": Q_mol_s, "x": x_other, "T": T_other, "Cp": np.dot(x_other, tank._cpg)})
        else:
            # No hay flujo
            valve._Qn_log.append(0.0)
            valve._t_log.append(t)

    # --- Suma de entradas y salidas (balances globales) ---
    Qin_total    = sum(e["Q"] for e in entradas)
    Qout_total   = sum(s["Q"] for s in salidas)

    xin_total    = sum(e["Q"] * e["x"] for e in entradas)
    xout_total   = sum(s["Q"] * s["x"] for s in salidas)

    Tin_total    = sum(e["Q"] * e["Cp"] * (e["T"] - Tref) for e in entradas)
    Tout_total   = sum(s["Q"] * s["Cp"] * (s["T"] - Tref) for s in salidas)

    Cp_mix = np.dot(x, tank._cpg)

    # --- Términos térmicos (Q_loss) ---
    if hasattr(tank, "_adi") and tank._adi is True:
        Q_loss = 0.0
    else:
        if all([hasattr(tank, "_hint"), hasattr(tank, "_hext"), hasattr(tank, "_kw"), hasattr(tank, "_Tamb")]):
            Rtot = (1/tank._hint + tank._e/tank._kw + 1/tank._hext)
            Q_loss = (T - tank._Tamb) * tank._Aint / Rtot
        else:
            Q_loss = 0.0
    if hasattr(tank, "_Qloss_log"):
        tank._Qloss_log.append(Q_loss)

    # --- ODEs ---
    dNdt  = Qin_total - Qout_total
    dxdt  = (xin_total - xout_total - x * dNdt) / N if N > 0 else np.zeros_like(x)
    dTdt  = (Tin_total - Tout_total - Q_loss) / (N * Cp_mix) - ((T - Tref) / N) * dNdt if N > 0 else 0.0
    dydt = np.concatenate((dNdt.flatten(), dxdt[:, :ncomp-1].flatten(), dTdt.flatten()))
    return dydt
