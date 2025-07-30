# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 04:23:42 2025

@author: MiguelCamaraSanz
"""

import numpy as np

def solveTanks(tank):

    def get_connected_pressure(other_unit, port):
        # Devuelve la presión relevante del equipo conectado
        if hasattr(other_unit, "_state_vars"):
            P_other = other_unit._state_vars["P"]
            if isinstance(P_other, (np.ndarray, list)):
                if port == "bottom":
                    return P_other[0]
                elif port == "top":
                    return P_other[-1]
            return P_other
        else:
            raise ValueError("El equipo conectado no tiene _state_vars['P']")

    ncomp = tank._ncomp    
    t    = tank._state_vars['t']
    N    = tank._state_vars['N']
    x    = tank._state_vars['x']
    T    = tank._state_vars['T']
    P    = tank._state_vars['P']

    # LOGGING (para plot y balances)
    tank._N_log.append((t, N))           
    tank._x_log.append((t, x))           
    tank._T_log.append((t, T))           
    tank._P_log.append((t, P))           

    Nm3_2_mol = 1.01325e5 / (8.314 * 273.15)
    Tref = 298.15

    entradas = []   
    salidas  = []   
    time_valve = t

    # --- Válvula de entrada (si existe) ---
    Valve_inlet = tank._conex.get("inlet", None)
    if Valve_inlet is not None:
        endTime_valve_inlet = (Valve_inlet.logic_params.get("start", 0) +
                               Valve_inlet.logic_params.get("duration", 0))  
        MW_gas = np.sum(tank._MW * tank._xin)
        Qn_in  = Valve_inlet._get_Qn_gas(time_valve, endTime_valve_inlet, tank._Pin, tank._Tin, P, MW_gas)
        Qin_mol_s = Qn_in * Nm3_2_mol / 3600.0
        Valve_inlet._Qn_log.append((t, Qn_in))
        Cp_in = np.dot(tank._xin, tank._cpg)
        entradas.append({"Q": Qin_mol_s, "x": tank._xin, "T": tank._Tin, "Cp": Cp_in})

    # --- Válvula de salida (si existe) ---
    Valve_outlet = tank._conex.get("outlet", None)
    if Valve_outlet is not None:
        endTime_valve_outlet = (Valve_outlet.logic_params.get("start", 0) +
                                Valve_outlet.logic_params.get("duration", 0)) 
        MW_gas = np.sum(tank._MW * x)
        Qn_out = Valve_outlet._get_Qn_gas(time_valve, endTime_valve_outlet, P, T, tank._Pout, MW_gas)
        Qout_mol_s = Qn_out * Nm3_2_mol / 3600.0
        Valve_outlet._Qn_log.append((t, Qn_out))
        Cp_mix = np.dot(x, tank._cpg)
        salidas.append({"Q": Qout_mol_s, "x": x, "T": T, "Cp": Cp_mix})

    # --- Válvulas interconectadas (bottom, top, side) ---
    for port in ["bottom", "top", "side"]:
        units  = tank._conex.get(f"units_{port}", [])
        valves = tank._conex.get(f"valves_{port}", [])
        for other_unit, valve in zip(units, valves):
            endTime_valve = (valve.logic_params.get("start", 0) + valve.logic_params.get("duration", 0)) # << MODIFICADO
            P_self   = P
            T_self   = T
            x_self   = x

            P_other  = get_connected_pressure(other_unit, port)  
            T_other  = getattr(other_unit, "_state_vars")["T"]
            x_other  = getattr(other_unit, "_state_vars")["x"]
            MW_other = np.sum(tank._MW * x_other)

            # Sentido del flujo por diferencia de presión
            if P_self > P_other:
                # Salida desde este tanque
                Qn = valve._get_Qn_gas(t, endTime_valve, P_self, T_self, P_other, MW_other)
                Q_mol_s = Qn * Nm3_2_mol / 3600.0
                valve._Qn_log.append((t, Qn))
                salidas.append({"Q": Q_mol_s, "x": x_self, "T": T_self, "Cp": np.dot(x_self, tank._cpg)})
            elif P_self < P_other:
                # Entrada desde otro tanque
                Qn = valve._get_Qn_gas(t, endTime_valve, P_other, T_other, P_self, MW_other)
                Q_mol_s = Qn * Nm3_2_mol / 3600.0
                valve._Qn_log.append((t, Qn))
                entradas.append({"Q": Q_mol_s, "x": x_other, "T": T_other, "Cp": np.dot(x_other, tank._cpg)})
            else:
                valve._Qn_log.append((t, 0.0))
                # No hay flujo

    # --- Suma de entradas y salidas (balances globales) ---
    
    Qin_total    = sum(e["Q"] for e in entradas)                       
    Qout_total   = sum(s["Q"] for s in salidas)                        

    xin_total    = (sum(e["Q"] * e["x"] for e in entradas) )#/ Qin_total) if Qin_total > 0 else np.zeros(ncomp)    
    xout_total   = (sum(s["Q"] * s["x"] for s in salidas) )#/ Qout_total) if Qout_total > 0 else np.zeros(ncomp)  # 

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
    tank._Qloss_log.append((t, N))           
    # --- ODEs ---
    dNdt  = Qin_total - Qout_total     
    dxdt  = (xin_total - xout_total - x * dNdt) / N
    dTdt  = (Tin_total - Tout_total - Q_loss) / (N * Cp_mix) - ((T - Tref) / N) * dNdt

    dydt = np.concatenate(([dNdt], dxdt[:-1], [dTdt]))  
    return dydt
