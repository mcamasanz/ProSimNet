# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 09:25:57 2025

@author: MiguelCamaraSanz
"""
import time 
import numpy as np
from numpy import trapz
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import simps

def solveTankNetwork(tanks,
                     valves,
                     connections,
                     species,
                     saveData,
                     starTime,
                     endTime,
                     solver='BDF',
                     rtol=1e-12,
                     atol=1e-12,
                     plot=False,
                     logBal=False,):

    assert isinstance(tanks, dict), "'tanks' debe ser un diccionario {nombre: objeto_Tank}"
    assert isinstance(valves, dict), "'valves' debe ser un diccionario {nombre: objeto_Valve}"
    assert isinstance(connections, list), "'connections' debe ser una lista de tuplas (tanque1, tanque2, valvula)"


    




def solve2TanksWithValve(Tank_A,
                         Valve_A_in,
                         Valve_A_out,
                         Tank_B,
                         Valve_B_in,
                         Valve_B_out,
                         Valve_AB,
                         saveData,
                         endTime,
                         solver='BDF',
                         rtol=1e-6,
                         atol=1e-6,
                         plot=True,
                         logBal=True):
    
    def _rhs(ti,y):
        ncomp = Tank_A._ncomp
        R = Tank_A._R
        Tref = 298.15
        Nm3_2_mol = 1.01325e5 / (R * 273.15)
        # --- Tanque A ---
        N_A = y[0]
        x_A = np.zeros(ncomp)
        x_A[:-1] = y[1:ncomp]
        x_A[-1] = 1 - np.sum(x_A[:-1])
        T_A = y[ncomp]
        P_A = (N_A * R * T_A) / Tank_A._vol
        Cp_A = np.dot(x_A, Tank_A._cpg)
        if Tank_A._adi==True:
            Q_Aloss = 0
        else:
            Rtot = (1/Tank_A._hint +
                    Tank_A._e/Tank_A._kw +
                    1/Tank_A._hext)
            Q_Aloss = (T_A - Tank_A._Tamb) * Tank_A._Aint / Rtot
        
        
        # --- Tanque B ---
        N_B = y[ncomp + 1]
        x_B = np.zeros(ncomp)
        x_B[:-1] = y[ncomp + 2:2 * ncomp + 1]
        x_B[-1] = 1 - np.sum(x_B[:-1])
        T_B = y[2 * ncomp + 1]
        P_B = (N_B * R * T_B) / Tank_B._vol
        Cp_B = np.dot(x_B, Tank_B._cpg)
        
        endTime_vA_in=(Valve_A_in.logic_params["start"] +
                       Valve_A_in.logic_params["duration"])
        endTime_vB_in=(Valve_B_in.logic_params["start"] +
                       Valve_B_in.logic_params["duration"])
        endTime_vA_out=(Valve_A_out.logic_params["start"] +
                       Valve_A_out.logic_params["duration"])
        endTime_vB_out=(Valve_B_out.logic_params["start"] +
                       Valve_B_out.logic_params["duration"])
        endTime_vAB=(Valve_AB.logic_params["start"] +
                       Valve_AB.logic_params["duration"])
        time_valve = ti 
        
        # === Flujo entre tanques A → B o B → A ===
        if P_A > P_B:
            P1, T1, x1 = P_A, T_A, x_A
            P0 = P_B
            MW = np.sum(Tank_A._MW * x1)
            Qn_AB = Valve_AB._get_Qn_gas(time_valve, endTime_vAB, P1, T1, P0, MW)
            n_dot = Qn_AB * Nm3_2_mol / 3600.0  # mol/s
    
            dN_A = -n_dot
            dN_B = +n_dot
            dx_A = (-x1 * dN_A) / N_A
            dx_B = (+x1 - x_B) * n_dot / N_B
            dT_A = -(Cp_A * (T_A - Tref) * n_dot) / (N_A * Cp_A)
            dT_B = (np.dot(x1, Tank_B._cpg) * (T1 - Tref) * n_dot - Cp_B * (T_B - Tref) * n_dot) / (N_B * Cp_B)
    
        elif P_B > P_A:
            P1, T1, x1 = P_B, T_B, x_B
            P0 = P_A
            MW = np.sum(Tank_B._MW * x1)
            Qn_AB = Valve_AB._get_Qn_gas(time_valve, endTime_vAB, P1, T1, P0, MW)
            n_dot = Qn_AB * Nm3_2_mol / 3600.0  # mol/s
    
            dN_B = -n_dot
            dN_A = +n_dot
            dx_B = (-x1 * dN_B) / N_B
            dx_A = (+x1 - x_A) * n_dot / N_A
            dT_B = -(Cp_B * (T_B - Tref) * n_dot) / (N_B * Cp_B)
            dT_A = (np.dot(x1, Tank_A._cpg) * (T1 - Tref) * n_dot - Cp_A * (T_A - Tref) * n_dot) / (N_A * Cp_A)
    
        else:
            dN_A = dN_B = 0
            dx_A = dx_B = np.zeros(ncomp)
            dT_A = dT_B = 0
        
        Valve_AB._Qn_log.append((ti, Qn_AB))
        """
        !!!
        
        !!!
        """

        dydt_A = np.concatenate(([dN_A], dx_A[:-1], [dT_A]))
        dydt_B = np.concatenate(([dN_B], dx_B[:-1], [dT_B]))

        return np.concatenate((dydt_A, dydt_B))
        
    def _solve():
        
        def _checkBalances(cpuTime):
            Nm3_2_mol = 1.01325e5 / (8.314 * 273.15)
            species = Tank_A._species
            ncomp = Tank_A._ncomp
        
            print("========== BALANCE CHECK =======================")
            # === 1. BALANCE DE MASA TOTAL TANQUES===
            # Variacion de moles en tanque
            N_A_ini = Tank_A._N[0]
            N_B_ini = Tank_B._N[0]
            N_A_fin = Tank_A._N[-1]
            N_B_fin = Tank_B._N[-1]
            
            delta_N_A = N_A_fin - N_A_ini
            delta_N_B = N_B_fin - N_B_ini
            
            # Moles entrada al tanque A
            # t_in = np.array(Valve_A_in._t)
            # Qn_A_in = np.array(Valve_A_in._Qn)
            # n_A_dot_in = Qn_A_in * Nm3_2_mol / 3600
            # N_A_in = simps(n_A_dot_in, t_in)
            N_A_in = 0
            # Moles salido al tanque A
            # t_out = np.array(Valve_A_out._t)
            # Qn_A_out = np.array(Valve_A_out._Qn)
            # n_A_dot_out = Qn_A_out * Nm3_2_mol / 3600
            # N_A_out = simps(n_A_dot_out, t_out)
            N_A_out = 0
            
            # Moles entrada al tanque B
            # t_in = np.array(Valve_B_in._t)
            # Qn_B_in = np.array(Valve_B_in._Qn)
            # n_B_dot_in = Qn_B_in * Nm3_2_mol / 3600
            # N_B_in = simps(n_B_dot_in, t_in)
            N_B_in=0
            # # Moles salido al tanque B
            # t_out = np.array(Valve_B_out._t)
            # Qn_B_out = np.array(Valve_B_out._Qn)
            # n_B_dot_out = Qn_B_out * Nm3_2_mol / 3600
            # N_B_out = simps(n_B_dot_out, t_out)
            N_B_out=0
            
            #Moles de AB o de BA
            # t_AB = np.array(Valve_AB._t)
            # Qn_AB = np.array(Valve_AB._Qn)
            # n_AB_dot = Qn_AB * Nm3_2_mol / 3600
            # N_AB = simps(n_AB_dot, t_AB)
            
            # if Tank_A._P[0] > Tank_B._P[0]:
            #     sentido = "A → B"
            #     N_AB_A = +N_AB
            #     N_AB_B = -N_AB
            # elif Tank_B._P[0] > Tank_A._P[0]:
            #     sentido = "B → A"
            #     N_AB_A = -N_AB
            #     N_AB_B = +N_AB
            # else:
            #     sentido = "Sin flujo"
            #     N_AB_A = N_AB_B = 0
            
            t_AB = np.array(Valve_AB._t)
            Qn_AB = np.array(Valve_AB._Qn)
            n_AB_dot = Qn_AB * Nm3_2_mol / 3600  # mol/s
            
            # Por convención:
            # - flujo positivo: entra a TankB, sale de TankA
            # - flujo negativo: entra a TankA, sale de TankB
            N_AB = simps(n_AB_dot, t_AB)  # moles netos hacia B
            
            N_AB_A = -N_AB  # moles netos que ha perdido TankA
            N_AB_B = +N_AB  # moles netos que ha ganado TankB
            
            if N_AB > 0:
                sentido = "A → B"
            elif N_AB < 0:
                sentido = "B → A"
            else:
                sentido = "Sin flujo"
                        
            
            # --- Balance individual ---
            err_A = N_A_in - N_A_out - N_AB_A - delta_N_A
            err_B = N_B_in - N_B_out - N_AB_B - delta_N_B
            err_total = (N_A_in + N_B_in) - (N_A_out + N_B_out) - (delta_N_A + delta_N_B)
            
            print("🔸 Balance de masa total (en moles):")
            print("===============================================")
            print(f"Tanque: {Tank_A._name}")
            print(f"  N_in   = {N_A_in:.3f} mol")
            print(f"  N_out  = {N_A_out:.3f} mol")
            print(f"  N_AB   = {N_AB:.3f} mol ")
            print(f"  N_INI  = {N_A_ini:.3f}")
            print(f"  N_END  = {N_A_fin:.3f}")
            print(f"  ΔN     = {delta_N_A:.3f} mol")
            print(f"  Error  = {err_A:.6f} mol")
            
            print(f"Tanque: {Tank_B._name}")
            print(f"  N_in   = {N_B_in:.3f} mol")
            print(f"  N_out  = {N_B_out:.3f} mol")
            print(f"  N_AB   = {-N_AB:.3f} mol")
            print(f"  N_INI  = {N_B_ini:.3f}")
            print(f"  N_END  = {N_B_fin:.3f}")
            print(f"  ΔN     = {delta_N_B:.3f} mol")
            print(f"  Error  = {err_B:.6f} mol")
            print("===============================================")
            print(f"Balance global:")
            print(f"  Error neto total = {err_total:.6f} mol")
            print("=================================================")
            delta_NA_curve = Tank_A._N - Tank_A._N[0]

            # N_AB integrado desde Qn_AB
            Qn_AB = np.array(Valve_AB._Qn) * Nm3_2_mol / 3600
            N_AB_integrated = np.cumsum(Qn_AB * np.gradient(Valve_AB._t))
            
            plt.plot(Valve_AB._t, N_AB_integrated, label="∫Qn_AB")
            plt.plot(Tank_A._t, delta_NA_curve, '--', label="ΔN_A")
            plt.legend()
            plt.xlabel("Time [s]")
            plt.ylabel("Mol")
            plt.title("Comparación ΔN_A vs ∫Qn_AB")
            plt.grid(True)
            
    
            return None
                
        if  Tank_A._actualTime == 0:
                Tank_A._initialize()
                Tank_B._initialize()
                Valve_AB._reset()
                Valve_A_in._reset()
                Valve_A_out._reset()
                Valve_B_in._reset()
                Valve_B_out._reset()
                #INICIALIZAR RESTP DE VALVULAS
                actualTime = 0

        else:
                actualTime=Tank_B._actualTime
                Tank_A._required['Results'] = False
                Valve_A_in._required['Results'] = False
                Valve_A_out._required['Results'] = False
                Tank_B._required['Results'] = False
                Valve_B_in._required['Results'] = False
                Valve_B_out._required['Results'] = False
                Valve_AB._required['Results'] = False
        
        N0_A = Tank_A._state_vars["N"]
        x0_A = np.array(Tank_A._state_vars["x"])
        T0_A = Tank_A._state_vars["T"]
        N0_B = Tank_B._state_vars["N"]
        x0_B = np.array(Tank_B._state_vars["x"])
        T0_B = Tank_B._state_vars["T"]
        R=8.314
        
        y0 = np.concatenate((
            [N0_A],
            x0_A[:-1],
            [T0_A],
            [N0_B],
            x0_B[:-1],
            [T0_B]
        ))    
        
        t_eval = np.linspace(actualTime, endTime, int((endTime-actualTime)/saveData+1))
        
        t_start = time.time()
        sol = solve_ivp(_rhs,
                        [actualTime,
                         endTime],
                         y0,
                         method=solver,
                         t_eval= t_eval,
                         atol = atol,
                         rtol = rtol)
     
        t_end = time.time()
        
        bs = Tank_A._ncomp + 1
    
        if sol.success:
            print(f"solve_ivp terminado con éxito. Tiempo simulado: {sol.t[-1] - sol.t[0]:.1f} s.")
            Tank_A._required['Results'] = True
            Valve_A_in._required['Results'] = True
            Valve_A_out._required['Results'] = True
            Tank_B._required['Results'] = True
            Valve_B_in._required['Results'] = True
            Valve_B_out._required['Results'] = True
            Valve_AB._required['Results'] = True

            cpuTime = t_end - t_start
            t=sol.t
            
            Tank_A._results = sol.y[0:bs, :]
            sol_A = sol.y[0:bs, :]
            # Tank_A._t = sol.t    
            # Tank_A._N = sol_A[0, :]
            # Tank_A._x = np.vstack([*sol_A[1:-1, :], 1 - np.sum(sol_A[1:-1, :], axis=0)])
            # Tank_A._T = sol_A[-1, :]
            # Tank_A._P = (Tank_A._N * R * Tank_A._T) / Tank_A._vol
    
            Tank_B._results = sol.y[bs:2*bs, :]
            sol_B = sol.y[bs:2*bs, :]    
            # Tank_B._t = sol.t
            # Tank_B._N = sol_B[0, :]
            # Tank_B._x = np.vstack([*sol_B[1:-1, :], 1 - np.sum(sol_B[1:-1, :], axis=0)])
            # Tank_B._T = sol_B[-1, :]
            # Tank_B._P = (Tank_B._N * R * Tank_B._T) / Tank_B._vol
            
            N_A = sol_A[0, :]
            N_B = sol_B[0, :]
            x_A = np.vstack([*sol_A[1:-1, :], 1 - np.sum(sol_A[1:-1, :], axis=0)])
            x_B = np.vstack([*sol_B[1:-1, :], 1 - np.sum(sol_B[1:-1, :], axis=0)])
            T_A = sol_A[-1, :]
            T_B = sol_B[-1, :]
            P_A = N_A * R * T_A / Tank_A._vol
            P_B = N_B * R * T_B / Tank_B._vol
            
            endTime_vA_in=(Valve_A_in.logic_params["start"] +
                           Valve_A_in.logic_params["duration"])
            endTime_vB_in=(Valve_B_in.logic_params["start"] +
                           Valve_B_in.logic_params["duration"])
            endTime_vA_out=(Valve_A_out.logic_params["start"] +
                           Valve_A_out.logic_params["duration"])
            endTime_vB_out=(Valve_B_out.logic_params["start"] +
                           Valve_B_out.logic_params["duration"])
            endTime_vAB=(Valve_AB.logic_params["start"] +
                           Valve_AB.logic_params["duration"])
                        
            for ti, Ni_A, Ti_A, xi_A, Pi_A, Ni_B, Ti_B, xi_B, Pi_B in zip(t, N_A, T_A, x_A.T, P_A, N_B, T_B, x_B.T, P_B):
                

                time_valve = ti 
                
                # solve A-B
                a_AB = Valve_AB._get_a(time_valve, endTime_vAB)
                Cv_AB = Valve_AB.Cv_max * Valve_AB._get_Cv(a_AB)
            
                
                if Pi_A - Pi_B < 1.E-1 :    
                    P0 = Pi_B
                    P1 = Pi_A
                    T1 = Ti_A
                    dP_AB = P1 - P0
                    MW_gas = np.sum(Tank_A._MW * xi_A)
                    
                    Qn_AB = Valve_AB._get_Qn_gas(time_valve, endTime_vAB, P0, T1, P1, MW_gas)
                elif Pi_B - Pi_A < 1.E-1 :
                    P0 = Pi_A
                    P1 = Pi_B
                    T1 = Ti_B
                    dP_AB = P1 - P0
                    MW_gas = np.sum(Tank_A._MW * xi_B)
                    Qn_AB = Valve_AB._get_Qn_gas(time_valve, endTime_vAB, P0, T1, P1, MW_gas)
                
                else:
                    P0 = min(Pi_A,Pi_B)
                    P1 = min(Pi_A,Pi_B)
                    T1 = (Ti_B+Ti_A)/2
                    dP_AB = 0
                    Qn_AB = 0 
                
                Valve_AB._storeData(ti, P1, dP_AB, a_AB, Cv_AB, Qn_AB)
                
                # solve INLET-A
                # ...
                # solve INLET-B
                # ...
                # solve OUTLET-A
                # ...
                # solve PUTLET-B
                # ...
               
            t_final = t[-1]
            y_A_final = sol_A[:, -1]
            N_A_final = y_A_final[0]
            x_A_final = np.zeros(Tank_A._ncomp)
            x_A_final[:-1] = y_A_final[1:-1]
            x_A_final[-1] = 1 - np.sum(x_A[:-1])
            T_A_final = y_A_final[-1]
            P_A_final = (N_A_final * R * T_A_final) / Tank_A._vol   
            Tank_A._storeData(t, P_A, T_A, x_A, N_A,
                t_final, P_A_final, T_A_final, x_A_final, N_A_final)
            
            y_B_final = sol_B[:, -1]
            N_B_final = y_B_final[0]
            x_B_final = np.zeros(Tank_B._ncomp)
            x_B_final[:-1] = y_B_final[1:-1]
            x_B_final[-1] = 1 - np.sum(x_B[:-1])
            T_B_final = y_B_final[-1]
            P_B_final = (N_B_final * R * T_B_final) / Tank_B._vol   
            Tank_B._storeData(t, P_B, T_B, x_B, N_B,
                t_final, P_B_final, T_B_final, x_B_final, N_B_final)
            
            print(f"\n⏱️ Simulation time (s): {cpuTime:.3f}")
            if logBal:
                _checkBalances(cpuTime)
            return None
    
        else:
        
            Tank_A._required['Results'] = False
            Valve_A_in._required['Results'] = False
            Valve_A_out._required['Results'] = False
            Tank_B._required['Results'] = False
            Valve_B_in._required['Results'] = False
            Valve_B_out._required['Results'] = False
            Valve_AB._required['Results'] = False
        
            print("ERROR: solve_ivp no terminó correctamente.")
            print(f"Mensaje: {sol.message}")
            
    # 
        return None  
    
    
    if Tank_A._actualTime < endTime:
        if not Tank_A._actualTime == Tank_B._actualTime:
            raise("Inicializar ambos tanques")
        else:
            _solve()
            # t_qn, qn_vals = zip(*Valve_AB._Qn_log)
            # N_AB_rhs = simps(qn_vals, t_qn)
            # delta_N_A = Tank_A._N - Tank_A._N[0]
            # print(f"🔍 Comparación ΔN_A = {delta_N_A:.3f} mol vs ∫ṅ_AB dt = {N_AB_rhs:.3f} mol")
            # print(f"🔍 Desviación = {delta_N_A - N_AB_rhs:.3f} mol")
            if plot:
                pass

def solveTankWithValves(Tank,
               Valve_inlet,
               Valve_outlet,
               endTime,
               startTime=0,
               saveData=1,
               solver='BDF',
               rtol=1e-3,
               atol=1e-3,
               plot=True,
               logBalances=True):
    
    def _rhs(ti,y):
        
        N = y[0]
        x = np.zeros(Tank._ncomp)
        x[:-1] = y[1:-1]
        x[-1] = 1 - np.sum(x[:-1])
        T = y[-1]
        P = (N * Tank._R * T) / Tank._vol  # Pa
        
        Tank._N_log.append((ti,N))
        Tank._x_log.append((ti,x))
        Tank._T_log.append((ti,T))
        Tank._P_log.append((ti,P))
        
        Nm3_2_mol = 1.01325e5 / (8.314 * 273.15)
        Tref = 298.15
        
        
        endTime_valve_outlet=(Valve_outlet.logic_params["start"] +
                       Valve_outlet.logic_params["duration"])
        endTime_valve_inlet=(Valve_inlet.logic_params["start"] +
                       Valve_inlet.logic_params["duration"])
        time_valve = ti 

        #inlet
        MW_gas = np.sum(Tank._MW * Tank._xin)
        Qn  = Valve_inlet._get_Qn_gas(time_valve,endTime_valve_inlet, Tank._Pin, Tank._Tin, P, MW_gas)
        Qin_mol_s = Qn * Nm3_2_mol / 3600.0  # mol/s
        Valve_inlet._Qn_log.append((ti, Qn))
    
        #outlet
        MW_gas = np.sum(Tank._MW * x)
        Qn = Valve_outlet._get_Qn_gas(time_valve,endTime_valve_outlet, P, T ,Tank._Pout, MW_gas) 
        Qout_mol_s = Qn * Nm3_2_mol/ 3600.0 # mol/s
        Valve_outlet._Qn_log.append((ti, Qn))
        
        # Cp mezcla actual y de entrada
        Cp_mix = np.dot(x, Tank._cpg)
        Cp_in = np.dot(Tank._xin, Tank._cpg)
        
        # Pérdidas de calor al ambiente
        Rtot = (1/Tank._hint + Tank._e/Tank._kw + 1/Tank._hext)
        
        if Tank._adi==True:
            Q_loss = 0
        else:
            Q_loss = (T - Tank._Tamb) * Tank._Aint / Rtot
        
        Tank._Qloss_log.append((ti,Q_loss))

        # Balance total
        dNdt = Qin_mol_s - Qout_mol_s
        # Balance de especies
        dxdt = (Qin_mol_s * Tank._xin - Qout_mol_s * x - x * dNdt) / N
        # Balance de energia 
        dTdt = (
                Qin_mol_s * Cp_in * (Tank._Tin - Tref)
                - Qout_mol_s * Cp_mix * (T - Tref)
                - Q_loss
                )/ (N * Cp_mix) - ((T - Tref) / N) * dNdt

        dydt = np.concatenate(([dNdt], dxdt[:-1], [dTdt]))
        return dydt
        
    def _plotAll():
        
        if (Tank._required["Results"] == True and
            Valve_inlet._required["Results"] == True and
            Valve_outlet._required["Results"] == True) :

            fig = plt.figure(figsize=(18, 10), constrained_layout=True)
            grid = fig.add_gridspec(3, 4)
        
            # --- Datos ---
            t = Tank._t
            P, T, N, x, species = Tank._P, Tank._T, Tank._N, Tank._x, Tank._species
            
            t_in, a_in, Qn_in,  = Valve_inlet._t, Valve_inlet._a, Valve_inlet._Qn, 
            
            # t_in2 = Valve_inlet._t2
            # Qn_in2 = Valve_inlet._Qn2
            
            
            t_out, a_out, Qn_out, = Valve_outlet._t, Valve_outlet._a, Valve_outlet._Qn, 
            # t_out2 = Valve_outlet._t2
            # Qn_out2 = Valve_outlet._Qn2
            
            a_range = np.linspace(0, 1, 100)
    
            # --- Válvula de entrada ---
            ax1 = fig.add_subplot(grid[0, 0])
            ax1.plot(a_range * 100, Valve_inlet._get_Cv(a_range) * 100, lw=2, color='red')
            ax1.set_xlabel("Apertura [%]")
            ax1.set_ylabel("Cv relativo [%]")
            ax1.set_title(f"Tipo de válvula: {Valve_inlet.valve_type}")
            ax1.grid()
    
            ax2 = fig.add_subplot(grid[1, 0])        
            ax2.plot(t_in, np.array(a_in) * 100, lw=2 , color='blue')
            ax2.set_ylabel("Apertura [%]")
            ax2.set_title("Lógica de control")
            ax2.grid()
            
            ax3 = fig.add_subplot(grid[2, 0])
            ax3.plot(t_in, Qn_in, lw=2, color='green')
            # ax3.scatter(t_in2, Qn_in2, lw=2, color='blue')
            ax3.set_title("Caudal IN [Nm3/h]")
            ax3.set_xlabel("Tiempo [s]")
            ax3.grid()
        
            # --- Tanque: Presión y Moles ---
            ax4 = fig.add_subplot(grid[0, 1])
            ax4.plot(t, P, lw=2)
            ax4.set_title("Presión tanque [Pa]")
            ax4.grid()
        
            ax5 = fig.add_subplot(grid[1, 1])
            ax5.plot(t, N, lw=2, color="lightgreen")
            ax5.set_title("Moles totales [mol]")
            ax5.set_xlabel("Tiempo [s]")
            ax5.grid()
        
            # --- Tanque: Temperatura y Fracciones ---
            ax6 = fig.add_subplot(grid[0, 2])
            ax6.plot(t, T, lw=2, color='orange')
            ax6.set_title("Temperatura [K]")
            ax6.grid()
            
            colors = [
                'orange',
                'cyan',
                'pink',
                'sky',
                'pale',
                'purple',
            ]
            
            ax7 = fig.add_subplot(grid[1, 2])
            for i, specie in enumerate(species):
                ax7.plot(t, x[i], label=specie, color=colors[i])
            ax7.set_title("Fracciones molares")
            ax7.set_xlabel("Tiempo [s]")
    
            ax7.legend()
            ax7.grid()
        
            # --- Válvula de salida ---
            ax8 = fig.add_subplot(grid[0, 3])
            ax8.plot(a_range * 100, Valve_outlet._get_Cv(a_range) * 100, lw=2, color='red')
            ax8.set_xlabel("Apertura [%]")
            ax8.set_ylabel("Cv relativo [%]")
            ax8.set_title(f"Tipo de válvula: {Valve_outlet.valve_type}")
            ax8.grid()
    
            ax9 = fig.add_subplot(grid[1, 3])        
            ax9.plot(t_out, np.array(a_out) * 100, lw=2 , color='blue')
            ax9.set_ylabel("Apertura [%]")
            ax9.set_title("Lógica de control")
            ax9.grid()
            
            ax10 = fig.add_subplot(grid[2, 3])
            ax10.plot(t_out, Qn_out, lw=2, color='green')
            # ax10.scatter(t_out2, Qn_out2, lw=2, color='blue')
            ax10.set_title("Caudal OUT [Nm3/h]")
            ax10.set_xlabel("Tiempo [s]")
            ax10.grid()
        
            fig.suptitle("Sistema Tanque + Válvulas", fontsize=16)
            plt.show()
    
    def _solve():
        
        def _checkBalances(cpuTime,actualTime):
            
            Nm3_2_mol = 1.01325e5 / (8.314 * 273.15)
            species = Tank._species
            ncomp = Tank._ncomp
        
            print("\n========== BALANCE CHECK =======================")
        
            # === 1. BALANCE DE MASA TOTAL ===
            N_ini = Tank._N2[0]
            N_fin = Tank._N2[-1]
            delta_N = N_fin - N_ini
        
            n_dot_in = np.array(Valve_inlet._Qn2) * Nm3_2_mol / 3600.0
            n_dot_out = np.array(Valve_outlet._Qn2) * Nm3_2_mol / 3600.0
            # n_dot_in[0]=0
            # n_dot_in[-1]=0
            # n_dot_out[0]=0
            # n_dot_out[-1]=0
            # N_in = simps(n_dot_in, t)
            
            t=np.array(Valve_inlet._t2)
            N_in = trapz(n_dot_in, t)
            N_out = trapz(n_dot_out, t)
            err = N_in - N_out - delta_N
            warnE = abs(err/max(N_in,N_out,delta_N))*100
            
            print("🔸 Balance de masa:")
            print("===============================================")
            if warnE < 3:
                print(f"    %Err    = {warnE:.2f}")
            else:
                print(f"  N_in      = {N_in:.2f} mol")
                print(f"  N_out     = {N_out:.2f} mol")
                print(f"  N_ini     = {N_ini:.2f} mol")
                print(f"  N_fin     = {N_fin:.2f} mol")
                print(f"  ΔN        = {delta_N:.2f} mol")
                print(f"  Error     = {err:.2f} mol")
                print(f"  %Err    = {warnE:.2f}")
            print("===============================================")
            # === 2. BALANCE DE ESPECIES ===
            x2 = np.array(Tank._x2)
            x_ini = x2[0,:]
            x_fin = x2[-1,:]
            Ni_ini = x_ini * N_ini
            Ni_fin = x_fin * N_fin
            delta_Ni = Ni_fin - Ni_ini
            
            xin = np.array(Tank._xin)
            n_dot_in_spec = xin[:, None] * n_dot_in
            Ni_in = np.array([trapz(n_dot_in_spec[i], t) for i in range(ncomp)])
            
            x_out = np.array(Tank._x2)  # ya tiene forma (ncomp, nt)
            n_dot_out_spec = []
            
            Tank._aux=x_out
            Tank._aux2=n_dot_out
            
            for i in range(ncomp):
                n_dot_out_spec.append(n_dot_out * x_out[:,i])
            Ni_out = np.array([trapz(n_dot_out_spec[i], t) for i in range(ncomp)])  
            
            
            print("🔸 Balance de especies:")
            print("===============================================")
            for i, sp in enumerate(species):
                err = Ni_in[i] - Ni_out[i] - delta_Ni[i]
                warnE = abs(err/max(Ni_in[i],Ni_out[i],delta_Ni[i]))*100
                if warnE < 3:
                    print(f"  {sp:<10}")    
                    print(f"    %Err    = {warnE:.2f}")
                else:
                    print(f"  {sp:<10}")
                    print(f"    In     = {Ni_in[i]:.2f}")
                    print(f"    tOut   = {Ni_out[i]:.2f}")
                    print(f"    Ni_ini = {Ni_ini[i]:.2f}")
                    print(f"    Ni_fin = {Ni_fin[i]:.2f}")
                    print(f"    ΔTank  = {delta_Ni[i]:.2f}")
                    print(f"    Err    = {err:.2f}")
                    print(f"    %Err    = {warnE:.2f}")
            print("===============================================")
            # === 3. BALANCE DE ENERGÍA ===
            Tref = 298.15
            Cp = np.array(Tank._cpg)
        
            T_ini = Tank._T2[0]
            T_fin = Tank._T2[-1]
            Cp_ini = np.dot(x_ini, Cp)
            Cp_fin = np.dot(x_fin, Cp)
        
            E_ini = N_ini * Cp_ini * (T_ini - Tref)
            E_fin = N_fin * Cp_fin * (T_fin - Tref)
            delta_E = E_fin - E_ini
        
            Cp_in = np.dot(xin, Cp)
            E_in = trapz(n_dot_in * Cp_in * (Tank._Tin - Tref), t)
        
            Cp_out_t = np.dot(x_out,Tank._cpg)
            T_out = np.array(Tank._T2)
            E_out = trapz(n_dot_out * Cp_out_t * (T_out - Tref), t)
        
            if Tank._adi:
                Q_loss = 0.0
            else:
                Qloss = np.array(Tank._Qloss2)
                Q_loss = simps(Qloss, Tank._t2)
        
            err_E = E_in - E_out - Q_loss - delta_E
            warnE = abs(err_E/max(E_in,E_out,delta_E))*100
            print("🔸 Balance de energía:")
            print("===============================================")
            if warnE < 3:
                print(f"  %Error  = {warnE:.2f} ")
            else:
                print(f"  E_in    = {E_in / 1000:.2f} KJ")
                print(f"  E_out:  = {E_out / 1000:.2f} KJ")
                print(f"  E_ini:  = {E_ini / 1000:.2f} KJ")
                print(f"  E_fin:  = {E_fin / 1000:.2f} KJ")
                print(f"  ΔE tank = {delta_E / 1000:.2f} KJ")
                print(f"  Q_loss: = {Q_loss / 1000:.2f} KJ")
                print(f"  Error   = {err_E / 1000:.2f} KJ")
                print(f"  %Error  = {warnE:.2f}")
            print("===============================================")
            
            return None
        
        if Tank._actualTime == 0:
            Tank._initialize()
            Valve_inlet._reset()
            Valve_outlet._reset()
            Valve_inlet._reset_logs()
            Valve_outlet._reset_logs()
            Tank._required['Results'] = False
            Valve_inlet._required['Results'] = False
            Valve_outlet._required['Results'] = False
        else:
            Tank._reset_logs()
            Valve_inlet._reset_logs()
            Valve_outlet._reset_logs()
            Tank._required['Results'] = False
            Valve_inlet._required['Results'] = False
            Valve_outlet._required['Results'] = False
        
            
        actualTime=Tank._actualTime
        # Condiciones iniciales
        N0 = Tank._state_vars["N"]
        x0 = np.array(Tank._state_vars["x"])
        T0 = Tank._state_vars["T"]
        y0 = np.concatenate(([N0], x0[:-1], [T0]))
        
        t_eval = np.linspace(actualTime, endTime, int((endTime-Tank._actualTime)/saveData))
    
        t_start = time.time()
        sol = solve_ivp(_rhs,
                        [actualTime,
                         endTime],
                         y0,
                         method=solver,
                         t_eval= t_eval,
                         atol = atol,
                         rtol = rtol)
     
        t_end = time.time()
    
        if sol.success:
            print(f"[{Tank._name}] solve_ivp terminado con éxito. Tiempo simulado: {sol.t[-1] - sol.t[0]:.1f} s.")
        
            Tank._results = sol
            cpuTime = t_end - t_start
            #units and valves _requiered['Results]=True
            Tank._required['Results'] = True
            Valve_inlet._required['Results'] = True
            Valve_outlet._required['Results'] = True
            
            #GUARDAR INFO PARA BALANCES
            t_log,QnIN_log  = Valve_inlet._clean_LOG_rhs(Valve_inlet._Qn_log)
            Valve_inlet._storeBal(t_log,QnIN_log)

            t_log,QnOUT_log= Valve_outlet._clean_LOG_rhs(Valve_outlet._Qn_log)
            Valve_outlet._storeBal(t_log,QnOUT_log)
            
            t_log,N_log  = Tank._clean_LOG_rhs(Tank._N_log)
            t_log,x_log  = Tank._clean_LOG_rhs(Tank._x_log)
            t_log,P_log  = Tank._clean_LOG_rhs(Tank._P_log)
            t_log,T_log  = Tank._clean_LOG_rhs(Tank._T_log)
            t_log,Qloss_log =Tank._clean_LOG_rhs(Tank._Qloss_log)
            Tank._storeBal(t_log,P_log,T_log,x_log,N_log,Qloss_log)
            #GUARDAR INFO PARA graficos  
            t=sol.t
            N = sol.y[0, :]
            x = np.zeros((Tank._ncomp, len(t)))
            x[:-1, :] = sol.y[1:-1, :]
            x[-1, :] = 1 - np.sum(x[:-1, :], axis=0)
            T = sol.y[-1, :]
            P = (N * Tank._R * T) / Tank._vol
            
            endTime_valve_inlet=(Valve_inlet.logic_params["start"] +
                           Valve_inlet.logic_params["duration"])
            
            endTime_valve_outlet=(Valve_outlet.logic_params["start"] +
                           Valve_outlet.logic_params["duration"])
            
            for ti, Ni, Ti, xi, Pi in zip(t, N, T, x.T, P):
                
                time_valve = ti 
                
                # Entrada
                a_in = Valve_inlet._get_a(time_valve, endTime_valve_inlet)
                Cv_in = Valve_inlet.Cv_max * Valve_inlet._get_Cv(a_in)
                dP_in = Tank._Pin - Pi
                MW_gas = np.sum(Tank._MW * Tank._xin)
                Qn_in = Valve_inlet._get_Qn_gas(time_valve, endTime_valve_inlet, Tank._Pin, Tank._Tin, Pi, MW_gas)
                Valve_inlet._storeData(ti, Pi, dP_in, a_in, Cv_in, Qn_in)
                
                # Salida
                a_out = Valve_outlet._get_a(time_valve, endTime_valve_outlet)
                Cv_out = Valve_outlet.Cv_max * Valve_outlet._get_Cv(a_out)
                dP_out = Pi - Tank._Pout
                MW_gas = np.sum(Tank._MW * xi)
                Qn_out = Valve_outlet._get_Qn_gas(time_valve, endTime_valve_outlet, Pi, Ti, Tank._Pout, MW_gas)
                Valve_outlet._storeData(ti, Pi, dP_out, a_out, Cv_out, Qn_out)
                
            t_final = t[-1]
            y_final = sol.y[:, -1]
            N_final = y_final[0]
            x_final = np.zeros(Tank._ncomp)
            x_final[:-1] = y_final[1:-1]
            x_final[-1] = 1 - np.sum(x_final[:-1])
            T_final = y_final[-1]
            P_final = (N_final * Tank._R * T_final) / Tank._vol   
            
            
            Tank._storeData(t      , P      , T      , x      , N      ,
                            t_final, P_final, T_final, x_final, N_final,)
            
            print(f"\n⏱️ Simulation time (s): {cpuTime:.3f}")
            if logBalances:
                _checkBalances(cpuTime,
                               actualTime)
            return None

        else:
        
            Tank._required['Results'] = False
            Valve_inlet._required['Results'] = False
            Valve_outlet._required['Results'] = False
            
            print(f"[{Tank._name}<] ERROR: solve_ivp no terminó correctamente.")
            print(f"Mensaje: {sol.message}")
            
    
        return None  
    
    if startTime == "lastTime":
        if Tank._actualTime < endTime:
            _solve()
            if plot:
                _plotAll()
            
    elif startTime == 0:

        Tank._initialize()
        Valve_inlet._initialize()
        Valve_outlet._initialize()
        
        if Tank._actualTime < endTime:
            _solve()
            if plot:
                _plotAll()

    elif startTime < Tank._actualTime:
        
        Tank._croopTime(startTime)
        Valve_inlet._croopTime(startTime)
        Valve_outlet._croopTime(startTime)
        
        if Tank._actualTime < endTime:
            _solve()
            if plot:
                _plotAll()
    
    else:
        raise ValueError(f"⛔ El tiempo solicitado {startTime:.2f} s es superior al actual del tanque {Tank._name}: {Tank._actualTime:.2f} s")

        
    
        
    
    
        
