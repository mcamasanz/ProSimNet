# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 09:25:48 2025

@author: MiguelCamaraSanz
"""
import time 
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
# =============================================================================
# 
# =============================================================================

# =============================================================================
# 

# =============================================================================
class Valve:
    
    def __init__(self,
                 Name,                  # Nombre del equipo
                 Cv_max=1.0,            # Coeficiente máximo de caudal [Nm3/h·bar^0.5]
                 valve_type="linear",   # Tipo de válvula: 'linear', 'quick-opening', etc.
                 a_max=1.,
                 a_min=0.,
                 logic="linear",        # Lógica de apertura: 'linear', 'step', 'custom'
                 logic_params=None,     # Parámetros para la lógica: {'t_open': x, 't_close': y}
                 opening_direction="oc", # Dirección de apertura: 'oc' o 'co'
                 ):
            
            self._name = Name
            self._t  = []   # Tiempos de simulación [s]
            self._t2 = []   # Tiempos de simulación [s]
            self._pi = []   # Presión en nodo 0 o 99 [Pa]
            self._a  = []   # Apertura de válvula [0-1]
            self._dP = []   # Diferencia de presión en la válvula [Pa]
            self._Cv = []   # Coeficiente de caudal instantáneo
            self._Qn = []   # Caudal instantáneo [Nm3/h]
            self._Qn_log=None # Guarda resultados desordendos de rhs
            self._Qn2  = None # Caudal instantáneo [Nm3/h] limpio y ordenado de rhs 
            
            # Parámetros de configuración
            self.a_max=a_max 
            self.a_min=a_min 
            self.Cv_max = Cv_max
            self.valve_type = valve_type.lower()
            self.logic = logic.lower()
            self.logic_params = logic_params if logic_params else {}
            self.opening_direction = opening_direction.lower()
            self._validate()  
            
            self._required = {'Design'        : True,
                              'Logical_info'  : True,
                              'Results'       : False}
            
            
            self._conex = {
                "type": None,  # tipo de válvula
                "unit_A": None,    # origen
                "unit_B": None,    # destino (None si es inlet u outlet)
                "port_A": None,    # "bottom" | "top" | "slide", # de donde sale/entra
                "port_B": None,    # "bottom" | "top" | "slide", # de donde sale/entra
                }
            

    def _initialize(self,):
        self._reset()
        self._reset_logs()
        self._required['Results'] = False
        return None
    
    def update(self,config=None, **kwargs):
        if config is None:
            config = kwargs
        else:
            config.update(kwargs)
    
        for k, v in config.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self._validate()
        return

        return

    def _validate(self,):
        allowed_valves = ["linear", "equal_percentage", "quick_opening", "custom"]
        allowed_logics = ["linear", "sigmoid", "poly", "step", "ramp", "sin"]
        allowed_dirs = ["oc", "co"]
        if self.valve_type not in allowed_valves:
            raise ValueError(f"Unsupported valve type. Choose from {allowed_valves}")
        if self.logic not in allowed_logics:
            raise ValueError(f"Unsupported logic type. Choose from {allowed_logics}")
        if self.opening_direction not in allowed_dirs:
            raise ValueError(f"Unsupported direction. Choose from {allowed_dirs}")
        if not (0 <= self.a_min < self.a_max <= 1):
            raise ValueError("a_min y a_max deben estar en el rango [0, 1] y cumplir a_min < a_max.")
        return
    
    def _get_a(self, t, t_step):
        
        p = self.logic_params
        start = p.get("start", 0)
        duration = p.get("duration", t_step)
        norm_time = np.clip((t - start) / duration, 0, 1)
        
        if self.logic == "linear":
            a = norm_time
        elif self.logic == "sigmoid":
            k = p.get("k", 10)
            x0 = p.get("x0", 0.5)
            a = 1 / (1 + np.exp(-k * (norm_time - x0)))
        elif self.logic == "poly":
            degree = p.get("degree", 2)
            a = norm_time ** degree
        elif self.logic == "step":
            steps = p.get("steps", 1)
            a = np.zeros_like(norm_time)
            step_time = duration / steps
            for i in range(steps):
                a += (t >= (start + i * step_time)) * (1 / steps)
            a = np.clip(a, 0, 1)
        elif self.logic == "ramp":
            height = p.get("height", 1)
            a = np.clip((t - start) / duration * height, 0, 1)
        elif self.logic == "sin":
            freq = p.get("freq", 1)
            a = 0.5 * (1 - np.cos(np.pi * norm_time * freq))
        else:
            a = norm_time
        
        if self.opening_direction == "oc": 
            apertura = 1 - a
        elif self.opening_direction == "co":
            apertura = a
        else:
            raise ValueError(f"opening_direction '{self.opening_direction}' must be 'co' or 'oc'")
        
        apertura = np.clip(apertura, self.a_min, self.a_max)
        return apertura
        
    def _get_Cv(self, a):
        if self.valve_type == "linear":
            Cv = a
        elif self.valve_type == "equal_percentage":
            R = 30
            Cv= (R**a - 1) / (R - 1)
        elif self.valve_type == "quick_opening":
            Cv= np.sqrt(a)
        elif self.valve_type == "custom":
            Cv=a**2 / (a**2 + 0.1)
        else:
            raise ValueError(f"valve_type '{self.valve_type}' is not recognized")
        return Cv
    
    def _get_Qn_gas(self, t,t_step, pIn, Tin, pOut, MW_gas):
        
        deltaP_bar = (pIn - pOut)/1e5
        
        if deltaP_bar <= 0:
            Qn = 0.0
        else:
            a = self._get_a(t, t_step)
            Cv_t = self._get_Cv(a)
            Cv = self.Cv_max * Cv_t
            MW_air = 28.96  # g/mol
            Sg = MW_gas / MW_air  # densidad relativa
        
            P1_bar = pIn / 1e5
        
            Qn = 414.97 * Cv * np.sqrt((deltaP_bar * P1_bar) / (Tin * Sg))

        return Qn
    
    def _get_Qn_liq(self, ):
        pass
        return None
    
    def _estimateCv(self,V,P0,P1,T0,T1,time):
        self._R = 8.314
        
        N1 = V * P1 / self._R / T1
        N0 = V * P0 / self._R / T0
        delta_N = abs(N1-N0)
        n_dot = delta_N / time  # mol/s
        Z=1.0
        # Conversión de mol/s a Nm3/h (usando condiciones normales)
        Tref = 273.15  # K
        Pref = 1.01325e5  # Pa
        Nm3_per_mol = self._R * Tref / Pref  # m3/mol
        Qn = n_dot * Nm3_per_mol * 3600  # Nm3/h
    
        # Caída de presión cuadrática (en Pa^2 / (K))
        deltaP_quad = (P1 ** 2 - P0 ** 2) / (T1 * Z)
    
        if deltaP_quad <= 0:
            raise ValueError("La presión de entrada debe ser mayor que la inicial del tanque.")
    
        Cv = Qn / np.sqrt(deltaP_quad) * 1e5  # normalización a unidades de Cv
        print(f"Se estima que se necesita Cv ≈ {Cv:.2f} Nm3/h/bar^0.5")
        return 
    
    def _reset(self,):
        self._t=[]
        self._pi=[]
        self._a=[]
        self._dP=[]
        self._Cv=[]
        self._Qn=[] 
        self._reset_logs()
        return None
    
    def _storeData(self, t, pi, dP, a, Cv,Qn,):
        "GUARDA LA INFORMACION DE TODOS LOS CICLOS PARA MOSTRAS POR PANTALLA "
        
        self._t.append(t)
        self._pi.append(pi)
        self._dP.append(dP)
        self._a.append(a)
        self._Cv.append(Cv)
        self._Qn.append(Qn)
        return None
    
    def _storeBal(self,t2,Qn2):
        "GUARDA LOS RESULTADOS LIMPIOS DE RHS PARA LA SIMULACION SIN ACUMULAR NECESARIO PARA CALCULAR LOS BALANCES DE LA SIMULACION PARA ESE CICLO"
        
        self._t2 = t2
        self._Qn2 = Qn2         
        return None
    
    def _reset_logs(self,):
        "GUARDA LOS RESULTADOS LIMPIOS DE RHS PARA LA SIMULACION SIN ACUMULAR NECESARIO PARA CALCULAR LOS BALANCES DE LA SIMULACION PARA ESE CICLO"
        
        self._t2 = []
        self._Qn2 = []  
        self._Qn_log=[]
        return None
    
    def _get_Data(self,):
        return {
            "t": self._t,
            "a": self._a,
            "Qn": self._Qn,
            "pi": self._pi,
            "dP": self._dP,
            "Cv": self._Cv,
        }
    
    def _clean_LOG_rhs(self,arrayLog):
    
        qn_by_time = defaultdict(list)
    
        # Recolecta todos los valores para cada tiempo
        for t, qn in arrayLog:
            qn_by_time[t].append(qn)
    
        # Conserva el último valor registrado para cada tiempo
        unique_times = list(qn_by_time.keys())
        unique_times.sort()
    
        t_clean = []
        VAR_clean = []
    
        for t in unique_times:
            t_clean.append(t)
            VAR_clean.append(qn_by_time[t][-1])  
    
        return t_clean, VAR_clean
    
    def _croopTime(self,startTime):
        idx_valid = np.where(np.array(self._t) <= startTime)[0]
        if len(idx_valid) == 0:
            raise ValueError(f"⛔ No hay datos previos en el tanque {self._name} para el tiempo de inicio {startTime:.2f} s")
        last_idx = idx_valid[-1]
        self._t = self._t[:last_idx + 1]
        self._pi = self._pi[:last_idx + 1]
        self._dP = self._dP[:last_idx + 1]
        self._Cv = self._Cv[:last_idx + 1]
        self._a = self._a[:last_idx + 1]
        self._Qn = self._Qn[:last_idx + 1]
    
        self._reset_logs()
        self._required['Results'] = False        
    
    def _plot(self,):
        
        if self._required["Results"] == True:
            fig = plt.figure(figsize=(12, 8))
            grid = fig.add_gridspec(3, 2, width_ratios=[2, 1])
            
            ax1 = fig.add_subplot(grid[0, 0])
            ax2 = fig.add_subplot(grid[1, 0])
            ax3 = fig.add_subplot(grid[2, 0])
            ax4 = fig.add_subplot(grid[:, 1])
    
            ax1.plot(self._t, np.array(self._a) * 100, color='blue', lw=2)
            ax1.set_ylabel("Apertura [%]")
            ax1.set_title("Lógica de control")
            ax1.grid()
    
            ax2.plot(self._t, self._Qn, lw=2, color='green')
            # ax2.plot(self._t2, self._Qn2, lw=2, color='blue')

            ax2.set_ylabel("Caudal [Nm³/h]")
            ax2.set_title("Caudal alimentado")
            ax2.grid()
        
            ax3.plot(self._t, self._pi, lw=2, color="orange")
            ax3.set_ylabel("Presión nodo  [Pa]")
            ax3.set_xlabel("Tiempo [s]")
            ax3.set_title("Presión en nodo 0")
            ax3.grid()
        
            a_range = np.linspace(0, 1, 100)
            ax4.plot(a_range * 100, self._get_Cv(a_range) * 100, lw=2, color='red')
            ax4.set_xlabel("Apertura [%]")
            ax4.set_ylabel("Cv relativo [%]")
            ax4.set_title(f"Tipo de válvula: {self.valve_type}")
            ax4.grid()
        
            plt.suptitle(f"Simulación de válvula - Tipo: {self.valve_type}, Lógica: {self.logic}")
            plt.tight_layout()
            plt.show()
        
# =============================================================================
# 
# =============================================================================
class Tank:
    
    def __init__(self,
                 Name,         # Nombre del equipo
                 Longitud,     # Longitud
                 Diametro,     # Diametro
                 Espesor,      # Espesor
                 prop_gas,     # info gases
                 ):
    
        self._R = 8.314   # Universal gas constant [J/mol/K : Pa*m^3/mol/K]
        self._name = Name
        
        #Design
        self._L = Longitud
        self._D = Diametro
        self._Ri = self._D /2
        self._e = Espesor
        
        self._vol = np.pi * (self._Ri )**2 * self._L
        self._Ai = np.pi * (self._Ri)**2
        self._Al = np.pi * self._D * self._L
        self._Aint = self._Al + self._Ai
        
        #Initial conditions
        self._P0 = None
        self._T0 = None
        self._x0 = None
        self._N0 = None
        self._prop_gas = prop_gas

        #Gas properties info
        self._species = prop_gas["species"]
        self._ncomp = len(prop_gas["species"])
        self._MW = prop_gas["MW"]        # Molecular Weight
        self._mu = prop_gas["mu"]        # Viscosity
        self._Dm = prop_gas["Dm"]        # Molecular diffusivity [m^2/s]
        self._cpg = prop_gas["Cp_molar"] # Specific heat of gas [J/mol/k]
        self._K = prop_gas["k"]          # Thermal conduction in gas phase [W/m/k]
        self._H = prop_gas["H"]          # Enthalpy [J/K]
        
        #Inlet Outlet conditions
        self._Pin=None
        self._Pout=None
        self._Tin=None
        self._xin=None
        
        #Variables
        self._time = None
        self._state_vars = None
        self._previous_vars = None
        self._results = None
        
        self._t=None
        self._P=None
        self._N=None
        self._T=None
        self._x=None
        self._Qloss=None
        
        self._P_log=[]
        self._N_log=[]
        self._T_log=[]
        self._x_log=[]
        self._Qloss_log=[]
        
        self._t2=None
        self._P2=None
        self._N2=None
        self._T2=None
        self._x2=None
        self._Qloss2=None
        
        self._required = {'Design':True,
                          'initialC_info'  : False,
                          'boundaryC_info' : False,
                          'thermal_info'   : False,
                          'Results'        : False}
        
        self._conex = {
            "inlet": None,                       # lista de válvulas de entrada (pueden ser múltiples)
            "where_inlet": None,    # 'top', 'bottom'-> posición relativa
            "outlet": None,           # lista de válvulas de salida
            "where_outlet": None ,
        
            "valves_top": [],       # válvulas conectadas en la parte superior hacia otros equipos
            "valves_bottom": [],    # válvulas conectadas en la parte inferior hacia otros equipos
            "valves_side": [],      # opcional, para mayor generalidad
            
            "units_top": [],       # válvulas conectadas en la parte superior hacia otros equipos
            "units_bottom": [],    # válvulas conectadas en la parte inferior hacia otros equipos
            "units_side": [],      # opcional, para mayor generalidad
                    } 
        
        self._actualTime = 0.0
        
    def initialC_info(self,P0,T0,x0,):
        
        self._P0=P0
        self._T0=T0
        self._x0=np.array(x0)
        self._N0=(self._P0 * self._vol) / (self._R * self._T0)
        self._required['initialC_info'] = True
        return
    
    def boundaryC_info(self,Pin,Tin,xin,Pout):
        self._Pin=Pin
        self._Pout=Pout
        self._xin=np.array(xin)
        self._Tin=Tin
        self._required['boundaryC_info'] = True
        return
    
    def thermal_info(self,adi,kw,hint,hext,Tamb,):
        self._adi=adi
        self._kw=kw
        self._hint=hint
        self._hext=hext
        self._Tamb=Tamb
        self._required['thermal_info'] = True
        return None
    
    def _initialize(self,):
        self._actualTime=0.0        
        self._state_vars = {
            't'   : self._actualTime,
            'P'   : self._P0,
            'T'   : self._T0,
            'x'   : self._x0,
            'N'   : self._N0,
            }
        
        self._t = None
        self._T = None
        self._P = None
        self._x = None
        self._N = None
        self._Qloss = None
        
        self._reset_logs()
        
        self._previous_vars = self._state_vars.copy()
        self._results = None
        self._required['Results'] = False
        
        return None
         
    def _storeData(self, t, P, T, x, N,
                   tend,Pend,Tend,xend,Nend):
        "GAURDA INFO PARA PASAR AL SIGUIENTE CICLO Y VER POR PANTALLA TODOS LOS CICLOS"
        
        if self._actualTime == 0:
            self._t = t
            self._N = N
            self._x = x
            self._T = T
            self._P = P
        else:
            self._t = np.concatenate([self._t, t])
            self._N = np.concatenate([self._N, N])
            self._x = np.hstack([self._x, x])  
            self._T = np.concatenate([self._T, T])
            self._P = np.concatenate([self._P, P])
        
        self._actualTime = tend
        self._previous_vars = self._state_vars.copy()
        self._state_vars = {
            't': tend,
            'P': Pend,
            'T': Tend,
            'x': np.array(xend),
            'N': Nend
        }
        return  None
    
    def _clean_LOG_rhs(self,arrayLog):
    
        qn_by_time = defaultdict(list)
    
        # Recolecta todos los valores para cada tiempo
        for t, qn in arrayLog:
            qn_by_time[t].append(qn)
    
        # Conserva el último valor registrado para cada tiempo
        unique_times = list(qn_by_time.keys())
        unique_times.sort()
    
        t_clean = []
        VAR_clean = []
    
        for t in unique_times:
            t_clean.append(t)
            VAR_clean.append(qn_by_time[t][-1])  
    
        return t_clean, VAR_clean
    
    def _reset_logs(self,):
        self._P_log=[]
        self._N_log=[]
        self._T_log=[]
        self._x_log=[]
        self._Qloss_log=[]
        
        self._t2=None
        self._P2=None
        self._N2=None
        self._T2=None
        self._x2=None
        self._Qloss2 = None
        
        self._aux=None
        self._aux2=None
        return None
    
    def _storeBal(self,t2,P2,T2,x2,N2,Qloss2):
        "GUARDA LOS RESULTADOS lIMPIOS DE RHS PARA LA SIMULACION SIN ACUMULAR NECESARIO PARA CALCULAR LOS BALANCES DE LA SIMULACION PARA ESE CICLO"
        self._t2 = t2
        self._N2 = N2
        self._x2 = x2
        self._T2 = T2
        self._P2 = P2
        self._Qloss2 = Qloss2
        return None
        
    def _croopTime(self, startTime):
        
        idx_valid = np.where(np.array(self._t) <= startTime)[0]
        if len(idx_valid) == 0:
            raise ValueError(f"⛔ No hay datos previos en el tanque {self._name} para el tiempo de inicio {startTime:.2f} s")
        last_idx = idx_valid[-1]
        
        self._t = self._t[:last_idx + 1]
        self._N = self._N[:last_idx + 1]
        self._T = self._T[:last_idx + 1]
        self._P = self._P[:last_idx + 1]
        self._x = self._x[:, :last_idx + 1]
    
        self._actualTime = startTime
        self._results = None
        self._state_vars = {
            't': self._t[-1],
            'N': self._N[-1],
            'T': self._T[-1],
            'P': self._P[-1],
            'x': self._x[:, -1],
        }
    
        self._reset_logs()
        self._required['Results'] = False
        
    def _plot(self,):
        if self._required["Results"] == True:

            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        
            axs[0, 0].plot(self._t, self._P, color='darkblue')
            axs[0, 0].set_title("Presión [Pa]")
            axs[0, 0].grid()
        
            axs[0, 1].plot(self._t, self._T, color='red')
            axs[0, 1].set_title("Temperatura [K]")
            axs[0, 1].grid()
            
            colors = [
                'orange',
                'olive',
                'pink',
                'sky',
                'pale',
                'purple',
            ]
            
            for i, specie in enumerate(self._species):
                axs[1, 0].plot(self._t, self._x[i], label=specie, color =colors[i])
            axs[1, 0].set_title("Fracción molar")
            axs[1, 0].legend()
            axs[1, 0].grid()
        
            axs[1, 1].plot(self._t, self._N, color='darkgreen')
            axs[1, 1].set_title("Moles totales")
            axs[1, 1].grid()
        
            plt.tight_layout()
            plt.show()
        
            return 
        
     
    