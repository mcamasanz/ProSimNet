# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 12:00:16 2025

@author: MiguelCamaraSanz
"""
import time 
from datetime import datetime
import numpy as np
from numpy import trapz
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class Network:
    
    def __init__(self,
                 prop_gas,
                 Units,
                 Valves,
                 ):
       # Diccionarios de equipos
       self._actualTime = 0.
       self._actualStep = None
       self._actualCycle = 0
       self._R = 8.314
       self._Tref = 298.15
       self._Nm3_2_mol = 1.01325e5 / (self._R * 273.15)
       
       self._unit_prefixes = {
            "T": "Tank",
            "C": "Column",
            "R": "Reactor",
            # en el futuro puedes añadir más, como:
            # "S": "Separator", "H": "Heater", ...
        }
           
       #Gas properties info
       self._species = prop_gas["species"]
       self._ncomp = len(prop_gas["species"])
       self._MW = prop_gas["MW"]        # Molecular Weight
       self._mu = prop_gas["mu"]        # Viscosity
       self._Dm = prop_gas["Dm"]        # Molecular diffusivity [m^2/s]
       self._cpg = prop_gas["Cp_molar"] # Specific heat of gas [J/mol/k]
       self._K = prop_gas["k"]          # Thermal conduction in gas phase [W/m/k]
       self._H = prop_gas["H"]          # Enthalpy [J/K]

       self.Tanks = []        
       self.Columns = []      
       self.Reactors = []
       self.Vinlet = []
       self.Voutlet = []
       self.VinterUnit = []
       
       self.Units = Units      
       self.Valves = Valves     

       # Conexiones entre equipos (por nombre)
       self.Networks = []        # [(equipo1, equipo2, valve_name, tipo_conexion)]
       self._state_mapping = []  # lista de (objeto, i_ini, i_fin, labels)
       self._state_labels  = {}  # dict index_global -> (objeto, label)
       self._n_state_vars  = 0 

       self.SimConfig = {
            "startTime": 0.0,
            "endTime": None,
            "saveData": 1.0,
            "solver": "BDF",
            "atol": 1e-12,
            "rtol": 1e-12,
            "plot": False,
            "logBal":False,
        }
       
       self._results=None
        
       t = time.time()
       dt = datetime.fromtimestamp(t)
       print(f"🧠 ProSimNet inicializado 📅 Día: {dt.strftime('%Y-%m-%d')} ⏰ Hora: {dt.strftime('%H:%M:%S')}")

    def _printNetwork(self,):
        print("=== CONEXIONES DE UNIDADES ===")
        for u in self.Units:
            print(f"Unidad: {u._name}")
            # Mostramos solo referencias por nombre para que sea legible
            conex = {}
            for k, v in u._conex.items():
                if isinstance(v, list):
                    conex[k] = [vi._name if hasattr(vi, "_name") else str(vi) for vi in v]
                elif hasattr(v, "_name"):
                    conex[k] = v._name
                else:
                    conex[k] = v
            print(conex)
            print()
        
        print("=== CONEXIONES DE VÁLVULAS ===")
        for v in self.Valves:
            print(f"Válvula: {v._name}")
            conex = {}
            for k, vv in v._conex.items():
                if hasattr(vv, "_name"):
                    conex[k] = vv._name
                elif vv is not None and hasattr(vv, "__class__") and vv.__class__.__name__ in ["Tank", "Column", "Reactor"]:
                    conex[k] = vv._name
                else:
                    conex[k] = vv
            print(conex)
            print()
        
        print("=== CONEXIONES DE LA RED (self.Networks) ===")
        for net in self.Networks:
            resumen = {}
            for k, v in net.items():
                if hasattr(v, "_name"):
                    resumen[k] = v._name
                elif isinstance(v, list):
                    resumen[k] = [vi._name if hasattr(vi, "_name") else str(vi) for vi in v]
                else:
                    resumen[k] = v
            print(resumen)      
            
        return None
    
    def _updateNetwork(self):
        unit_dict = {u._name: u for u in self.Units}
        self.Networks = []
        self.Tanks = []
        self.Columns = []
        self.Reactors = []
        self.Vinlet = []
        self.Voutlet = []
        self.VinterUnit = []
        self._reset_conex()
        # Clasificamos los equipos
        for u in self.Units:
            if u._name.startswith("T"):
                self.Tanks.append(u)
            elif u._name.startswith("C"):
                self.Columns.append(u)
            elif u._name.startswith("R"):
                self.Reactors.append(u)
    
        for valve in self.Valves:
            name = valve._name
            if not name.startswith("v"):
                continue
    
            name_core = name[1:]
            entry = {
                "valve": name,
                "type": None,
                "unit_A": None,
                "port_A": None,
                "unit_B": None,
                "port_B": None,
            }
    
            # inlet?
            if name_core.endswith("i"):
                entry["type"] = "inlet"
                unit_port = name_core[:-1]  
                entry["unit_A"] = unit_port[:-1]
                entry["port_A"] = "bottom" if unit_port[-1] == "b" else "top"
                self.Vinlet.append(valve)
            # outlet?
            elif name_core.endswith("o"):
                entry["type"] = "outlet"
                unit_port = name_core[:-1]
                entry["unit_A"] = unit_port[:-1]
                entry["port_A"] = "bottom" if unit_port[-1] == "b" else "top"
                self.Voutlet.append(valve)
            # interunit
            else:
                entry["type"] = "interunit"
                # Busco la posición de la segunda letra de equipo (T,C,R,...)
                split_points = [i for i, c in enumerate(name_core) if c in "TCR"]
                if len(split_points) >= 2:
                    A_start, B_start = split_points[:2]
                    unitA_raw = name_core[A_start:B_start]
                    unitB_raw = name_core[B_start:]
                    entry["unit_A"] = unitA_raw[:-1]
                    entry["port_A"] = "bottom" if unitA_raw[-1] == "b" else "top"
                    entry["unit_B"] = unitB_raw[:-1]
                    entry["port_B"] = "bottom" if unitB_raw[-1] == "b" else "top"
                    self.VinterUnit.append(valve)
    
            # --- Cambios aquí ---
            # Convertimos los nombres en objetos
            uA = unit_dict.get(entry["unit_A"])
            uB = unit_dict.get(entry["unit_B"])
            entry["unit_A"] = uA
            entry["unit_B"] = uB
    
            # Registro para debug y trazabilidad de conexiones
            self.Networks.append(entry)
    
            # actualizo valve._conex
            valve._conex = {k: entry[k] for k in ["type", "unit_A", "unit_B", "port_A", "port_B"]}
    
            # actualizo cada unidad (ojo: ahora uA y uB son objetos o None)
            if entry["type"] == "inlet" and uA:
                uA._conex["inlet"] = valve
                uA._conex["where_inlet"] = entry["port_A"]
    
            elif entry["type"] == "outlet" and uA:
                uA._conex["outlet"] = valve
                uA._conex["where_outlet"] = entry["port_A"]
    
            elif entry["type"] == "interunit":
                # Válvulas
                if uA:
                    key_valves_A = f"valves_{entry['port_A']}"
                    uA._conex[key_valves_A].append(valve)
                if uB:
                    key_valves_B = f"valves_{entry['port_B']}"
                    uB._conex[key_valves_B].append(valve)
                # Equipos conectados
                if uA and uB:
                    key_units_A = f"units_{entry['port_A']}"
                    key_units_B = f"units_{entry['port_B']}"
                    uA._conex[key_units_A].append(uB)
                    uB._conex[key_units_B].append(uA)
        self._mapping_state()
        return None
    
    def addUnits(self,units_list):
        """
        Añade unidades (tanques, columnas, reactores, etc.) a la red si no existen.
        unit_list: lista de objetos con atributo `_name` único.
        """
        existing = {u._name for u in self.Units}
        for unit in units_list:
            if unit._name not in existing:
                self.Units.append(unit)
        self._updateNetwork()        
        return None
    
    def addValves(self, valve_list):
        """
        Añade válvulas a la red si no existen.
        valve_list: lista de objetos con atributo `_name` único.
        """
    
        existing = {v._name for v in self.Valves}
        for valve in valve_list:
            if valve._name not in existing:
                self.Valves.append(valve)
        
        self._updateNetwork()
        return None
    
    def _mapping_state(self,):
        idx = 0
        label_dict = {}
        mapping = []
        for u in self.Units:
            n_vars, labels = u._get_mapping()
            mapping.append((u, idx, idx + n_vars, labels))
            for i, label in enumerate(labels):
                label_dict[idx + i] = (u, label)
            idx += n_vars
        self._state_mapping = mapping       # lista de (objeto, i_ini, i_fin, labels)
        self._state_labels = label_dict     # dict index_global -> (objeto, label)
        self._state_n_vars = idx            # tamaño total del vector de estado
        
        return None
    
    def _initialize(self):
        for unit in self.Units:
            unit._initialize()   # Cada clase debe tener su propio _initialize
        for valve in self.Valves:
            valve._initialize()  # Igual para válvulas
        self._actualTime = 0.0
        return None
    
    def _croopTime(self,startTime):
        for unit in self.Units:
            unit._croopTime(startTime)
        for valve in self.Valves:
            valve._croopTime(startTime)
        self._actualTime = startTime
    
    def _reset_conex(self,):
        for u in self.Units:
            u._reset_conex()
            
        for v in self.Valves:
            v._reset_conex()
        return None
         
    def _clean_logs(self):
        for unit in self.Units:
            unit._clean_LOG_unit()

        for valve in self.Valves:
            valve._clean_LOG_valve()
        return None    
    
    def _reset_logs(self):
        for unit in self.Units:
            unit._reset_logs()
        for valve in self.Valves:
            valve._reset_logs()
        return None    
    
    def _storeData(self,):
        for unit in self.Units:
            idx = [m for m in self._state_mapping if m[0] is unit]
            if idx:
                i_ini, i_fin = idx[0][1], idx[0][2]
                t = self._results.t
                y = self._results.y[i_ini:i_fin, :]
                unit._storeData(t, y)
        
        for valve in self.Valves:
            valve._storeData(self.SimConfig["startTime"],self.SimConfig["endTime"])
            valve._required['Results'] = True

        return None
            
    def _plotAll(self,):
        pass
    
    def _plotUnit(self,):
        pass
    
    def _plotValve(self,):
        pass
    
    def _rhs(self,ti,y):
        dy = np.zeros_like(y)
        for (unit, i_init, i_fin, labels) in self._state_mapping:
            y_local = y[i_init:i_fin]
            unit._set_State(ti,y_local)
        
        for (unit, i_init, i_fin, labels) in self._state_mapping:
            dy_local = unit._rhs()
            dy[i_init:i_fin] = dy_local

        return dy
    
    def _solve(self,):
        y0 = np.zeros(self._state_n_vars)
        for (unit, i_ini, i_fin, labels) in self._state_mapping:
            y0[i_ini:i_fin] = unit._get_State()
            
        
        t_eval = np.linspace(self.SimConfig['startTime'],
                             self.SimConfig['endTime'],
                             int((self.SimConfig['endTime']-self.SimConfig['startTime'])/
                                 self.SimConfig['saveData']+1))
        
        t_start = time.time()
        sol = solve_ivp(self._rhs,
                        [self.SimConfig['startTime'],
                         self.SimConfig['endTime']],
                         y0,
                         method=self.SimConfig['solver'],
                         t_eval= t_eval,
                         atol = self.SimConfig['atol'],
                         rtol = self.SimConfig['rtol'])
     
        t_end = time.time()
        if sol.success:
            self._results=sol
            self._clean_logs()
            self._storeData()
            
            print(f"solve_ivp terminado con éxito.\nTiempo simulado: {sol.t[-1] - sol.t[0]:.1f} s.\nTiempo simulado: {t_end - t_start:.2f} s.")
        else:
            raise RuntimeError(f"⛔ solve_ivp falló: {sol.message}")
        
    
        return None
    
    def Run(self,
                startTime,
                endTime,
                saveData,
                solver,
                atol,
                rtol,
                plot,
                logBal):
        
        self.SimConfig = {
             "startTime": startTime,
             "endTime": endTime,
             "saveData": saveData,
             "solver": solver,
             "atol": atol,
             "rtol": rtol,
             "plot": plot,
             "logBal" : logBal
         } 
        
        if startTime == 0 or startTime is None:
            self._initialize()
            self._solve()
            if plot:
                self._plotAll()
            self._actualTime=endTime
            
        elif isinstance(startTime, (float, int)) and startTime < self._actualTime:
            self._croopTime(startTime)
            self._solve()
            if plot:
                self._plotAll()
            self._actualTime=endTime
            
        elif (startTime == "lastTime") or (startTime == self._actualTime):
            startTime = self._actualTime
            self.SimConfig["startTime"]=startTime
            self._reset_logs()
            self._solve()
            if plot:
                self._plotAll()
            self._actualTime=endTime
            
        else:
            raise ValueError(f"⛔ El tiempo solicitado {startTime} s es superior al actual de la simulación: {self._actualTime:.2f} s")
            
        return None
            
