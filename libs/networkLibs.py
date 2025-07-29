# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 12:00:16 2025

@author: MiguelCamaraSanz
"""
import time 
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

class Network:
    
    def __init__(self,):
       # Diccionarios de equipos
       self._actualTime = 0.
       self._actualStep = None
       self._actualCycle = 0
       
       self._unit_prefixes = {
            "T": "Tank",
            "C": "Column",
            "R": "Reactor",
            # en el futuro puedes añadir más, como:
            # "S": "Separator", "H": "Heater", ...
        }
            
       self.Tanks = []        
       self.Columns = []      
       self.Reators = []       
       self.Units = []       
       self.Valves = []       

       # Conexiones entre equipos (por nombre)
       self.Networks = []  # [(equipo1, equipo2, valve_name, tipo_conexion)]
       
       self.SimConfig = {
            "start": 0.0,
            "end": None,
            "saveData": 1.0,
            "solver": "BDF",
            "atol": 1e-12,
            "rtol": 1e-12,
            "plot": False,
            "logBal":False,
        }
        
       t = time.time()
       dt = datetime.fromtimestamp(t)
       print(f"🧠 ProSimNet inicializado 📅 Día: {dt.strftime('%Y-%m-%d')} ⏰ Hora: {dt.strftime('%H:%M:%S')}")
    
    def updateNetwork(self):
        unit_dict = {u._name: u for u in self.Units}
        self.Networks = []
    
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
    
            # outlet?
            elif name_core.endswith("o"):
                entry["type"] = "outlet"
                unit_port = name_core[:-1]
                entry["unit_A"] = unit_port[:-1]
                entry["port_A"] = "bottom" if unit_port[-1] == "b" else "top"
    
            # interunit
            else:
                entry["type"] = "interunit"
                # busco la posición de la segunda letra de equipo (T,C,R,...)
                split_points = [i for i,c in enumerate(name_core) if c in "TCR"]
                if len(split_points) >= 2:
                    A_start, B_start = split_points[:2]
                    unitA_raw = name_core[A_start:B_start]
                    unitB_raw = name_core[B_start:]
                    entry["unit_A"] = unitA_raw[:-1]
                    entry["port_A"] = "bottom" if unitA_raw[-1] == "b" else "top"
                    entry["unit_B"] = unitB_raw[:-1]
                    entry["port_B"] = "bottom" if unitB_raw[-1] == "b" else "top"
    
            # registro
            self.Networks.append(entry)
    
            # actualizo valve._conex
            valve._conex = { k: entry[k] for k in ["type","unit_A","unit_B","port_A","port_B"] }
    
            # actualizo cada unidad
            uA = unit_dict.get(entry["unit_A"])
            uB = unit_dict.get(entry["unit_B"])
    
            if entry["type"] == "inlet" and uA:
                uA._conex["inlet"]      = valve._name
                uA._conex["where_inlet"]= entry["port_A"]
    
            elif entry["type"] == "outlet" and uA:
                uA._conex["outlet"]      = valve._name
                uA._conex["where_outlet"]= entry["port_A"]
    
            elif entry["type"] == "interunit":
                # Válvulas
                if uA:
                    key_valves_A = f"valves_{entry['port_A']}"
                    uA._conex[key_valves_A].append(valve._name)
                if uB:
                    key_valves_B = f"valves_{entry['port_B']}"
                    uB._conex[key_valves_B].append(valve._name)

                # Equipos conectados
                if uA and entry['unit_B']:
                    key_units_A = f"units_{entry['port_A']}"
                    uA._conex[key_units_A].append(entry['unit_B'])
                if uB and entry['unit_A']:
                    key_units_B = f"units_{entry['port_B']}"
                    uB._conex[key_units_B].append(entry['unit_A'])
    
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
        return None
    
    def _initialize(self):
        for unit in self.Units:
            unit._initialize()   # Cada clase debe tener su propio _initialize
        for valve in self.Valves:
            valve._initialize()  # Igual para válvulas
        self.actualTime = 0.0
        return None
    
    def _croopTime(self,startTime):
        for unit in self.Units:
            unit._croopTime(startTime)
        for valve in self.Valves:
            valve._croopTime(startTime)
        self.actualTime = startTime
    
    def _plotAll(self,):
        pass
    
    def _plotUnit(self,):
        pass
    
    def _plotValve(self,):
        pass
    
    
    def _solve():
        pass
    
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
             "start": startTime,
             "end": endTime,
             "saveData": saveData,
             "solver": solver,
             "atol": atol,
             "rtol": rtol,
             "plot": plot,
             "logBal" : logBal
         } 
        
        if startTime == 0 or startTime is None:
            self._initialize()
            self,_actualTime=0.
            self._solve()
            if plot:
                self._plotAll()
            
        elif isinstance(startTime, (float, int)) and startTime < self.actualTime:
            for unit in self.Units:
                unit._croopTime(startTime)  # recorta logs y estado al tiempo indicado
            for valve in self.Valves:
                valve._croopTime(startTime)
            self.actualTime = startTime
        elif (startTime == "lastTime") or (startTime == self.actualTime):
            pass
        else:
            raise ValueError(f"⛔ El tiempo solicitado {startTime} s es superior al actual de la simulación: {self.actualTime:.2f} s")
            
