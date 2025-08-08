import os,time 
from datetime import datetime
import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class Network:

# =============================================================================
#     # 1. Constructoe
# =============================================================================    
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


# =============================================================================
#     # 2. Configuración    
# =============================================================================    

    def _printNetwork(self,): # MEJOR TABLA POR UNIDADES COMO EL CHECKBLANCE!!!
        
        """
        ¿Qué hago?
        ----------
        Recolecto y organizo toda la información de conexiones de la red en dos DataFrames:
        - Uno por unidad: resume todas las conexiones de cada equipo, con nombres legibles.
        - Otro por válvula: lista todas las válvulas y a qué unidades/puertos conectan.
        
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        - Para análisis, depuración o exportación clara de la topología de la red.
        - Facilita inspección visual, trazabilidad y comparación de configuraciones complejas.
        - Pensado para ingenieros, desarrolladores o cualquier usuario que necesite documentación estructurada de la red.
        
        Ejemplo de uso
        --------------
        >>> df_units, df_valves = net.get_connections_dataframes()
        >>> print(df_units)
        >>> print(df_valves)
        """
        # ----------- Por UNIDAD ------------
        units_data = []
        for u in self.Units:
            conex = {}
            for k, v in u._conex.items():
                if isinstance(v, list):
                    conex[k] = [vi._name if hasattr(vi, "_name") else str(vi) for vi in v]
                elif hasattr(v, "_name"):
                    conex[k] = v._name
                else:
                    conex[k] = v
            row = {"Unidad": u._name}
            row.update(conex)
            units_data.append(row)
        df_units = pd.DataFrame(units_data).fillna("")
    
        # ----------- Por VÁLVULA ------------
        valves_data = []
        for v in self.Valves:
            conex = {}
            for k, vv in v._conex.items():
                if hasattr(vv, "_name"):
                    conex[k] = vv._name
                elif isinstance(vv, list):
                    conex[k] = [vi._name if hasattr(vi, "_name") else str(vi) for vi in vv]
                else:
                    conex[k] = vv
            row = {"Valvula": v._name}
            row.update(conex)
            valves_data.append(row)
        df_valves = pd.DataFrame(valves_data).fillna("")
        
        print(df_units)
        print(df_valves)
        
        return None

    
    def _mapping_state(self,):
        """
        ¿Qué hago?:
        -----------
        Construye el mapeo global de variables de estado de todas las unidades de la red, estableciendo la relación
        entre el vector plano de integración global (ODE) y las variables locales de cada equipo.
        Genera internamente los índices de inicio y fin de cada subvector correspondiente a cada unidad, así como
        el diccionario de etiquetas y la longitud total del vector de estado.
        
        ¿Cuándo se usa?:
        ----------------
        - Siempre que se añaden o eliminan unidades a la red, o cuando se modifica la estructura interna de equipos.
        - Antes de lanzar cualquier simulación dinámica o integración ODE global (solve_ivp).
        - Recomendado tras modificar el mapping local de alguna unidad (por ejemplo, tras cambiar nodos, especies, etc.).
        
        ¿Qué afecta/modifica?:
        ----------------------
        - Actualiza los atributos internos:
            * self._state_mapping: lista de tuplas (objeto, i_ini, i_fin, labels) para cada unidad.
            * self._state_labels: diccionario index_global -> (objeto, label), útil para debug o acceso rápido.
            * self._state_n_vars: entero con el tamaño total del vector de estado global de la red.
        - No modifica las variables de estado física de los equipos ni los resultados de simulación.
        
        Parámetros:
        -----------
        None
        
        Return:
        -------
        None
        
        Ejemplo de uso:
        ---------------
        >>> net._mapping_state()
        >>> print(net._state_n_vars)
        >>> print(net._state_mapping)
        >>> print(net._state_labels)
        
        Notas:
        ------
        - Es fundamental que el mapping esté actualizado antes de cualquier integración global (ODE) para evitar errores de desalineación.
        - Si una unidad cambia su mapping interno, es obligatorio volver a llamar a este método.
        - El mapping facilita la reconstrucción y almacenamiento eficiente de resultados y estados en simulaciones multi-equipo.
        """
        idx = 0
        label_dict = {}
        mapping = []
        for u in self.Units:
            n_vars, labels = u._get_mapping_()
            mapping.append((u, idx, idx + n_vars, labels))
            for i, label in enumerate(labels):
                label_dict[idx + i] = (u, label)
            idx += n_vars
        self._state_mapping = mapping       # lista de (objeto, i_ini, i_fin, labels)
        self._state_labels = label_dict     # dict index_global -> (objeto, label)
        self._state_n_vars = idx            # tamaño total del vector de estado
        
        return None

    
    def _updateNetwork(self):
        """
        ¿Qué hago?:
        -----------
        Reconstruye y actualiza toda la estructura de conexiones internas de la red de procesos, incluyendo:
        - Clasificación y asignación de equipos (tanques, columnas, reactores, ...).
        - Detección y mapeo de válvulas de entrada, salida e inter-unitarias.
        - Asignación recíproca de conexiones en cada unidad y válvula (puertos, válvulas conectadas, equipos conectados).
        - Construcción de la lista `self.Networks` con la trazabilidad de todas las conexiones.
      
        ¿Cuándo se usa?:
        ----------------
        - Siempre que se añaden, eliminan o modifican equipos o válvulas en la red.
        - Antes de lanzar cualquier simulación dinámica o realizar operaciones de control/rediseño de la red.
        - Tras cualquier cambio manual en los atributos `Units` o `Valves`.
      
        ¿Qué afecta/modifica?:
        ----------------------
        - Actualiza todos los atributos de estructura y conexión:
            * `self.Tanks`, `self.Columns`, `self.Reactors`: listas actualizadas por tipo.
            * `self.Vinlet`, `self.Voutlet`, `self.VinterUnit`: válvulas clasificadas por función.
            * `self.Networks`: lista de diccionarios con la info de cada conexión (equipos, puertos, tipo...).
            * Los atributos `_conex` de cada unidad y válvula (estableciendo referencias cruzadas por objeto).
        - Resetea todas las conexiones previas antes de actualizar.
        - Llama internamente a `_mapping_state` para reconstruir el mapping global del vector de estado.
      
        Parámetros:
        -----------
        None
      
        Return:
        -------
        None
      
        Ejemplo de uso:
        ---------------
        >>> net._updateNetwork()
        >>> net._printNetwork()
      
        Notas:
        ------
        - Es fundamental llamar a este método después de cualquier cambio estructural para evitar inconsistencias o errores en las simulaciones.
        - El sistema reconoce automáticamente los roles de cada válvula por el sufijo de su nombre (`i`, `o` o composición entre equipos).
        - Para redes complejas o arquitecturas personalizadas, este método puede extenderse fácilmente añadiendo nuevas lógicas de clasificación.
        - Si una unidad o válvula no sigue la convención de nombres, puede quedar desconectada (no se lanza excepción por defecto).
        """
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

    
    def _addUnits(self, units_list):
        """
        ¿Qué hago?:
        -----------
        Añade nuevas unidades de proceso (tanques, columnas, reactores, etc.) a la red, garantizando que no se repitan nombres de unidad.
        - Verifica que cada objeto de `units_list` tiene un nombre único respecto a las unidades ya presentes.
        - Agrega únicamente aquellas unidades que aún no existen en la red.
        - Actualiza automáticamente la estructura de la red y todas las conexiones llamando a `_updateNetwork()`.
    
        ¿Cuándo se usa?:
        ----------------
        - Al construir o ampliar la red de proceso, añadiendo nuevos equipos (por ejemplo, tras definir tanques o columnas adicionales).
        - Antes de lanzar simulaciones que requieran incorporar nuevas unidades.
        - Durante la configuración dinámica de la red en aplicaciones modulares o entornos gráficos.
    
        ¿Qué afecta/modifica?:
        ----------------------
        - Modifica el atributo `self.Units`, añadiendo las nuevas unidades.
        - Actualiza toda la estructura de red y conexiones internas mediante `_updateNetwork()`.
        - Puede modificar los atributos `Tanks`, `Columns`, `Reactors`, `Networks`, etc., como consecuencia indirecta.
        - No afecta las válvulas existentes ni las conexiones previas de las unidades ya presentes.
    
        Parámetros:
        -----------
        units_list : list
            Lista de objetos unidad a añadir (deben tener el atributo `_name` único en la red).
    
        Return:
        -------
        None
    
        Ejemplo de uso:
        ---------------
        >>> net.addUnits([T1, C1, R1])
    
        Notas:
        ------
        - Si alguna unidad de `units_list` ya existe en la red (por nombre), se ignora sin error ni sobrescritura.
        - El método es seguro frente a duplicados pero **no** comprueba la validez estructural de los objetos (asume que tienen `_name`).
        - Si se añaden varias unidades con el mismo nombre en la misma llamada, solo la primera se añade (comportamiento estándar de set).
        - Llamar a este método siempre fuerza la actualización total de la red.
        """
        existing = {u._name for u in self.Units}
        for unit in units_list:
            if unit._name not in existing:
                self.Units.append(unit)
        self._updateNetwork()
        return None


    def _addValves_(self, valve_list):
        """
        ¿Qué hago?:
        -----------
        Añade nuevas válvulas a la red de proceso, evitando duplicados por nombre.
        - Comprueba que cada válvula de `valve_list` tiene un nombre único respecto a las válvulas ya presentes en la red.
        - Solo se añaden aquellas válvulas que aún no existen.
        - Tras la inserción, actualiza la estructura de conexiones y la topología de la red llamando a `_updateNetwork()`.
    
        ¿Cuándo se usa?:
        ----------------
        - Al construir o ampliar la red de proceso, añadiendo nuevas válvulas de conexión entre equipos.
        - Antes de lanzar simulaciones, para garantizar que la red contiene todas las válvulas definidas en la topología.
        - Durante la configuración dinámica de la red en aplicaciones modulares o entornos gráficos.
    
        ¿Qué afecta/modifica?:
        ----------------------
        - Modifica el atributo `self.Valves`, añadiendo las nuevas válvulas.
        - Actualiza toda la estructura de red y conexiones internas mediante `_updateNetwork()`.
        - Puede modificar los atributos `Vinlet`, `Voutlet`, `VinterUnit`, y `Networks`, como consecuencia indirecta.
        - No afecta las unidades existentes ni las conexiones previas de las válvulas ya presentes.
    
        Parámetros:
        -----------
        valve_list : list
            Lista de objetos válvula a añadir (deben tener el atributo `_name` único en la red).
    
        Return:
        -------
        None
    
        Ejemplo de uso:
        ---------------
        >>> net._addValves_([vT1b, vT2t, vT1bT2t])
    
        Notas:
        ------
        - Si alguna válvula de `valve_list` ya existe en la red (por nombre), se ignora sin error ni sobrescritura.
        - El método es seguro frente a duplicados pero **no** comprueba la validez estructural de los objetos (asume que tienen `_name`).
        - Si se añaden varias válvulas con el mismo nombre en la misma llamada, solo la primera se añade.
        - Llamar a este método siempre fuerza la actualización total de la red.
        """
        existing = {v._name for v in self.Valves}
        for valve in valve_list:
            if valve._name not in existing:
                self.Valves.append(valve)
        self._updateNetwork()
        return None
    
    
 # =============================================================================
 #      # 3. Inicilizar/Retroceder/Resets/Leer data
 # =============================================================================
    def _reset_conex(self,):
        """
        ¿Qué hago?:
        -----------
        Resetea (borra) todas las conexiones internas de las unidades (tanques, columnas, reactores, etc.) y de las válvulas
        dentro de la red. Deja todos los atributos de conexiones en cada objeto exactamente igual que tras su construcción,
        eliminando cualquier enlace o referencia a otros equipos/válvulas.
    
        ¿Cuándo se usa?:
        ----------------
        - Antes de redefinir la estructura de la red (añadir/quitar unidades o válvulas, cambiar conexiones, etc).
        - Al reconstruir la red desde cero o tras cargar un nuevo caso de simulación.
        - Para asegurar que no quedan residuos, bucles o referencias cruzadas de simulaciones previas.
    
        ¿Qué afecta/modifica?:
        ----------------------
        - Llama internamente a `_reset_conex_()` en cada unidad y válvula, eliminando todas sus conexiones.
        - Deja todos los equipos y válvulas sin enlaces ni referencias a otros objetos de la red.
        - No afecta a las condiciones iniciales, de frontera, ni a los resultados simulados.
    
        Parámetros:
        -----------
        None
    
        Return:
        -------
        None
    
        Ejemplo de uso:
        ---------------
        >>> net._reset_conex()
        # Deja toda la red "desconectada", lista para una nueva topología.
    
        Notas:
        ------
        - Es **imprescindible** llamar a este método antes de reconstruir la red si se van a modificar conexiones, evitarás referencias cruzadas antiguas.
        - Si alguna clase no implementa `_reset_conex_()`, lanzará un error.
        """
        for u in self.Units:
            u._reset_conex_()
        for v in self.Valves:
            v._reset_conex_()
        return None
 
 
    def _reset_logs(self):
        """
        ¿Qué hago?:
        -----------
        Resetea (borra) todos los logs y resultados temporales de simulación almacenados en las unidades (tanques, columnas, etc.)
        y en las válvulas de la red. Deja vacíos todos los registros y arrays asociados a la última simulación realizada.
    
        ¿Cuándo se usa?:
        ----------------
        - Antes de iniciar una nueva simulación, ciclo o escenario.
        - Tras un recorte temporal (`croopTime`), si se va a relanzar la simulación desde ese punto y se quieren eliminar residuos de pasos anteriores.
        - Cuando se requiere liberar memoria antes de una nueva integración.
    
        ¿Qué afecta/modifica?:
        ----------------------
        - Llama internamente a `_reset_logs()` en cada unidad y válvula, eliminando todos los arrays de resultados y logs temporales.
        - No afecta a las condiciones iniciales, de frontera ni a la configuración de las unidades/válvulas.
        - No afecta los datos persistentes ni los parámetros de setup.
    
        Parámetros:
        -----------
        None
    
        Return:
        -------
        None
    
        Ejemplo de uso:
        ---------------
        >>> net._reset_logs()
        # Elimina todos los resultados temporales y logs antes de relanzar la simulación.
    
        Notas:
        ------
        - Es fundamental llamar a este método para evitar acumulación de memoria cuando se realizan múltiples ciclos o escenarios de simulación.
        - No debe llamarse si se quiere conservar resultados previos para análisis posteriores.
        - Si alguna clase no implementa `_reset_logs()`, lanzará un error.
        """
        for unit in self.Units:
            unit._reset_logs_()
        for valve in self.Valves:
            valve._reset_logs_()
        return None
 

    def _initialize(self):
        """
        ¿Qué hago?:
        -----------
        Inicializa el estado de todas las unidades (tanques, columnas, reactores, etc.) y válvulas de la red.
        - Llama internamente al método `_initialize_()` de cada unidad y válvula para reiniciar su estado, arrays internos y variables de simulación.
        - Deja la red lista para arrancar una nueva simulación, partiendo siempre del estado inicial configurado en cada equipo.
        - Reinicia el tiempo interno de la red (`self._actualTime`) a cero.
    
        ¿Cuándo se usa?:
        ----------------
        - Siempre **antes de lanzar una simulación completa** o tras modificar el setup de cualquier equipo o válvula.
        - Obligatorio tras cambios en condiciones iniciales, de frontera, parámetros térmicos, lógica de control, etc.
        - Antes de reiniciar una campaña de simulaciones o arrancar desde un estado limpio.
    
        ¿Qué afecta/modifica?:
        ----------------------
        - Resetea y deja en estado inicial todos los arrays de variables dinámicas y logs de cada unidad y válvula.
        - Borra cualquier residuo de simulaciones anteriores, evitando contaminaciones de estado.
        - El tiempo de simulación global (`self._actualTime`) se resetea a 0.0.
        - No afecta la definición de conexiones ni la topología de la red.
    
        Parámetros:
        -----------
        None
    
        Return:
        -------
        None
    
        Ejemplo de uso:
        ---------------
        >>> net._initialize()
        # Deja toda la red lista para simular desde t=0
    
        Notas:
        ------
        - Es imprescindible llamar a este método tras definir/ajustar el setup y **antes de cualquier integración ODE**.
        - Cada clase de equipo y válvula debe implementar correctamente su método `_initialize_` para garantizar compatibilidad.
        - Si algún equipo no tiene el método `_initialize_`, lanzará un error de atributo.
        """
        for unit in self.Units:
            unit._initialize_()   # Cada clase debe tener su propio _initialize
        for valve in self.Valves:
            valve._initialize_()  # Igual para válvulas
        self._updateNetwork()
        self._actualTime = 0.0
        return None

    
    def _croopTime(self, startTime):
        """
        ¿Qué hago?:
        -----------
        Recorta (trunca) todos los arrays de resultados y logs de todas las unidades (tanques, columnas, reactores, etc.)
        y válvulas de la red hasta el tiempo 'startTime'. Elimina cualquier información posterior a ese instante,
        dejando la red lista para relanzar la simulación desde ese punto, por ejemplo, con nuevas condiciones de operación
        o lógica de control.
    
        ¿Cuándo se usa?:
        ----------------
        - Cuando necesitas retroceder la simulación a un instante anterior y relanzar desde ahí:
            * Para aplicar una lógica de control dependiente del estado en cierto momento.
            * Para optimizar condiciones a partir de un punto concreto.
            * Para limpiar residuos de simulaciones anteriores y evitar acumulación de memoria.
    
        ¿Qué afecta/modifica?:
        ----------------------
        - Llama internamente a `_croopTime_()` en cada unidad y válvula, recortando sus resultados y logs hasta `startTime`.
        - Actualiza el atributo `self._actualTime` al nuevo instante de simulación.
        - Borra todos los datos de simulación posteriores a `startTime` en la red y sus componentes.
        - No afecta a la configuración base ni al setup de los equipos.
    
        Parámetros:
        -----------
        startTime : float
            Nuevo instante final de la simulación (en segundos). Todo lo que ocurra después de este tiempo se elimina.
    
        Return:
        -------
        None
    
        Ejemplo de uso:
        ---------------
        >>> net._croopTime(25.0)
        # Elimina todos los resultados y logs posteriores a 25 segundos en toda la red.
    
        Notas:
        ------
        - Es fundamental para relanzar simulaciones desde un estado parcial, optimizar escenarios o aplicar control avanzado.
        - Lanza error si alguna unidad o válvula no tiene datos previos para el tiempo solicitado.
        - Asegúrate de llamar a este método antes de reiniciar la integración si quieres evitar inconsistencias temporales.
        """
        for unit in self.Units:
            unit._croopTime_(startTime)
        for valve in self.Valves:
            valve._croopTime_(startTime)
        self._actualTime = startTime
    

    def _readCase(self,):
        return None


    def _readData(self,):
        return None
    
    
    def _readCaseData(self,):
        return None
    

# =============================================================================
#     # 4.Limpieza
# =============================================================================
    def _clean_logs(self):
        """
        ¿Qué hago?:
        -----------
        Limpia y sincroniza todos los logs temporales de integración de la red, procesando los registros de
        cada unidad (tanque, columna, reactor, etc.) y de cada válvula conectada. 
        Elimina duplicados temporales, solapamientos y deja los resultados listos para análisis de balances,
        visualización y postprocesado. **Al finalizar, marca la simulación como completada para cada unidad.**
        
        ¿Cuándo se usa?:
        ----------------
        - Al finalizar una simulación dinámica, antes de calcular balances o visualizar resultados.
        - Siempre antes de exportar datos o hacer integrales sobre la historia temporal.
        - Es obligatorio antes de calcular balances de masa o energía perfectamente cerrados.
        - **Este método marca el final efectivo de la simulación: tras ejecutarlo, todos los resultados y logs están preparados para su análisis.**
        
        ¿Qué afecta/modifica?:
        ----------------------
        - Llama a `unit._clean_LOG_unit()` en todas las unidades (tanques, columnas, etc.), procesando sus logs internos.
        - Llama a `valve._clean_LOG_valve()` en todas las válvulas, limpiando sus logs de apertura, caudal y estado.
        - Actualiza internamente los arrays de resultados temporales (`_t2`, `_N2`, etc.) en cada objeto.
        - **Marca el flag interno `RunProces=True` en cada unidad, indicando que la simulación ha finalizado correctamente.**
        
        Parámetros:
        -----------
        None
        
        Return:
        -------
        None
        
        Ejemplo de uso:
        ---------------
        >>> net._clean_logs()
        # Después de esto puedes analizar los resultados o comprobar los balances.
        
        Notas:
        ------
        - Fundamental llamar a este método tras cada simulación, para asegurar la consistencia temporal
          de todos los registros, especialmente cuando hay múltiples pasos en el mismo tiempo.
        - Si se omite, los balances pueden ser incorrectos por duplicidad de datos.
        - **Tras ejecutar este método, la red y todas las unidades quedan listas para análisis, validaciones o visualización final.**
        """
        for unit in self.Units:
            unit._clean_LOG_unit_()
            unit._required["RunProces"] = True
    
        for valve in self.Valves:
            valve._clean_LOG_valve_()
        return None

    
# =============================================================================
#     # 6.Dinamica 
# =============================================================================
    def _rhs(self, ti, y):
        """
        Calcula el vector de derivadas del sistema completo (right-hand side, RHS) para el solver ODE.
    
        Este método recorre todas las unidades de la red, actualiza sus estados locales a partir del vector global `y`, 
        calcula la derivada local (`dy_local`) de cada unidad y compone el vector derivada global `dy`.
    
        Parameters
        ----------
        ti : float
            Tiempo actual de integración.
        y : ndarray
            Vector de estado global del sistema, concatenando los estados de todas las unidades según el mapeo definido en 
            `self._state_mapping`.
    
        Returns
        -------
        dy : ndarray
            Vector de derivadas global, del mismo tamaño y orden que `y`, con las derivadas calculadas por cada unidad.
    
        Notas
        -----
        - El mapeo `self._state_mapping` define para cada unidad el rango de índices en el vector global.
        - Cada unidad debe implementar los métodos `_set_State_ylocal_(ti, y_local)` y `_rhs_()`.
        - Este método está diseñado para ser utilizado como función de derivadas en integradores ODE tipo `solve_ivp`.
        """
        
        dy = np.zeros_like(y)
        for (unit, i_init, i_fin, labels) in self._state_mapping:
            y_local = y[i_init:i_fin]
            unit._set_State_ylocal_(ti, y_local)
        
        for (unit, i_init, i_fin, labels) in self._state_mapping:
            dy_local = unit._rhs_()
            dy[i_init:i_fin] = dy_local
    
        return dy


    def _solve(self,):
        """
        Resuelve la dinámica completa del sistema acoplado integrando las ecuaciones diferenciales en el tiempo.
        
        Inicializa el vector de estado global `y0` a partir del estado actual de cada unidad, define el vector de tiempos
        de guardado, y ejecuta el solver numérico configurado (por defecto `solve_ivp`). Al finalizar, almacena los resultados
        y registros asociados.
        
        Returns
        -------
        None
        
        Guarda
        ------
        self._results : OdeResult
            Objeto resultado de `solve_ivp` con los tiempos, estados y meta-información de la simulación.
        self._logs, self._storeData()
            Llama a métodos auxiliares para limpiar logs y almacenar datos en disco o en memoria según configuración.
        
        Raises
        ------
        RuntimeError
            Si la integración numérica falla (`solve_ivp` no tiene éxito).
        
        Notas
        -----
        - El estado inicial global `y0` se construye concatenando los estados locales de cada unidad del sistema.
        - Los parámetros de simulación se definen en `self.SimConfig`, que debe incluir:
            - 'startTime': tiempo inicial
            - 'endTime': tiempo final
            - 'saveData': intervalo de guardado de resultados
            - 'solver': método numérico para solve_ivp (e.g. 'BDF', 'RK45', etc.)
            - 'atol', 'rtol': tolerancias absolutas y relativas del solver
        - La función de derivadas usada es `self._rhs`.
        - El método imprime resumen del tiempo simulado y tiempo computacional.
        
        Ejemplo
        -------
        >>> pro_net = ProSimNet()
        >>> pro_net._solve()
        >>> results = pro_net._results
        
        """
        
        y0 = np.zeros(self._state_n_vars)
        for (unit, i_ini, i_fin, labels) in self._state_mapping:
            y0[i_ini:i_fin] = unit._get_State_ylocal_()
            
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
                         t_eval=t_eval,
                         atol=self.SimConfig['atol'],
                         rtol=self.SimConfig['rtol'])
    
        t_end = time.time()
        if sol.success:
            self._results = sol
            self._clean_logs()
            self._storeData()
            print(f"\nProSimNet  terminado con éxito. Tiempo simulado//computacional: {sol.t[-1] - sol.t[0]:.1f}//{t_end - t_start:.2f} s.")
        else:
            raise RuntimeError(f"⛔ solve_ivp falló: {sol.message}")
    
        return None
    
    
    def _checkBalances(self):
        """
        Calcula y presenta el balance global de masa (total y por especie) y de energía para todas las unidades del sistema.
    
        Para cada unidad del sistema se recopilan los principales términos de balance de moles y energía (inicial, final, entradas, salidas, intercambio con otras unidades y error relativo), tanto de forma global como desglosado por especie química. El resultado se presenta en un DataFrame organizado con unidades asociadas, facilitando la auditoría del cierre de balances de simulación.
    
        Returns
        -------
        None
    
        Imprime
        -------
        df_balances : pandas.DataFrame
            Tabla resumen con los balances de masa (totales y por especie) y energía de todas las unidades, incluyendo los errores relativos.
    
        Notas
        -----
        - La columna 'Units' especifica las unidades físicas asociadas a cada variable del balance.
        - Para cada especie química en `self._species` se añaden columnas específicas de balance.
        - Cada unidad debe implementar el método `_checkBalances_()`, que debe devolver un diccionario de resultados con los términos de balance.
        - Es útil para validar la conservación de masa y energía, detectar fugas numéricas o inconsistencias en simulaciones acopladas.
    
        Ejemplo
        -------
        >>> net = ProSimNet()
        >>> net._checkBalances()
        # Muestra por pantalla la tabla de balances para revisión manual.
        """
        unidades = {
            "N_init": "mol",
            "N_end": "mol",
            "deltaN": "mol",
            "N_in": "mol",
            "N_out": "mol",
            "N_intercon": "mol",
            "N_total": "mol",
            "%Error_N": "%",
        }
        
        # Añade dinámicamente para cada especie
        for s in self._species:
            unidades[f"Ni_init_{s}"]      = "mol"
            unidades[f"Ni_end_{s}"]       = "mol"
            unidades[f"deltaNi_{s}"]      = "mol"
            unidades[f"Ni_in_{s}"]        = "mol"
            unidades[f"Ni_out_{s}"]       = "mol"
            unidades[f"Ni_intercon_{s}"]  = "mol"
            unidades[f"Ni_total_{s}"]     = "mol"
            unidades[f"%Error_Ni_{s}"]    = "%"
        
        # Energía (todas en kJ salvo %Error_H)
        unidades.update({
            "H_init": "kJ",
            "H_end": "kJ",
            "deltaH": "kJ",
            "H_in": "kJ",
            "H_out": "kJ",
            "H_intercon": "kJ",
            "H_total": "kJ",
            "%Error_H": "%",
        })
        
        
        results_dict = {}
        for unit in self.Units:
            results_dict[unit._name] = unit._checkBalances_()
        results_dict["Units"] = unidades
        df_balances = pd.DataFrame(results_dict)
        
        print(df_balances)

        return None
        
    
    def _run(self,
             startTime,
             endTime,
             saveData,
             solver,
             atol,
             rtol,
             plot,
             logBal):
        
        """
        Ejecuta una simulación completa del sistema según la configuración temporal y numérica indicada.
    
        Inicializa el sistema o recorta/restaura estados según el punto de partida, lanza el integrador numérico y, 
        opcionalmente, genera gráficos y/o calcula los balances finales. Permite simular desde t=0, continuar una simulación 
        previa, o recortar el historial hasta un tiempo anterior.
    
        Parameters
        ----------
        startTime : float, int o str
            Tiempo inicial de la simulación. Puede ser 0, un valor menor que el tiempo actual (para recorte), 
            'lastTime' o el tiempo actual (para continuar).
        endTime : float o int
            Tiempo final de la simulación.
        saveData : float
            Intervalo temporal de guardado de resultados.
        solver : str
            Nombre del método numérico para el integrador ODE (e.g., 'BDF', 'RK45').
        atol : float
            Tolerancia absoluta para el solver.
        rtol : float
            Tolerancia relativa para el solver.
        plot : bool
            Si True, genera gráficos de los resultados al finalizar la simulación.
        logBal : bool
            Si True, calcula e imprime el balance global de masa y energía al finalizar.
    
        Returns
        -------
        None
    
        Efectos
        -------
        - Almacena la configuración de simulación en `self.SimConfig`.
        - Ejecuta los métodos internos `_initialize`, `_solve`, `_plotAll`, `_checkBalances`, según corresponda.
        - Actualiza el atributo `self._actualTime` al tiempo final de la simulación.
        - Puede lanzar excepciones en caso de errores de configuración temporal.
    
        Raises
        ------
        ValueError
            Si el tiempo de inicio solicitado es superior al tiempo actual de la simulación.
    
        Notas
        -----
        - Permite distintos escenarios: simulación desde cero, recorte temporal, o continuación desde el último estado.
        - El flujo de simulación es totalmente automático según los parámetros introducidos.
        - Los balances calculados ayudan a validar la calidad de la simulación.
    
        Ejemplo
        -------
        >>> net = ProSimNet()
        >>> net._run(
                startTime=0,
                endTime=100,
                saveData=1,
                solver='BDF',
                atol=1e-6,
                rtol=1e-4,
                plot=True,
                logBal=True
            )
        """
        
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
            if logBal:
                self._checkBalances()
            self._actualTime=endTime
            
        elif isinstance(startTime, (float, int)) and startTime < self._actualTime:
            self._croopTime(startTime)
            self._solve()
            if plot:
                self._plotAll()
            if logBal:
                self._checkBalances()
            self._actualTime=endTime
            
        elif (startTime == "lastTime") or (startTime == self._actualTime):
            startTime = self._actualTime
            self.SimConfig["startTime"]=startTime
            self._reset_logs()
            self._solve()
            if plot:
                self._plotAll()
            if logBal:
                self._checkBalances()
            self._actualTime=endTime
            
        else:
            raise ValueError(f"⛔ El tiempo solicitado {startTime} s es superior al actual de la simulación: {self._actualTime:.2f} s")
            
        return None
            

# =============================================================================
#     # 7.Save Simulation/Case/Data
# =============================================================================
    def _storeData(self,):
        for unit in self.Units:
            idx = [m for m in self._state_mapping if m[0] is unit]

            if idx:
                i_ini, i_fin = idx[0][1], idx[0][2]
                t = self._results.t
                y = self._results.y[i_ini:i_fin, :]
                unit._storeData_(t, y)
        
        for valve in self.Valves:
            valve._storeData_(self.SimConfig["startTime"], self.SimConfig["endTime"])
    
        return None


    def _writeCase(self, ruta, name_archivo_case): # EN DESARROLLO !!
        if not os.path.isdir(ruta):
            raise FileNotFoundError(f"⛔ La ruta '{ruta}' no existe.")
    
        lista = []
        for u in self.Units:
            name, code = u._writeCase_()
            lista.append({"type": "unit", "name": name, "case_code": code})
    
        for v in self.Valves:
            name, code = v._writeCase_()
            lista.append({"type": "valve", "name": name, "case_code": code})
    
        df = pd.DataFrame(lista, columns=["type", "name", "case_code"])
    
        fullpath = os.path.join(ruta, name_archivo_case+"_case")
        df.to_csv(fullpath, sep="\t", index=False)
        print(f"✅ Caso guardado correctamente en: {fullpath}")
    
        return None


    def _writeData(self, ruta, name_archivo_data): # EN DESARROLLO !!

        if not os.path.isdir(ruta):
            raise FileNotFoundError(f"⛔ La ruta '{ruta}' no existe.")
    
        lista = []
        for u in self.Units:
            name, code = u._writeData_()
            lista.append({"type": "unit", "name": name, "data_code": code})
    
        for v in self.Valves:
            name, code = v._writeData_()
            lista.append({"type": "valve", "name": name, "data_code": code})
    
        df = pd.DataFrame(lista, columns=["type", "name", "data_code"])
    
        fullpath = os.path.join(ruta, name_archivo_data+"_data")
        df.to_csv(fullpath, sep="\t", index=False)
        print(f"✅ Datos de simulación guardados correctamente en: {fullpath}")
    
        return None

            
    def _writeCaseData(self, ruta, name_archivo):
        self.writeCase(ruta, name_archivo)
        self.writeData(ruta, name_archivo)
        print(f"✅ Setup y datos de simulación guardados en: {ruta}")
        return None


# =============================================================================
#     # 8.Plots y postProceso
# =============================================================================   
    def _plotAll(self,):
        for unit in self.Units:
            unit._plot_()
    
    
    def _plotUnit(self,): # EN DESARROLLO !!
        pass
    
    
    def _plotValve(self,): # EN DESARROLLO !!
        pass
    
    
    
