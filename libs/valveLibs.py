import numpy as np
import matplotlib.pyplot as plt
import pickle,base64

class Valve:
# =============================================================================
#     # 1. Constructor    
# =============================================================================
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
        self._R = 8.314

        self._t  = None   # Array de tiempos de simulación [s]
        self._pA = None   # Presión en unidad A [Pa]
        self._pB = None   # Presión en unidad B [Pa]
        self._a  = None   # Apertura de válvula [0-1]
        self._dP = None   # Diferencia de presión [Pa]
        self._Cv = None   # Coeficiente de caudal instantáneo
        self._Qn = None   # Caudal instantáneo [Nm³/h]
        self._Qn_log = None # Guarda resultados desordenados de rhs
        self._t_log = None # Guarda resultados desordenados de rhs
        self._t2 = None   # Array de tiempos limpio de simulación [s]
        self._Qn2  = None # Caudal instantáneo [Nm³/h] limpio y ordenado de rhs 
        
        # Parámetros de configuración
        self.a_max = a_max 
        self.a_min = a_min 
        self.Cv_max = Cv_max
        self.valve_type = valve_type.lower()
        self.logic = logic.lower()
        self.logic_params = logic_params if logic_params else {}
        self.opening_direction = opening_direction.lower()
        self._validate_()  
        
        self._required = {
            'Design': True,
            'Logical_info': True,
            'Results': False
        }
        
        self._conex = {
            "type": None,     # tipo de válvula (inlet, outlet, inter-unit, etc.)
            "unit_A": None,   # equipo origen
            "unit_B": None,   # equipo destino (None si es inlet u outlet)
            "port_A": None,   # "bottom" | "top" | "side"
            "port_B": None,   # "bottom" | "top" | "side"
        }

        self._setup_vars = [
            '_name', 'Cv_max', 'valve_type', 'a_max', 'a_min', 'logic',
            'logic_params', 'opening_direction'
        ]
        
        
        self.case=None
        self.data=None

    def _validate_(self):
        """
        ¿Qué hago?
        -----------
        Compruebo que la configuración de la válvula es válida, revisando que los parámetros críticos (tipo de válvula, lógica de apertura y dirección de apertura) están dentro de los valores permitidos. Además, verifico que los límites de apertura mínima y máxima sean coherentes y estén dentro del rango [0, 1].
    
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        Este método asegura la robustez y la integridad de la definición del objeto Valve antes de cualquier simulación o uso. Previene errores de usuario y evita que la simulación se ejecute con configuraciones incompatibles o físicamente imposibles. Es fundamental para debugging, validación de setups y para mantener la coherencia en el modelado de procesos modulares tipo DWSIM/Aspen.
    
        Ejemplo de uso
        --------------
        >>> v = Valve("vT1b", valve_type="linear", logic="sigmoid", opening_direction="co")
        >>> v._validate_()  # No lanza errores: todo correcto
        >>> v2 = Valve("vT2t", valve_type="no-existe", logic="linear")
        >>> v2._validate_() # Lanza ValueError: tipo de válvula no permitido
    
        Avisos
        ------
        - Si algún parámetro no es válido, lanza una excepción ValueError explicativa y detiene la ejecución.
        - Los valores de apertura (`a_min`, `a_max`) deben cumplir: 0 ≤ a_min < a_max ≤ 1.
        - Este método se llama automáticamente al instanciar una válvula, pero puede ejecutarse manualmente tras cambiar atributos críticos.
        - No modifica el estado del objeto; solo verifica y, si hay error, lanza excepción.
        """
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
        return None

            
# =============================================================================
#     # 2. Configuración    
# =============================================================================
    def update(self, config=None, **kwargs):
        """
        ¿Qué hago?
        -----------
        Actualizo la configuración del objeto válvula (`Valve`) de forma flexible, permitiendo modificar uno o varios de sus atributos críticos (como `Cv_max`, `valve_type`, `logic`, `a_min`, `a_max`, etc.) tras la creación del objeto. Acepto la configuración tanto como diccionario (`config`) como mediante argumentos clave-valor (`**kwargs`).  
        Al finalizar, valido automáticamente la coherencia de la nueva configuración llamando a `_validate_()`.
    
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        Este método permite modificar la configuración de la válvula de manera sencilla y programática después de su creación, facilitando la construcción dinámica de redes, el ajuste de parámetros durante la preparación de un caso, o la automatización de simulaciones. Es especialmente útil en entornos donde la topología y los parámetros pueden cambiar entre simulaciones o para estudios paramétricos.
    
        Ejemplo de uso
        --------------
        >>> v = Valve("vT1b", valve_type="linear", logic="linear", Cv_max=1.0)
        >>> v.update({"Cv_max": 2.5, "logic": "sigmoid"})
        >>> v.update(valve_type="equal_percentage", a_min=0.2)
        # Si introduces un parámetro inválido, se lanza un ValueError (por ejemplo, valve_type="no-existe")
    
        Avisos
        ------
        - Si algún parámetro no existe en la clase `Valve`, simplemente se ignora (no lanza error).
        - Tras cualquier cambio, la configuración se valida automáticamente.
        - Si los nuevos valores no son compatibles (por ejemplo, `a_min >= a_max`), se lanzará un `ValueError` y el objeto puede quedar en estado inconsistente si no capturas la excepción.
        - No retorna nada; los cambios afectan directamente al objeto en memoria.
        """

        if config is None:
            config = kwargs
        else:
            config.update(kwargs)
    
        for k, v in config.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self._validate_()
        return None


# =============================================================================
#     # 3. Inicilizar/Retroceder/Resets/Leer data
# =============================================================================
    def _reset_conex_(self):
        """
        ¿Qué hago?
        -----------
        Reseteo completamente el diccionario interno de conexiones (`_conex`) de la válvula, eliminando cualquier referencia previa a unidades conectadas, puertos o tipo de conexión. El estado del atributo `_conex` vuelve a su estado inicial (todas las claves a `None`).
        
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        Se utiliza para asegurar que la válvula no guarda conexiones "fantasma" de simulaciones anteriores o de cambios en la topología de la red. Es esencial antes de reconfigurar la red de proceso, cargar un nuevo caso, o reconstruir las conexiones entre equipos y válvulas, evitando referencias cruzadas o inconsistencias.
        
        Ejemplo de uso
        --------------
        >>> v = Valve("vT1b")
        >>> v._conex["unit_A"] = myTank
        >>> v._reset_conex_()
        >>> print(v._conex)
        {'type': None, 'unit_A': None, 'unit_B': None, 'port_A': None, 'port_B': None}
        
        Avisos
        ------
        - No elimina ni afecta otras variables o resultados del objeto válvula, solo el atributo `_conex`.
        - Es recomendable llamarlo siempre antes de reconstruir la red o al limpiar/recargar un caso.
        - No retorna nada.
        """
        self._conex = {
           "type": None,
           "unit_A": None,
           "unit_B": None,
           "port_A": None,
           "port_B": None,
         }
        return None


    def _reset_logs_(self):
        """
        ¿Qué hago?
        -----------
        Elimino y reseteo todos los logs y arrays internos relacionados con el historial temporal de la simulación de la válvula.
        Esto incluye los registros de tiempo y caudal instantáneo (`_t_log`, `_Qn_log`) y los arrays "limpios" de resultados (`_t2`, `_Qn2`).
        Además, marco el flag de resultados como no disponible (`_required['Results'] = False`).
        
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        Se utiliza para dejar la válvula en un estado listo para una nueva simulación, evitando acumulaciones, residuos o mezclas de datos de simulaciones anteriores. Es fundamental antes de relanzar simulaciones, retroceder en el tiempo, o liberar memoria entre escenarios.
        
        Ejemplo de uso
        --------------
        >>> v = Valve("vT1b")
        >>> v._t_log = [1,2,3]
        >>> v._Qn_log = [10,20,30]
        >>> v._reset_logs_()
        >>> print(v._t_log, v._Qn_log, v._t2, v._Qn2)
        [] [] None None
        
        Avisos
        ------
        - Este método borra toda la información de logs y resultados limpios previos de la válvula.
        - Debe llamarse antes de iniciar una nueva simulación, retroceder el tiempo o liberar memoria.
        - No afecta la configuración ni el estado de la válvula fuera de los resultados/arrays de logs.
        - No retorna nada.
        """
        self._t_log = []
        self._Qn_log = []
        self._t2 = None
        self._Qn2 = None
        self._required['Results'] = False
        return None


    def _initialize_(self):
        """
        ¿Qué hago?
        -----------
        Inicializo y vacío todos los arrays principales de la válvula que almacenan el historial temporal de simulación: tiempo (`_t`), presiones de ambos lados (`_pA`, `_pB`), apertura (`_a`), diferencia de presión (`_dP`), coeficiente de caudal (`_Cv`) y caudal instantáneo (`_Qn`).  
        Además, llamo a `_reset_logs_()` para borrar todos los logs y resultados limpios previos.
        
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        Este método prepara la válvula para iniciar una **nueva simulación dinámica**, asegurando que no quede ningún dato residual de ejecuciones anteriores. Es esencial para reiniciar el estado de la válvula antes de cualquier integración ODE o campaña de simulación.
        
        Ejemplo de uso
        --------------
        >>> v = Valve("vT1b")
        >>> v._t = np.array([1,2,3])
        >>> v._initialize_()
        >>> print(v._t, v._Qn)
        [] []
        
        Avisos
        ------
        - El método borra *toda* la información dinámica previa de la válvula.
        - Debe ejecutarse siempre antes de lanzar una nueva simulación o tras cambios en configuración.
        - No afecta a la configuración estructural ni a los parámetros de la válvula.
        - No retorna nada.
        """

        self._t = np.array([])
        self._pA = np.array([])
        self._pB = np.array([])
        self._a = np.array([])
        self._dP = np.array([])
        self._Cv = np.array([])
        self._Qn = np.array([])
        self._reset_logs_()
        return None


    def _croopTime(self, target_time):
        """
        ¿Qué hago?
        ----------
        Recorto (trunco) todos los arrays dinámicos de la válvula (tiempos, presiones, apertura, dP, Cv, caudal, etc.) hasta el instante `target_time`, eliminando cualquier dato posterior. Dejo la válvula lista para relanzar la simulación desde ese instante concreto.
        
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        Permite **retroceder la simulación** de la válvula a un estado anterior y relanzar desde ahí, lo que resulta esencial para:
        - Optimizar escenarios desde un punto concreto.
        - Repetir simulaciones con lógica de control avanzada.
        - Evitar acumulación de memoria con datos temporales innecesarios.
        Lo usa cualquier usuario o función que gestione reinicios parciales de la simulación en redes dinámicas.
        
        Ejemplo de uso
        --------------
        >>> v = Valve("vT1b")
        >>> v._t = np.array([0, 10, 20, 30])
        >>> v._Qn = np.array([0, 1, 2, 3])
        >>> v._croopTime(15)
        >>> print(v._t, v._Qn)
        [0 10] [0 1]
        
        Avisos
        ------
        - Si no hay datos anteriores al tiempo solicitado, lanza una excepción.
        - El método elimina todos los datos dinámicos posteriores a `target_time`, incluidas presiones y caudales.
        - Llama internamente a `_reset_logs()` para asegurar que los logs limpios también quedan vacíos.
        - No modifica la configuración fija de la válvula, solo los arrays de resultados de simulación.
        - No retorna nada.
        """

        if self._t is None or len(self._t) == 0:
            raise ValueError(f"⛔ No hay datos previos en la válvula {self._name} para el tiempo de inicio {target_time:.2f} s")
        idx_valid = np.where(self._t <= target_time)[0]
        if len(idx_valid) == 0:
            raise ValueError(f"⛔ No hay datos previos en la válvula {self._name} para el tiempo de inicio {target_time:.2f} s")
        last_idx = idx_valid[-1]
    
        # Recorta arrays principales
        self._t   = self._t[:last_idx + 1]
        self._pA  = self._pA[:last_idx + 1]
        self._pB  = self._pB[:last_idx + 1]
        self._a   = self._a[:last_idx + 1]
        self._dP  = self._dP[:last_idx + 1]
        self._Cv  = self._Cv[:last_idx + 1]
        self._Qn  = self._Qn[:last_idx + 1]
    
        # Borra logs y resultados limpios para garantizar limpieza de memoria y resultados
        self._reset_logs()
        self._required['Results'] = False
        return None

            
    def _readCase_(self,case): # EN DESARROLLO!!!!
        pass
    
    
    def _readData_(self,data): # EN DESARROLLO!!!!
        pass
    
    
    def readCaseData(self,case,data):
        self._readCase_(case)
        self._readData_(data)
        return None


# =============================================================================
#     # 4.Limpieza
# =============================================================================
    def _clean_LOG_(self, t_log, VAR_log):
        """
        ¿Qué hago?
        ----------
        Proceso y limpio dos listas paralelas (tiempos y valores de variable) para eliminar duplicados temporales, garantizando que para cada instante de tiempo solo se conserva el **último valor registrado**. Devuelvo arrays sincronizados, listos para postprocesado, balances o visualización.
    
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        Este método es fundamental cuando:
        - Se generan múltiples registros para el mismo tiempo durante la integración (por callbacks, interpolaciones, etc.).
        - Se necesita un historial de la variable libre de solapamientos o repeticiones, clave para cálculos de balances e integrales temporales.
        - El usuario o el postprocesado requieren arrays **sin duplicados** y perfectamente alineados.
        Es utilizado internamente en funciones de limpieza de logs antes de calcular balances de masa, energía, etc.
    
        Ejemplo de uso
        --------------
        >>> t_log = [0, 1, 1, 2, 3, 3]
        >>> var_log = [10, 20, 21, 30, 40, 41]
        >>> t_clean, var_clean = valve._clean_LOG_(t_log, var_log)
        >>> print(t_clean)  # [0, 1, 2, 3]
        >>> print(var_clean)  # [10, 21, 30, 41]
    
        Avisos
        ------
        - Solo se conserva el **último valor** para cada tiempo duplicado.
        - El orden temporal queda garantizado (de menor a mayor).
        - Puede aplicarse a cualquier log paralelo tiempo-valor, no solo a caudales.
        - No modifica los arrays originales, retorna nuevos arrays limpios.
        - Fundamental llamar antes de integraciones, balances o exportación de datos para asegurar consistencia.
        """
        from collections import defaultdict
    
        var_by_time = defaultdict(list)
        for t, v in zip(t_log, VAR_log):
            var_by_time[t].append(v)
    
        # Ordena los tiempos únicos
        unique_times = sorted(var_by_time.keys())
        t_clean = []
        VAR_clean = []
    
        for t in unique_times:
            t_clean.append(t)
            VAR_clean.append(var_by_time[t][-1])  # el último valor para ese tiempo
    
        return t_clean, VAR_clean

            
    def _clean_LOG_valve_(self):
        """
        ¿Qué hago?
        ----------
        Limpio y sincronizo los logs temporales de la válvula, procesando los registros de tiempo y caudal (_t_log y _Qn_log) para eliminar duplicados y conservar un único valor por instante de tiempo. Los datos limpios se almacenan en los arrays de resultados definitivos (_t2, _Qn2) mediante el método _storeBal_.
        
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        Este método es fundamental para:
        - Garantizar que los históricos de la válvula están libres de duplicidades temporales antes de análisis, balances o exportación.
        - Preparar los arrays de tiempo y caudal para el postprocesado y el cálculo de integrales o balances de masa/energía.
        - Marcar el final de una simulación o integración sobre la válvula, asegurando consistencia de los datos.
        Se utiliza internamente después de una simulación o antes de calcular balances.
        
        Ejemplo de uso
        --------------
        >>> valve._t_log = [0, 1, 1, 2]
        >>> valve._Qn_log = [5, 10, 11, 20]
        >>> valve._clean_LOG_valve_()
        >>> print(valve._t2)    # [0, 1, 2]
        >>> print(valve._Qn2)   # [5, 11, 20]
        
        Avisos
        ------
        - Solo se conserva el **último valor** registrado para cada tiempo duplicado.
        - El método debe llamarse **siempre** tras la integración antes de cualquier análisis o exportación de resultados.
        - Los arrays originales de logs no se modifican, pero los resultados limpios (_t2, _Qn2) sobrescriben valores previos.
        - Si los logs están vacíos o mal formateados, puede lanzar un error o dejar resultados vacíos.
        """

        t_Q, Qn2 = self._clean_LOG_(self._t_log, self._Qn_log)
        self._storeBal_(t_Q, Qn2)
        return None


    def _storeBal_(self, t2, Qn2):
        """
        ¿Qué hago?
        ----------
        Almaceno en los atributos internos de la válvula (_t2, _Qn2) los resultados procesados y sincronizados de tiempo y caudal volumétrico, normalmente tras limpiar los logs temporales. Estos arrays representan la historia definitiva de la válvula para postprocesado, balances o exportación.
      
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        Este método se utiliza para:
        - Guardar de manera estructurada y definitiva los resultados limpios de la válvula tras una simulación.
        - Permitir un acceso rápido y seguro a los datos de tiempo y caudal sin riesgo de duplicidades o inconsistencias.
        - Facilitar el análisis posterior, como el cálculo de balances de masa o la visualización.
        Suele llamarse automáticamente desde métodos como `_clean_LOG_valve_`, pero puede usarse manualmente si se dispone de arrays procesados.
      
        Ejemplo de uso
        --------------
        >>> valve._storeBal_([0, 1, 2], [10, 20, 30])
        >>> print(valve._t2)    # [0, 1, 2]
        >>> print(valve._Qn2)   # [10, 20, 30]
      
        Avisos
        ------
        - Este método **sobrescribe** los valores actuales de `_t2` y `_Qn2` sin pedir confirmación.
        - No realiza comprobaciones de coherencia: asume que los arrays `t2` y `Qn2` son correctos y tienen la misma longitud.
        - Si se pasan arrays vacíos o de distinto tamaño, los atributos resultantes pueden ser inconsistentes o vacíos.
        """
        self._t2 = np.array(t2)
        self._Qn2 = np.array(Qn2)         
        return None

            
# =============================================================================
#     # 5.Cálculo y control (físico/lógico)
# =============================================================================
    def _get_a_(self, t, t_step):
        """
        ¿Qué hago?
        ----------
        Calculo el grado de apertura (`apertura`) de la válvula en el instante `t`, según la lógica temporal definida en `self.logic` y los parámetros de control de apertura (`logic_params`). El resultado tiene en cuenta el tipo de lógica (lineal, sigmoide, polinómica, escalón, rampa, senoidal, etc.), la dirección de apertura (`co` o `oc`) y los límites de apertura mínima y máxima.
    
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        - Para determinar de manera programática y flexible cómo evoluciona la apertura de la válvula en el tiempo, adaptándose a diferentes estrategias de control o simulación.
        - Es útil tanto en simulaciones dinámicas (cada paso de integración) como en estudios de respuesta o pruebas de lógica de válvulas.
        - Se emplea internamente cada vez que se necesita saber la apertura instantánea para calcular el caudal.
    
        Ejemplo de uso
        --------------
        >>> valve.logic = "sigmoid"
        >>> valve.logic_params = {"start": 0, "duration": 10, "k": 8, "x0": 0.4}
        >>> valve.opening_direction = "co"
        >>> valve.a_min = 0.1
        >>> valve.a_max = 0.9
        >>> t = 5
        >>> t_step = 10
        >>> a = valve._get_a_(t, t_step)
        >>> print(a)  # Valor de apertura en t=5, entre 0.1 y 0.9 según la sigmoide
    
        Avisos
        ------
        - El método asume que los parámetros de lógica (`logic_params`) y los atributos de la válvula (`logic`, `opening_direction`, etc.) están correctamente configurados antes de llamar.
        - Si el atributo `logic` no está en la lista soportada, se aplicará comportamiento "lineal" por defecto.
        - La apertura resultante se recorta siempre entre `a_min` y `a_max`.
        - Si `opening_direction` no es "co" (cierre→apertura) ni "oc" (apertura→cierre), lanza un error.
        - Si la lógica "step" se usa con un valor bajo de `steps`, la apertura puede variar bruscamente.
        """
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


    def _get_Cv_(self, a):
        """
        ¿Qué hago?
        ----------
        Calculo el coeficiente de caudal (Cv) de la válvula en función del grado de apertura `a` (normalizado entre 0 y 1), utilizando la ecuación característica correspondiente al tipo de válvula especificado en `self.valve_type`. Soporta válvulas lineales, de igual porcentaje, de apertura rápida o personalizadas.
    
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        - Para modelar de manera realista cómo varía la capacidad de paso de la válvula según su apertura, permitiendo simular válvulas reales de distintos tipos.
        - Es fundamental para el cálculo posterior del caudal que atraviesa la válvula en cada instante, ya que Cv determina la relación entre apertura y caudal.
        - Este método es invocado internamente en cada paso de simulación o cuando se necesita conocer el Cv asociado a una determinada apertura.
    
        Ejemplo de uso
        --------------
        >>> valve.valve_type = "equal_percentage"
        >>> a = 0.5
        >>> Cv = valve._get_Cv_(a)
        >>> print(Cv)  # Valor del coeficiente Cv para apertura 0.5 en válvula equal_percentage
    
        Avisos
        ------
        - El parámetro de entrada `a` debe estar en el rango [0, 1]. No se recorta automáticamente, así que se recomienda asegurarlo antes de llamar al método.
        - Si `valve_type` no es uno de los soportados ("linear", "equal_percentage", "quick_opening", "custom"), lanza un ValueError.
        - Para la curva "equal_percentage" se usa por defecto un radio R=30, adecuado para la mayoría de válvulas industriales, pero este valor puede ajustarse según fabricante.
        - La opción "custom" usa la fórmula a**2 / (a**2 + 0.1) a modo de ejemplo, pero puede adaptarse a una curva real experimental.
        """
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


    def _get_Qn_(self, t, t_step, pIn, Tin, pOut, MW_gas):
        deltaP_bar = (pIn - pOut)/1e5
        
        if (deltaP_bar) <= 0:
            Qn = 0.0
        else:
            a = self._get_a_(t, t_step)
            Cv_t = self._get_Cv_(a)
            Cv = self.Cv_max * Cv_t
            MW_air = 28.96  # g/mol
            Sg = MW_gas / MW_air  # densidad relativa
        
            P1_bar = pIn / 1e5
        
            Qn = 414.97 * Cv * np.sqrt((deltaP_bar * P1_bar) / (Tin * Sg))

        return Qn


    def _estimateCv_(self, V, P0, P1, T0, T1, tsim):
        """
        ¿Qué hago?
        ----------
        Estimo el coeficiente de caudal Cv necesario para vaciar (o llenar) un volumen `V` desde una presión inicial `P0` hasta una presión final `P1` (con temperaturas inicial y final `T0`, `T1`) en un tiempo total `tsim`. El cálculo se basa en balances globales de masa y en la ecuación cuadrática típica para caudal a través de una válvula, asumiendo condiciones ideales.
    
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        - Para dimensionar válvulas en la fase de diseño preliminar, determinando qué valor de Cv permitirá alcanzar el cambio de presión deseado en el tiempo objetivo.
        - Es útil tanto para ingenieros de proceso que están eligiendo válvulas, como para validar si una válvula instalada cumple con el requerimiento de vaciado/llenado rápido.
        - Este método puede emplearse como herramienta de estimación rápida antes de lanzar simulaciones más complejas.
    
        Ejemplo de uso
        --------------
        >>> valve._estimateCv_(V=2.5, P0=5e5, P1=1e5, T0=300, T1=295, tsim=60)
        Se estima que se necesita Cv ≈ 13.52 Nm3/h/bar^0.5
    
        Avisos
        ------
        - Las presiones `P0` y `P1` deben estar en Pa, las temperaturas en K, el volumen en m³ y el tiempo `tsim` en segundos.
        - El cálculo supone comportamiento ideal del gas y Z=1.0 (sin corrección de compresibilidad).
        - El resultado Cv es una **estimación aproximada** válida solo para la hipótesis de vaciado/llenado rápido, sin considerar dinámica de apertura ni resistencias adicionales.
        - La fórmula utilizada es válida para gases, y Cv se reporta en unidades de Nm³/h/bar^0.5.
        - Si la diferencia cuadrática de presiones resulta negativa (presión final mayor que la inicial), lanza un ValueError.
        - El método imprime por pantalla el valor estimado de Cv y **no lo retorna** (retorna None). Si necesitas el valor, adapta la función.
        """
    
        N1 = V * P1 / self._R / T1
        N0 = V * P0 / self._R / T0
        delta_N = abs(N1-N0)
        n_dot = delta_N / tsim  # mol/s
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
        return None
    

# =============================================================================
#     # 6.Helper
# =============================================================================
    def _get_unitVar_(self, unit, var, time=None, nodo=None, port=None, especie=None    ):
        """
          ¿Qué hago?
          ----------
          Recupero el valor de una variable dinámica (`_var`) de una unidad (tanque, columna, etc.) de forma flexible, permitiendo filtrar por tiempo, nodo, puerto (top/bottom/side) y especie química. Devuelvo el array o el valor escalar correspondiente, facilitando la extracción de resultados para cualquier variable simulada (presión, temperatura, fracciones, moles, etc.) en cualquier instante o localización de la unidad.
        
          ¿Por qué/para qué/quién lo hago?
          --------------------------------
          - Permite acceder de manera unificada y sencilla a cualquier variable interna de un equipo durante el análisis de simulación, postproceso, cálculos de balances o para la lógica de control.
          - Evita tener que recordar la estructura interna de cada variable (dimensiones, nombres, etc.), centralizando el acceso.
          - Es fundamental para el postprocesado de simulaciones complejas, la elaboración de gráficos, la implementación de lazos de control o la extracción de perfiles espaciales/temporales.
          - Lo utilizan tanto otros métodos internos del framework como usuarios avanzados que analizan o customizan simulaciones.
        
          Ejemplo de uso
          --------------
          >>> P = net._get_unitVar_(T1, "P2", time=100, port="top")          # Presión en el nodo superior en t=100 s
          >>> x_CO2 = net._get_unitVar_(T1, "x2", time=250, port="bottom", especie="CO2")  # Fracción de CO2 en la entrada a t=250 s
          >>> T_all = net._get_unitVar_(T1, "T2")                            # Temperatura en todos los nodos/todos los tiempos
          >>> x_all = net._get_unitVar_(T1, "x2", nodo=0)                    # Fracciones de todas las especies en nodo 0
        
          Avisos
          ------
          - El argumento `var` debe coincidir con el nombre interno del atributo de la unidad, por ejemplo `"P2"`, `"x2"`, `"T"`, etc.
          - Si se solicita por tiempo, el método interpola el valor más cercano (no necesariamente exacto si el array de tiempos no es uniforme).
          - Si la variable es una composición (`x`), puedes filtrar por nodo, puerto o especie.
          - Si el resultado es un array de tamaño 1, devuelve un escalar (float) en vez de un array.
          - Lanza ValueError o AttributeError si se solicita un puerto/nodo/especie no existente o si falta el atributo.
          - En composiciones, si `especie` es str, busca el índice en el atributo `_species` de la unidad (debe estar definido).
          - El método **no modifica** ningún estado interno; es solo lectura.
          - Puede devolver distintos tipos (float, np.ndarray) según los filtros aplicados.
        
          """
        # --- Selecciona array de datos ---
        arr = getattr(unit, f"_{var}", None)
        if arr is None:
            raise AttributeError(f"La unidad no tiene atributo _{var}")
    
        arr = np.asarray(arr)
    
        # --- Selecciona el array de tiempos adecuado ---
        if var.endswith("2"):
            t_arr = getattr(unit, "_t2", None)
            t_var_name = "_t2"
        else:
            t_arr = getattr(unit, "_t", None)
            t_var_name = "_t"
    
        # --- Selección de tiempo ---
        if time is not None:
            if t_arr is None:
                raise AttributeError(
                    f"La unidad no tiene variable de tiempo {t_var_name} (requerido para var '{var}')"
                )
            t_arr = np.asarray(t_arr)
            idx = (np.abs(t_arr - float(time))).argmin() if not isinstance(time, int) else int(time)
            arr = arr[idx]
    
        # --- Composición (x): [ntimes, nnodos, ncomp] o [nnodos, ncomp] ---
        if var.lower().startswith("x"):
            # Forzamos a 3D: [ntimes, nnodos, ncomp]
            if arr.ndim == 2:
                arr = arr[None, ...]
            # --- Selección de puerto/nodo ---
            if port is not None:
                port = port.lower()
                if port == "top":
                    arr = arr[..., -1, :]  # Último nodo
                elif port == "bottom":
                    arr = arr[..., 0, :]   # Primer nodo
                elif port == "side":
                    if nodo is None:
                        raise ValueError("Si port='side', necesitas nodo")
                    arr = arr[..., nodo, :]
                else:
                    raise ValueError(f"Puerto desconocido: {port}")
            elif nodo is not None:
                arr = arr[..., nodo, :]
    
            # --- Selección de especie ---
            if especie is not None:
                if isinstance(especie, str):
                    # Buscar índice si unit tiene ._species
                    if hasattr(unit, "_species"):
                        idx_especie = list(unit._species).index(especie)
                    else:
                        raise ValueError("unit no tiene atributo _species para buscar nombre de especie")
                else:
                    idx_especie = int(especie)
                arr = arr[..., idx_especie]
    
            return arr  # Puede ser [ntimes, ncomp], [nnodos,], [ncomp], etc. según filtros
    
        # --- Variables escalares: [ntimes, nnodos] o [nnodos] ---
        else:
            if arr.ndim == 1:
                arr = arr[None, ...]
            if port is not None:
                port = port.lower()
                if port == "top":
                    arr = arr[..., -1]
                elif port == "bottom":
                    arr = arr[..., 0]
                elif port == "side":
                    if nodo is None:
                        raise ValueError("Si port='side', necesitas nodo")
                    arr = arr[..., nodo]
                else:
                    raise ValueError(f"Puerto desconocido: {port}")
            elif nodo is not None:
                arr = arr[..., nodo]
            # --- Asegúrate de que si es un escalar (array tamaño 1), devuelves float ---
            if isinstance(arr, np.ndarray) and arr.size == 1:
                return float(arr)
            return arr

# =============================================================================
#     # 7.Save Simulation/Case/Data
# =============================================================================
    def _storeData_(self, start, end):
        """
        ¿Qué hago?
        ----------
        Recorro el historial de tiempos de simulación y calculo todos los arrays dinámicos de la válvula (presión aguas arriba/abajo, diferencial de presión, apertura, coeficiente de caudal, caudal volumétrico, etc.) en cada instante del intervalo especificado `[start, end]`. Guardo estos resultados en los atributos internos de la válvula, listos para graficar, postprocesar o validar balances.
      
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        - Para reconstruir y almacenar el comportamiento dinámico de la válvula a partir de los resultados simulados de los equipos conectados (tanques, columnas, etc.).
        - Es esencial para poder visualizar la evolución temporal de la válvula, analizar su operación o comparar con datos experimentales.
        - Lo utilizan internamente los métodos de postproceso del sistema y cualquier usuario que quiera exportar o visualizar los resultados de la válvula de forma autónoma.
      
        Ejemplo de uso
        --------------
        >>> vT1b._storeData_(start=0, end=100)
        >>> # Ahora puedes acceder a vT1b._Qn, vT1b._a, vT1b._pA, etc. para análisis o gráficos
        >>> plt.plot(vT1b._t, vT1b._Qn)
      
        Avisos
        ------
        - El método debe llamarse **después** de la simulación completa, cuando ya están disponibles los arrays de resultados de las unidades conectadas (`_t`, `_P`, `_T`, `_x`, ...).
        - Los tiempos utilizados se corresponden con el historial de la unidad aguas arriba (`unit_A`), y solo se procesan los pasos comprendidos entre `start` y `end`.
        - Si el tipo de válvula es "interunit" compara presiones de ambos lados y adapta el sentido del caudal automáticamente.
        - Lanza error si los arrays necesarios están vacíos o si no hay datos en el intervalo solicitado.
        - Los resultados **se acumulan** si ya existía información previa: puedes llamar varias veces con distintos intervalos y no se sobrescribe lo anterior.
        - Este método **no modifica** las variables de estado de las unidades conectadas, solo procesa y guarda resultados de la válvula.
        """
        unit_A = self._conex.get("unit_A")
        unit_B = self._conex.get("unit_B")
        port_A = self._conex.get("port_A")
        port_B = self._conex.get("port_B")
        vtype  = self._conex.get("type")        
    
        t_ref = unit_A._t
        mask = (t_ref > start) & (t_ref <= end)
        indices = np.where(mask)[0]
        endTime_valve = self.logic_params.get("start", 0) + self.logic_params.get("duration", 1)
    
        # Arrays locales de resultados
        t_list, pA_list, pB_list, a_list, dP_list, Cv_list, Qn_list = [], [], [], [], [], [], []
        for i in indices:
            
            ti = t_ref[i]
            if vtype == "inlet":
                Pi = unit_A._Pin
                Ti = unit_A._Tin
                Pj= self._get_unitVar_(unit_A, 'P', time=ti, port=port_A)
                # Pj = unit_A._P[i][0]
                MW_gas = np.sum(unit_A._MW * unit_A._xin)
                PA, PB = Pi, Pj
            
            elif vtype == "outlet":
                Pi = self._get_unitVar_(unit_A, 'P', time=ti, port=port_A)
                Ti = self._get_unitVar_(unit_A, 'T', time=ti, port=port_A)
                # Pi = unit_A._P[i][0]
                # Ti = unit_A._T[i][0]
                Pj = unit_A._Pout
                MW_gas = np.sum(unit_A._MW * unit_A._x[i,:])
                PA, PB = Pi, Pj
            
            else:
                Pi = self._get_unitVar_(unit_A, 'P', ti, port=port_A)
                Pj = self._get_unitVar_(unit_B, 'P', ti, port=port_B)
                if Pi > Pj:
                    Ti = self._get_unitVar_(unit_A, 'T', ti, port=port_A)
                    MW_gas = np.sum(unit_A._MW * unit_A._x[i,:,:])
                    PA, PB = Pi, Pj

                else:
                    Ti = self._get_unitVar_(unit_B, 'T', ti, port=port_B)
                    MW_gas = np.sum(unit_B._MW * unit_B._x[i,:,:])
                    PA, PB = Pj, Pi

            a  = self._get_a_(ti, endTime_valve)
            Cv = self.Cv_max * self._get_Cv_(a)
            dP = (PA - PB)
            
            Qn = self._get_Qn_(ti, self.logic_params.get("duration", ti), PA, Ti, PB, MW_gas)

        
            t_list.append(ti)
            pA_list.append(PA)
            pB_list.append(PB)
            dP_list.append(dP)
            a_list.append(a)
            Cv_list.append(Cv)
            Qn_list.append(Qn)
    
        # Los nuevos arrays para este segmento temporal
        t_arr  = np.array(t_list)
        pA_arr = np.array(pA_list)
        pB_arr = np.array(pB_list)
        dP_arr = np.array(dP_list)
        a_arr  = np.array(a_list)
        Cv_arr = np.array(Cv_list)
        Qn_arr = np.array(Qn_list)
    
        # ACUMULA: si la variable ya existe y tiene datos, concatena, si no, asigna
        def append_or_create(old, new):
            if old is None or old.size == 0:
                return new
            else:
                return np.concatenate([old, new])
    
        self._t  = append_or_create(self._t,  t_arr)
        self._pA = append_or_create(self._pA, pA_arr)
        self._pB = append_or_create(self._pB, pB_arr)
        self._dP = append_or_create(self._dP, dP_arr)
        self._a  = append_or_create(self._a,  a_arr)
        self._Cv = append_or_create(self._Cv, Cv_arr)
        self._Qn = append_or_create(self._Qn, Qn_arr)
        self._required['Results'] = True

        return None
          
   
    def _writeCase_(self):   # EN DESARROLLO
        # ----- Qué variables exporto -----
        setup_vars = self._setup_vars
        setup_dict = {var: getattr(self, var, None) for var in setup_vars}
    
        # Serializa las conexiones _conex (por nombre)
        setup_dict['conex'] = {}
        for k, v in self._conex.items():
            if isinstance(v, list):
                value_str = [getattr(obj, '_name', str(obj)) for obj in v if obj is not None]
            elif hasattr(v, '_name'):
                value_str = v._name
            else:
                value_str = v
            setup_dict['conex'][k] = value_str
    
        # Serialización binaria + base64
        binary = pickle.dumps(setup_dict)
        setup_str = base64.b64encode(binary).decode('ascii')
        self._case = setup_str  # opcional: guarda una copia interna
        return self._name, setup_str
    

    def _writeData_(self):   # EN DESARROLLO
        data_dict = {
            't':   self._t.flatten()   if self._t is not None else None,
            'pA':  self._pA.flatten()  if self._pA is not None else None,
            'pB':  self._pB.flatten()  if self._pB is not None else None,
            'a':   self._a.flatten()   if self._a is not None else None,
            'dP':  self._dP.flatten()  if self._dP is not None else None,
            'Cv':  self._Cv.flatten()  if self._Cv is not None else None,
            'Qn':  self._Qn.flatten()  if self._Qn is not None else None,
        }
        binary = pickle.dumps(data_dict)
        data_str = base64.b64encode(binary).decode('ascii')
        self._data = data_str  # opcional, uso interno
        return self._name, data_str
    
    
    def _writeCaseData_(self,):        
        self._writeCase_()
        self._writeData_()


# =============================================================================
#     # 8.Plots y postProceso
# =============================================================================

    def _plot_(self):
        """
        ¿Qué hago?
        ----------
        Genero un panel de gráficos que muestra de forma visual el comportamiento dinámico de la válvula simulada:
          - Evolución temporal de la apertura (lógica de control).
          - Caudal volumétrico alimentado/salida en cada instante.
          - Presiones aguas arriba y abajo, y diferencial de presión (dP).
          - Curva característica del coeficiente de caudal (Cv) según la apertura.
    
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        - Para facilitar el **análisis visual** del funcionamiento de la válvula, su lógica de control, su respuesta hidráulica y su interacción con las unidades conectadas.
        - Permite detectar fácilmente anomalías, errores de configuración, cuellos de botella o comportamientos no deseados durante la simulación.
        - Es útil tanto para el desarrollador/modelador (debug, validación) como para el usuario final (interpretación de resultados, informes, tuning de parámetros).
    
        Ejemplo de uso
        --------------
        >>> vT1b._plot_()
        # Muestra la figura con los cuatro gráficos principales de la válvula:
        #  - Apertura vs tiempo
        #  - Caudal vs tiempo
        #  - Presión y diferencial de presión vs tiempo
        #  - Curva Cv relativa vs apertura
    
        Avisos
        ------
        - Es imprescindible haber ejecutado antes la simulación y haber llamado a `_storeData_`, de lo contrario el método avisará que no hay resultados.
        - Si alguno de los arrays principales está vacío (por ejemplo, no hay tiempos registrados), también lo notificará y **no lanzará excepción**.
        - Los datos limpios (`_t2`, `_Qn2`) pueden añadirse manualmente si se quiere comparar "raw vs clean".
        - La curva de Cv es relativa (normalizada a Cv_max=1); para comparar con datos reales, hay que escalarla.
        - No devuelve ningún objeto, solo muestra la figura interactiva con `plt.show()`.
        - Si el entorno de ejecución no soporta gráficos interactivos (por ejemplo, ejecución en background), la figura puede no visualizarse correctamente.
        """
        if not getattr(self, '_required', {}).get("Results", False):
            print("⛔ No hay resultados para mostrar. Ejecuta primero la simulación.")
            return
    
        # Check arrays para evitar errores
        if (self._t is None or len(self._t) == 0
            or self._a is None or len(self._a) == 0):
            print("⛔ No hay datos en los arrays principales. Ejecuta primero la simulación.")
            return
    
        fig = plt.figure(figsize=(12, 8))
        grid = fig.add_gridspec(3, 2, width_ratios=[2, 1])
    
        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[1, 0])
        ax3 = fig.add_subplot(grid[2, 0])
        ax4 = fig.add_subplot(grid[:, 1])
    
        ax1.plot(self._t, np.array(self._a) * 100, color='purple', lw=2)
        ax1.set_ylabel("Apertura [%]")
        ax1.set_title("Lógica de control")
        ax1.grid()
    
        ax2.plot(self._t, self._Qn, lw=2, color='green')
        # Si quieres, puedes añadir también los datos limpios (self._t2, self._Qn2) aquí
    
        ax2.set_ylabel("Caudal [Nm³/h]")
        ax2.set_title("Caudal alimentado")
        ax2.grid()
    
        ax3.plot(self._t, self._pA, lw=2, color="red", label='Unit-A')
        ax3.plot(self._t, self._pB, lw=2, color="blue", label='Unit-B')
        ax3.plot(self._t, self._dP, lw=2, color="yellow", label='dP')
        ax3.set_ylabel("Presión nodo  [Pa]")
        ax3.set_xlabel("Tiempo [s]")
        ax3.set_title("Presión")
        ax3.grid()
        ax3.legend(loc="best")
    
        a_range = np.linspace(0, 1, 100)
        ax4.plot(a_range * 100, self._get_Cv_(a_range) * 100, lw=2, color='orange')
        ax4.set_xlabel("Apertura [%]")
        ax4.set_ylabel("Cv relativo [%]")
        ax4.set_title(f"Tipo de válvula: {self.valve_type}")
        ax4.grid()
    
        plt.suptitle(f"Simulación de válvula {self._name} -> Conexion: {self._conex['type']}, Tipo: {self.valve_type}, Lógica: {self.logic}",fontsize=24)
        plt.tight_layout()
        plt.show()
    
