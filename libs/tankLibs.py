#
import numpy as np
from scipy.integrate import trapz
import itertools
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle,base64

from solveLibs import solveTanks
from commonLibs import *

# =============================================================================
# 
# =============================================================================
class Tank:
    
# =============================================================================
#     # 1. Constructor    
# =============================================================================
    def __init__(self,
                 Name,          # Nombre del equipo
                 design_info,   # info design
                 prop_gas,      # info gases
                 init_info=None,     # initial info,
                 boundary_info=None, # boundary_info,
                 thermal_info=None,  # thermal_info
                 ):

        
        self._R = 8.314
        self._name = Name
        
        #Design
        self._L      = design_info["Longitud"]    # Longitud total (m)
        self._D      = design_info["Diametro"]    # Diámetro total (m)
        self._Ri     = self._D / 2
        self._e      = design_info["Espesor"]     # Espesor (m)
        self._Vol    = np.pi * (self._Ri)**2 * self._L
        self._Ai     = np.pi * (self._Ri)**2              # Área transversal interna
        self._Al     = np.pi * self._D * self._L          # Área lateral
        self._Aint   = self._Al + self._Ai                # Área total interna
        self._nodos  = 1      # Número de nodos
        # Discretización (ejemplo: nodos igualmente repartidos)
        # Si quieres longitudes desiguales, cambia el array _Lx
        self._Lx  = np.ones(self._nodos) * self._L / self._nodos       # Longitud por nodo [nodos]
        self._D_x = np.ones(self._nodos) * self._D                     # Diámetro por nodo [nodos]
        self._Rix = self._D_x / 2                                      # Radio por nodo [nodos]
        self._Aix = np.pi * (self._Rix)**2                             # Área interna por nodo [nodos]
        self._Alx = np.pi * self._D_x * self._Lx                       # Área lateral por nodo [nodos]
        self._Aintx = self._Aix + self._Alx                            # Área interna total por nodo [nodos]
        self._Volx = self._Aix * self._Lx  
        
        #Gas properties info
        self._prop_gas = prop_gas
        self._species = prop_gas["species"]
        self._ncomp = len(prop_gas["species"])
        
        self._MW = prop_gas["MW"]       # Molecular Weight
        self._mu = prop_gas["mu"]       # Viscosity
        self._sigmaLJ = prop_gas["sigmaLJ"] # diámetro Lennard-Jones [Å]      
        self._epskB = prop_gas["epskB"]   # energía de pozo LJ / kB [K]      
        self._cpg = prop_gas["Cp_molar"] # Specific heat of gas [J/mol/k]
        self._K = prop_gas["k"]  # Thermal conduction in gas phase [W/m/k]
        self._H = prop_gas["H"]  # Enthalpy [J/K]
        
        
        # Requerimientos
        self._required = {'Design':True,
                          'prop_gas':True,
                          'initialC_info'  : False,
                          'boundaryC_info' : False,
                          'thermal_info'   : False,
                          'PreProces'      : False,
                          'RunProces'      : False,
                          'Results'        : False}
    
        #Initial conditions
        if init_info == None:
            self._P0 = None
            self._T0 = None
            self._x0 = None
            self._N0 = None
        else:
            self._P0 = init_info['P0']
            self._T0 = init_info['T0']
            self._x0 = np.array(init_info['x0'],dtype=float)
            self._N0 = self._P0*self._Vol/self._R /self._T0
            self._required['initialC_info'] = True
        
        
        #Inlet Outlet conditions
        if boundary_info == None:
            self._Pin=None
            self._Pout=None
            self._Tin=None
            self._xin=None
        else:
            self._Pin=boundary_info['Pin']
            self._Pout=boundary_info['Pout']
            self._Tin=boundary_info['Tin']
            self._xin=np.array(boundary_info['xin'],dtype=float)
            self._required['boundaryC_info'] = True
            
        #Thermal conditions
        if thermal_info == None:
            self._adi=None
            self._hext=None
            self._hint=None
            self._Tamb=None
            self._kw=None
        else:
            self._adi=thermal_info['adi']
            self._hext=thermal_info['hext']
            self._hint=thermal_info['hint']
            self._Tamb=thermal_info['Tamb']
            self._kw=thermal_info['kw']
            self._required['thermal_info'] = True

            
        #Variables  actuales (valores actuales y de la ultima iteracion)
        self._state_cell_vars = None
        self._previous_vars = None
        ## Variables globales (valores de todas las simulaciones)
        self._t=None
        self._P=None
        self._N=None
        self._T=None
        self._x=None
        self._Qloss=None
        ## Variables locales (valores de la ultima simulacion)
        self._t2=None
        self._P2=None
        self._N2=None
        self._T2=None
        self._x2=None
        self._Qloss2=None
        ## Logging de la ultima simulacion
        self._t_log = None
        self._P_log = None
        self._N_log = None
        self._T_log = None
        self._x_log = None
        self._Qloss_log = None
        ## Conexiones del equipo
        self._conex =   {
                        "inlet": None,
                        "where_inlet": None,
                        "outlet": None,
                        "where_outlet": None,
                        "valves_top": [],
                        "valves_bottom": [],
                        "valves_side": [],
                        "units_top": [],
                        "units_bottom": [],
                        "units_side": [],
                        }
        
        # Mapping vars
        self._nVars=None
        self._labelVars=None
        # Resultados y actualTime 
        self._results = None
        self._actualTime = None
        self._case = None
        self._setup_vars=[ '_name', '_L', '_D', '_Ri', '_e', '_vol',
                          '_Ai', '_Al', '_Aint', '_nodos','_P0', '_T0', 
                          '_x0', '_N0','_prop_gas', '_species', '_ncomp',
                          '_MW', '_mu', '_Dm', '_cpg', '_K', '_H','_Pin',
                          '_Tin', '_xin', '_Pout','_adi', '_kw', '_hint',
                          '_hext', '_Tamb','_nVars', '_labelVars'
                          ]
        self._setup_code=None


     
    def __str__(self):
        str_return = '[[Current information included here]] \n'
        for kk in self._required.keys():
            str_return = str_return + '{0:16s}'.format(kk)
            if type(self._required[kk]) == type('  '):
                str_return = str_return+ ': ' + self._required[kk] + '\n'
            elif self._required[kk]:
                str_return = str_return + ': True\n'
            else:
                str_return = str_return + ': False\n' 
        return str_return

    
# =============================================================================
#     # 2. Configuración    
# =============================================================================
    def initialC_info(self, P0, T0, x0):
        """
        ¿Qué hago?
        ----------
        Configuro las condiciones iniciales del tanque, definiendo la presión inicial `P0`, la temperatura inicial `T0` y el vector de fracciones molares `x0` (composición) para el gas contenido en el tanque. Calcula automáticamente el número inicial de moles `N0` en cada nodo usando la ecuación de gases ideales, y marca el flag interno de condiciones iniciales como definido.
    
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        - Es **imprescindible** para definir el estado de partida de cualquier simulación dinámica, llenado, vaciado o balance de un tanque.
        - Permite especificar el punto de partida físico y químico del sistema, garantizando que todos los cálculos posteriores tengan una base coherente y reproducible.
        - El método es requerido tanto por el usuario (setup manual) como por rutinas automáticas de inicialización previa a la simulación.
    
        Ejemplo de uso
        --------------
        >>> T1 = Tank('T1', 10, 1, 0.01, prop_gas)
        >>> T1.initialC_info(P0=2e5, T0=300.0, x0=[1.0, 0.0])
        # Deja el tanque T1 con P0=2 bar, T0=300 K, y composición 100% primer componente.
    
        Avisos
        ------
        - El vector `x0` debe ser de longitud igual al número de componentes definido en `prop_gas['species']` y sumar 1 (o muy cercano, se corrige internamente).
        - Si se llama a este método varias veces, **sobrescribe** el estado inicial definido anteriormente.
        - Es **obligatorio** llamar a este método antes de lanzar la simulación; si no se hace, la inicialización fallará.
        - No valida físicamente la coherencia de los valores (puedes poner presiones o temperaturas poco realistas, y lo aceptará).
        - No ejecuta ningún cálculo dinámico, solo fija el estado de partida.
        """
        self._P0 = P0
        self._T0 = T0
        self._x0 = np.array(x0, dtype=float)
        # Calcula N0 con la ecuación de gases ideales (por nodo, solo 1 nodo aquí)
        self._N0 = (self._P0 * self._Volx) / (self._R * self._T0)
        self._required['initialC_info'] = True
        return None


    def boundaryC_info(self, Pin, Tin, xin, Pout):
        """
        ¿Qué hago?
        ----------
        Defino las condiciones de frontera (boundary conditions) para el tanque, estableciendo la presión y temperatura de entrada (`Pin`, `Tin`), la composición del gas de entrada (`xin`), y la presión de salida (`Pout`). Marca internamente que las condiciones de frontera han sido correctamente configuradas.
    
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        - Es **esencial** para describir cómo interactúa el tanque con el entorno (otras unidades, alimentación, extracción, etc.) durante una simulación dinámica.
        - Permite especificar el gas que entra (presión, temperatura y composición) y la presión de referencia a la que se descargará el gas cuando haya una válvula de salida.
        - Es usado tanto por el usuario para configurar escenarios experimentales, como por los controladores automáticos del sistema antes de cada simulación.
    
        Ejemplo de uso
        --------------
        >>> T1.boundaryC_info(Pin=3e5, Tin=320.0, xin=[0.7, 0.3], Pout=1e5)
        # Configura el tanque para ser alimentado a 3 bar, 320 K, mezcla 70/30, y con presión de descarga de 1 bar.
    
        Avisos
        ------
        - El vector `xin` debe tener el mismo número de componentes que `prop_gas['species']` y normalmente sumar 1.
        - Este método **no realiza comprobaciones físicas** de coherencia entre valores (puedes poner temperaturas o presiones no realistas).
        - Si alguna variable no se define (por ejemplo, sólo te interesa alimentar o sólo descargar), debes asegurarte de poner un valor apropiado; el modelo espera que estén definidos los cuatro parámetros.
        - Es **obligatorio** llamar a este método antes de simular, especialmente si hay flujo de entrada o salida; si se omite, la simulación lanzará error de setup incompleto.
        - Llamar varias veces a este método **sobrescribe** los valores anteriores.
        """

        self._Pin = Pin
        self._Tin = Tin
        self._xin = np.array(xin, dtype=float)
        self._Pout = Pout
        self._required['boundaryC_info'] = True
        return None


    def thermal_info(self, adi, kw, hint, hext, Tamb):
        """
        ¿Qué hago?
        ----------
        Defino los parámetros térmicos del tanque, especificando si el tanque es adiabático (`adi`), la conductividad térmica de la pared (`kw`), los coeficientes de transferencia de calor interna (`hint`) y externa (`hext`), y la temperatura ambiente (`Tamb`). Marca internamente que la información térmica ha sido configurada.
        
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        - Permite controlar cómo se intercambia calor entre el gas, la pared del tanque y el ambiente, lo que es fundamental para simulaciones realistas con balances de energía.
        - Es imprescindible para modelos donde la temperatura evoluciona de forma dinámica o donde interesa analizar pérdidas/ganancias de calor.
        - Es usado tanto por el usuario para experimentar con diferentes condiciones térmicas, como por scripts automáticos de setup de simulación.
        
        Ejemplo de uso
        --------------
        >>> T1.thermal_info(adi=False, kw=15.0, hint=100.0, hext=20.0, Tamb=298.15)
        # Configura el tanque con transmisión térmica realista: kw=15 W/m/K, coeficiente interno 100, externo 20, y temperatura ambiente 25°C.
        
        Avisos
        ------
        - Si `adi=True`, el tanque se considera perfectamente aislado y **no se aplica** intercambio de calor con el entorno, ignorando el resto de parámetros.
        - Si `adi=False`, es obligatorio definir `kw`, `hint`, `hext` y `Tamb` con valores físicos coherentes para que el modelo calcule correctamente las pérdidas o ganancias de calor.
        - El método **no valida** automáticamente los valores de los parámetros (puedes poner negativos o no realistas), revisa tu setup.
        - Es obligatorio llamar a este método antes de simular balances de energía o temperaturas no constantes; de lo contrario, el setup se marcará como incompleto.
        """
        self._adi = adi
        self._kw = kw
        self._hint = hint
        self._hext = hext
        self._Tamb = Tamb
        self._required['thermal_info'] = True
        return None


# =============================================================================
#      # 3. Inicilizar/Retroceder/Resets/Leer data
# =============================================================================
    def _reset_conex_(self):
        """
        ¿Qué hago?
        ----------
        Borro y reinicio todas las conexiones del tanque con otras unidades y válvulas, dejando vacíos los atributos del diccionario `_conex`.
        Elimino cualquier referencia a válvulas de entrada/salida, válvulas de conexión, o unidades conectadas en cualquier puerto.
        
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        - Para dejar el tanque completamente "aislado" y listo para redefinir su topología de conexiones en la red.
        - Evita residuos de conexiones anteriores que puedan causar errores o referencias cruzadas inesperadas al reconstruir la red.
        - Usado por métodos internos de gestión de red y recomendado antes de cargar nuevas configuraciones.
        
        Ejemplo de uso
        --------------
        >>> T1._reset_conex_()
        # El tanque T1 ya no está conectado a ninguna válvula ni a otras unidades.
        
        Avisos
        ------
        - No modifica condiciones iniciales, resultados ni propiedades físicas, solo la información de conexiones.
        - **Obligatorio** llamar antes de redefinir/rediseñar la red o al cargar nuevos escenarios para evitar bugs por conexiones antiguas.
        - Si otras variables dependen de las conexiones, revisa su estado tras ejecutar este método.
        """

        self._conex = {
            "inlet": None,
            "where_inlet": None,
            "outlet": None,
            "where_outlet": None,
            "valves_top": [],
            "valves_bottom": [],
            "valves_side": [],
            "units_top": [],
            "units_bottom": [],
            "units_side": [],
        }
        return None
    

    def _reset_logs_(self):
        """
        ¿Qué hago?
        ----------
        Borro y reinicio todos los arrays y listas de logs/resultados temporales del tanque, tanto los datos limpios (`_t2`, `_P2`, etc.)
        como los logs crudos (`_t_log`, `_P_log`, etc.). Dejo todos estos atributos vacíos o en None, como tras construir el objeto.
        
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        - Para preparar el tanque antes de una nueva simulación o tras recortar el historial de datos.
        - Evita acumulación de resultados de simulaciones previas que puedan contaminar nuevos cálculos, gráficos o balances.
        - Fundamental en workflows iterativos, campañas de simulación o cuando se reinicia el solver en distintos escenarios.
        
        Ejemplo de uso
        --------------
        >>> T1._reset_logs_()
        # T1 ya no tiene datos de simulaciones previas, sus logs y arrays de resultados están limpios.
        
        Avisos
        ------
        - No afecta a las condiciones iniciales, físicas o de frontera del tanque, solo a los datos dinámicos almacenados.
        - **Recomendado** llamar siempre antes de una nueva integración dinámica (solve_ivp) o al recortar temporalmente los resultados.
        - Si olvidas llamarlo, puedes mezclar resultados viejos y nuevos en análisis o visualización.
        """
        self._t2 = None
        self._P2 = None
        self._N2 = None
        self._T2 = None
        self._x2 = None
        self._Qloss2 = None
        self._required['Results'] = False
        
        self._t_log = []
        self._P_log = []
        self._N_log = []
        self._T_log = []
        self._x_log = []
        self._Qloss_log = []
        return None


    def _initialize_(self):
        """
        ¿Qué hago?
        ----------
        Inicializo (preparo) todos los arrays y variables dinámicas del tanque al estado inicial, dejándolo listo para comenzar
        una simulación desde t=0. Verifico que las condiciones iniciales, de frontera y térmicas estén correctamente definidas;
        en caso contrario, lanzo un error informativo. Reinicio los arrays principales de tiempo, presión, temperatura, moles,
        composición, y pérdidas térmicas, así como el estado instantáneo y el historial temporal.
        
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        - Para garantizar que el tanque arranca una simulación desde un estado consistente y limpio, sin residuos de ejecuciones anteriores.
        - Es esencial antes de lanzar una integración dinámica (`solve_ivp`), especialmente en simulaciones por lotes o campañas.
        - Ayuda a evitar errores difíciles de depurar por estados incompletos o datos heredados.
        - Pensado tanto para usuarios finales como para desarrolladores que usan la clase en pipelines automáticos.
        
        Ejemplo de uso
        --------------
        >>> T1.initialC_info(P0=1e5, T0=300, x0=[1.0, 0.0])
        >>> T1.boundaryC_info(Pin=2e5, Tin=310, xin=[1.0, 0.0], Pout=1e5)
        >>> T1.thermal_info(adi=False, kw=45, hint=200, hext=150, Tamb=298)
        >>> T1._initialize_()
        # El tanque T1 está listo para simular desde t=0
        
        Avisos
        ------
        - Si falta alguna condición (inicial, de frontera o térmica), lanza un `RuntimeError` y muestra en detalle qué falta.
        - Solo resetea el estado dinámico y los arrays de simulación; no toca los parámetros de diseño ni condiciones base.
        - **Obligatorio** llamar a este método antes de cada integración numérica para evitar residuos de simulaciones previas.
        - Si tienes varias simulaciones en cadena, llama a `_reset_logs_()` y después a `_initialize_()` antes de cada ciclo.
        """

        
        if not (self._required.get('initialC_info', False) and
            self._required.get('boundaryC_info', False) and
            self._required.get('thermal_info', False)):
            raise RuntimeError(
            f"⛔ No se puede inicializar el tanque '{self._name}' sin definir:\n"
            f"\tinitialC_info  : {self._required['initialC_info']}\n"
            f"\tboundaryC_info : {self._required['boundaryC_info']}\n"
            f"\tthermal_info   : {self._required['thermal_info']}"
            )
        self._required['PreProces'] = True
        
        self._results = None    
        n_nodes = self._nodos
        self._actualTime = 0.0
        
        self._t = np.array([0.0])
        self._T = np.full((1, n_nodes), self._T0) 
        self._P = np.full((1, n_nodes), self._P0)
        self._N = np.full((1, n_nodes), self._N0)
        self._Qloss = np.full((1, n_nodes), 0.0)
        self._x = np.tile(self._x0, (1, n_nodes, 1))
        
        self._state_cell_vars = {
                            't': self._actualTime,
                            'N': self._N[0].copy(),   # [nodo]
                            'P': self._P[0].copy(),   # [nodo]
                            'T': self._T[0].copy(),   # [nodo]
                            'x': self._x[0].flatten().copy()   # [nodo * ncomp]
                            }
        self._previous_vars = self._state_cell_vars.copy()
        self._reset_logs_()
        return None
        
     
    def _croopTime_(self, target_time):
        """
        ¿Qué hago?
        ----------
        Recorto todos los arrays de resultados y estados del tanque hasta el instante `target_time`, eliminando cualquier
        información posterior a ese momento. Dejo el estado instantáneo del tanque sincronizado con el nuevo tiempo final,
        permitiendo relanzar o continuar simulaciones desde ese punto. Limpio también los logs y resultados temporales 
        para garantizar un estado consistente y sin residuos de simulaciones posteriores.
        
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        - Para retroceder la simulación a un instante anterior y poder relanzar, optimizar o modificar condiciones a partir de ahí.
        - Útil en control avanzado, análisis de escenarios, optimización o campañas en las que se exploran bifurcaciones del mismo caso.
        - Permite liberar memoria y asegurar que los resultados y el estado están perfectamente sincronizados con el tiempo de corte.
        - Pensado tanto para usuarios que ajustan simulaciones en tiempo real como para scripts automáticos o notebooks interactivos.
        
        Ejemplo de uso
        --------------
        >>> T1._croopTime_(25.0)
        # El tanque T1 elimina cualquier resultado posterior a t=25 s y queda listo para relanzar desde ahí
        
        Avisos
        ------
        - Si el tanque no tiene datos previos o el tiempo solicitado está fuera del rango, lanza un `ValueError` informativo.
        - Tras recortar, el estado interno (`_state_cell_vars`) se actualiza al último valor disponible antes de `target_time`.
        - El método también limpia logs y borra resultados temporales (`_reset_logs_`), de modo que cualquier simulación posterior empieza limpio.
        - No afecta la configuración estructural ni las condiciones base del tanque, solo el estado dinámico y arrays de simulación.
        """

        idx_valid = np.where(self._t <= target_time)[0]
        if len(idx_valid) == 0:
            raise ValueError(f"⛔ No hay datos en el tanque '{self._name}' para el tiempo solicitado {target_time:.2f} s")
        last_idx = idx_valid[-1]
    
        # Recorta arrays principales
        self._t = self._t[:last_idx+1]
        self._N = self._N[:last_idx+1, ...]
        self._T = self._T[:last_idx+1, ...]
        self._P = self._P[:last_idx+1, ...]
        self._x = self._x[:last_idx+1, ...]
        if self._Qloss is not None:
            self._Qloss = self._Qloss[:last_idx+1, ...]
    
        # Actualiza el estado actual
        self._actualTime = self._t[-1]
        self._state_cell_vars['t'] = self._actualTime
        self._state_cell_vars['N'] = self._N[-1, :].copy()
        self._state_cell_vars['T'] = self._T[-1, :].copy()
        self._state_cell_vars['P'] = self._P[-1, :].copy()
        self._state_cell_vars['x'] = self._x[-1, :, :].flatten().copy()
    
        self._reset_logs_()    
        self._results = None
        self._required['Results'] = False
        return None


    def _readCase_(self,case): #EN DESARROLLO
        setup_str = case
        setup_dict = pickle.loads(base64.b64decode(setup_str.encode('ascii')))
    
        # Asigna todas las variables del setup al objeto
        for var, val in setup_dict.items():
            setattr(self, var, val)
    
        # Guarda internamente el código si quieres acceso rápido
        self._setup = setup_str
    
        return None

    
    def _readData_(self,data): #EN DESARROLLO
        # Decodifica y deserializa
        data_dict = pickle.loads(base64.b64decode(data.encode('ascii')))
    
        # Arrays principales de simulación
        for key in ['t', 'P', 'T', 'N', 'x']:
            val = data_dict.get(key, None)
            if val is not None:
                setattr(self, f'_{key}', np.array(val))
    
        # Estado instantáneo
        if 'state_N' in data_dict:
            if self._state_cell_vars is None:
                self._state_cell_vars = {}
            self._state_cell_vars['N'] = np.array(data_dict['state_N'])
        if 'state_P' in data_dict:
            self._state_cell_vars['P'] = np.array(data_dict['state_P'])
        if 'state_T' in data_dict:
            self._state_cell_vars['T'] = np.array(data_dict['state_T'])
        if 'state_x' in data_dict:
            self._state_cell_vars['x'] = np.array(data_dict['state_x'])
    
        # Configuración de simulación
        if data_dict.get('simConfig', None) is not None:
            self._simConfig = data_dict['simConfig']
    
        # actualTime
        if data_dict.get('actualTime', None) is not None:
            self._actualTime = data_dict['actualTime']
    
        # Guarda internamente el string si quieres (opcional)
        self._data = data
    
        return None

    
    def _readCaseData_(self,case,data):
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
        Elimino duplicados y solapamientos en los registros temporales (`logs`) de la simulación, procesando dos listas paralelas: 
        tiempos (`t_log`) y valores asociados (`VAR_log`). Para cada instante de tiempo único, conservo solo el **último** valor registrado,
        garantizando que cada tiempo tenga un solo valor representativo y ordenado.
        
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        - Para limpiar y sincronizar los logs de variables almacenadas durante la integración, evitando repeticiones que pueden ocurrir por pasos 
          de integración irregulares, reinicios o acumulaciones.
        - Fundamental antes de realizar análisis de balances, visualización o exportación de datos, donde cada tiempo debe estar asociado a un solo valor.
        - Pensado para usuarios que quieren resultados consistentes y limpios tras la simulación, sin artefactos numéricos.
        
        Ejemplo de uso
        --------------
        >>> t_clean, Qn_clean = valve._clean_LOG_(valve._t_log, valve._Qn_log)
        >>> print(t_clean)   # Tiempos únicos ordenados
        >>> print(Qn_clean)  # Último valor de caudal para cada tiempo
        
        Avisos
        ------
        - Si hay múltiples registros para el mismo instante de tiempo, **solo se guarda el último** (no la media ni el primero).
        - La salida son dos listas: los tiempos únicos ordenados y los valores asociados.
        - Es fundamental aplicar este método antes de hacer integrales, gráficos o balances para evitar errores por datos duplicados.
        - No modifica los logs originales, solo devuelve versiones limpias.
        """
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

    
    def _clean_LOG_unit_(self):
        """
        ¿Qué hago?
        ----------
        Limpio y sincronizo todos los logs temporales del tanque (o unidad), eliminando duplicados y solapamientos. 
        Procesa los arrays de tiempos y variables (N, x, P, T, Qloss) usando `_clean_LOG_`, asegurando que para cada tiempo único
        solo quede un valor representativo por variable. Comprueba que todos los arrays de tiempo limpios coinciden, 
        y almacena los resultados limpios y sincronizados para análisis, balances o visualización.
    
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        - Para dejar el historial de resultados de la unidad perfectamente limpio antes de calcular balances, analizar datos o graficar.
        - Evita errores causados por registros múltiples para un mismo instante, que pueden afectar balances o integrales.
        - Fundamental tras simulaciones largas o con reinicios, para que todos los arrays (N, x, P, T, Qloss) tengan los mismos tiempos y tamaño.
        - Para usuarios que necesitan resultados numéricamente coherentes y fiables tras la simulación.
    
        Ejemplo de uso
        --------------
        >>> tank._clean_LOG_unit_()
        # Deja el tanque con todos los arrays limpios, listos para análisis o exportación.
    
        Avisos
        ------
        - Si los tiempos limpios de los diferentes logs **no coinciden exactamente**, lanza un assert (esto debe ser revisado si ocurre).
        - No modifica los logs originales, solo almacena las versiones limpias en las variables terminadas en `2` (por ejemplo, `self._N2`).
        - Obligatorio llamar a este método antes de calcular balances globales o exportar los resultados de la unidad.
        - Si el historial de tiempos no está bien alineado entre variables, se detectará aquí antes de causar errores posteriores.
        """
        
        
        t_N, N2 = self._clean_LOG_(self._t_log, self._N_log)
        t_x, x2 = self._clean_LOG_(self._t_log, self._x_log)
        t_P, P2 = self._clean_LOG_(self._t_log, self._P_log)
        t_T, T2 = self._clean_LOG_(self._t_log, self._T_log)
        t_Q, Qloss2 = self._clean_LOG_(self._t_log, self._Qloss_log)
    
        # Comprueba que todos los tiempos coinciden (muy importante para los balances)
        assert (t_N == t_x == t_P == t_T == t_Q), "Tiempos limpios no coinciden, revisa logs"
    
        # Almacena los resultados finales sincronizados
        self._storeBal_(t_N, P2, T2, x2, N2, Qloss2)
        return None


    def _storeBal_(self, t2, P2, T2, x2, N2, Qloss2):
        """
        ¿Qué hago?
        ----------
        Almaceno y sincronizo en arrays definitivos (limpios y listos para análisis) todos los resultados principales de la simulación
        del tanque: tiempos (`t2`), número de moles (`N2`), composición (`x2`), temperatura (`T2`), presión (`P2`) y pérdidas térmicas (`Qloss2`).
        Reorganizo las dimensiones para asegurar que cada variable tiene la forma adecuada: [tiempos, nodos, (ncomp)].
        
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        - Para tener disponibles todos los resultados limpios, sin duplicados ni inconsistencias, listos para análisis de balances,
          exportación, validaciones o graficado.
        - Esta función es llamada internamente tras limpiar los logs, como parte final de la simulación o al preparar los resultados para análisis.
        - Facilita el acceso seguro y ordenado a los datos, garantizando dimensiones consistentes para cualquier posprocesado.
        - Para usuarios que necesitan trabajar con arrays perfectamente formateados y sincronizados tras la simulación.
        
        Ejemplo de uso
        --------------
        >>> tank._storeBal_(t2, P2, T2, x2, N2, Qloss2)
        # Los arrays tank._t2, tank._N2, tank._x2, tank._T2, tank._P2 y tank._Qloss2 quedan listos y sincronizados.
        
        Avisos
        ------
        - Se espera que los argumentos de entrada ya estén limpios y alineados en tiempo (sin duplicados, todos con igual longitud).
        - Si las dimensiones de entrada no coinciden con [ntimes, nodos, (ncomp)], pueden surgir errores de `reshape`.
        - Llama siempre a este método desde el flujo de limpieza (`_clean_LOG_unit_`), no manualmente salvo que sepas lo que haces.
        - Tras este método, los resultados de la simulación están listos para análisis de balances de masa, energía, exportación o visualización.
        """

        ntimes=len(t2)
        nodos=self._nodos
        ncomp=self._ncomp
        self._t2 = np.array(t2)
        self._N2 = np.array(N2).reshape(ntimes,nodos)
        self._x2 = np.array(x2).reshape(ntimes,nodos,ncomp)
        self._T2 = np.array(T2).reshape(ntimes,nodos)
        self._P2 = np.array(P2).reshape(ntimes,nodos)
        self._Qloss2 = np.array(Qloss2).reshape(ntimes,nodos)
        return None


# =============================================================================
#     # 5.Helpers/Getters/Setters
# ============================================================================= 
    def _get_mapping_(self):
        """
        ¿Qué hago?
        ----------
        Calculo y devuelvo el número total de variables de estado necesarias para describir el estado dinámico completo del tanque,
        así como las etiquetas (labels) asociadas a cada variable. Para cada nodo, se incluyen: número de moles (N), composición (x para todas las especies menos la última) y temperatura (T).
        
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        - Para definir la estructura del vector de estado global que será integrado numéricamente en la simulación dinámica.
        - Facilita la correspondencia entre el vector plano ODE y las variables físicas del tanque, esencial para la integración y el post-procesado.
        - Es utilizado automáticamente por la red o por el solver al construir el mapeo global de todas las unidades.
        - Permite un acceso claro e indexado a cada variable por su nombre (útil para debugging y análisis avanzado).
        
        Ejemplo de uso
        --------------
        >>> n_vars, labels = tank._get_mapping_()
        >>> print(n_vars)    # Número total de variables de estado del tanque
        >>> print(labels)    # Lista de etiquetas ["N_0", "x0_0", ..., "T_0", ..., "N_n", ...]
        
        Avisos
        ------
        - El orden de las variables es: [N, x1...x(ncomp-1), T] para cada nodo, y se repite para todos los nodos.
        - El último componente de la composición x no se almacena explícitamente (se calcula por diferencia, para garantizar suma 1).
        - El método también guarda internamente `self._nVars` y `self._labelVars` para referencia rápida posterior.
        - Es importante llamar a este método después de definir el número de nodos y especies del tanque.
        """

        n_vars_per_node = 1 + (self._ncomp - 1) + 1  # N, x1..x(ncomp-1), T por nodo
        n_vars = self._nodos * n_vars_per_node
        labels = []
        for n in range(self._nodos):
            labels.append(f"N_{n}")
            for i in range(0, self._ncomp-1):
                labels.append(f"x{i}_{n}")
            labels.append(f"T_{n}")
        # Guarda como atributos para referencia rápida
        self._nVars = n_vars
        self._labelVars = labels
        return n_vars, labels


    def _set_State_ylocal_(self, ti, ylocal):
        
        nodos = self._nodos
        ncomp = self._ncomp
        N = np.zeros(nodos)
        T = np.zeros(nodos)
        x = np.zeros((nodos, ncomp))
        
        for idx, label in enumerate(self._labelVars):
            if label.startswith("N_"):
                nodo = int(label.split("_")[1])
                N[nodo] = ylocal[idx]
            elif label.startswith("T_"):
                nodo = int(label.split("_")[1])
                T[nodo] = ylocal[idx]
            elif label.startswith("x"):
                especie, nodo = label[1:].split("_")
                especie = int(especie)
                nodo = int(nodo)
                x[nodo, especie] = ylocal[idx]
    
        # Completa la última fracción molar de cada nodo (cierra sumas a 1)
        for nodo in range(nodos):
            x[nodo, -1] = 1.0 - np.sum(x[nodo, :-1])
        x = x.flatten()
        P = N * self._R * T / self._Volx   
        
        self._state_cell_vars = {
            't': ti,
            'N': N,
            'P': P,
            'x': x,
            'T': T,
        }
        return None
    
                    
    def _get_State_ylocal_(self):
        """
        ¿Qué hago?
        ----------
        Construyo y devuelvo un vector plano (`ylocal`) que contiene todas las variables de estado locales del tanque, ordenadas según el mapping definido en `_get_mapping_()`.
        Extraigo de `self._state_cell_vars` los valores actuales de número de moles, fracciones molares y temperaturas de cada nodo, y los organizo en el orden requerido para la integración ODE.
        
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        - Este método es esencial para la integración dinámica global (solve_ivp), donde se necesita un vector plano con todas las variables de todos los equipos.
        - Permite empaquetar el estado interno del tanque en el formato compatible con los solvers numéricos.
        - Se utiliza al iniciar la simulación o cada vez que se requiere conocer el estado actual como vector.
        - Es clave en sistemas multi-equipo y para el checkpointing/restauración de estados.
        
        Ejemplo de uso
        --------------
        >>> ylocal = tank._get_State_ylocal_()
        # Devuelve el subvector de estado del tanque, listo para ensamblarse en el vector global de integración.
        
        Avisos
        ------
        - El orden y la longitud del vector resultante están determinados por `_labelVars`, definido en `_get_mapping_()`.
        - Si `self._state_cell_vars` no está sincronizado (por ejemplo, después de modificar manualmente variables internas), el vector puede no reflejar el estado físico real.
        - Las fracciones molares de la última especie **sí** se incluyen, ya que en este vector está toda la información explícita.
        - Asegúrate de usar este método siempre que se requiera el estado plano para integración ODE o restauración de estado.
        """

        ylocal = []
        x=self._state_cell_vars["x"].reshape(self._nodos,self._ncomp)
        for idx, label in enumerate(self._labelVars):
            if label.startswith("N_"):
                nodo = int(label.split("_")[1])
                ylocal.append(self._state_cell_vars["N"][nodo])
            elif label.startswith("T_"):
                nodo = int(label.split("_")[1])
                ylocal.append(self._state_cell_vars["T"][nodo])
            elif label.startswith("x"):
                especie, nodo = label[1:].split("_")
                especie = int(especie)
                nodo = int(nodo)
                ylocal.append(x[nodo][especie])
                
        return np.array(ylocal, dtype=float)


    def _get_unitVar_(self, unit, var, time=None, nodo=None, port=None, especie=None    ):
        """
        ¿Qué hago?
        ----------
        Devuelvo el valor (o subarray) de una variable interna de una unidad (tanque, columna, etc.) en el formato y localización solicitados.
        Permite extraer el valor de una variable interna (presión, temperatura, composición, etc.) para un instante de tiempo concreto, un nodo, puerto o especie específicos, devolviendo siempre el array filtrado y orientado al caso de uso.
        
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        - Facilita la consulta genérica y flexible de resultados o estados internos en cualquier unidad, tanto durante simulaciones como en postproceso.
        - Esencial para rutinas de análisis, conexiones entre equipos, cálculos de caudal por válvula, visualización y validación.
        - Permite obtener fácilmente el valor actual, histórico, o localizado de cualquier variable clave sin acoplarse al formato interno de la clase.
        - Pensado para uso interno del framework, rutinas de postproceso y lógica de red.
        
        Ejemplo de uso
        --------------
        >>> # Presión en el nodo superior en t=12.5 s
        >>> P_top = net._get_unitVar_(T1, 'P', time=12.5, port='top')
        >>> # Composición de CO2 en nodo 0 para t=0
        >>> x_CO2 = net._get_unitVar_(T1, 'x', time=0, nodo=0, especie='CO2')
        >>> # Temperatura de todos los nodos para el último instante simulado
        >>> T_all = net._get_unitVar_(T1, 'T', time=-1)
        
        Avisos
        ------
        - El nombre de variable (`var`) debe coincidir con los atributos internos de la unidad (por ejemplo, 'T', 'P', 'x', 'N').
        - Si se usa un tiempo (`time`), el método busca el instante más cercano en el array de tiempos correspondiente (`_t` o `_t2`).
        - Para variables de composición ('x'), el método soporta filtrado adicional por puerto, nodo y especie (por nombre o índice).
        - Para variables escalares (como presión o temperatura), los argumentos `port` y `nodo` seleccionan el nodo o ubicación deseada.
        - Si el resultado es un único valor, se devuelve como float para máxima comodidad; si no, se devuelve el array filtrado.
        - Lanza error si el atributo no existe, el puerto es incorrecto, o faltan argumentos obligatorios (por ejemplo, nodo para 'side').
        - La función es muy general: asegúrate de pasar argumentos consistentes con la estructura de la unidad para evitar errores silenciosos.
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
           
    
    def _get_StateVar_(self, var, nodo=None, port=None):
        """
        ¿Qué hago?
            ----------
        Devuelvo el valor de una variable de estado interna del objeto tanque, como 'N', 'T', 'P' o 'x',
        permitiendo filtrar por nodo o por puerto lógico ('top', 'bottom', 'side').
        Reconstruye el formato apropiado (vector, matriz o escalar) según la variable solicitada y los argumentos proporcionados.
    
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        - Para obtener de forma rápida y consistente el valor actual de cualquier variable de estado en el tanque, 
          tanto durante la integración ODE como en análisis, control o conexiones con otros equipos.
        - Permite abstraer el acceso a las variables, independientemente del número de nodos o de la estructura interna.
        - Usado internamente en rutinas de caudal, cálculo de balances, lógica de válvulas, y por el usuario para análisis avanzado.
    
        Ejemplo de uso
        --------------
        >>> # Obtener fracciones molares del nodo superior
        >>> x_top = T1._get_StateVar_('x', port='top')
        >>> # Obtener temperatura en nodo 0
        >>> T_nodo0 = T1._get_StateVar_('T', nodo=0)
        >>> # Obtener presión en el fondo del tanque
        >>> P_bottom = T1._get_StateVar_('P', port='bottom')
        >>> # Obtener matriz completa de composiciones
        >>> x_full = T1._get_StateVar_('x')
    
        Avisos
        ------
        - El parámetro `var` debe ser una clave de self._state_cell_vars: 'N', 'T', 'P', 'x'.
        - Para variables de composición ('x'), el resultado es siempre un array de forma [nodo, ncomp].
          Si se filtra por puerto o nodo, sigue siendo [1, ncomp]; si no, es [nodos, ncomp].
        - Para variables escalares ('N', 'T', 'P'), devuelve el valor para el nodo/puerto solicitado, o el array completo si no se filtra.
        - Si solo hay un nodo, puede devolver un escalar.
        - Lanza error si `port='side'` pero no se especifica `nodo`.
        - Los valores devueltos están **sin aplanar**: ideales para ser usados directamente en balances, caudal, etc.
        """
       
        val = self._state_cell_vars[var]
        nodos = self._nodos
        ncomp = self._ncomp
    
        if var == 'x':
            # Siempre lo reconstruye a [nodos, ncomp]
            if val.ndim == 1:
                x_mat = val.reshape(nodos, ncomp)
            else:
                x_mat = val
            if port is not None:
                port = port.lower()
                if port == "bottom":
                    return x_mat[[0], :]   # [1, ncomp]
                elif port == "top":
                    return x_mat[[-1], :]  # [1, ncomp]
                elif port == "side":
                    if nodo is None:
                        raise ValueError("Si port='side', debes especificar nodo")
                    return x_mat[[nodo], :]
                else:
                    raise ValueError(f"Port '{port}' no reconocido (usa 'bottom', 'top', 'side')")
            elif nodo is not None:
                return x_mat[[nodo], :]   # [1, ncomp]
            else:
                return x_mat  # [nodos, ncomp]
        else:
            # Para 'N', 'T', 'P', etc.
            if isinstance(val, (np.ndarray, list)):
                if port is not None:
                    port = port.lower()
                    if port == "bottom":
                        return val[0]
                    elif port == "top":
                        return val[-1]
                    elif port == "side":
                        if nodo is None:
                            raise ValueError("Si port='side', debes especificar nodo")
                        return val[nodo]
                    else:
                        raise ValueError(f"Port '{port}' no reconocido (usa 'bottom', 'top', 'side')")
                elif nodo is not None:
                    return val[nodo]
                else:
                    return val
            else:
                # Si es escalar (tanque 1 nodo)
                return val
     
# =============================================================================
#     # 6.Dinamica 
# =============================================================================
    def _rhs_(self):
        """
        ¿Qué hago?
        ----------
        Calculo el **vector de derivadas de estado** para el tanque, es decir, la tasa de cambio de cada variable dinámica 
        (N, x, T) en el formato requerido por el integrador ODE (solve_ivp), en función del estado actual, entradas, salidas, 
        transferencia térmica y lógicas de válvula. Si el tanque no está correctamente inicializado (preprocesado), lanza un error.
        
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        - Este método es **clave en la integración dinámica**: se invoca automáticamente en cada paso de integración temporal.
        - Permite simular la evolución temporal del tanque bajo cualquier configuración, calculando balances de materia y energía en tiempo real.
        - Garantiza que no se pueda avanzar en la simulación si faltan datos esenciales de setup, evitando errores silenciosos.
        - Es utilizado tanto internamente por la clase de red (`Network`) como en test unitarios o simulaciones individuales.
        
        Ejemplo de uso
        --------------
        >>> # Dentro de un loop ODE o solver:
        >>> dydt = T1._rhs_()
        >>> # O, indirectamente, desde la red:
        >>> net._rhs(t, y)  # que llama internamente a T1._rhs_() para cada equipo
        
        Avisos
        ------
        - Si el tanque no ha sido inicializado correctamente (faltan condiciones iniciales, de frontera o térmicas), lanza un RuntimeError.
        - El método asume que todas las variables requeridas para el balance están correctamente definidas y actualizadas.
        - El formato de salida es un vector plano compatible con el mapeo de integración de la red, acorde al método _get_mapping_().
        - No actualiza ningún estado interno, solo calcula y devuelve las derivadas instantáneas para el paso actual.
        """

        if not self._required.get('PreProces', False):
            raise RuntimeError(
                    f"⛔ No se puede calcular _rhs en el tanque '{self._name}' sin haber realizado el preprocesado (_initialize).\n"
        "Asegúrate de que se han definido las condiciones iniciales, de frontera y térmicas, y que se ha ejecutado correctamente el método _initialize()."
    )
        return solveTanks(self)


# =============================================================================
#     # 7.Save Simulation/Case/Data
# =============================================================================
    def _storeData_(self, t, ylocal):
        """
        ¿Qué hago?
        ----------
        Reconstruyo y almaceno los **arrays temporales** de las variables principales del tanque (N, x, T, P) a partir de los resultados
        planos (`ylocal`) de la integración ODE, usando la lista de etiquetas (`self._labelVars`). Organizo estos datos en arrays
        multidimensionales [tiempo, nodo, (componente)], cierro la fracción molar, recalculo la presión y actualizo el estado actual.
    
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        - Permite convertir los resultados planos de la integración (propios de `solve_ivp`) en arrays estructurados fáciles de analizar y graficar.
        - Garantiza que todos los resultados de la simulación quedan almacenados de forma sincronizada y reutilizable.
        - Se invoca automáticamente tras cada integración ODE, pero también puede llamarse para añadir nuevos tramos temporales en simulaciones por etapas.
        - Es esencial para el postprocesado, visualización y cálculos de balances del tanque.
    
        Ejemplo de uso
        --------------
        >>> # Tras integrar con solve_ivp:
        >>> t = sol.t                   # Vector de tiempos
        >>> ylocal = sol.y[ini:fin, :]  # Subvector de resultados de este tanque
        >>> T1._storeData_(t, ylocal)   # Reconstruye y almacena resultados
    
        Avisos
        ------
        - Si ya existen datos previos (self._actualTime > 0), los nuevos resultados se concatenan, extendiendo los arrays temporales.
        - El método cierra siempre la última fracción molar para asegurar la consistencia (∑x_i = 1 en cada nodo y tiempo).
        - La presión (P) se recalcula siempre a partir de N, T y el volumen total por nodo.
        - Actualiza el estado actual del tanque (`self._state_cell_vars`) al último instante disponible.
        - Marca el flag 'Results' como True para permitir el postproceso y la visualización.
        """
            
        # Reconstruye los arrays multinodo a partir de ylocal y self._labelVars
        nodos = self._nodos
        ncomp = self._ncomp
        ntimes = t.size
    
        N = np.zeros((ntimes, nodos))
        T = np.zeros((ntimes, nodos))
        x = np.zeros((ntimes, nodos, ncomp))
    
        # Recorrido mapping para rellenar los arrays
        for var_idx, label in enumerate(self._labelVars):
            if label.startswith("N_"):
                nodo = int(label.split("_")[1])
                N[:, nodo] = ylocal[var_idx, :]
            elif label.startswith("T_"):
                nodo = int(label.split("_")[1])
                T[:, nodo] = ylocal[var_idx, :]
            elif label.startswith("x"):
                especie, nodo = label[1:].split("_")
                especie = int(especie)
                nodo = int(nodo)
                x[:, nodo, especie] = ylocal[var_idx, :]
            
        # Cierra la fracción molar de cada nodo en cada tiempo
        x[:, :, -1] = 1.0 - np.sum(x[:, :, :-1], axis=2)
        P = (N * self._R * T) / self._Volx
        
        # Guarda en atributos
        if self._actualTime == 0:
            self._t = t
            self._N = N
            self._T = T
            self._x = x
            self._P = P
        else:
            self._t = np.concatenate([self._t, t])
            self._N = np.concatenate([self._N, N])
            self._x = np.concatenate([self._x, x],axis=0)  
            self._T = np.concatenate([self._T, T])
            self._P = np.concatenate([self._P, P])
                        
        self._required['Results'] = True
        self._actualTime = t[-1]
        # Actualiza el estado actual (último instante)
        self._state_cell_vars = {
            't': t[-1],
            'N': N[-1, :],
            'T': T[-1, :],
            'x': x[-1, :, :].flatten(),
            'P': P[-1, :]
        }
        return None


    def _writeCase_(self,): # EN DESARROLLO !!!
        setup_vars = self._setup_vars
        setup_dict = {var: getattr(self, var, None) for var in setup_vars}
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
        self._setup_code = setup_str
        return self._name, setup_str


    def _writeData_(self,): # EN DESARROLLO !!!
        data_dict = {}
        # Historial principal como arrays planos
        data_dict['t'] = self._t.flatten() if self._t is not None else None
        data_dict['P'] = self._P.flatten() if self._P is not None else None
        data_dict['T'] = self._T.flatten() if self._T is not None else None
        data_dict['N'] = self._N.flatten() if self._N is not None else None
        data_dict['x'] = self._x.flatten() if self._x is not None else None  # [ntimes, nodes, ncomp] -> 1D
    
        # Último estado (vectores planos)
        data_dict['state_N'] = self._state_cell_vars['N'].copy() if self._state_cell_vars is not None else None
        data_dict['state_P'] = self._state_cell_vars['P'].copy() if self._state_cell_vars is not None else None
        data_dict['state_T'] = self._state_cell_vars['T'].copy() if self._state_cell_vars is not None else None
        data_dict['state_x'] = self._state_cell_vars['x'].copy() if self._state_cell_vars is not None else None
    
        # Configuración de simulación (puedes adaptar el nombre del dict)
        if hasattr(self, '_simConfig'):
            data_dict['simConfig'] = self._simConfig
        else:
            data_dict['simConfig'] = None
    
        data_dict['actualTime'] = self._actualTime
    
        # Serializa y codifica
        binary = pickle.dumps(data_dict)
        data_str = base64.b64encode(binary).decode('ascii')
        self._data = data_str
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
        Grafico la evolución temporal de las principales variables del tanque (presión, temperatura, moles, fracciones molares) y
        de todas las válvulas conectadas (apertura y caudal instantáneo), mostrando la historia completa de la simulación para análisis y validación.
    
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        - Permite visualizar de forma clara y rápida cómo evoluciona el tanque y sus conexiones a lo largo del tiempo.
        - Es útil tanto para diagnóstico en desarrollo como para análisis de resultados, control de calidad y comunicación de simulaciones.
        - Incluye la información de **todas** las válvulas conectadas, diferenciando colores según el tipo (entrada, salida, inter-unit).
        - Es fundamental antes de analizar balances o compartir resultados con otros usuarios.
    
        Ejemplo de uso
        --------------
        >>> T1._plot_()
        # Abre una ventana con seis subgráficos: presión, temperatura, composición, moles totales,
        # % de apertura de válvulas y caudales instantáneos (entrada/salida).
    
        Avisos
        ------
        - Es imprescindible que el método `_storeData_` se haya ejecutado correctamente (flag "Results" activado) antes de graficar; si no, avisa y no grafica.
        - Para tanques con varias válvulas interunitarias, el sentido del caudal se corrige según la diferencia de presión en cada instante.
        - Si faltan datos en arrays críticos (`self._t`, `self._T`, `self._P`, etc.) puede dar errores o graficar incompleto.
        - La visualización distingue colores para válvulas de entrada (azul), salida (rojo) e inter-unitarias (paleta variada).
        - Los caudales se representan positivos al entrar al tanque y negativos al salir.
        - Si necesitas personalizar colores, leyendas o formato, puedes modificar el código antes de la entrega final.
        """
        
        if not self._required.get("Results", False):
            print("⛔ No hay resultados disponibles para graficar. Ejecuta la simulación primero.")
            return
    
        t = np.array(self._t)            # [ntimes]
        ncomp = self._ncomp
        P = np.array(self._P).flatten()  # [ntimes]
        T = np.array(self._T).flatten()
        N = np.array(self._N).flatten()
        x = np.array(self._x).reshape(-1, ncomp)  # [ntimes, ncomp]
    
        colors_species = [
            'orange', 'olive', 'pink', 'skyblue', 'palegreen', 'purple', 'brown', 'red', 'blue', 'grey'
        ]
    
        # Válvulas conectadas
        valves_all = []
        for k in ["inlet", "outlet", "valves_top", "valves_bottom", "valves_side"]:
            v = self._conex.get(k, None)
            if isinstance(v, list):
                valves_all.extend(v)
            elif v is not None:
                valves_all.append(v)
        valves_all = list(set(valves_all))
    
        fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    
        # 1. Presión
        axs[0, 0].plot(t, P, color='navy',lw=2)
        axs[0, 0].set_title("Presión [Pa]")
        axs[0, 0].set_ylabel("P [Pa]")
        axs[0, 0].grid()
    
        # 2. Temperatura
        axs[0, 1].plot(t, T, color='crimson',lw=2)
        axs[0, 1].set_title("Temperatura [K]")
        axs[0, 1].set_ylabel("T [K]")
        axs[0, 1].grid()
    
        # 3. Fracciones molares
        for i in range(ncomp):
            axs[1, 0].plot(t, x[:, i], label=f"{self._species[i]}", color=colors_species[i % len(colors_species)],lw=2)
        axs[1, 0].set_title("Fracción molar")
        axs[1, 0].set_ylabel("x [-]")
        axs[1, 0].legend(fontsize=8)
        axs[1, 0].grid()
    
        # 4. Moles totales
        axs[1, 1].plot(t, N, color='darkgreen',lw=2)
        axs[1, 1].set_title("Moles totales")
        axs[1, 1].set_ylabel("N [mol]")
        axs[1, 1].grid()
    
        COLOR_INLET = 'blue'
        COLOR_OUTLET = 'red'
        
        # Paleta para el resto (sin azul ni rojo)
        otros_colores = [
            'green', 'orange', 'purple', 'brown', 'magenta', 'cyan', 'olive', 'gold', 'grey'
        ]
        color_cycle = itertools.cycle(otros_colores)
        
        for v in valves_all:
            t_v = np.array(v._t)
            a_v = np.array(v._a*100)
            
            tipo = v._conex.get("type", "")
            if tipo == "inlet":
                color = COLOR_INLET
            elif tipo == "outlet":
                color = COLOR_OUTLET
            else:
                color = next(color_cycle)
            
            axs[2, 0].plot(t_v, a_v, label=v._name, lw=2, color=color)
        
        axs[2, 0].set_title("% Apertura válvulas")
        axs[2, 0].set_xlabel("Tiempo [s]")
        axs[2, 0].set_ylabel("Apertura [%]")
        axs[2, 0].legend(fontsize=8)
        axs[2, 0].grid()
        
        # 6. Caudal instantáneo válvulas (+entrada, -salida)
        color_cycle2 = itertools.cycle(otros_colores)  # Nuevo ciclo para evitar coincidencias si hay muchas válvulas
        
        for v in valves_all:
            t_v = np.array(v._t)
            qn_v = np.array(v._Qn).copy()
            tipo = v._conex.get("type", "")
            if tipo == "inlet":
                color = COLOR_INLET
            elif tipo == "outlet":
                color = COLOR_OUTLET
                qn_v*=-1.
            else:
                color = next(color_cycle2)
                qn_v_cor=[]
                other = v._conex["unit_B"] if v._conex["unit_A"]._name == self._name else v._conex["unit_A"]
                for i, ti in enumerate(t_v):            
                    
                    p_self = self._get_unitVar_(self, "P", time=ti)
                    p_other = self._get_unitVar_(other, "P", time=ti)
                    
                    if p_self > p_other:
                        qn_v_cor.append(-qn_v[i])  # Sale: negativo
                    else:
                        qn_v_cor.append(qn_v[i])   # Entra: positivo
                qn_v = np.array(qn_v_cor)
                    
            axs[2, 1].plot(t_v, qn_v, label=v._name, lw=2, color=color)
        
        self._aux = valves_all
        axs[2, 1].set_title("Caudal instantáneo válvulas (+entrada, -salida) [Nm³/h]")
        axs[2, 1].set_xlabel("Tiempo [s]")
        axs[2, 1].set_ylabel("Qn [Nm³/h]")
        axs[2, 1].legend(fontsize=8)
        axs[2, 1].grid()
        plt.suptitle(f"\nSimulación del tanque {self._name}",fontsize=24)
        plt.tight_layout()
        
    def _checkBalances_(self):
        """
        ¿Qué hago?
        ----------
        Calculo los balances completos de masa, especies y energía del tanque, integrando entradas, salidas y transferencias interunitarias
        (por válvulas) a lo largo de toda la simulación. Devuelvo un diccionario con los valores iniciales, finales, integrales de entrada/salida/interconexión,
        balances netos y errores relativos (%) respecto al cambio real en el sistema.
        
        ¿Por qué/para qué/quién lo hago?
        --------------------------------
        - Permite validar que la simulación cierra correctamente los balances fundamentales (masa total, cada especie y energía).
        - Es crucial para el postprocesado: diagnóstico de pérdidas numéricas, fugas, errores de configuración o integración.
        - Pensado para el usuario avanzado, desarrollador o auditor que requiere asegurar la conservación física y la calidad de los resultados.
        - Este método puede ser llamado automáticamente desde la red o manualmente tras una simulación.
        
        Ejemplo de uso
        --------------
        >>> balances = T1._checkBalances_()
        >>> for k, v in balances.items():
        ...     print(f"{k}: {v}")
        
        Avisos
        ------
        - Es imprescindible que se haya ejecutado `_clean_LOG_unit_` antes para sincronizar los arrays limpios (`*_2`), si no los resultados serán incorrectos.
        - Los balances de masa consideran la suma sobre **todos** los nodos del tanque, para cada especie y para la masa total.
        - Las entradas y salidas se integran usando los caudales limpios (`_Qn2`) y los arrays de tiempo `t_ref` (igual para cada válvula conectada).
        - El sentido de caudal interunitario se calcula en cada instante según el gradiente de presión entre unidades conectadas.
        - El error porcentual puede ser positivo o negativo, según la dirección del desbalance respecto al sistema.
        - Todos los resultados de energía están en **kJ** (kilojulios), salvo los errores (%) que son adimensionales.
        - Si alguna válvula no tiene los arrays limpios (`_Qn2` y `_t2`) correctamente calculados, el resultado del balance puede ser incorrecto o lanzar error.
        - Si `print_table = True`, imprime por pantalla los resultados de forma detallada y legible, útil para depuración manual.
        - Si cambias la topología o los nodos después de simular, **recalcula logs y ejecuta este método de nuevo**.
        - Devuelve un **diccionario** con los resultados, que puede ser tabulado (por ejemplo, con Pandas DataFrame).
        """

        # Válvulas conectadas
        valves_all = []
        for k in ["inlet", "outlet", "valves_top", "valves_bottom", "valves_side"]:
            v = self._conex.get(k, None)
            if isinstance(v, list):
                valves_all.extend(v)
            elif v is not None:
                valves_all.append(v)
        valves_all = list(set(valves_all))
        
        Nm3_2_mol = 1.01325e5 / (8.314 * 273.15)
        t_ref = np.array(self._t2)
        ncomp  = self._ncomp
        nodos  = self._nodos
        
        # list to np.array([ntimes][nodoes])
        # Inicializamos acumuladores
        N_init     = np.sum(self._N2[0])
        N_end      = np.sum(self._N2[-1]) 
        N_in_tot   = 0.0
        N_out_tot  = 0.0
        N_intercon = 0.0
        N_total    = 0.0
        warnE_N      = 0.0 
        
        Ni_init     = np.zeros_like(self._xin)
        Ni_end      = np.zeros_like(self._xin)
        Ni_in_tot   = np.zeros_like(self._xin)
        Ni_out_tot  = np.zeros_like(self._xin)
        Ni_intercon = np.zeros_like(self._xin)
        Ni_total    = np.zeros_like(self._xin)
        warnE_Ni    = np.zeros_like(self._xin)
        
        Tref = 273.15
        Cp = np.array(self._cpg).copy()
        
        T_init = self._T2[0]
        T_end = self._T2[-1]
        H_init     = 0.0
        H_end      = 0.0
        H_in_tot   = 0.0
        H_out_tot  = 0.0
        H_intercon = 0.0
        H_total    = 0.0
        warnE_H    = 0.0 
        
        for j in range(0,ncomp):
            for i in range(0,nodos):    
                    Ni_init[j] += self._N2[0][i]*self._x2[0][i][j]
                    Ni_end[j]  +=  self._N2[-1][i]*self._x2[-1][i][j]
            
            H_init = np.sum(np.dot(Ni_init,Cp)*(T_init-Tref))
            H_end  = np.sum(np.dot(Ni_end,Cp)*(T_end-Tref))
        
        deltaN  = N_end - N_init
        deltaNi = Ni_end - Ni_init
        deltaH = H_end - H_init
        
        for v in valves_all:
            qn_v = np.array(v._Qn2)
            n_dot = qn_v * Nm3_2_mol / 3600.0  # mol/s
            tipo = v._conex.get("type", None)
            if tipo == "inlet":
                N_in_tot += trapz(n_dot, t_ref)
                port = v._conex.get("port_A", None)
                xin = self._xin
                Tin = self._Tin
                for i in range(0,ncomp):
                    Ni_in_tot[i]+=trapz(n_dot*xin[i], t_ref)
                
                H_in_tot+=trapz(n_dot*np.dot(xin,Cp)*(Tin-Tref), t_ref)
                
                                    
            elif tipo == "outlet":
                N_out_tot += trapz(n_dot, t_ref)
                port = v._conex.get("port_A", None)
                xout = self._get_unitVar_(self, "x2",port=port)
                Tout = self._get_unitVar_(self, "T2",port=port)
                for i in range(0,ncomp):
                    Ni_out_tot[i]+=trapz(n_dot*xout[:,i], t_ref)
                    H_out = n_dot*xout[:,i]*Cp[i]
                    H_out_tot+=trapz(H_out*(Tout-Tref), t_ref)
                    
            else:
                # Interconectada: determinar sentido en cada instante
                if v._conex["unit_A"]._name == self._name:
                    other = v._conex["unit_B"]
                    pother = v._conex["port_B"]
                    pself = v._conex["port_A"]                
                elif v._conex["unit_B"]._name == self._name:
                    other = v._conex["unit_A"]
                    pother = v._conex["port_A"]
                    pself = v._conex["port_B"]
                    
                Pself = self._get_unitVar_(self, "P2", port=pself)
                Tself = self._get_unitVar_(self, "T2", port=pself)
                xself = self._get_unitVar_(self, "x2", port=pself)
                Pother = other._get_unitVar_(other, "P2", port=pother)
                Tother = other._get_unitVar_(other, "T2", port=pother)
                xother = other._get_unitVar_(other, "x2", port=pother)
                signo = np.sign(Pother - Pself)
                N_intercon += trapz(n_dot * signo, t_ref)
                
                for i in range(0, ncomp):
                    
                    n_dot_sign = n_dot * signo
                    
                    x_inter = np.where(signo > 0, n_dot_sign *xother[:,i],
                                                  n_dot_sign*xself[:,i])
                    
                    Ni_intercon[i] += trapz(x_inter, t_ref)  
                    
                    H_inter = np.where(signo > 0, n_dot_sign*xother[:,i]*Cp[i]*(Tother-Tref),
                                                  n_dot_sign*xself[:,i]*Cp[i]*(Tself-Tref))
                
                    H_intercon+=trapz(H_inter, t_ref)
                
        # Balances neto:
        N_total = N_in_tot - N_out_tot + N_intercon
        H_total = H_in_tot - H_out_tot + H_intercon
        
        if  max(abs(N_in_tot),abs(N_out_tot),abs(N_intercon)) > 0 :
            warnE_N = (abs(N_total) - abs(deltaN))/max(abs(N_in_tot),abs(N_out_tot),abs(N_intercon))*100
        else:
            warnE_N = 0.
        
        if max(abs(H_in_tot),abs(H_out_tot),abs(H_intercon)) > 0:
            warnE_H = (abs(H_total) - abs(deltaH))/max(abs(H_in_tot),abs(H_out_tot),abs(H_intercon))*100
        else:
            warnE_H = 0.
            
        for i in range(0,ncomp):    
            Ni_total[i] = Ni_in_tot[i] - Ni_out_tot[i] + Ni_intercon[i]
            if max(abs(Ni_in_tot[i]),abs(Ni_out_tot[i]),abs(Ni_intercon[i])) > 0.:
                warnE_Ni[i] = (abs(Ni_total[i]) - abs(deltaNi[i]))/max(abs(Ni_in_tot[i]),abs(Ni_out_tot[i]),abs(Ni_intercon[i]))*100
            else:
                warnE_Ni[i]=0
        
        print_table = False
        if print_table == True:
            print("============================================")
            print(f" BALANCE MASA : {self._name} ")
            print("============================================")
            if abs(warnE_N) > -1.E-12:
                print(f" N_init (iniciales) = {N_init:.2f} mol")
                print(f" N_end (finales)    = {N_end:.2f} mol")
                print("============================================")
                print(f" deltaN (entrada)   = {deltaN:.2f} mol")
                print("============================================")
                print(f" N_in (entrada)     = {N_in_tot:.2f} mol")
                print(f" N_out (salida)     = {N_out_tot:.2f} mol")
                print(f" N_intercon (neto)  = {N_intercon:.2f} mol")
                print(f" Balance            = {N_total:.2f} mol")
                print("============================================")
            print(f" %Error             = {warnE_N:.2f} %")
            print("============================================")
    
            print("============================================")
            print(f" BALANCE ESPECIES : {self._name} ")
            print("============================================")
            for i in range(0,ncomp):
                print(f"{self._species[i]}")
                if abs(warnE_Ni[i]) > -1.E-12:
                    print(f"\tNi_init (iniciales) = {Ni_init[i]:.2f} mol")
                    print(f"\tNi_end (finales)    = {Ni_end[i]:.2f} mol")
                    print("============================================")
                    print(f"\tdeltaNi (entrada)   = {deltaNi[i]:.2f} mol")
                    print("============================================")
                    print(f"\tN_in (entrada)      = {Ni_in_tot[i]:.2f} mol")
                    print(f"\tN_out (salida)      = {Ni_out_tot[i]:.2f} mol")
                    print(f"\tN_intercon (neto)   = {Ni_intercon[i]:.2f} mol")
                    print(f"\tBalance             = {Ni_total[i]:.2f} mol")
                print("============================================")
                print(f"\t%Error              = {warnE_Ni[i]:.2f} %")
                print("============================================")
                
            print("============================================")
            print(f" BALANCE ENERGIA   : {self._name} ")
            print("============================================")
            if abs(warnE_H) > -1.E-12:
                print(f"H_init (iniciales)  = {H_init/1000:.2f} kJ")
                print(f"H_end (finales)     = {H_end/1000:.2f} kJ")
                print("============================================")
                print(f"deltaH (entrada)    = {deltaH/1000:.2f} kJ")
                print("============================================")
                print(f" H_in (entrada)     = {H_in_tot/1000:.2f} kJ")
                print(f" H_out (salida)     = {H_out_tot/1000:.2f} kJ")
                print(f" H_intercon (neto)  = {H_intercon/1000:.2f} kJ")
                print(f" Balance            = {H_total/1000:.2f} kJ")
            print("============================================")
            print(f" %Error             = {warnE_H:.2f} %")
            print("============================================")
        
        results = {
            "N_init": round(N_init, 2),
            "N_end": round(N_end, 2),
            "deltaN": round(deltaN, 2),
            "N_in": round(N_in_tot, 2),
            "N_out": round(N_out_tot, 2),
            "N_intercon": round(N_intercon, 2),
            "N_total": round(N_total, 2),
            "%Error_N": round(warnE_N, 2),
            # Especies:
            **{f"Ni_init_{self._species[i]}": round(Ni_init[i], 2) for i in range(ncomp)},
            **{f"Ni_end_{self._species[i]}": round(Ni_end[i], 2) for i in range(ncomp)},
            **{f"deltaNi_{self._species[i]}": round(deltaNi[i], 2) for i in range(ncomp)},
            **{f"Ni_in_{self._species[i]}": round(Ni_in_tot[i], 2) for i in range(ncomp)},
            **{f"Ni_out_{self._species[i]}": round(Ni_out_tot[i], 2) for i in range(ncomp)},
            **{f"Ni_intercon_{self._species[i]}": round(Ni_intercon[i], 2) for i in range(ncomp)},
            **{f"Ni_total_{self._species[i]}": round(Ni_total[i], 2) for i in range(ncomp)},
            **{f"%Error_Ni_{self._species[i]}": round(warnE_Ni[i], 2) for i in range(ncomp)},
            # Energía en kJ
            "H_init": round(H_init / 1000., 2),
            "H_end": round(H_end / 1000., 2),
            "deltaH": round(deltaH / 1000., 2),
            "H_in": round(H_in_tot / 1000., 2),
            "H_out": round(H_out_tot / 1000., 2),
            "H_intercon": round(H_intercon / 1000., 2),
            "H_total": round(H_total / 1000., 2),
            "%Error_H": round(warnE_H, 2),  # El error no lo dividas por 1000, ya está en %
        }

        
        return results