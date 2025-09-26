# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 15:24:02 2025

@author: MiguelCamaraSanz
"""

import numpy as np
from collections import defaultdict
import pickle, base64

from solveLibs import solveAdsColumn
from commonLibs import _propGas_
from commonLibs import _DijAll_, _DimMix_, _DknMix_, _DporMix_, _DeffMix_, _Dz_, _lamz_
from commonLibs import _avg_face_arith_, _avg_face_harm_matrix_
from commonLibs import _darcy_ergun_velocity_faces_, _ergun_velocity_faces_
from commonLibs import _vFaces_to_vCells_, _phiFaces_to_phiCells_, _peclet_faces_
from commonLibs import _Re_, _Pr_, _Sc_, _Nu_, _Sh_, _hc_, _hrw_, _kc_, _kldf_, _Da_ext_, _Da_ldf_, _Bi_g_, _Bi_c_, _U_global_

class gasifierColumn:
# =============================================================================
#     # 1. Constructor (solo lee parámetros y crea atributos; NADA de cálculos)
# =============================================================================
    def __init__(self,
                 Name,
                 design_info,
                 packed_info,
                 prop_gas,
                 prop_solid,
                 prop_kinetics,
                 init_info=None,
                 boundary_info=None,
                 thermal_info=None,
                 ):
        self._R = 8.314
        self._name = Name

        # --- Diseño y malla axial ---
        self._L       = design_info["Longitud"]
        self._Hin     = design_info["InletBio"]      # Cota fija de boca de alimentación de biomasa
        self._D       = design_info["Diametro"]
        self._e       = design_info["Espesor"]
        self._nodos   = design_info["Nodos"]
        self._znodos  = self._nodos + 2

        self._Ri      = self._D / 2
        self._Vol     = np.pi * (self._Ri)**2 * self._L
        self._Ai      = np.pi * (self._Ri)**2
        self._Al      = np.pi * self._D * self._L
        self._Aint    = self._Al + self._Ai

        # Discretización por nodo (geométrica, OK calcular aquí)
        self._Lx       = np.ones(self._nodos) * self._L / self._nodos
        self._Lxf      = self._Lx[:-1]
        self._xfaces   = np.concatenate([[0.0], np.cumsum(self._Lx)])
        self._xcenters = 0.5 * (self._xfaces[1:] + self._xfaces[:-1])

        self._D_x   = np.ones(self._nodos) * self._D
        self._Rix   = self._D_x / 2
        self._Aix   = np.pi * (self._Rix)**2
        self._Alx   = np.pi * self._D_x * self._Lx
        self._Aintx = self._Aix + self._Alx

        # --- Lecho (parámetros base) ---
        self._Lpb0  = packed_info["Longitud"]     # Altura inicial de lecho (estado inicial)
        self._eps0  = packed_info["Porosidad"]    # Porosidad de lecho base (estado inicial)
        self._tau0  = packed_info["Tortuosidad"]  # Tortuosidad de lecho base (estado inicial)

        # Variables DINÁMICAS (se inicializarán en initialize(), no aquí)
        self._H          = None     # Altura actual de lecho [m]; H ← Lpb0 en initialize()
        self._mask_bed   = None     # Máscara booleana/ fraccional θ_j (0–1) de ocupación de lecho por nodo
        self._theta_bed  = None     # (opcional) Versión fraccional de máscara para celdas parcialmente ocupadas

        # Volúmenes efectivos por nodo (se calculan al fijar H y eps)
        self._Volfx = None  # Volumen de gas por nodo:
                            #   V_g[j] = A_ix[j]*Lx[j]*( (1 - theta[j]) + theta[j]*eps[j] )
        self._Volsx = None  # Volumen de sólido por nodo:
                            #   V_s[j] = A_ix[j]*Lx[j]*( theta[j]*(1 - eps[j]) )

        # --- Gas (propiedades) ---
        self._prop_gas = prop_gas
        self._species  = prop_gas["species"]
        self._ncomp    = len(self._species)
        self._i        = None  # dict de índices de especies: { 'O2':i, 'CO2':i, ... } (se crea en initialize())

        # Backend de propiedades dependientes de T-P-x:
        # self._prop_gas = _propGas_  # handler a usar en _rhs_ (no se llama aquí)

        # --- Sólido: composición y propiedades por componente (datos base) ---
        self._prop_solid = prop_solid
        self._bioName    = prop_solid["Name"]

        # Fracciones másicas de la MATRIZ SÓLIDA DE ENTRADA (excluye agua líquida en poros)
        self._y_bio  = prop_solid["y_biomass"]  # Biomasa seca (antes de pirólisis)
        self._y_char = prop_solid["y_char"]
        self._y_ash  = prop_solid["y_ash"]
        # Humedad líquida asociada al sólido alimentado (inventario aparte)
        self._y_h2o  = prop_solid["y_h2o"]

        # Densidades de cada constituyente (kg/m3)
        self._rho_bio  = prop_solid["rho_biomass"]
        self._rho_char = prop_solid["rho_char"]
        self._rho_ash  = prop_solid["rho_ash"]
        self._rho_h2o  = prop_solid["rho_h2o"]

        # Propiedades térmicas de cada constituyente
        self._k_bio,  self._k_char,  self._k_ash,  self._k_h2o  = (
            prop_solid["k_biomass"], prop_solid["k_char"], prop_solid["k_ash"], prop_solid["k_h2o"]
        )
        self._cp_bio, self._cp_char, self._cp_ash, self._cp_h2o = (
            prop_solid["cp_biomass"], prop_solid["cp_char"], prop_solid["cp_ash"], prop_solid["cp_h2o"]
        )

        # Geometría de partícula / intrapartícula (base)
        self._d_s_0   = prop_solid["diam"]      # dp0
        self._eps_s_0 = prop_solid["eps"]       # porosidad intrapartícula (esqueleto)
        self._r_p_0   = prop_solid["rp"]        # radio de partícula (si aplica)
        self._sphere  = prop_solid["sphere"]    # esfericidad

        # Variables DINÁMICAS de partícula/lecho (arrays por nodo; se crean en initialize())
        self._d_s  = None   # dp(z)
        self._eps  = None   # eps_bed(z)  (porosidad de LECHO)
        self._tau  = None   # tau(z)
        self._a_s  = None   # área específica lecho(z): a_s = 6*(1 - eps)/dp  (solo en zona de lecho)

        # --- Densidades del sólido (placeholders + FÓRMULAS DOCUMENTADAS) ---

        # (1) Densidad del ESQUELETO SECO de partícula, ρ_skel  [kg/m3]
        #     No incluye agua líquida; depende de la mezcla {biomasa seca, char, ceniza}
        #     Normalizando las fracciones sólidas secas:  wB~, wC~, wAsh~
        #     1/ρ_skel = wB~/ρ_bio + wC~/ρ_char + wAsh~/ρ_ash
        #     (Se calcula más adelante a partir de inventarios w_B, w_C, w_Ash en cada nodo)
        self._rho_skel = None

        # (2) Densidad APARENTE de partícula, ρ_p  [kg/m3]
        #     Incluye agua líquida ocupando poros intrapartícula (si se modela saturación sW en ε_p)
        #     ρ_p = ρ_skel * (1 - ε_p) + ρ_H2O * sW * ε_p
        #     (ε_p ≈ ε_s, sW∈[0,1]; si no se usa sW, puede omitirse el cálculo de ρ_p)
        self._rho_p = None

        # (3) Densidad EFECTIVA DEL SÓLIDO CONTINUO EN LECHO por celda, ρ_s,eff  [kg/m3]
        #     Es la que entra en balances de masa/energía del “sólido continuo”.
        #     Se calcula con inventarios por celda:
        #       ρ_s,eff[j] = (m_B + m_C + m_Ash + m_Wl)_j  /  V_s[j]
        #     donde  V_s[j] = A_ix[j]*Lx[j] * theta[j] * (1 - eps[j])
        self._rho_s = None  # aquí guardaremos ρ_s,eff(z) cuando haya inventarios

        # (4) Densidad BULK de lecho de referencia (opcional, para informes), ρ_pb0  [kg/m3]
        #     ρ_sa0: densidad del esqueleto seco "de diseño" (mezcla base de entrada)
        #     ρ_pb0 = (1 - eps0) * ρ_sa0
        self._rho_sa0 = None
        self._rho_pb0 = None

        # --- Propiedades térmicas EFECTIVAS del sólido continuo (placeholders) ---
        # cp_s,eff(z): mezcla ponderada por masa de {B, C, Ash, Wl} en cada celda
        # k_s,eff(z):  puedes empezar con mezcla lineal simple o un modelo lecho (k_s*(1-eps) + k_g*eps)
        self._cp_s = None
        self._k_s  = None

        # --- Transporte/cierres y cinética (guardados; no se evalúan aquí) ---
        self._kinetics = prop_kinetics   # dict con {drying, pyrolysis, char_O2, char_H2O, char_CO2, homog: {...}}
        self._transport = None           # (opcional) dict con Ergun, Nu, Dax, etc. si decides pasarlo luego

        # --- Condiciones iniciales, frontera y térmicas (solo guardar referencias si llegan) ---
        self._init_info     = init_info
        self._boundary_info = boundary_info
        self._thermal_info  = thermal_info

        # --- Puertos / Conexiones (se usarán en networkLibs) ---
        self._conex = {
            "inlet_gas": None,     "where_inlet_gas": None,
            "outlet_gas": None,    "where_outlet_gas": None,
            "inlet_solid": None,   "where_inlet_solid": None,
            "outlet_solid": None,  "where_outlet_solid": None,
            "valves_top": [], "valves_bottom": [], "valves_side": [],
            "units_top": [],  "units_bottom": [],  "units_side": [],
        }

        # --- Flags de requisitos (misma filosofía que AdsorptionColumn) ---
        self._required = {
            'Design': True,
            'prop_packed': True,
            'prop_gas': True,
            'prop_solid': True,
            'kinetics_info': True,
            'initialC_info': False,
            'boundaryC_info': False,
            'thermal_info': False,
            'PreProces': False,
            'RunProces': False,
            'Results': False,
        }

        # --- Estructuras de estado/resultados/logs (placeholders, se llenarán luego) ---
        self._t = self._v = self._P = self._N = None
        self._Tg = self._Ts = None
        self._x  = None  # fracciones molares gas [ntimes, nodos, ncomp]

        # Inventarios de sólido por especie (por nodo, en el tiempo)
        self._wB = self._wC = self._wAsh = self._wWl = None  # masas o fracciones (según elijas en initialize)

        # Hidráulica y geometría dinámica
        self._dP = None  # caída de presión total / perfil, según definas
        # self._eps, self._d_s, self._a_s ya definidos como placeholders

        # Locales “última simulación”
        self._t2 = self._v2 = self._P2 = self._N2 = None
        self._Tg2 = self._Ts2 = self._x2 = None
        self._wB2 = self._wC2 = self._wAsh2 = self._wWl2 = None

        # Logs mínimos
        self._t_log = []; self._Tg_log = []; self._Ts_log = []
        self._x_log = []; self._wB_log = []; self._wC_log = []; self._wAsh_log = []; self._wWl_log = []
        self._H_log = []; self._dP_log = []; self._eps_log = []; self._dp_log = []

        # Estados/derivadas para solver
        self._state_cell_vars = None
        self._previous_cell_vars = None
        self._state_cell_properties = None
        self._state_face_vars = None
        self._previous_face_vars = None
        self._state_face_properties = None
        self._derivates = None

        # Mapping integración
        self._nVars = None
        self._labelVars = None

        # Resultados y meta
        self._results = None
        self._actualTime = 0.0
        self._case = None
        self._setup_code = None

        # Nota: existirá un método initialize() donde:
        #  - Se fijará H := Lpb0, se construirán mask_bed y (opcional) theta_bed
        #  - Se crearán arrays nodales dinámicos: eps[:]=eps0, d_s[:]=d_s_0, a_s[:]=6*(1-eps)/d_s
        #  - Se calcularán V_s y V_g por nodo con las fórmulas comentadas arriba
        #  - Se construirán inventarios iniciales (w_B, w_C, w_Ash, w_Wl) y se evaluarán:
        #       ρ_skel, ρ_p (si aplica), ρ_s,eff
        #  - Se definirá self._i (índices de especies) y cualquier cierre adicional

    def kinetic_info(self,):
        rgas=0.
        rsolid=0.
        return rgas,rsolid

    def initialC_info(self, P0, Tg0, Ts0, x0):
        self._P0 = P0
        self._Tg0 = Tg0
        self._Ts0 = Ts0
        self._x0 = np.array(x0, dtype=float)
        self._required['initialC_info'] = True
        return None
    
    def boundaryC_info(self, Pin, Tin, xin, Pout):
        self._Pin= Pin
        self._Pout=Pout
        self._Tin=Tin
        self._xin=np.array(xin,dtype=float)
        self._required['boundaryC_info'] = True
        return None

    def thermal_info(self, adi, kw, hint, hext, Tamb):
        self._adi = adi
        self._kw = kw
        self._hint = hint
        self._hext = hext
        self._Tamb = Tamb
        self._required['thermal_info'] = True
        return None