# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 11:58:26 2025

@author: MiguelCamaraSanz
"""
import numpy as np
from collections import defaultdict
import pickle,base64

from solveLibs import solveAdsColumn
from commonLibs import _set_propertyTable_,_get_propertyTable_
from commonLibs import _mwMix_,_rhoMix_,_muMix_,_cpMix_,_kMix_,_propGas_
from commonLibs import _DijAll_,_DimMix_,_DknMix_,_DporMix_,_DeffMix_,_Dz_,_lamz_
from commonLibs import _avg_face_arith_,_avg_face_harm_matrix_
from commonLibs import _darcy_ergun_velocity_faces_,_ergun_velocity_faces_
from commonLibs import _vFaces_to_vCells_,_phiFaces_to_phiCells_,_peclet_faces_
from commonLibs import _Re_,_Pr_,_Sc_,_Nu_,_Sh_,_hc_,_hrw_,_kc_,_kldf_,_Da_ext_,_Da_ldf_,_Bi_g_,_Bi_c_,_U_global_

class AdsorptionColumn :
    
# =============================================================================
#     # 1. Constructor    
# =============================================================================
    def __init__(self,
                 Name,
                 design_info,
                 packed_info,
                 prop_gas,
                 prop_solid,
                 prop_kmtl,
                 prop_isoFuns,
                 init_info=None,     
                 boundary_info=None, 
                 thermal_info=None,  
                 ):
        
        self._R = 8.314
        self._name = Name

        # --- Diseño y malla axial ---
        self._L      = design_info["Longitud"]
        self._D      = design_info["Diametro"]
        self._e      = design_info["Espesor"]
        self._nodos  = design_info["Nodos"]
        self._znodos = self._nodos + 2

        self._Ri     = self._D / 2
        self._Lpb    = packed_info["Longitud"]
        self._eps    = packed_info["Porosidad"]
        self._tau    = packed_info["Tortuosidad"]
        
        self._Vol    = np.pi * (self._Ri)**2 * self._L
        self._Ai     = np.pi * (self._Ri)**2
        self._Al     = np.pi * self._D * self._L
        self._Aint   = self._Al + self._Ai

        # Discretización por nodo
        self._Lx  = np.ones(self._nodos) * self._L / self._nodos
        self._Lxf = self._Lx[:-1]
        self._xfaces   = np.concatenate([[0.0], np.cumsum(self._Lx)])
        self._xcenters = 0.5 * (self._xfaces[1:] + self._xfaces[:-1])
        x_min = self._L/2 - self._Lpb/2
        x_max = self._L/2 + self._Lpb/2
        self._mask_ads = (self._xcenters >= x_min) & (self._xcenters <= x_max)
        
        self._D_x = np.ones(self._nodos) * self._D
        self._Rix = self._D_x / 2
        self._Aix = np.pi * (self._Rix)**2
        self._Alx = np.pi * self._D_x * self._Lx
        self._Aintx = self._Aix + self._Alx
                 
        # --- Propiedades de gas (por componente) ---    
        self._prop_gas = prop_gas
        self._species  = prop_gas["species"]
        self._ncomp    = len(self._species)
        self._MW       = prop_gas["MW"]          
        self._mu       = prop_gas["mu"]
        self._sigmaLJ       = prop_gas["sigmaLJ"] 
        self._epskB       = prop_gas["epskB"]
        self._cpg      = prop_gas["Cp_molar"]
        self._cpg2      = prop_gas["Cp_mass"]
        self._K        = prop_gas["k"]
        self._H        = prop_gas["H"]
        

        # A FUTURO! METODO PARA OBTENER PROPIEDADES DEPENDIENTE DE T Y P
        # SE GENERA UNA TABLA EN PARA COMPONENTES PUROS Y SE ITNERPOLA 
        # self._T_range=np.linspace(295, 315, 20)
        # self._P_range=np.linspace(1e1, 10e6, 20)
        
        # self._prop_table, self._prop_summary = _set_propertyTable_(
        #                                         self._species,
        #                                         self._T_range,
        #                                         self._P_range,)
        
        # OTRO METODO PODRIA SER UNA RED NEURAL ENTRENADA CON MUCHAS VARIABLES!!
        
        # --- Propiedades del sólido y adsorbente ---
        self._prop_solid =  prop_solid
        self._adsName = prop_solid["Name"] 
        self._rho_s = prop_solid["rho"]  
        self._eps_s = prop_solid["eps"] 
        self._d_s = prop_solid["diam"]
        self._r_p = prop_solid["rp"]
        self._sphere = prop_solid["sphere"] 
        self._cp_s  = prop_solid["cp"]
        self._k_s =  prop_solid["k"]
        self._Volfx = self._Aix * self._Lx * self._eps
        self._Volsx = self._Aix * self._Lx * (1.-self._eps)
        self._ap = 6*(1-self._eps)/(self._d_s*self._sphere)
        
        #
        self._rho_sa = self._rho_s * (1.0 - self._eps_s)     # densidad esqueleto [kg/m3]
        self._rho_pb = (1.0 - self._eps) * self._rho_sa      # [kg/m3]
        self._a_s    = 6.0 * (1.0 - self._eps) / self._d_s 

        self._kmtl = prop_kmtl["kmtl"]
        self._isoFuncs = prop_isoFuns
        
        self._required = {'Design':True,
                          'prop_packed': True,
                          'prop_gas': True,
                          'prop_solid':True,
                          'mass_trans_info': True,
                          'isoFunctions_info': True,
                          'initialC_info' : False,
                          'boundaryC_info' :False,
                          'thermal_info' : False,
                          'economic_info' : False,
                          'PreProces'      : False,
                          'RunProces'      : False,
                          'Results'        : False}
        
        # --- Condiciones iniciales (a rellenar con métodos) ---#
        if init_info == None:
            self._P0 = None
            self._Tg0 = None
            self._x0 = None
            self._N0 = None
            self._q0 = None  
            self._Ts0 = None  
        else:
            self._P0 = init_info['P0']
            self._Tg0 = init_info['Tg0']
            self._x0 = np.array(init_info['x0'],dtype=float)
            self._Ts0 = init_info['Ts0']  # Temperatura sólido inicial [nodos]
            self._required['initialC_info'] = True
        
        # --- Condiciones de frontera (inlet, outlet, etc.) ---
        if boundary_info == None:
            self._Pin  = None
            self._Pout = None
            self._Tin  = None
            self._xin  = None
        else:
            self._Pin  = boundary_info['Pin']
            self._Pout = boundary_info['Pout']
            self._Tin  = boundary_info['Tin']
            self._xin  = boundary_info['xin']
            self._required['boundaryC_info'] = True

        # --- Térmicas externas ---
        if thermal_info == None:
            self._adi  = None
            self._hext = None
            self._hint = None
            self._Tamb = None
            self._kw   = None
        else:
            self._adi  = thermal_info['adi']
            self._hext = thermal_info['hext']
            self._hint = thermal_info['hint']
            self._Tamb = thermal_info['Tamb']
            self._kw   = thermal_info['kw']
            self._required['thermal_info'] = True



        # --- Variables globales para registro de resultados ---
        self._t    = None
        self._v    = None
        self._P    = None
        self._N    = None
        self._Tg   = None     # Temperatura gas [ntimes, nodos]
        self._Ts   = None     # Temperatura sólido [ntimes, nodos]
        self._x    = None     # Fracción molar gas [ntimes, nodos, ncomp]
        self._q    = None     # Adsorbed species [ntimes, nodos, ncomp]
        self._Qloss = None

        # --- Variables locales (última simulación) ---
        self._t2    = None
        self._v2    = None
        self._P2    = None
        self._N2    = None
        self._Tg2   = None
        self._Ts2   = None
        self._x2    = None
        self._q2    = None
        self._Qloss2 = None

        # --- Logging de la simulación ---
        self._t_log    = []
        self._v_log    = []
        self._P_log    = []
        self._N_log    = []
        self._Tg_log   = []
        self._Ts_log   = []
        self._x_log    = []
        self._q_log    = []
        self._Qloss_log = []

        # --- Conexiones de la columna (igual que tanque, pero adaptado) ---
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
        
        # --- Variables locales nodales (a rellenar cada paso) ---
        self._state_cell_vars = None
        self._previous_cell_vars = None
        self._state_cell_properties = None
       
        self._state_face_vars = None
        self._previous_face_vars = None
        self._state_face_properties = None
        
        # --- Derivadas temporales y espaciales ---
        self._derivates = None

        # --- Mapping vars para integración numérica ---
        self._nVars = None
        self._labelVars = None

        # --- Resultados, time info, case y setup vars ---
        self._results = None
        self._actualTime = 0.0
        self._case = None
        self._setup_code = None

        # --- Setup/atributos de utilidad ---
        self._setup_vars = [
            '_name', 
        ]
        
            
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
    def massTransfer_info(self,kmtl):
        self._kmtl=kmtl
        self._required['mass_trans_info'] = True
        return None
    

    def isoThermFunc_info(self,isoFuncs):
        self._isoFuncs=isoFuncs        
        self._required['isoFunctions_info'] = True
        return None        
            
    
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
    

    def _isotherm(self, x, P, T,mask, method="IAST", Lamb=None, C=0.0, drop_thresh=1E-3):
        
        iso_fun = self._isoFuncs['iso_fun']  # lista de funciones una por especie
        nnodos, ncomp = x.shape
        q_all = np.zeros((nnodos, ncomp), dtype=float)
    
        # imports locales para no depender de globales
        if method.lower() == "iast":
            from isoLibs import IAST
        elif method.lower() == "rast":
            from isoLibs import rast
    
        for node in range(nnodos):
            x_node = np.asarray(x[node, :], dtype=float)
            Ptot_kPa = float(P[node]) / 1000.0  # kPa
            Tnode = float(T[node])
    
            # especies relevantes
            idx_keep = np.where(x_node >= drop_thresh)[0]
            if idx_keep.size == 0 or Ptot_kPa <= 0.0:
                # nada relevante o P≈0 → q = 0
                continue
    
            x_keep = x_node[idx_keep]
            x_keep /= x_keep.sum()  # renormaliza
    
            funcs_keep = [iso_fun[i] for i in idx_keep]
    
            if method.lower() == "naive":
                # q_i = isoterma_i(P_i, T), con P_i = y_i * Ptot
                q_keep = np.array([funcs_keep[i](x_keep[i] * Ptot_kPa, Tnode)
                                   for i in range(len(funcs_keep))], dtype=float)
    
            elif method.lower() == "iast":
                # IAST espera vector de presiones parciales (kPa)
                P_i_kPa = x_keep * Ptot_kPa
                q_keep = IAST(funcs_keep, P_i_kPa, Tnode)
    
            elif method.lower() == "rast":
                if Lamb is None:
                    raise ValueError("RAST requiere matriz Lamb NxN.")
                Lamb_keep = np.asarray(Lamb)[np.ix_(idx_keep, idx_keep)]
                P_i_kPa = x_keep * Ptot_kPa
                q_keep = rast(funcs_keep, P_i_kPa, Tnode, Lamb_keep, C)
    
            else:
                raise ValueError("Método no reconocido: use 'naive', 'IAST' o 'RAST'.")
    
            # reubicar en vector completo
            q_node = np.zeros(ncomp, dtype=float)
            q_node[idx_keep] = q_keep
            q_all[node, :] = q_node
            
        q=q_all * mask[:, None]
        eps = drop_thresh
        q[np.abs(q) < eps] = 0.0
        np.clip(q, 0.0, None, out=q)
        
        return q

# =============================================================================
#      # 3. Inicilizar/Retroceder/Resets/Leer data
# =============================================================================
    def _reset_conex_(self):
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
        self._t2    = None
        self._v2    = None
        self._P2    = None
        self._N2    = None
        self._Tg2   = None
        self._Ts2   = None
        self._x2    = None
        self._q2    = None
        self._Qloss2 = None

        self._t_log    = []
        self._v_log    = []
        self._P_log    = []
        self._N_log    = []
        self._Tg_log   = []
        self._Ts_log   = []
        self._x_log    = []
        self._q_log    = []
        self._Qloss_log = []
        return None

    
    def _updateVars_(self, t=None, N=None, v=None, P=None, Tg=None, Ts=None, x=None, q=None):
        S = getattr(self, "_state_cell_vars", {})
        for k, val in (( 't', t), ('N', N), ('v', v), ('P', P),
                       ('Tg', Tg), ('Ts', Ts), ('x', x), ('q', q)):
            if val is not None:
                S[k] = val.reshape(-1) if k in ('x', 'q') else val
        self._state_cell_vars = S
        return None

    
    def _updateProperties_(self, P, T, x):
        n, nc = self._nodos, self._ncomp
        x_mat = np.asarray(x, float).reshape(n, nc)
        T = np.asarray(T, float).ravel()
        P = np.asarray(P, float).ravel()
    
        # --- Mezcla (nodal)
        #METODO1
        MWmix  = _mwMix_(x_mat, self._MW)                 # [n]
        rho    = _rhoMix_(P, T, MWmix)                    # [n]
        mu     = _muMix_(x_mat, self._mu, self._MW)       # [n]
        cp_molar   = _cpMix_(x_mat, self._cpg)                # molar 
        cp_mass     = _cpMix_(x_mat, self._cpg2)               # MASA 
        k      = _kMix_(x_mat, self._K, self._MW)         # [n]
        
        #METODO 2
        #...
        MWmix,rho,mu,cp_mass,k=_propGas_(P,T,x_mat,self._species)        
        alpha  = k / np.maximum(rho*cp_mass, 1e-30)            # [n] m2/s
    
        Dij    = _DijAll_(T, P, self._MW, self._sigmaLJ, self._epskB)  # [n,nc,nc]
        Dim    = _DimMix_(x_mat, Dij)                     # [n,nc]
        Dkn    = _DknMix_(T, self._MW, self._r_p)         # [n,nc]
        Dpor   = _DporMix_(Dim, Dkn)                      # [n,nc]
        Deff   = _DeffMix_(Dpor, self._eps_s, self._tau)  # [n,nc]
    
        # --- Promedios a CARA (usar cp MASS)
        rho_f   = _avg_face_arith_(rho)                   # [n-1]
        mu_f    = _avg_face_arith_(mu)                    # [n-1]
        cp_mass_f    = _avg_face_arith_(cp_mass)                    # [n-1]  MASS
        cp_molar_f    = _avg_face_arith_(cp_molar)                 # [n-1]  MASS
        k_f     = _avg_face_arith_(k)                     # [n-1]
        alpha_f = _avg_face_arith_(alpha)                 # [n-1]
        Dim_f   = _avg_face_harm_matrix_(Dim)             # [n-1, nc]
    
        # --- Velocidad en CARAS por Ergun
        v_faces, dPdz_faces, flowDir = _ergun_velocity_faces_(
            P=P, rho_f=rho_f, mu_f=mu_f, eps=self._eps, d_p=self._d_s,
            xcenters=self._xcenters, g=9.81
        )

        # v_faces, dPdz_faces, flowDir, inbed = _darcy_ergun_velocity_faces_(
        #     P=P, rho_f=rho_f, mu_f=mu_f, eps=self._eps, d_p=self._d_s,
        #     xcenters=self._xcenters,mask=self._mask_ads, g=9.81
        # )                                                     # v_faces con signo
    
        # Magnitud superficial en cara e INTERSTICIAL para Pe
        v_faces  = np.abs(v_faces)                         # [n-1]
        vStar_f  = v_faces / max(self._eps, 1e-30)         # [n-1]
    
        # --- Péclet en CARA
        Pe_s_f = _peclet_faces_(vStar_f, self._Lxf, Dim_f)   # [n-1,nc]
        Pe_e_f = _peclet_faces_(vStar_f, self._Lxf, alpha_f) # [n-1]
    
        # --- Dispersión axial (caras)
        Dz_f   = _Dz_(Dim_f, vStar_f, self._d_s, Pe_s_f)     # [n-1,nc]
        lamz_f = _lamz_(rho_f, cp_mass_f, alpha_f, vStar_f, self._d_s, Pe_e_f)  # [n-1]
    
        # --- Pasar CARAS -> CELDAS cuando toque
        v_cells   = _vFaces_to_vCells_(v_faces)            # [n]
        Pe_s      = _phiFaces_to_phiCells_(Pe_s_f)         # [n,nc]
        Pe_e      = _phiFaces_to_phiCells_(Pe_e_f)         # [n]
        Dz        = _phiFaces_to_phiCells_(Dz_f)           # [n,nc]
        lamz      = _phiFaces_to_phiCells_(lamz_f)         # [n]
    
        # --- Números adim. (usar u* = u/eps)
        vStar = v_cells / max(self._eps, 1e-30)            # [n]
        Reg   = _Re_(vStar, rho, mu, self._Lx)             # [n]  (por Lx)
        Rec   = _Re_(vStar, rho, mu, self._d_s)            # [n]  (por dp)
        Pr    = _Pr_(mu, cp_mass, k)                            # [n]
        Sc    = _Sc_(mu, rho, Dim)                         # [n,nc]
    
        # --- Correlaciones película
        Nuw = _Nu_(Reg, Pr)                                # [n]
        Nuc = _Nu_(Rec, Pr)                                # [n]
        Sh  = _Sh_(Rec, Sc)                                # [n,nc]
        hw  = _hc_(Nuw, k, self._Lx)                       # [n] pared por Lx (si lo quieres)
        hc  = _hc_(Nuc, k, self._d_s)                      # [n] película gas-partícula
        kc  = _kc_(Sh,  Dim, self._d_s)                    # [n,nc]
        kldf = _kldf_(Deff, self._d_s)
        
        # --- Biot & U pared
        Big = _Bi_g_(hc, self._d_s, self._k_s)             # [n]
        Bic = _Bi_c_(kc, self._d_s, Deff)                  # [n,nc]
        
        U = _U_global_(hw,self._e,self._kw,self._hext,self._adi)
    
        # --- Damköhler externos (y opcional LDF/kmtl luego)
        Da_ext = _Da_ext_(kc,self._a_s,self._L,self._eps,v_cells)     # [n,nc]
        Da_ldf = _Da_ldf_(kldf, self._Lx, v_cells, self._eps)  # [n, nc]
        
        # ---- Guardado (CELDAS)
        S = self._state_cell_properties
        S['MW'], S['rho'], S['mu'], S['k'],S['cp_molar'], S['cp_mass'], S['alpha'] = MWmix, rho, mu, k, cp_molar, cp_mass, alpha
        S['Dij'], S['Dim'], S['Dkn'], S['Dpor'], S['Deff'] = Dij, Dim, Dkn, Dpor, Deff
        S['Pe_s'], S['Pe_e'] = Pe_s, Pe_e
        S['Dz'], S['lamz'] = Dz, lamz
        S['hw'], S['hc'], S['kc'], S['kldf'] = hw, hc, kc, kldf
        S['Reg'], S['Rec'], S['Pr'], S['Sc'] = Reg, Rec, Pr, Sc
        S['Nuw'], S['Nuc'], S['Sh'] = Nuw, Nuc, Sh
        S['Big'], S['Bic'], S['U'] = Big, Bic, U
        S['Da_ext'],S['Da_ldf'] = Da_ext,Da_ldf
        S['U']=U
        
    
        # ---- Guardado (CARAS)
        SF = self._state_face_properties
        VF = self._state_face_vars
        VF['u'] = v_faces
        SF['rho'], SF['mu'], SF['k'], SF['cp_molar'], SF['cp_mass'] = rho_f, mu_f, k_f, cp_molar_f, cp_mass_f
        SF['alpha'], SF['Dim'] = alpha_f, Dim_f
        SF['Dz'], SF['lamz'] = Dz_f, lamz_f
        SF['dPdz'], SF['flowDir'] = dPdz_faces, flowDir
        SF['Pe_s'], SF['Pe_e'] = Pe_s_f, Pe_e_f
        # SF['inbed']=inbed
    
        # velocidad superficial nodal (magnitud)
        self._updateVars_(v=v_cells)


    def _initialize_(self):
        if not (self._required.get('mass_trans_info',False) and      
            self._required.get('initialC_info', False) and
            self._required.get('boundaryC_info', False) and
            self._required.get('thermal_info', False)):
            raise RuntimeError(
            f"⛔ No se puede inicializar la columna '{self._name}' sin definir:\n"
            f"\tmassTransfer_info  : {self._required['mass_trans_info']}\n"
            f"\tinitialC_info  : {self._required['initialC_info']}\n"
            f"\tboundaryC_info : {self._required['boundaryC_info']}\n"
            f"\tthermal_info   : {self._required['thermal_info']}"
            )
        self._required['PreProces'] = True
        self._results = None    
        self._actualTime = 0.0
        self._t = np.array([0.0])
        n=self._nodos  
        nc=self._ncomp
        t=self._actualTime
        
        self._state_cell_vars = {
            't' : t,
            'N' : np.ones(n),
            'v' : np.ones(n),
            'P' : np.ones(n),
            'Tg' : np.ones(n),
            'Ts' : np.ones(n),
            'x'  : np.ones(n*nc),
            'q'  : np.ones(n*nc)}
        
        self._previous_cell_vars = self._state_cell_vars.copy()
        
        # --- Propiedades locales nodales (a rellenar cada paso) ---
        self._state_cell_properties = {
            # Mezcla
            'MW'    : np.ones(n),              # [kg/mol] mezcla
            'rho'   : np.ones(n),              # [kg/m3]
            'mu'    : np.ones(n),              # [Pa·s]
            'cp_mass'    : np.ones(n),              # [J/kg/K]
            'cp_molar'    : np.ones(n),              # [J/mol/K] 
            'k'     : np.ones(n),              # [W/m/K]
            'alpha' : np.ones(n),              # [m2/s]
        
            # Difusión/disp./efectivas (por especie, 2D)
            'Dij'  : np.ones((n, nc, nc)),  # [n,nc,nc]
            'Dim'  : np.ones((n, nc)),      # [n,nc]
            'Dkn'  : np.ones((n, nc)),      # [n,nc]
            'Dpor' : np.ones((n, nc)),      # [n,nc]
            'Deff' : np.ones((n, nc)),      # [n,nc]
        
            # Péclet (celda) y transporte axial
            'Pe_s' : np.ones((n, nc)),      # [n,nc]
            'Pe_e' : np.ones(n),            # [n]
            'Dz'   : np.ones((n, nc)),      # [n,nc]
            'lamz' : np.ones(n),            # [n]
        
            # Adimensionales
            'Reg'  : np.ones(n),            # [n]
            'Rec'  : np.ones(n),            # [n]
            'Pr'   : np.ones(n),            # [n]
            'Sc'   : np.ones((n, nc)),      # [n,nc]
            'Sh'   : np.ones((n, nc)),      # [n,nc]
            'Nuw'  : np.ones(n),            # [n]  (si lo usas para pared por L)
            'Nuc'  : np.ones(n),            # [n]  (Nu por dp)
            'Da_ext': np.ones((n, nc)),     # [n,nc] <-- corregido (por especie)
            'Da_ldf': np.ones((n, nc)),     # [n,nc] <-- corregido (por especie)
            'Big'  : np.ones(n),            # [n]
            'Bic'  : np.ones((n, nc)),      # [n,nc]
        
            # Pared
            'hwr'   : np.ones(n),            # [n]
            'hw'   : np.ones(n),            # [n]
            'U'    : np.ones(n),            # [n]
            
        }
            
        self._state_face_vars = {
            't': self._actualTime,
            'u': np.ones(n-1)}
        
        self._previous_face_vars = self._state_face_vars.copy()
        
        self._state_face_properties = {
            'rho'   : np.ones(n-1),           # [n-1]
            'mu'    : np.ones(n-1),           # [n-1]
            'k'     : np.ones(n-1),           # [n-1]
            'cp_mass' : np.ones(n-1),           # [n-1]  
            'cp_molar' : np.ones(n-1),           # [n-1]  
            'alpha' : np.ones(n-1),           # [n-1]
            'Dim'   : np.ones((n-1, nc)),     # [n-1,nc]
            'Dz'    : np.ones((n-1, nc)),     # [n-1,nc]
            'lamz'  : np.ones(n-1),           # [n-1] 
            'dPdz'  : np.ones(n-1),           # [n-1]
            'flowDir': np.ones(n-1, dtype=int), # [n-1]  (-1/ +1)
            'Pe_s'  : np.ones((n-1, nc)),     # [n-1,nc]  (tu Pe “num” o “phys”, el que elijas)
            'Pe_e'  : np.ones(n-1),           # [n-1]
            'bed'  : np.ones(n-1),           # [n-1]
        }


        # --- Derivadas temporales y espaciales ---
        self._derivates = {
                    #Temporal derivates
                    'dNdt'         : np.zeros(n) ,    
                    'dPdt'         : np.zeros(n) ,    
                    'dqdt'         : np.zeros(n * nc) ,   
                    'dxdt'         : np.zeros(n * nc) ,   
                    'dTgdt'        : np.zeros(n) ,      
                    'dTsdt'        : np.zeros(n) ,      
                    #Spatial derivatives
                    'dPdz'         : np.zeros(n) ,   
                    'dPdzh'        : np.zeros(n-1) ,   
                    'dxdz'         : np.zeros(n * nc) ,   
                    'd2xdz2'       : np.zeros(n * nc) ,   
                    'dTgdz'        : np.zeros(n) ,   
                    'd2Tgdz2'      : np.zeros(n) ,
                    'dTsdz'        : np.zeros(n) ,   
                    'd2Tsdz2'      : np.zeros(n)
                    }

        
        self._reset_logs_()
        
        
        self._Tg = np.full((1, n), self._Tg0)
        Tg= self._Tg[-1]
        self._Ts = np.full((1, n), self._Ts0)
        Ts= self._Ts[-1]
        self._P = np.full((1, n), self._P0)
        P= self._P[-1]
        self._N = self._P * self._Volfx / self._R / self._Tg
        N= self._N[-1]
        self._Qloss = np.full((1, n), 0.0)
        self._x = np.tile(self._x0, (1, n, 1))
        x= self._x[-1]
        q= self._isotherm(x=x,
                              P=P,
                              T=(Tg+Ts)/2.,
                              method="IAST",
                              Lamb=None,
                              C=0.,
                              mask=self._mask_ads,
                              drop_thresh=1E-3)  #[nodos][species]
        
        self._q = q[None, :, :]#[times][nodos][species]
        
        self._updateProperties_(P,Tg,x)
        self._updateVars_(t=self._actualTime,N=N,P=P,Tg=Tg,Ts=Ts,x=x,q=q)        
        
        return None

    
    def _croopTime_(self, target_time):
        idx_valid = np.where(self._t <= target_time)[0]
        if len(idx_valid) == 0:
            raise ValueError(f"⛔ No hay datos en la columna '{self._name}' para el tiempo solicitado {target_time:.2f} s")
        last_idx = idx_valid[-1]
        
        # Recorta arrays principales
        self._t = self._t[:last_idx+1]
        self._N = self._N[:last_idx+1, ...]
        self._Tg = self._Tg[:last_idx+1, ...]
        self._Ts = self._Ts[:last_idx+1, ...]
        self._P = self._P[:last_idx+1, ...]
        self._x = self._x[:last_idx+1, ...]
        self._q = self._q[:last_idx+1, ...]

        if self._Qloss is not None:
            self._Qloss = self._Qloss[:last_idx+1, ...]
            
        # Actualiza el estado actual
        self._actualTime = self._t[-1]
        self._state_cell_vars['t'] = self._actualTime
        self._state_cell_vars['N'] = self._N[-1, :].copy()
        self._state_cell_vars['Tg'] = self._Tg[-1, :].copy()
        self._state_cell_vars['Ts'] = self._Ts[-1, :].copy()
        self._state_cell_vars['P'] = self._P[-1, :].copy()
        self._state_cell_vars['x'] = self._x[-1, :, :].flatten().copy()
        self._state_cell_vars['q'] = self._q[-1, :, :].flatten().copy()
        
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
        for key in ['t', 'P', 'Tg','Ts', 'N', 'x', 'q']:
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
        if 'state_Tg' in data_dict:
            self._state_cell_vars['Tg'] = np.array(data_dict['state_Tg'])
        if 'state_Ts' in data_dict:
            self._state_cell_vars['Ts'] = np.array(data_dict['state_Ts'])
        if 'state_x' in data_dict:
            self._state_cell_vars['x'] = np.array(data_dict['state_x'])
        if 'state_q' in data_dict:
            self._state_cell_vars['q'] = np.array(data_dict['state_q'])
    
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
        t_N, N2 = self._clean_LOG_(self._t_log, self._N_log)
        t_P, P2 = self._clean_LOG_(self._t_log, self._P_log)
        t_Ts, Ts2 = self._clean_LOG_(self._t_log, self._Ts_log)
        t_Tg, Tg2 = self._clean_LOG_(self._t_log, self._Tg_log)
        t_x, x2 = self._clean_LOG_(self._t_log, self._x_log)
        t_q, q2 = self._clean_LOG_(self._t_log, self._q_log)
        t_Q, Qloss2 = self._clean_LOG_(self._t_log, self._Qloss_log)
    
        # Comprueba que todos los tiempos coinciden (muy importante para los balances)
        assert (t_N == t_x == t_q == t_P == t_Tg == t_Ts == t_Q), "Tiempos limpios no coinciden, revisa logs"
    
        # Almacena los resultados finales sincronizados
        self._storeBal_(t_N, P2, Tg2, Ts2, x2, q2, N2, Qloss2)
        return None

    
    def _storeBal_(self, t2, P2, Tg2, Ts2, x2, q2, N2, Qloss2):
        ntimes=len(t2)
        nodos=self._nodos
        ncomp=self._ncomp
        self._t2 = np.array(t2)
        self._N2 = np.array(N2).reshape(ntimes,nodos)
        self._x2 = np.array(x2).reshape(ntimes,nodos,ncomp)
        self._q2 = np.array(q2).reshape(ntimes,nodos,ncomp)
        self._Tg2 = np.array(Tg2).reshape(ntimes,nodos)
        self._Ts2 = np.array(Ts2).reshape(ntimes,nodos)
        self._P2 = np.array(P2).reshape(ntimes,nodos)
        self._Qloss2 = np.array(Qloss2).reshape(ntimes,nodos)
        return None
    

# =============================================================================
#     # 5.Helpers/Getters/Setters
# =============================================================================
    # def updateProperties(self,P,T,x):
    #     self._state_properties['rho'] =
    #     self._state_properties['mu']  = 
    #     self._state_properties['k']   = 
    #     self._state_properties['cp']  =
    #     #Reynolds
    #     self._state_properties['Reg'] =  
    #     self._state_properties['Rec'] =
    #     #Prandtl
    #     self._state_properties['Prg'] = 
    #     self._state_properties['Prc'] =        #Resto de numeros
        
        
    #     return None
        


    def _get_mapping_(self):
        n_vars_per_node = 1 + (self._ncomp - 1) + self._ncomp + 1 + 1  # N, x1..x(ncomp-1),q..q(ncomp) Tg + Ts por nodo
        n_vars = self._nodos * n_vars_per_node
        labels = []
        for n in range(self._nodos):
            labels.append(f"N_{n}")
            for i in range(0, self._ncomp-1):
                labels.append(f"x{i}_{n}")
            for i in range(0, self._ncomp):
                labels.append(f"q{i}_{n}")
            labels.append(f"Tg_{n}")
            labels.append(f"Ts_{n}")
        # Guarda como atributos para referencia rápida
        self._nVars = n_vars
        self._labelVars = labels
        return n_vars, labels
    
    
    def _set_State_ylocal_(self, ti, ylocal):

        nodos = self._nodos
        ncomp = self._ncomp
        N = np.zeros(nodos)
        Tg = np.zeros(nodos)
        Ts = np.zeros(nodos)
        x = np.zeros((nodos, ncomp))
        q = np.zeros((nodos, ncomp))

        for idx, label in enumerate(self._labelVars):
            if label.startswith("N_"):
                nodo = int(label.split("_")[1])
                N[nodo] = ylocal[idx]
            elif label.startswith("Tg_"):
                nodo = int(label.split("_")[1])
                Tg[nodo] = ylocal[idx]
            elif label.startswith("Ts_"):
                nodo = int(label.split("_")[1])
                Ts[nodo] = ylocal[idx]
            elif label.startswith("x"):
                especie, nodo = label[1:].split("_")
                especie = int(especie)
                nodo = int(nodo)
                x[nodo, especie] = ylocal[idx]
            elif label.startswith("q"):
                especie, nodo = label[1:].split("_")
                especie = int(especie)
                nodo = int(nodo)
                q[nodo, especie] = ylocal[idx]
    
        # Completa la última fracción molar de cada nodo (cierra sumas a 1)
        for nodo in range(nodos):
            x[nodo, -1] = 1.0 - np.sum(x[nodo, :-1])
        x = x.flatten()
        q = q.flatten()
        P = N * self._R * Tg / self._Volfx   
        
        self._state_cell_vars = {
            't': ti,
            'N': N,
            'P': P,
            'x': x,
            'q': q,
            'Tg': Tg,
            'Ts': Ts
        }
        return None
    
    
    def _get_State_ylocal_(self):
        ylocal = []
        x=self._state_cell_vars["x"].reshape(self._nodos,self._ncomp)
        q=self._state_cell_vars["q"].reshape(self._nodos,self._ncomp)
        for idx, label in enumerate(self._labelVars):
            if label.startswith("N_"):
                nodo = int(label.split("_")[1])
                ylocal.append(self._state_cell_vars["N"][nodo])
            elif label.startswith("Tg_"):
                nodo = int(label.split("_")[1])
                ylocal.append(self._state_cell_vars["Tg"][nodo])
            elif label.startswith("Ts_"):
                nodo = int(label.split("_")[1])
                ylocal.append(self._state_cell_vars["Ts"][nodo])
            elif label.startswith("x"):
                especie, nodo = label[1:].split("_")
                especie = int(especie)
                nodo = int(nodo)
                ylocal.append(x[nodo][especie])
            elif label.startswith("q"):
                especie, nodo = label[1:].split("_")
                especie = int(especie)
                nodo = int(nodo)
                ylocal.append(q[nodo][especie])
                
        return np.array(ylocal, dtype=float)
    
    
    def _get_unitVar_(self, unit, var, time=None, nodo=None, port=None, especie=None    ):
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
        if var.lower().startswith("x") or var.lower().startswith("q") :
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
        val = self._state_cell_vars[var]
        nodos = self._nodos
        ncomp = self._ncomp
    
        if var == 'x' or var == 'q':
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
        
        if not self._required.get('PreProces', False):
            raise RuntimeError(
                    f"⛔ No se puede calcular _rhs en la columna '{self._name}' sin haber realizado el preprocesado (_initialize).\n"
        "Asegúrate de que se han definido las condiciones iniciales, de frontera y térmicas, y que se ha ejecutado correctamente el método _initialize()."
    )
        return solveAdsColumn(self)
    
    
# =============================================================================
#     # 7.Save Simulation/Case/Data
# =============================================================================
    def _storeData_(self, t, ylocal):
        # Reconstruye los arrays multinodo a partir de ylocal y self._labelVars
        nodos = self._nodos
        ncomp = self._ncomp
        ntimes = t.size
    
        N = np.zeros((ntimes, nodos))
        Tg = np.zeros((ntimes, nodos))
        Ts = np.zeros((ntimes, nodos))
        x = np.zeros((ntimes, nodos, ncomp))
        q = np.zeros((ntimes, nodos, ncomp))

    
        # Recorrido mapping para rellenar los arrays
        for var_idx, label in enumerate(self._labelVars):
            if label.startswith("N_"):
                nodo = int(label.split("_")[1])
                N[:, nodo] = ylocal[var_idx, :]
            elif label.startswith("Tg_"):
                nodo = int(label.split("_")[1])
                Tg[:, nodo] = ylocal[var_idx, :]
            elif label.startswith("Ts_"):
                nodo = int(label.split("_")[1])
                Ts[:, nodo] = ylocal[var_idx, :]
            elif label.startswith("x"):
                especie, nodo = label[1:].split("_")
                especie = int(especie)
                nodo = int(nodo)
                x[:, nodo, especie] = ylocal[var_idx, :]
            elif label.startswith("q"):
                especie, nodo = label[1:].split("_")
                especie = int(especie)
                nodo = int(nodo)
                q[:, nodo, especie] = ylocal[var_idx, :]
            
        # Cierra la fracción molar de cada nodo en cada tiempo
        x[:, :, -1] = 1.0 - np.sum(x[:, :, :-1], axis=2)
        P = (N * self._R * Tg) / self._Volfx
        
        # Guarda en atributos
        if self._actualTime == 0:
            self._t = t
            self._N = N
            self._Tg = Tg
            self._Ts = Ts
            self._x = x
            self._q = q
            self._P = P
        else:
            self._t = np.concatenate([self._t, t])
            self._N = np.concatenate([self._N, N])
            self._x = np.concatenate([self._x, x],axis=0)  
            self._q = np.concatenate([self._q, q],axis=0)  
            self._Tg = np.concatenate([self._Tg, Tg])
            self._Ts = np.concatenate([self._Ts, Ts])
            self._P = np.concatenate([self._P, P])
                        
        self._required['Results'] = True
        self._actualTime = t[-1]
        # Actualiza el estado actual (último instante)
        self._state_cell_vars = {
            't': t[-1],
            'N': N[-1, :],
            'Tg': Tg[-1, :],
            'Ts': Ts[-1, :],
            'x': x[-1, :, :].flatten(),
            'q': q[-1, :, :].flatten(),
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
        data_dict['Tg'] = self._Tg.flatten() if self._Tg is not None else None
        data_dict['Ts'] = self._Ts.flatten() if self._Ts is not None else None
        data_dict['N'] = self._N.flatten() if self._N is not None else None
        data_dict['x'] = self._x.flatten() if self._x is not None else None  # [ntimes, nodes, ncomp] -> 1D
        data_dict['q'] = self._q.flatten() if self._q is not None else None  # [ntimes, nodes, ncomp] -> 1D
    
        # Último estado (vectores planos)
        data_dict['state_N'] = self._state_cell_vars['N'].copy() if self._state_cell_vars is not None else None
        data_dict['state_P'] = self._state_cell_vars['P'].copy() if self._state_cell_vars is not None else None
        data_dict['state_Tg'] = self._state_cell_vars['Tg'].copy() if self._state_cell_vars is not None else None
        data_dict['state_Ts'] = self._state_cell_vars['Ts'].copy() if self._state_cell_vars is not None else None
        data_dict['state_x'] = self._state_cell_vars['x'].copy() if self._state_cell_vars is not None else None
        data_dict['state_q'] = self._state_cell_vars['q'].copy() if self._state_cell_vars is not None else None
    
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



