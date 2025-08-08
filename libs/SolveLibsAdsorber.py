import os,time
import tempfile
import glob
import numpy as np
import scipy as sp
import pandas as pd
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from IsoLib import *

class PSA:
    
    def __init__(self,
                 Lenght,                                                       # Length of the column [m]
                 Radio_interior,                                               # Interior radio of the column [m]
                 Radio_exterior,                                               # Exterior radio of the column [m]
                 nComponentes,                                                 # Number of components [m]
                 nParColumns,                                                  # Par columns number [-]
                 Nodes):                                                       # Number of nodes [-]
    
        #Design info        
        self._L = Lenght
        self._R = 8.314   # Universal gas constant [J/mol/K : Pa*m^3/mol/K]
        self._ri = Radio_interior
        self._Ai = np.pi*pow(self._ri,2)
        self._ro = Radio_exterior
        self._Ao = np.pi*pow(self._ro,2)
        self._nPCol = nParColumns
        self._ncomp = nComponentes
        self._N = Nodes
        self._z = np.linspace(0,self._L,self._N)
        
        self.DesignProp = {"L":self._L,
                      "ri":self._ri,
                      "Ai":self._Ai,
                      "ro":self._ro,
                      "Ao":self._Ao}   
        
        #Espacial discretization
        self._dz= 1./self._N 
        
        #Gas properties info
        self._species = []
        self._WM = []                                                          # Molecular Weight
        self._mu = []                                                          # Viscosity
        self._Dm = []                                                          # Molecular diffusivity [m^2/s]
        self._cpg = []                                                         # Specific heat of gas [J/mol/k]
        self._K = []                                                           # Thermal conduction in gas phase [W/m/k]
        
        #Adsorbation info
        self._isoFuncs = None
        
        self._rho_cry = 0.                                                     # Density of the Crystal  [kg/m^3]
        self._epsilon = 0.                                                     # Void fraction [-]
        self._ep = 0.                                                          # Porosity of pellets [-]
        self._rp = 0.                                                          # Radio of pellets [m]
        self._tau_s = 0.                                                       # Tortuosity factor [-]
        self._cpa = 0.                                                         # Specific heat capacity of the adsorbent [J/kg/K]
        self._rho_ads = 0.                                                     # Density of the Adsorbent [kg/m^3]
        self._q_s = 0.                                                         # Molar loading scaling factor [mol/kg]
        self._q_s0 = 0.                                                        # Molar loading scaling factor [mol/m3]
        
        #Mass trasnfer info
        self._a_surf = 0.
        self._k_mtc = []
        
        #Boundary conditions 
        self._M1  = 0.                                                         # Mass flow inlet [kg/s]
        self._N1  = 0.                                                         # Molar flow inlet [mol/s]
        self._y0 = []                                                          # Mass fraction initial [kg_i/KgT]  
        self._x0 = []                                                          # Molar fraction initial [mol_i/molT] 
        self._y1 = []                                                          # Mass fraction inlet [kg_i/KgT]  
        self._x1 = []                                                          # Molar fraction inlet [mol_i/molT] 
        self._v1 = 0.                                                          # Velocity inlet
        self._v0 = 0.                                                          # Velocity initial   
        
        self._tAds  = 0.                                                       # Time of adsorption step [s]
        self._tPre  = 0.                                                       # Maximum/time of pressurization step [s]
        self._tDPreCo  = 0.                                                    # Maximum/time of depressurization CoCorrent step [s]
        self._tDPreCn  = 0.                                                    # Maximum/time of depressurization CnCorrent step [s]
        self._tLRF  = 0.                                                       # Time of light reflux step [s]
        self._tHRF  = 0.                                                       # Time of heavy reflux step [s]
        self._rLR  = 0.                                                        # Light product reflux ratio [-]
        self._rHR  = 0.                                                        # Heavy product reflux ratio [-]

        self._PH  = 0.                                                         # Pressure of feed gas at the inlet of the adsorption step
        self._PM  = 0.                                                         # Intermediate pressure [Pa] (Not used in modified skarstrom cycle)
        self._PL  = 0.                                                         # Purge Pressure [Pa]
        self._Ptau  = 0.                                                       # Parameter used for determining speed of pressure change
        
        self._actualStep = None
        self._actualTime = None
        self._actualCycle = None
        self._saveData = 25
        self._saveCycle = 1
        self._log = False
        self._ncycle = 0
        self._solvers = ['RK45', 'RK23', 'BDF', 'LSODA', 'Radau', 'DOP853']
        
        self._reljP = 1.
        self._reljT = 1.
        self._reljx = 1.
        self._reljq = 1.
        
        n = self._N + 2
        ns = self._ncomp
        self._Ptol = 1
        self._Ttol = 0.01
        self._xtol = 1.E-5
        self._qtol = 1.E-5
        
        self._atol = np.zeros(2*n + 2*ns*n)
        self._atol[0:n] = self._Ptol  
        self._atol[n:2*n] = self._Ttol
        self._atol[2*n:2*n + ns*n] = self._xtol
        self._atol[2*n + ns*n:] = self._qtol
        
        self._state_properties = { 'rho' : np.ones(self._N+2),
                           'mu'  : np.ones(self._N+2),
                           'cp'  : np.ones(self._N+2),
                           'k'   : np.ones(self._N+2),
                           'Reg'  : np.ones(self._N+2),
                           'Rec' : np.ones(self._N+2),
                           'Pe'  : np.ones(self._N+2),
                           'Prg'  : np.ones(self._N+2),
                           'Prc'  : np.ones(self._N+2),
                           'Sc'  : np.ones(self._N+2),
                           'Sh'  : np.ones(self._N+2),
                           'Big'  : np.ones(self._N+2),
                           'Bic'  : np.ones(self._N+2),
                           'Da'  : np.ones(self._N+2),
                           'Nug'  : np.ones(self._N+2),
                           'Nuc'  : np.ones(self._N+2)}

        
        self._state_vars = {'P' : np.ones(self._N+2),
                    'v' : np.ones(self._N+2),
                    'Tg' : np.ones(self._N+2),
                    'Ts' : np.ones(self._N+2),
                    'Tw' : np.ones(self._N+2),
                    'x'  : np.ones(self._ncomp * (self._N+2)),
                    'q'  : np.ones(self._ncomp * (self._N+2)),
                    }
        
        self._previous_vars = {'P' : np.ones(self._N+2),
                       'v' : np.ones(self._N+2),
                       'Tg' : np.ones(self._N+2),
                       'Ts' : np.ones(self._N+2),
                       'Tw' : np.ones(self._N+2),
                       'x'  : np.ones(self._ncomp * (self._N+2)),
                       'q'  : np.ones(self._ncomp * (self._N+2)),
                    }
        
        self._derivates = {
                    #Temporal derivates
                    'dPdt'        : np.zeros(self._N+2) ,    
                    'dqdt'        : np.zeros(self._ncomp * (self._N+2)) ,   
                    'dxdt'       : np.zeros(self._ncomp * (self._N+2)) ,   
                    'dTgdt'        : np.zeros(self._N+2) ,      
                    #Spatial derivatives
                    'dPdz'        : np.zeros(self._N+2) ,   
                    'dPdzh'       : np.zeros(self._N+1) ,   
                    'dxdz'        : np.zeros(self._ncomp * (self._N+2)) ,   
                    'd2xdz2'      : np.zeros(self._ncomp * (self._N+2)) ,   
                    'dTgdz'        : np.zeros(self._N+2) ,   
                    'd2Tgdz2'      : np.zeros(self._N+2)
                    }
        
        self._bc_LRF_N=None
        self._bc_LRF_P=None
        self._bc_LRF_Tg=None
        self._bc_LRF_x=None
        
        self._bc_HRF_N=None
        self._bc_HRF_P=None
        self._bc_HRF_Tg=None
        self._bc_HRF_x=None
        
        self._yPre = None
        self._yAds = None
        self._yHRF = None
        self._yLRF = None
        self._yDPCn = None
        self._yDPCo = None
        
        self._required = {'Design':True,
                          'adsorbent_info':False,
                          'gas_prop_info': False,
                          'mass_trans_info': False,
                          'thermal_info' : False,
                          'boundaryC_info' :False,
                          'initialC_info' : False,
                          'economic_info' : False,
                          'simulation_info': False
                          }
        
        self.results = {}
        
        #clean monitor        
        root = tk.Tk()
        root.withdraw()  # Oculta la ventana principal
        self._imgwidth = root.winfo_screenwidth()
        self._imgheight = root.winfo_screenheight()
        root.destroy()
        
        base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self._color_species = (base_colors * ((self._ncomp // len(base_colors)) + 1))[:self._ncomp]
        self._color_data = (base_colors * ((self._saveData // len(base_colors)) + 1))[:self._saveData]
        self._color_cycle = (base_colors * ((self._saveCycle // len(base_colors)) + 1))[:self._saveCycle]
      
        
        self._dbg_1dfunc = True
        self._dbg_Plots = True
        self._namecase="debbuging"

    def updateProperties(self,):

        self._state_properties['rho'] =((1./self._R)*self._state_vars['P'] /self._state_vars['Tg'] 
                                          *np.sum(self._state_vars['x'].reshape(self._ncomp,self._N+2).T*self._WM,axis=1))
        self._state_properties['mu']  = self.Wilke_c(self._state_vars['x'].reshape(self._ncomp,(self._N+2)),self._mu)
        self._state_properties['k']  = self.Wilke_c(self._state_vars['x'].reshape(self._ncomp,(self._N+2)),self._K)
        self._state_properties['cp']  = np.sum(self._state_vars['x'].reshape(self._ncomp,(self._N+2)).T*self._cpg,axis=1)

        #Reynolds
        self._state_properties['Reg']  = self._state_properties['rho']  * self._state_vars['v']  * self._ri * 2. / self._state_properties['mu'] 
        self._state_properties['Rec']  = self._rho_ads * self._state_vars['v']  * self._rp * 2. / self._state_properties['mu'] 
        
        #Prandtl
        self._state_properties['Prg']  = self._state_properties['mu']  * self._state_properties['cp']  / self._state_properties['k'] 
        self._state_properties['Prc']  = self._state_properties['mu']  * self._cpa / self._ka
        
        return 
    
    def updateVars(self,P,Tg,v,x,q):
        self._previous_vars['P'][:]=self._state_vars['P'][:]
        # self._state_vars['P'][:]=P*self._P1
        self._state_vars['P'][:]=P

        self._previous_vars['Tg'][:]=self._state_vars['Tg'][:]
        # self._state_vars['Tg'][:]=Tg*self._TG1
        self._state_vars['Tg'][:]=Tg
        
        self._previous_vars['v'][:]=self._state_vars['v'][:]
        # self._state_vars['v'][:]=v*self._v1
        self._state_vars['v'][:]=v
        
        self._previous_vars['x'][:]=self._state_vars['x'][:]
        self._state_vars['x']=x
        
        self._previous_vars['q'][:]=self._state_vars['q'][:]
        # self._state_vars['q']=q*self._q_s0/self._rho_ads
        self._state_vars['q']=q
        
        self.updateProperties()
        
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
                
    def gas_prop_info(self,
                      Species,
                      Weight_Molar,
                      Viscosity,
                      Molecular_Diffusivity,
                      Specific_Heat,
                      Thermal_Conductivity):
        stack_true = 0
        if len(Species) == self._ncomp:
            stack_true = stack_true + 1
        else:
            raise ValueError('The input variable [Species] should be a list/narray with shape ({0:d}, ).'.format(self._ncomp))
        
        if len(Weight_Molar) == self._ncomp:
            stack_true = stack_true + 1
        else:
            raise ValueError('The input variable [Weight_Molar] should be a list/narray with shape ({0:d}, ).'.format(self._ncomp))
        if len(Viscosity) == self._ncomp:
            stack_true = stack_true + 1
        else:
            raise ValueError('The input variable [Viscosity] should be a list/narray with shape ({0:d}, ).'.format(self._ncomp))
        if len(Molecular_Diffusivity) == self._ncomp:
            stack_true = stack_true + 1
        else:
            raise ValueError('The input variable [Molecular_Diffusivity] should be a list/narray with shape ({0:d}, ).'.format(self._ncomp))
        if len(Specific_Heat) == self._ncomp:
            stack_true = stack_true + 1
        else:
            raise ValueError('The input variable [Specific_heat] should be a list/narray with shape ({0:d}, ).'.format(self._ncomp))
        if len(Thermal_Conductivity) == self._ncomp:
            stack_true = stack_true + 1
        else:
            raise ValueError('The input variable [Thermal_Conductivity] should be a list/narray with shape ({0:d}, ).'.format(self._ncomp))       
        
        if stack_true == 6:
            self._species = Species                                            # Name Species
            self._WM = Weight_Molar                                            # Molecular Weight
            self._mu = Viscosity                                               # Viscosity
            self._phiIJ = np.zeros((self._ncomp,self._ncomp))
            
            for i in range(0,self._ncomp):
                for j in range(0,self._ncomp):
                    self._phiIJ[i,j]=(((1 + (self._mu[i] / self._mu[j])**0.5 * 
                               (self._WM[j] / self._WM[i])**0.25)**2) / 
                    (np.sqrt(8 * (1 + self._WM[i] / self._WM[j]))))
                    
            self._Dm = Molecular_Diffusivity                                   # Molecular diffusivity [m^2/s]
            self._cpg = Specific_Heat                                          # Specific heat of gas [J/mol/k]
            self._K = Thermal_Conductivity                                     # Thermal conduction in gas phase [W/m/k]
            self._required['gas_prop_info'] = True
            
            self.GasProp = {"species" : self._species,
                            "WM"      : self._WM,
                            "mu"      : self._mu,
                            "Dm"      : self._Dm,
                            "cp"      : self._cpg,
                            "k"       : self._K}
            
        
        return 
            
    def Wilke_f(self,xh,phi):
        mu_mixture = np.zeros(self._N+1)

        for node in range(self._N+1):
            sum_x_mu = 0
            sum_x_phi = np.zeros(self._ncomp)
    
            for i in range(self._ncomp):
                for j in range(self._ncomp):
                    sum_x_phi[i] += xh[j, node] * self._phiIJ[i, j]
    
                sum_x_mu += (xh[i, node] * phi[i]) / sum_x_phi[i]
    
            mu_mixture[node] = sum_x_mu
    
        return mu_mixture
    
    def Wilke_c(self,xh,phi):
        mu_mixture = np.zeros(self._N+2)

        for node in range(self._N+2):
            sum_x_mu = 0
            sum_x_phi = np.zeros(self._ncomp)
    
            for i in range(self._ncomp):
                for j in range(self._ncomp):
                    sum_x_phi[i] += xh[j, node] * self._phiIJ[i, j]
    
                sum_x_mu += (xh[i, node] * phi[i]) / sum_x_phi[i]
    
            mu_mixture[node] = sum_x_mu
    
        return mu_mixture
         
    def adsorbent_info(self,
                       Name,
                       isoFuncs,
                       Rho_Cristal,
                       Void_fraction,
                       Porosity_pellet,
                       Radio_pellet,
                       Toruosity_factor,
                       Sphericity,
                       Specific_Heat,
                       Condutivity,
                       ):
           
        self._isoFuncs = isoFuncs
        self._catName = Name                                   
        self._rho_cry = Rho_Cristal                                    # Density of the Crystal  [kg/m^3]
        self._epsilon = Void_fraction                                  # Void fraction [-]
        self._ep = Porosity_pellet                                     # Porosity of pellets [-]
        self._rp = Radio_pellet                                        # Radio of pellets [m]
        self._tau_s = Toruosity_factor                                 # Tortuosity factor [-]
        self._sphere = Sphericity                                      # Esfericidad del caralizador [-]
        self._cpa = Specific_Heat                                      # Specific heat capacity of the adsorbent [J/kg/K]
        self._ka = Condutivity
        self._rho_ads = self._rho_cry*(1-self._ep)                     # Density of the Adsorbent [kg/m^3]
        
        self._q_s = max(isoFuncs['iso_qs0'])
        self._q_s0 = self._q_s * self._rho_ads 
        
        
        self._required['adsorbent_info'] = True
        
        self.AdsProp ={"Name" : self._catName,
                       "rho"  : self._rho_ads,
                       "eps"  : self._epsilon,
                       "por"  : self._ep,
                       "rp"   : self._rp,
                       "tau"  : self._tau_s,
                       "sph"  : self._sphere,
                       "cp"   : self._cpa,
                       "k"    : self._ka}

        return        
    
    def isotherm(self,x,P,T):
        
        isoFun = self._isoFuncs
        q = np.array([isoFun['iso_fun'][i](x[i]*P/1000.0, T) for i in range(len(x))])
                
        return q
           
    def mass_trans_info(self,
                        k_mass_transfer,
                        a_specific_surf):
        
        self._kmtc = k_mass_transfer
        self._a_surf = a_specific_surf
        self._required['mass_trans_info'] = True 
            
        return
    
    def thermal_info(self,
                 Exterior_Heat_Coeficient,
                 Interior_Heat_Coeficient,
                 Ambient_Temperature):
        self._hext=Exterior_Heat_Coeficient
        self._hint=Interior_Heat_Coeficient
        self._Tamb=Ambient_Temperature
        self._required['thermal_info'] = True

        return
        
    def boundaryC_info(self,
                       Volumetric_normal_rate,
                       Temperture_Inlet,
                       Molar_fraction_Inlet,
                       Time_Adsorption,
                       Time_Pressurization,
                       Time_CoDepressurization,
                       Time_CnDepressurization,
                       Time_Light_Reflux,
                       Time_Heavy_Reflux,
                       Ratio_Reflux_Light_Product,
                       Ratio_Reflux_heavy_Product,
                       Pressure_High,
                       Pressure_Medium,
                       Pressure_Low,
                       Pressure_Speed_Change): 
    
        self._Q1 = Volumetric_normal_rate
        self._TG1 = Temperture_Inlet
        self._PH = Pressure_High
        self._PM = Pressure_Medium
        self._PL =  Pressure_Low
        self._x1 = np.array(Molar_fraction_Inlet)
        self._N1= self._Q1/80.64/self._nPCol
        self._y1= self._x1*self._WM/sum(self._WM*self._x1)
        self._C1= self._PH/self._R/self._TG1
        self._v1s = self._N1/self._C1/self._Ai
        self._v1i = self._v1s/self._epsilon
        self._M1 = self._N1 * np.sum(self._x1 * self._WM)
        self._tAds = Time_Adsorption
        self._tPre = Time_Pressurization
        self._tCoDepre = Time_CoDepressurization
        self._tCnDepre = Time_CnDepressurization
        self._tLRF = Time_Light_Reflux
        self._tHRF = Time_Heavy_Reflux
        self._rLR = Ratio_Reflux_Light_Product
        self._rHR = Ratio_Reflux_heavy_Product

        self._Ptau = Pressure_Speed_Change
        self._flowDir = None
        self._required['boundaryC_info'] = True

    def initialC_info(self,
                      Initial_Velocity,
                      Initial_Pressure,
                      Inital_Molar_Fraction,
                      Initial_Temperature_Gas,
                      Initial_Temperature_Solid,
                      Initial_Temperatura_Wall):
        
        self._v0 = Initial_Velocity
        self._P0 = Initial_Pressure
        self._TG0 = Initial_Temperature_Gas
        self._TS0 = Initial_Temperature_Solid
        self._TW0 = Initial_Temperatura_Wall
        self._x0 = np.array(Inital_Molar_Fraction)
        self._y0= self._x0*self._WM/sum(self._WM*self._x0)
        self._required['initialC_info'] = True
        return
    
    def economic_info(self,
                      Desired_flow,
                      Electricity_cost,
                      Hours_to_year_conversion,
                      Life_span_equipament,
                      Life_span_adsorbent,
                      CEPCI):
        
        self._Desired_flow=Desired_flow
        self._Electricity_cost=Electricity_cost
        self._Hours_to_year_conversion=Hours_to_year_conversion
        self._Life_span_equipament=Life_span_equipament
        self._Life_span_adsorbent=Life_span_adsorbent
        self._CEPCI = CEPCI
        self._required['economic_info'] = True
        
        return
    
    def _WENO(self,
                 Flux_c,
                 FlowDir):
    
        delta = 1e-10  # Small value to avoid division by zero
        N = len(Flux_c)
        N -= 1  # Adjust for boundary conditions
        Flux_w = np.zeros(N)
        
        if FlowDir.lower() == "upwind":
            for j in range(1, N - 1):
                alpha_0 = (2/3) / ((Flux_c[j+1] - Flux_c[j] + delta) ** 4)
                alpha_1 = (1/3) / ((Flux_c[j] - Flux_c[j-1] + delta) ** 4)
                
                Flux_w[j] = (alpha_0 / (alpha_0 + alpha_1)) * (0.5 * (Flux_c[j] + Flux_c[j+1])) \
                            + (alpha_1 / (alpha_0 + alpha_1)) * (1.5 * Flux_c[j] - 0.5 * Flux_c[j-1])
        
        elif FlowDir.lower() == "downwind":
            for j in range(1, N - 1):
                alpha_0 = (2/3) / ((Flux_c[j] - Flux_c[j+1] + delta) ** 4)
                alpha_1 = (1/3) / ((Flux_c[j+1] - Flux_c[j+2] + delta) ** 4)
                
                Flux_w[j] = (alpha_0 / (alpha_0 + alpha_1)) * (0.5 * (Flux_c[j] + Flux_c[j+1])) \
                            + (alpha_1 / (alpha_0 + alpha_1)) * (1.5 * Flux_c[j+1] - 0.5 * Flux_c[j+2])
        
        elif FlowDir.lower() == "central":
            for j in range(1, N - 1):
                Flux_w[j] = 0.5 * (Flux_c[j] + Flux_c[j+1])
        
        else:
            raise ValueError("FlowDir must be 'upwind', 'downwind', or 'central'")
        
        return Flux_w
        
    def initialColumn(self,):
        self._actualTime=0.
        self._actualCycle = 0.
        self._actualStep=None
        self._actualFlow=None
        self.results={}
        self._state_vars['P'] = np.ones(self._N+2)*self._P0
        self._state_vars['Tg'] = np.ones_like(self._state_vars['P'])*self._TG0
        self._state_vars['Ts'] = np.ones_like(self._state_vars['P'])*self._TS0
        self._state_vars['Tw'] = np.ones_like(self._state_vars['P'])*self._TW0
        self._state_vars['v'] = np.ones_like(self._state_vars['P'])*self._v0
        
        ii=0
        for j in range(0,len(self._x0)):
            for i in range(0,self._N+2):
                self._state_vars['x'][ii]=self._x0[j]
                ii+=1  
                
        self._state_vars['q'] =  (self.isotherm(
            x=self._state_vars['x'].reshape(self._ncomp,self._N+2),
            P=self._state_vars['P'],
            T=self._state_vars['Tg'],
            )).reshape(self._ncomp*(self._N+2))
    
        self._state_vars['q'] *= 0
        self._previous_vars['P'] = self._state_vars['P']
        self._previous_vars['Tg'] = self._state_vars['Tg']
        self._previous_vars['Ts'] = self._state_vars['Ts']
        self._previous_vars['Tw'] = self._state_vars['Tw']
        self._previous_vars['q'] =  self._state_vars['q']
        self._previous_vars['v'] =  self._state_vars['v']

    def nextStep(self,):
        if self._flowDir == "coCurrent":
            y = np.concatenate([
            self._state_vars['P'],
            self._state_vars['Tg'],
            self._state_vars['x'].reshape(self._ncomp*(self._N+2)),
            self._state_vars['q'].reshape(self._ncomp*(self._N+2))])   
        
        elif self._flowDir == "cnCurrent":
            x = self._state_vars['x'].reshape(self._ncomp, self._N+2)[:, ::-1]
            q = self._state_vars['q'].reshape(self._ncomp, self._N+2)[:, ::-1]
            y = np.concatenate([
                self._state_vars['P'][::-1],
                self._state_vars['Tg'][::-1],
                x.reshape(self._ncomp * (self._N+2)),
                q.reshape(self._ncomp * (self._N+2))])
        else:
            print("nexStep error!",self._actualStep)
        return y
    
    def uploadStep(self,cycle,preStep):
        cycle=cycle
        step=preStep
        case_dir = os.path.join(tempfile.gettempdir(), self._namecase, f"cycle_{cycle}")
        fname    = f"{step}_{self._flowDir}.csv"  # o busca con os.listdir si no sabes el flowDir
        filepath = os.path.join(case_dir, fname)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"No encuentro {filepath}")
    
        # 2) Cargo todo y me quedo con la última fila
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)  # (nt, M)
        flow = fname[len(step)+1:-4]
        
        last = data[-1]
        Z    = self._N+2
        ns   = self._ncomp
        idx  = 0
        
        t_last   = last[idx];            
        idx += 1
        P_last   = last[idx: idx+Z];     
        idx += Z
        Tg_last  = last[idx: idx+Z];     
        idx += Z
        v_last   = last[idx: idx+Z];     
        idx += Z
        x_flat   = last[idx: idx+ns*Z];  
        idx += ns*Z
        q_flat   = last[idx: idx+ns*Z];  
        idx += ns*Z
        x_last   = x_flat.reshape(ns, Z)    
        q_last   = q_flat.reshape(ns, Z)
        
        if self._flowDir == "coCurrent":
            y0 = np.concatenate([
                P_last.copy(),
                Tg_last.copy(),
                x_last.copy(),
                q_last.copy()
            ])
        elif self._flowDir == "cnCurrent":
            # invierto orden de nodo a nodo
            x = x_last.copy().reshape(self._ncomp, self._N+2)[:, ::-1]
            q = q_last.copy().reshape(self._ncomp, self._N+2)[:, ::-1]
            y0 = np.concatenate([
                P_last[::-1].copy(),
                Tg_last[::-1].copy(),
                x.reshape(self._ncomp * (self._N+2)),
                q.reshape(self._ncomp * (self._N+2))
            ])
        else:
            raise ValueError(f"FlowDir inesperado: '{self._flowDir}'")

        return y0
  
    def solveStep(self,time,y0):
        step = self._actualStep
        P = y0[0:self._N+2]
        #P = np.clip(P, self._PL/self._PH, self._PH/self._PH) 
        P = P * self._reljP + self._previous_vars['P']*(1.-self._reljP)
         
        Tg = y0[self._N+2:2*(self._N+2)]
        #Tg = np.clip(Tg, self._TG0/self._TG1, self._TG1/self._TG1)        
        Tg = Tg * self._reljT + self._previous_vars['Tg']*(1.-self._reljT)
         
        x = y0[2*(self._N+2):2*(self._N+2) + self._ncomp*(self._N+2)]   
        # for i in range(0, self._ncomp):
        #     x[i] = np.clip(x[i],0,1.)    
            
        x = x * self._reljx + self._previous_vars['x']*(1.-self._reljx)
        x=x.reshape(self._ncomp, self._N+2)
        
        q = y0[2*(self._N+2) + self._ncomp*(self._N+2):]
        # for i in range(0, self._ncomp):
        #     qei = self.isotherm(x=x, P=P*self._P1, T=Tg*self._TG1,)[i]*self._rho_ads/self._q_s0
        #     q[i] = np.clip(q[i],0,qei[i])
            
        q = q * self._reljq + self._previous_vars['q']*(1.-self._reljx)
        q=q.reshape(self._ncomp, self._N+2)
        
        Ph = np.zeros(self._N+1)
        Tgh = np.zeros(self._N+1)
        xh = np.zeros((self._ncomp, self._N+1))
        v=np.zeros_like(P)
        
        dPdz = self._derivates['dPdz']
        dPdzh = self._derivates['dPdzh']
        dxdz = self._derivates['dxdz'].reshape(self._ncomp,self._N+2) 
        dTgdz = self._derivates['dTgdz']
        d2xdz2 = self._derivates['d2xdz2'].reshape(self._ncomp,self._N+2)
        dqdt=self._derivates['dqdt'].reshape(self._ncomp,self._N+2)
        dTgdt = self._derivates['dTgdt']
        dPdt = self._derivates['dPdt']
        dxdt = self._derivates['dxdt'].reshape(self._ncomp,self._N+2)
        d2xdz2 = self._derivates['d2xdz2'].reshape(self._ncomp,self._N+2)
        d2Tgdz2 = self._derivates['d2Tgdz2']
        dPdt = self._derivates['dPdt']
        dxdt = self._derivates['dxdt'].reshape(self._ncomp,self._N+2)
        dqdt=self._derivates['dqdt'].reshape(self._ncomp,self._N+2)
        dTgdt = self._derivates['dTgdt']
        
        if step == "Pressurization":
            
            if self._flowDir=="coCurrent":

                dt = time - self._actualTime
                alpha = 5
                tau = self._tCoDepre
                
                if dt < tau:
                    P[0]   = self._PM + (self._PH - self._PM) * (1.0 - np.exp(-alpha * dt / tau))
                    dPdt[0]= alpha * (self._PH - self._PM)/tau * np.exp(-alpha * dt / tau)
                else:
                    P[0]   = self._PH
                    dPdt[0]= 0.0
                    
                x[:,0]    = self._x1
                Tg[0]     = self._TG1
                v[0]      = v[1]
                
                x[:,-1]   = x[:,-2]    
                P[-1]     = P[-2]
                Tg[-1]    = Tg[-2]
                v[-1]     = 0.
                
                rho_molar = P / (self._R * Tg)
            
            elif self._flowDir=="cnCurrent":
                
                dt = time - self._actualTime
                alpha = 25 
                tau = self._tCnDepre
                Ninlet,Pinlet,Tinlet,xinlet = self.bc_HRF(time)
                
                if dt < tau:
                    P[0]   = self._PL + (self._PM - self._PL) * (1.0 - np.exp(-alpha * dt / tau))
                    dPdt[0]= alpha * (self._PM - self._PL)/tau * np.exp(-alpha * dt / tau)
                        
                else:
                    P[0]   = Pinlet
                    dPdt[0]= 0.0
                    
                # P[0]      = Pinlet
                
                x[:,0]    = xinlet
                Tg[0]     = Tinlet
                C1 = Pinlet/self._R/Tinlet
                v1i= Ninlet/C1/self._Ai/self._epsilon
                v1s= max(Ninlet/C1/self._Ai,1E-6)
                v[0] = v1s
                
                x[:,-1]   = x[:,-2]    
                P[-1]     = P[-2]
                Tg[-1]    = Tg[-2]
                v[-1]     = 0.
                
                rho_molar = P / (self._R * Tg)
            
        elif step == "DePressurization":
            
            if self._flowDir=="cnCurrent":
                dt = time - self._actualTime
                tau = self._tCnDepre
                alpha = 5 
                
                if dt < tau:
                    P[-1] = self._PM + (self._PL -self._PM) * (1.0 - np.exp(-alpha * dt / tau))
                    dPdt[-1] = (self._PL - self._PM) * (alpha / tau) * np.exp(-alpha * dt / tau)
                else:
                    P[-1] = self._PM
                    dPdt[-1] = 0.0 
                
                x[:,-1]    = x[:,-2]
                Tg[-1]      = Tg[-2]
                v[-1]     = v[-2]
                
                P[0] = P[1]
                Tg[0] = Tg[1]
                x[:,1] = x[:,0]
                v[0] = v[1]
                
                rho_molar = P / (self._R * Tg)
        
            elif self._flowDir=="coCurrent":
                dt = time - self._actualTime
                tau = self._tCoDepre
                alpha = 5 
                
                if dt < tau:
                    P[-1] = self._PH + (self._PM -self._PH) * (1.0 - np.exp(-alpha * dt / tau))
                    dPdt[-1] = (self._PM - self._PH) * (alpha / tau) * np.exp(-alpha * dt / tau)
                else:
                    P[-1] = self._PM
                    dPdt[-1] = 0.0 
                
                x[:,-1]    = x[:,-2]
                Tg[-1]      = Tg[-2]
                v[-1]     = v[-2]
                
                P[0] = P[1]
                Tg[0] = Tg[1]
                x[:,1] = x[:,0]
                v[0] = v[1]
                
                rho_molar = P / (self._R * Tg)
                
        elif step == "Purge":
            
            Ninlet,Pinlet,Tinlet,xinlet = self.bc_LRF(time)

            x[:,0] = xinlet
            Tg[0] = Tinlet
            P[0] = self._PL
            
            C1 = Pinlet/self._R/Tinlet
            v1i= Ninlet/C1/self._Ai/self._epsilon
            v1s= max(Ninlet/C1/self._Ai,1E-6)
            
            
            mu = self._state_properties['mu']
            MW = np.sum(self._x1 * self._WM)
            term_mu  = (150. * mu[0] * (1.- self._epsilon) ** 2) / (4. * self._rp**2 * self._epsilon**2 * self._sphere**2)
            term_kin = 1.75 * (1. - self._epsilon) / (2. * self._rp * self._epsilon * self._sphere)
            
            # #vnew
            factor = 0.5 * self._L / self._R / Tg[0]
            numerador = term_mu * v1s * 0.5 * self._L 
            denominador = 1. - factor * MW * term_kin * (v1s)**2
            P[-1] =  (P[0] - numerador) * denominador 
        
            # P[-1] = self._PH
            # P[0] = P[-1]/denominador + numerador 
        
            x[:,-1]   = x[:,-2]    
            Tg[-1]    = Tg[-2]
            v[-1]     = v[-2]
            
            rho_molar =  self._PL / (self._R * Tg)
        
        elif step =="Adsorption":
            x[:,0]    = self._x1
            Tg[0]     = self._TG1
            v1s=self._v1s
            
            
            mu = self._state_properties['mu']
            MW = np.sum(self._x1 * self._WM)
            term_mu  = (150. * mu[0] * (1.- self._epsilon) ** 2) / (4. * self._rp**2 * self._epsilon**2 * self._sphere**2)
            term_kin = 1.75 * (1. - self._epsilon) / (2. * self._rp * self._epsilon * self._sphere)
            
            # #vnew
            factor = 0.5 * self._L / self._R / self._TG1
            numerador = term_mu * v1s * 0.5 * self._L 
            denominador = 1. - factor * MW * term_kin * (v1s)**2
            
            P[-1] =  (P[0] - numerador) * denominador 
        
            # P[-1] = self._PH
            # P[0] = P[-1]/denominador + numerador 
        
            x[:,-1]   = x[:,-2]    
            Tg[-1]    = Tg[-2]
            v[-1]     = v[-2]
            
            rho_molar = self._PH / (self._R * Tg)
            
        else:
            print("error!x1")
            print(self._actualStep)
        
        self.updateVars(P=P,
                v=v,
                Tg=Tg,
                x=x.reshape(self._ncomp*(self._N+2)),
                q=q.reshape(self._ncomp*(self._N+2)))
        
        # rho = self._state_properties['rho'] 
        mu = self._state_properties['mu']
        cp = self._state_properties['cp']
        k = self._state_properties['k'] 
        phi = self._R * (1 - self._epsilon) / self._epsilon
        
        dP = P[1:] - P[:-1]
        idx_f = dP <= 0
        idx_b = dP > 0 
        
        Ph_f=self._WENO(P,'upwind')
        Ph_b=self._WENO(P,'downwind')
        Tgh_f=self._WENO(Tg,'upwind')
        Tgh_b=self._WENO(Tg,'downwind')
        xh_f = np.array([self._WENO(xi, 'upwind') for xi in x])
        xh_b = np.array([self._WENO(xi, 'downwind') for xi in x])
        
        Ph[idx_f]= Ph_f[idx_f]
        Ph[idx_b]= Ph_b[idx_b]
        xh[:,idx_f] = xh_f[:,idx_f]
        xh[:,idx_b] = xh_b[:,idx_b]
        Tgh[idx_f]=Tgh_f[idx_f]
        Tgh[idx_b]=Tgh_b[idx_b]
        
        Ph[0]=P[0]
        Ph[-1]=P[-1]
        
        if P[0] > P[1]:
            Tgh[0] = Tg[0]
            xh[:,0] = x[:,0]
        else:
            Tgh[0] = Tg[1]
            xh[:,0] = x[:,1]
        
        if P[-2] > P[-1]:
            xh[:,-1]=x[:,-1]
            Tgh[-1]=Tg[-1]
        else:
            xh[:,-1]=x[:,-2]
            Tgh[-1]=Tg[-2]
        
        dPdz[1:self._N+1] = ( Ph[1:self._N+1] - Ph[:self._N] ) / self._dz   
        dPdz[0]  = 0. #
        dPdz[-1] = 0. # 
        dPdzh[1:self._N] = ( P[2:self._N+1] - P[1:self._N] ) / self._dz
        dPdzh[0]  = 2. * ( P[1] - P[0] ) / self._dz
        dPdzh[-1] = 2. * ( P[-1] - P[-2] ) / self._dz
        dxdz[:,1:self._N+1]= ( xh[:,1:self._N+1] - xh[:,:self._N] ) / self._dz         
        dxdz[:,0]  = 2. * ( x[:,1] - x[:,0] ) / self._dz
        dxdz[:,-1] = 2. * ( x[:,-1] - x[:,-2] ) / self._dz        
        dTgdz[1:self._N+1] = ( Tgh[1:self._N+1] - Tgh[:self._N] ) / self._dz
        dTgdz[0]  = 2. * ( Tg[1] - Tg[0] ) / self._dz
        dTgdz[-1] = 2. * ( Tg[-1] - Tg[-2] ) / self._dz
        
        if step == "Pressurization":
            rho_faces=(Ph[0:self._N+1] * np.sum(xh.reshape(self._ncomp,self._N+1).T * self._WM,axis=1) / Tgh[0:self._N+1] / self._R)
            
        elif step == "DePressurization":
            rho_faces=(Ph[0:self._N+1] * np.sum(xh.reshape(self._ncomp,self._N+1).T * self._WM,axis=1) / Tgh[0:self._N+1] / self._R)
        
        elif step == "Adsorption":
            rho_faces=(self._PH * np.sum(xh.reshape(self._ncomp,self._N+1).T * self._WM,axis=1) / Tgh[0:self._N+1] / self._R)
            
        elif step == "Purge":
            rho_faces=(self._PL * np.sum(xh.reshape(self._ncomp,self._N+1).T * self._WM,axis=1) / Tgh[0:self._N+1] / self._R)
        
        else:
            print("error!x2")
            print(self._actualStep)
        
        mu_faces = self.Wilke_f(xh,self._mu)
        term_mu = (150. * mu_faces * (1. - self._epsilon)**2) / (4. * self._rp**2 * self._epsilon**2 * self._sphere**2)
        term_kin = (1.75 * rho_faces * (1. - self._epsilon)) / (2. * self._rp * self._epsilon * self._sphere)
        vh = -np.sign(dPdzh) * (-term_mu + np.sqrt(term_mu**2 + 4. * term_kin * np.abs(dPdzh))) / (2. * term_kin)
        vh = np.clip(vh,1E-6,None)
        
        if step == "Pressurization":

            vh[-1] = 1E-6    
            v[-1] = vh[-1]
            v[1:self._N+1] = 0.5 * (vh[0:self._N] + vh[1:self._N+1])
            
            v[1]= v[2]+abs(v[3]-v[4])
            v[0]= v[1]+abs(v[2]-v[3])
                    
        elif step == "DePressurization":

            vh[0] = 1E-6    
            v[0] = vh[0]
            v[1:self._N+1] = 0.5 * (vh[0:self._N] + vh[1:self._N+1])
            v[-2]= v[-3]+abs(v[-4]-v[-5])
            v[-1]= v[-2]+abs(v[-3]-v[-4])
            
        elif step == "Adsorption":
            
            vh[0]=v1s
            v[1:self._N+1] = 0.5 * (vh[0:self._N] + vh[1:self._N+1])
            v[1]= v[2]+abs(v[3]-v[4])
            v[0]= (v[1]+vh[0])/2.
            v[-2]= v[-3]+abs(v[-4]-v[-5])
            v[-1]= v[-2]+abs(v[-3]-v[-4])

        elif step == "Purge":
            
            vh[0]=v1s
            v[1:self._N+1] = 0.5 * (vh[0:self._N] + vh[1:self._N+1])
            v[1]= v[2]-abs(v[3]-v[4])
            v[0]= v[1]-abs(v[2]-v[3])
            v[-2]= v[-3]+abs(v[-4]-v[-5])
            v[-1]= v[-2]+abs(v[-3]-v[-4])           
            
        else:
            print("error!x3")
            print(self._actualStep)
        
        Dl = 0.7 * np.mean(self._Dm) + v*self._rp
        Pe = v * self._dz / Dl
        qi_eq = self.isotherm(x=x, P=P, T=Tg) # [mol/kg]
        
        if step == "Pressurization":
            kmtc =  np.array([0.005, 0.0 ,0.0, 0.0, 0.0, 0.0, 0.0004])
        elif step == "CoDePressurization":
            kmtc =  np.array([0.005, 0.0 ,0.0, 0.0, 0.0, 0.0, 0.0004])
        elif step == "DePressurization":
            kmtc =  np.array([0.005, 0.0 ,0.0, 0.0, 0.0, 0.0, 0.0004])
        elif step == "Purge":
            kmtc =  np.array([0.005, 0.0 ,0.0, 0.0, 0.0, 0.0, 0.0004])
        elif step == "Adsorption":
            kmtc =  np.array([0.005, 0.0 ,0.0, 0.0, 0.0, 0.0, 0.0004])
        else:
            print("error!x4")
            print(self._actualStep)

        dqdt[:,1:self._N+1] =  (kmtc
                       *(
                          + qi_eq[:,1:self._N+1].T
                          - q[:,1:self._N+1].T)
                       ).T 
                       
        dqdt[:,0]=0.
        dqdt[:,-1]=0.
        
        sink_term= (
            (1. - self._epsilon) * (self._rho_ads * self._cpa + self._q_s0 * cp)  # sólido y carga adsorbida
            + self._epsilon * rho_molar * cp                                      # gas
            )                                                                     # unidades: J/m³·K
        
        keff = (
            k * self._ep
            + self._ka * (1. - self._ep) * (1. - self._epsilon)
            + k * (1. - self._ep) * self._epsilon
            )                                                                     # unidades:  W/m·K
        
        PvT = Ph[:self._N+1] * vh[:self._N+1] / Tgh[:self._N+1]                # [Pa·m/s / K]
        Pv  = Ph[:self._N+1] * vh[:self._N+1]                                  # [Pa·m/s]
        
        generation_term = np.sum([
            (1. - self._epsilon) * (-dH_i) * dqdt[i, 1:self._N+1] * self._rho_ads 
            for i, dH_i in enumerate(self._isoFuncs['iso_dH'])                 # dH_i en J/mol 
            ], axis=0)                                                         # unidades[W/m³] (J/s/m³)   
        
        dTgdt[1:self._N+1] = keff[1:self._N+1] * d2Tgdz2[1:self._N+1] / sink_term[1:self._N+1]
        dTgdt[1:self._N+1] -= ( self._epsilon * cp[1:self._N+1] * rho_molar[1:self._N+1]
                               * ((Pv[1:self._N+1] - Pv[:self._N]) - Tg[1:self._N+1] * (PvT[1:self._N+1] - PvT[:self._N])) 
                               / self._dz) / sink_term[1:self._N+1]
        
        dTgdt[1:self._N+1] += generation_term / sink_term[1:self._N+1]
        
        dPdt[1:self._N+1] = -Tg[1:self._N+1] * (PvT[1:self._N+1] - PvT[:self._N]) / self._dz
        dPdt[1:self._N+1] -= phi * Tg[1:self._N+1] * np.sum(dqdt[:,1:self._N+1], axis=0) * self._rho_ads
        dPdt[1:self._N+1] += P[1:self._N+1] * dTgdt[1:self._N+1] / Tg[1:self._N+1]
        
        dxdt[:, 1:self._N+1] = (1 / Pe[1:self._N+1]) * ( d2xdz2[:, 1:self._N+1] + dxdz[:, 1:self._N+1] * dPdz[1:self._N+1] / P[1:self._N+1] -
                                        dxdz[:, 1:self._N+1] * dTgdz[1:self._N+1] / Tg[1:self._N+1])

        dxdt[:, 1:self._N+1] -= vh[1:self._N+1] * dxdz[:, 1:self._N+1]
        dqdt_total = np.sum(dqdt[:, 1:self._N+1], axis=0)
        dxdt[:, 1:self._N+1] += (phi * Tg[1:self._N+1] / P[1:self._N+1]) * (x[:, 1:self._N+1] * dqdt_total - dqdt[:, 1:self._N+1]) * self._rho_ads
        
        if step == "Pressurization":
            dxdt[:, 0] = 0  
            dxdt[:, -1] = dxdt[:,-2]
            dTgdt[0] = 0
            dTgdt[-1] = dTgdt[-2]
            dPdt[-1] = dPdt[-2]
            dqdt[:,0] = 0
            dqdt[:,-1] = 0 
            
        elif step == "DePressurization":
            dxdt[:, -1] = dxdt[:, -2]  
            dxdt[:, 0] = dxdt[:,1]
            dTgdt[0] = dTgdt[1]
            dTgdt[-1] = dTgdt[-2]
            dPdt[0] = dPdt[-1]
            dqdt[:,0] = 0
            dqdt[:,-1] = 0 
                
        elif step == "Adsorption":
            dxdt[:, 0] = 0  
            dxdt[:, -1] = dxdt[:,-2]
            dTgdt[0] = 0
            dTgdt[-1] = dTgdt[-2]
            dPdt[-1] = dPdt[-2]
            dqdt[:,0] = 0
            dqdt[:,-1] = 0 
            
        elif step == "Purge":
            dxdt[:, 0] = 0  
            dxdt[:, -1] = dxdt[:,-2]
            dTgdt[0] = 0
            dTgdt[-1] = dTgdt[-2]
            dPdt[-1] = dPdt[-2]
            dqdt[:,0] = 0
            dqdt[:,-1] = 0 
            
        else:
            print("error!x5")
            print(self._actualStep)
        
        self.updateVars(P=P,
                v=v,
                Tg=Tg,
                x=x.reshape(self._ncomp*(self._N+2)),
                q=q.reshape(self._ncomp*(self._N+2)))
        
        self._derivates['dPdt'] = dPdt
        self._derivates['dTgdt'] = dTgdt
        self._derivates['dxdt'] = dxdt
        self._derivates['dqdt'] = dqdt
        
        return np.concatenate([dPdt, dTgdt, dxdt.reshape(self._ncomp *(self._N+2)), dqdt.reshape(self._ncomp * (self._N+2))])

    def bc_LRF(self,time,):
        if self._bc_LRF_N == None:
            return
        N   = max(float(self._bc_LRF_N(time))*self._rLR,1E-5)
        P   = float(self._bc_LRF_P(time))
        Tg  = float(self._bc_LRF_Tg(time))
        x   = np.array([f_i(time) for f_i in self._bc_LRF_x])
        
        return N, P, Tg, x
    
    def bc_HRF(self,time,):
        
        if self._bc_HRF_N == None:
            return
        N   = max(float(self._bc_HRF_N(time))*self._rHR,1E-5)
        P   = float(self._bc_HRF_P(time))
        Tg  = float(self._bc_HRF_Tg(time))
        x   = np.array([f_i(time) for f_i in self._bc_HRF_x])
        
        return N, P, Tg, x

    def updateStep(self,):

        data = self.results[self._actualStep]
        t = np.array(data['t']) 
        outlet = self._N+1
        inlet = 0
        
        P_out  = np.array([P_list[outlet]  for P_list  in data['P']])   # Pa
        Tg_out = np.array([Tg_list[outlet] for Tg_list in data['Tg']])  # K
        v_out  = np.array([v_list[outlet]  for v_list  in data['v']])   # m/s
        x_out = np.array([x_list[:, outlet] for x_list in data['x']])
        
        C_out = P_out / (self._R * Tg_out)  # mol/m³
        N_out = C_out * v_out * self._Ai
        
        P_in  = np.array([P_list[inlet]  for P_list  in data['P']])   # Pa
        Tg_in = np.array([Tg_list[inlet] for Tg_list in data['Tg']])  # K
        v_in  = np.array([v_list[inlet]  for v_list  in data['v']])   # m/s
        x_in  = np.stack([x_list[:, inlet] for x_list in data['x']], axis=0)    # (nt, ncomp)
        
        C_in = P_in / (self._R * Tg_in)  # mol/m³
        N_in = C_in * v_in * self._Ai
        
        if self._actualStep == "DePressurization" and self._flowDir == "coCurrent":
            
            tHRF = t + self._tCnDepre + self._tCoDepre + self._tLRF
            self._bc_HRF_N=interp1d(tHRF, N_out,  kind='linear', fill_value='extrapolate')
            self._bc_HRF_P=interp1d(tHRF, P_out,  kind='linear', fill_value='extrapolate')
            self._bc_HRF_Tg=interp1d(tHRF, Tg_out, kind='linear', fill_value='extrapolate')
            self._bc_HRF_x = []
            
            for i in range(self._ncomp):
                f_i = interp1d(tHRF, x_out[:, i], kind='linear', fill_value='extrapolate')
                self._bc_HRF_x.append(f_i)
                
            if self._dbg_1dfunc == True:
                tplot = np.linspace(tHRF[0], tHRF[-1], 500)
                
                fig, axs = plt.subplots(2, 2, figsize=(10, 8))
                plt.subplots_adjust(left=0.05, right=0.95,
                                    top=0.95, bottom=0.05,
                                    hspace=0.2, wspace=0.2)

                
                ax1, ax2, ax3, ax4 = axs.flatten()
                
                ax1.plot(tplot, self._bc_HRF_N(tplot)* 3.6, '-',color='r', label='interp')
                ax1.plot(t,       N_out * 3.6,       'o',color='r', label='original')
                ax1.set_xlabel('t [s]')
                ax1.set_ylabel('N [kmol/h]')
                ax1.set_title('Flujo molar')
                ax1.grid(True) 
                
                ax2.plot(tplot, self._bc_HRF_P(tplot)/100000, '-',color='b', label='interp')
                ax2.plot(t,       P_out/100000,       'o',color='b', label='original')
                ax2.set_xlabel('t [s]')
                ax2.set_ylabel('P [kPa]')
                ax2.set_title('Presión')
                ax2.grid(True)  
                
                ax3.plot(tplot, self._bc_HRF_Tg(tplot), '-',color='g', label='interp')
                ax3.plot(t,       Tg_out,       'o',color='g', label='original')
                ax3.set_xlabel('t [s]')
                ax3.set_ylabel('Tg [K]')
                ax3.set_title('Temperatura')
                ax3.grid(True)      
            
                for i, name in enumerate(self._species):
                    f_i = self._bc_HRF_x[i]
                    ax4.plot(tplot, f_i(tplot),   '-', color = self._color_species[i], label=f'{name}')
                    ax4.plot(t,      x_out[:, i], 'o', color = self._color_species[i] )
                ax4.set_xlabel('t [s]')
                ax4.set_ylabel('y (molar)')
                ax4.set_title('Composición')
                ax4.legend(ncol=2)
                ax4.grid(True)
                plt.show()
            
        elif self._actualStep == "Adsorption":
            
            tLRF = t + self._tCnDepre + self._tCoDepre + self._tAds
            
            self._bc_LRF_N=interp1d(tLRF, N_out,  kind='linear', fill_value='extrapolate')
            self._bc_LRF_P=interp1d(tLRF, P_out,  kind='linear', fill_value='extrapolate')
            self._bc_LRF_Tg=interp1d(tLRF, Tg_out, kind='linear', fill_value='extrapolate')
            self._bc_LRF_x = []

            
            for i in range(self._ncomp):
                f_i = interp1d(tLRF, x_out[:, i], kind='linear', fill_value='extrapolate')
                self._bc_LRF_x.append(f_i)
                
            if self._dbg_1dfunc == True:
                tplot = np.linspace(tLRF[0], tLRF[-1], 200)
                
                fig, axs = plt.subplots(2, 2, figsize=(10, 8))
                plt.subplots_adjust(left=0.05, right=0.95,
                                    top=0.95, bottom=0.05,
                                    hspace=0.2, wspace=0.2)

                
                ax1, ax2, ax3, ax4 = axs.flatten()
                
                ax1.plot(tplot, self._bc_LRF_N(tplot)* 3.6, '-',color='r', label='interp')
                ax1.plot(t,       N_out * 3.6,       'o',color='r', label='original')
                ax1.set_xlabel('t [s]')
                ax1.set_ylabel('N [kmol/h]')
                ax1.set_title('Flujo molar')
                ax1.grid(True) 
                
                ax2.plot(tplot, self._bc_LRF_P(tplot)/100000, '-',color='b', label='interp')
                ax2.plot(t,       P_out/100000,       'o',color='b', label='original')
                ax2.set_xlabel('t [s]')
                ax2.set_ylabel('P [kPa]')
                ax2.set_title('Presión')
                ax2.grid(True)  
                
                ax3.plot(tplot, self._bc_LRF_Tg(tplot), '-',color='g', label='interp')
                ax3.plot(t,       Tg_out,       'o',color='g', label='original')
                ax3.set_xlabel('t [s]')
                ax3.set_ylabel('Tg [K]')
                ax3.set_title('Temperatura')
                ax3.grid(True)      
            
                for i, name in enumerate(self._species):
                    f_i = self._bc_LRF_x[i]
                    ax4.plot(tplot, f_i(tplot),   '-', color = self._color_species[i], label=f'{name}')
                    ax4.plot(t,      x_out[:, i], 'o', color = self._color_species[i] )
                ax4.set_xlabel('t [s]')
                ax4.set_ylabel('y (molar)')
                ax4.set_title('Composición')
                ax4.legend(ncol=2)
                ax4.grid(True)
                plt.show()
            
        return
    
    def readStep(self,cycle,step,flowDir):
        case_dir = os.path.join(tempfile.gettempdir(), self._namecase)
        cycle_dir = os.path.join(case_dir, f"cycle_{cycle}")
        if not os.path.join(case_dir, f"cycle_{cycle}"):
            raise FileNotFoundError(f"No encuentro el directorio {cycle_dir}")
        if flowDir == "coCurrent":
            prefix = 'Co'
        elif flowDir == "cnCurrent":
            prefix = 'Cn'
        else:
            print("erorx10! readStep") 
        
        fname = f"{prefix}_{step}.csv"
        filepath =  os.path.join(cycle_dir, fname)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"No encuentro el archivo {filepath}")
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        nt = data.shape[0]
        Z = self._N +2
        ns = self._ncomp
        t_list   = data[:, 0].tolist()
        P_list   = []
        Tg_list  = []
        Ts_list  = []  # si Ts es distinto de Tg, adapta aquí
        v_list   = []
        x_list   = []
        q_list   = []

        for i in range(nt):
            row = data[i]
            idx = 1
            P_i  = row[   idx : idx+Z].copy();            idx += Z
            Tg_i = row[   idx : idx+Z].copy();            idx += Z
            v_i  = row[   idx : idx+Z].copy();            idx += Z
            x_i  = row[idx : idx+ns*Z].reshape(ns, Z).copy(); idx += ns*Z
            q_i  = row[idx : idx+ns*Z].reshape(ns, Z).copy(); idx += ns*Z

            P_list .append(P_i)
            Tg_list.append(Tg_i)
            Ts_list.append(Tg_i)   # si tienes Ts_list diferente, reemplaza por esa
            v_list .append(v_i)
            x_list .append(x_i)
            q_list .append(q_i)

        if step not in self.results:
            self.results[step] = {}
            
        self.results[step] = {
            't':    t_list,
            'P':    P_list,
            'Tg':   Tg_list,
            'Ts':   Ts_list,
            'x':    x_list,
            'q':    q_list,
            'v':    v_list,


            'Flow': flowDir
        }
        
        return
    
    def writeStep(self,t_list,P_list,Tg_list,Ts_list,x_list,q_list,v_list):
     
        actualStep  = self._actualStep
        actualCycle = int(self._actualCycle)
        actualFlow  = self._flowDir
        
        t = t_list
        nt = len(t)
        case_dir = os.path.join(tempfile.gettempdir(), self._namecase)
        os.makedirs(case_dir, exist_ok=True)
        base_path    = os.path.join(case_dir, f"cycle_{actualCycle}")
        os.makedirs(base_path, exist_ok=True)
        P_arr  = np.stack(P_list, axis=1).T      # (nt, N+2)
        Tg_arr = np.stack(Tg_list, axis=1).T     # (nt, N+2)
        v_arr  = np.stack(v_list, axis=1).T      # (nt, N+2)
        x_arr  = np.stack(x_list, axis=0)        # (nt, ncomp, N+2)
        q_arr  = np.stack(q_list, axis=0)        # (nt, ncomp, N+2)
        
        
        cols   = [t.reshape(nt,1), P_arr, Tg_arr, v_arr,
                  x_arr.reshape(nt, -1),
                  q_arr.reshape(nt, -1)]
        bigmat = np.hstack(cols)
        
        # cabecera
        hdr = ["t"]
        Z   = self._N + 2
        hdr += [f"P_z{z}"   for z in range(Z)]
        hdr += [f"Tg_z{z}"  for z in range(Z)]
        hdr += [f"v_z{z}"   for z in range(Z)]
        for s in self._species:
            hdr += [f"x_{s}_z{z}" for z in range(Z)]
        for s in self._species:
            hdr += [f"q_{s}_z{z}" for z in range(Z)]
        
        if actualFlow == "coCurrent":
            outfile = os.path.join(base_path, f"Co_{actualStep}.csv")
        elif actualFlow == "cnCurrent": 
            outfile = os.path.join(base_path, f"Cn_{actualStep}.csv")
            
        np.savetxt(
            outfile,
            bigmat,
            delimiter=",",
            header=",".join(hdr),
            comments=""
        )
        return
        
    def readCycle(self,cycle_number):
        case_dir = os.path.join(tempfile.gettempdir(), self._namecase)
        base_path = os.path.join(case_dir, f"cycle_{cycle_number}")
        if not os.path.isdir(base_path):
            raise FileNotFoundError(f"No existe la carpeta de ciclo: {base_path}")
        Z = self._N+2
        species = self._species
        reconstructed = {}

        # Itera sobre cada archivo STEP_FLOW.csv
        for fn in sorted(glob.glob(os.path.join(base_path, "*.csv"))):
            fname = os.path.basename(fn)[:-4]
            # separa “STEP” de “FLOW”
            step = fname
            flow = 'coCurrent'

            df = pd.read_csv(fn)
            t  = df["t"].values
            nt = len(t)

            # Reconstruye P, Tg, v
            P_arr  = df[[f"P_z{z}"  for z in range(Z)]].values   # (nt, Z)
            Tg_arr = df[[f"Tg_z{z}" for z in range(Z)]].values
            v_arr  = df[[f"v_z{z}"  for z in range(Z)]].values

            # Reconstruye x y q
            ncomp = len(species)
            x_arr = np.zeros((nt, ncomp, Z))
            q_arr = np.zeros((nt, ncomp, Z))
            for i, s in enumerate(species):
                x_cols = [f"x_{s}_z{z}" for z in range(Z)]
                q_cols = [f"q_{s}_z{z}" for z in range(Z)]
                x_arr[:, i, :] = df[x_cols].values
                q_arr[:, i, :] = df[q_cols].values

            # Pasa todo a listas de instantes
            P_list  = [P_arr[i, :].copy() for i in range(nt)]
            Tg_list = [Tg_arr[i, :].copy() for i in range(nt)]
            v_list  = [v_arr[i, :].copy() for i in range(nt)]
            x_list  = [x_arr[i, :, :].copy() for i in range(nt)]
            q_list  = [q_arr[i, :, :].copy() for i in range(nt)]

            reconstructed[step] = {
                "t":   t,
                "P":   P_list,
                "Tg":  Tg_list,
                "Ts":  Tg_list,    # si guardabas Ts igual que Tg
                "x":   x_list,
                "q":   q_list,
                "v":   v_list,
                "Flow": flow
            }

        self.results = reconstructed
        
        return
                
    def storeStep(self,):
        t = self._yData.t
        nt = len(t)
        y = self._yData.y
     
        P_list  = []
        Tg_list = []
        Ts_list = []
        x_list  = []
        q_list  = []
        v_list = []
        
        for i in range(nt):
            y_i   = y[:, i]
            P     = y_i[0:self._N + 2].copy()
            Tg    = y_i[self._N + 2:2 * (self._N + 2)].copy()
            xflat = y_i[2 * (self._N + 2):2 * (self._N + 2) + self._ncomp * (self._N + 2)]
            qflat = y_i[2 * (self._N + 2) + self._ncomp * (self._N + 2):2 * (self._N + 2) + 2 * self._ncomp * (self._N + 2)]
     
            x     = xflat.reshape((self._ncomp, self._N + 2)).copy()
            q     = qflat.reshape((self._ncomp, self._N + 2)).copy()
            
            if self._actualStep== "Pressurization" and self._flowDir == "coCurrent": 
                
                if P[0] < P[1]:
                    P[0] = P[1]
                    Tg[0]     = Tg[1]
                    x[:, 0]   = x[:, 1]
                    q[:, 0]   = 0
                else:
                    Tg[0]     = self._TG1
                    x[:,0]    = self._x1
                    q[:, 0]   = 0
                     
                P[-1]     = P[-2]
                Tg[-1]    = Tg[-2]
                x = np.clip(x, 0.0, 1.0)
                x[:, -1]  = x[:, -2]
                q[:, -1]  = 0
                                
                rho   = (P / (self._R * Tg)) * np.sum(x.T * self._WM, axis=1)
                mu    = self.Wilke_c(x, self._mu)    
                dPdzh = np.gradient(P, self._dz)
                term_mu = ((150. * mu * (1. - self._epsilon) ** 2)
           / (4. * self._rp ** 2 * self._epsilon ** 2 * self._sphere ** 2))
                term_kin = ((1.75 * rho * (1. - self._epsilon))
            / (2. * self._rp * self._epsilon * self._sphere))                
                
                v = (-np.sign(dPdzh) *
                      (-term_mu + np.sqrt(term_mu ** 2 + 4. * term_kin * np.abs(dPdzh)))
                      / (2. * term_kin))
                
                v[1]= v[2]+abs(v[3]-v[4])
                v[0]= v[1]+abs(v[2]-v[3])
                v[-1]= 0.
                
                Tg[0]     = self._TG1
                x[:,0]    = self._x1
                q[:, 0]   = 0
            
            elif self._actualStep== "Pressurization" and self._flowDir == "cnCurrent": 
                
                Ninlet,Pinlet,Tinlet,xinlet = self.bc_HRF(t[i])
                C1 = Pinlet/self._R/Tinlet
                v1i= Ninlet/C1/self._Ai/self._epsilon
                v1s= Ninlet/C1/self._Ai
                
                if P[0] < P[1]:
                    P[0] = P[1]
                    Tg[0]     = Tg[1]
                    x[:, 0]   = x[:, 1]
                    q[:, 0]   = 0
                else:
                    Tg[0]     = self._TG1
                    x[:,0]    = self._x1
                    q[:, 0]   = 0
                     
                P[-1]     = P[-2]
                Tg[-1]    = Tg[-2]
                x = np.clip(x, 0.0, 1.0)
                x[:, -1]  = x[:, -2]
                q[:, -1]  = 0
                                
                rho   = (P / (self._R * Tg)) * np.sum(x.T * self._WM, axis=1)
                mu    = self.Wilke_c(x, self._mu)    
                dPdzh = np.gradient(P, self._dz)
                term_mu = ((150. * mu * (1. - self._epsilon) ** 2)
           / (4. * self._rp ** 2 * self._epsilon ** 2 * self._sphere ** 2))
                term_kin = ((1.75 * rho * (1. - self._epsilon))
            / (2. * self._rp * self._epsilon * self._sphere))                
                
                v = (-np.sign(dPdzh) *
                      (-term_mu + np.sqrt(term_mu ** 2 + 4. * term_kin * np.abs(dPdzh)))
                      / (2. * term_kin))
                
                v[1]= v[2]+abs(v[3]-v[4])
                v[0]= v[1]+abs(v[2]-v[3])
                v[-1]= 0.
                
                Tg[0]     = Tinlet
                x[:,0]    = xinlet
                q[:, 0]   = 0
            
            elif self._actualStep== "DePressurization" and self._flowDir == "coCurrent":
                
                if P[-1] < P[-2]:
                    P[-1] = P[-2]
                    Tg[-1]     = Tg[-2]
                    x[:, -1]   = x[:, -2]
                    q[:, -1]   = 0
                else:
                    P[-1] = P[-2]
                    Tg[-1]     = Tg[-2]
                    x[:,-1]    = x[:, -2]
                    q[:, -1]   = 0
                     
                P[0]     = P[1]
                Tg[0]    = Tg[1]
                x = np.clip(x, 0.0, 1.0)
                x[:, 0]  = x[:, 1]
                q[:, 0]  = 0
                                
                rho   = (P / (self._R * Tg)) * np.sum(x.T * self._WM, axis=1)
                mu    = self.Wilke_c(x, self._mu)    
                dPdzh = np.gradient(P, self._dz)
                term_mu = ((150. * mu * (1. - self._epsilon) ** 2)
           / (4. * self._rp ** 2 * self._epsilon ** 2 * self._sphere ** 2))
                term_kin = ((1.75 * rho * (1. - self._epsilon))
            / (2. * self._rp * self._epsilon * self._sphere))                
                
                v = (-np.sign(dPdzh) *
                      (-term_mu + np.sqrt(term_mu ** 2 + 4. * term_kin * np.abs(dPdzh)))
                      / (2. * term_kin))
                
                v[-2]= v[-3]+abs(v[-4]-v[-5])
                v[-1]= v[-2]+abs(v[-3]-v[-4])
                v[0]= 0.
            
            elif self._actualStep== "DePressurization" and self._flowDir == "cnCurrent":
                
                if P[-1] < P[-2]:
                    P[-1] = P[-2]
                    Tg[-1]     = Tg[-2]
                    x[:, -1]   = x[:, -2]
                    q[:, -1]   = 0
                else:
                    P[-1] = P[-2]
                    Tg[-1]     = Tg[-2]
                    x[:,-1]    = x[:, -2]
                    q[:, -1]   = 0
                     
                P[0]     = P[1]
                Tg[0]    = Tg[1]
                x = np.clip(x, 0.0, 1.0)
                x[:, 0]  = x[:, 1]
                q[:, 0]  = 0
                                
                rho   = (P / (self._R * Tg)) * np.sum(x.T * self._WM, axis=1)
                mu    = self.Wilke_c(x, self._mu)    
                dPdzh = np.gradient(P, self._dz)
                term_mu = ((150. * mu * (1. - self._epsilon) ** 2)
           / (4. * self._rp ** 2 * self._epsilon ** 2 * self._sphere ** 2))
                term_kin = ((1.75 * rho * (1. - self._epsilon))
            / (2. * self._rp * self._epsilon * self._sphere))                
                
                v = (-np.sign(dPdzh) *
                      (-term_mu + np.sqrt(term_mu ** 2 + 4. * term_kin * np.abs(dPdzh)))
                      / (2. * term_kin))
                
                v[-2]= v[-3]+abs(v[-4]-v[-5])
                v[-1]= v[-2]+abs(v[-3]-v[-4])
                v[0]= 0.
                
            elif self._actualStep == "Adsorption":
                if P[0] < P[1]:
                    P[0] = P[1]
                    Tg[0]     = Tg[1]
                    x[:, 0]   = x[:, 1]
                    q[:, 0]   = q[:, 1]
                else:
                    P[0]      = self._PH
                    Tg[0]     = self._TG1
                    x[:,0]    = self._x1
                    q[:, 0]   = 0
                    
                P[-1]     = P[-2]
                Tg[-1]    = Tg[-2]
                x = np.clip(x, 0.0, 1.0)
                x[:, -1]  = x[:, -2]
                q[:, -1]  = 0
                                
                rho   = (P / (self._R * Tg)) * np.sum(x.T * self._WM, axis=1)
                mu    = self.Wilke_c(x, self._mu)    
                dPdzh = np.gradient(P, self._dz)
                term_mu = ((150. * mu * (1. - self._epsilon) ** 2)
           / (4. * self._rp ** 2 * self._epsilon ** 2 * self._sphere ** 2))
                term_kin = ((1.75 * rho * (1. - self._epsilon))
            / (2. * self._rp * self._epsilon * self._sphere))                
                
                v = (-np.sign(dPdzh) *
                      (-term_mu + np.sqrt(term_mu ** 2 + 4. * term_kin * np.abs(dPdzh)))
                      / (2. * term_kin))
                
                v[1]= v[2]+abs(v[3]-v[4])
                v[0]= v[1]+abs(v[2]-v[3])
                v[-2]= v[-3]+abs(v[-4]-v[-5])
                v[-1]= v[-2]+abs(v[-3]-v[-4])
                
                P[0]      = self._PH
                Tg[0]     = self._TG1
                x[:,0]    = self._x1
                q[:, 0]   = 0

            elif self._actualStep == "Purge":
                
                Ninlet,Pinlet,Tinlet,xinlet = self.bc_LRF(t[i])
                C1 = Pinlet/self._R/Tinlet
                v1i= Ninlet/C1/self._Ai/self._epsilon
                v1s= Ninlet/C1/self._Ai
                
                if P[0] < P[1]:
                    P[0] = P[1]
                    Tg[0]     = Tg[1]
                    x[:, 0]   = x[:, 1]
                    q[:, 0]   = q[:, 1]
                else:
                    P[0]      = self._PL
                    Tg[0]     = Tinlet
                    x[:,0]    = xinlet
                    q[:, 0]   = 0
                    
                P[-1]     = P[-2]
                Tg[-1]    = Tg[-2]
                x = np.clip(x, 0.0, 1.0)
                x[:, -1]  = x[:, -2]
                q[:, -1]  = 0
                                
                rho   = (P / (self._R * Tg)) * np.sum(x.T * self._WM, axis=1)
                mu    = self.Wilke_c(x, self._mu)    
                dPdzh = np.gradient(P, self._dz)
                term_mu = ((150. * mu * (1. - self._epsilon) ** 2)
           / (4. * self._rp ** 2 * self._epsilon ** 2 * self._sphere ** 2))
                term_kin = ((1.75 * rho * (1. - self._epsilon))
            / (2. * self._rp * self._epsilon * self._sphere))                
                
                v = (-np.sign(dPdzh) *
                      (-term_mu + np.sqrt(term_mu ** 2 + 4. * term_kin * np.abs(dPdzh)))
                      / (2. * term_kin))
                
                v[1]= v[2]-abs(v[3]-v[4])
                v[0]= v[1]-abs(v[2]-v[3])
                
                v[-2]= v[-3]+abs(v[-4]-v[-5])
                v[-1]= v[-2]+abs(v[-3]-v[-4])
                
                P[0]      = self._PL
                Tg[0]     = Tinlet
                x[:,0]    = xinlet
                q[:, 0]   = 0  
                
            else:  
                print("error!x6")
                print(self._actualStep)
            
            if self._flowDir == "coCurrent":
                P_list.append(P)
                Tg_list.append(Tg)
                Ts_list.append(Tg)
                x_list.append(x)
                q_list.append(q)
                v_list.append(v)
                
            elif self._flowDir == "cnCurrent":
                P_list.append(P[::-1])
                Tg_list.append(Tg[::-1])
                Ts_list.append(Tg[::-1])
                x = x.reshape(self._ncomp, self._N+2)[:, ::-1]
                q = q.reshape(self._ncomp, self._N+2)[:, ::-1]
                x_list.append(x)
                q_list.append(q)
                v_list.append(v[::-1])
                
        P_last=P_list[-1].copy()
        Tg_last=Tg_list[-1].copy()
        v_last=v_list[-1].copy()
        x_last=x_list[-1].copy()
        q_last=q_list[-1].copy()
        
        self.updateVars(P=P_last,
                        Tg=Tg_last,
                        v=v_last,
                        x=x_last.reshape(self._ncomp*(self._N+2)), 
                        q=q_last.reshape(self._ncomp*(self._N+2)))
                
        
        time = t
        actualStep=self._actualStep
        actualFlow=self._flowDir
        results ={
            't': time,
            'P': P_list,
            'Tg': Tg_list,
            'Ts': Ts_list,
            'x': x_list,
            'q': q_list,
            'v': v_list,
            'Flow': actualFlow  
        }
        
        #GUARDAR INFO EN MEMOMERIA
        if actualStep not in self.results:
            self.results[actualStep] = {}
            self.results[actualStep] = {
                't': time,
                'P': P_list,
                'Tg': Tg_list,
                'Ts': Ts_list,
                'x': x_list,
                'q': q_list,
                'v': v_list,
                'Flow': actualFlow  
            }
        else:
            self.results[actualStep] = {
                't': time,
                'P': P_list,
                'Tg': Tg_list,
                'Ts': Ts_list,
                'x': x_list,
                'q': q_list,
                'v': v_list,
                'Flow': actualFlow  
            }
            
        self.writeStep(t_list=t,
                       P_list=P_list,
                       Tg_list=Tg_list,
                       Ts_list=Ts_list,
                       x_list=x_list,
                       q_list=q_list,
                       v_list=v_list)
        
        
        if self._dbg_Plots == True:
            self.dbgPlotStep(results=results,dim="3d")
        
        return 

    def dbgPlotStep(self,results,dim):
        
        data = results
        tiempos = data['t']
        longitud = np.linspace(0, self._L, self._N + 2)
        Tgrid, Zgrid = np.meshgrid(tiempos, longitud)
    
        # Convertir listas a arrays con ejes consistentes
        P  = np.stack(data['P'], axis=1)     # (N+2, nt)
        Tg = np.stack(data['Tg'], axis=1)
        Ts = np.stack(data['Ts'], axis=1)
        v  = np.stack(data['v'], axis=1)
    
        x_array = np.stack(data['x'], axis=0)   # (nt, ncomp, N+2)
        q_array = np.stack(data['q'], axis=0)
    
        # Reorganizar x y q para plot: list of 2D arrays por componente
        x = [x_array[:, i, :].T for i in range(self._ncomp)]  # cada uno (N+2, nt)
        q = [q_array[:, i, :].T for i in range(self._ncomp)]
    
        # Identificar CO2
        try:
            idx_CO2 = self._species.index("CO2")
        except ValueError:
            idx_CO2 = 0  # fallback
    
        xCO2 = x[idx_CO2]
        qCO2 = q[idx_CO2]
        xRESTO = sum([x[i] for i in range(self._ncomp) if i != idx_CO2])
        qRESTO = sum([q[i] for i in range(self._ncomp) if i != idx_CO2])
    
        variables = [
            (P / 1000, 'Presión [kPa]'),
            (v / self._epsilon, 'Velocidad [m/s]'),
            (Tg, 'Temperatura Gas [K]'),
            (Ts, 'Temperatura Sólido [K]'),
            (xCO2, 'yCO₂'),
            (xRESTO, 'yResto'),
            (qCO2, 'qCO₂ [mol/kg]'),
            (qRESTO, 'qResto [mol/kg]')
        ]
    
    
        if dim == "3d":
            fig = plt.figure(figsize=(self._imgwidth/100, self._imgheight/100))
            axes = []
            for i, (var, label) in enumerate(variables):
                ax = fig.add_subplot(2, 4, i+1, projection='3d')
                for t in tiempos:
                    # 2) Ahora que "tiempos" es array, .tolist() funciona
                    idx_t = list(tiempos).index(t)
                    color = next(ax._get_lines.prop_cycler)['color']
                    ax.plot(longitud, [t]*len(longitud), var[:, idx_t], color=color)
                    ax.scatter(longitud[0], t, var[0, idx_t], color=color, marker='s', s=40)
                    ax.scatter(longitud[-1], t, var[-1, idx_t], color=color, marker='>', s=70)
    
                ax.set_xlabel('L [m]')
                ax.set_ylabel('t [s]')
                ax.set_title(label)
                ax.set_box_aspect([8,8,8])
                ax.grid(True)
                axes.append(ax)
    
            def on_move(event):
                if event.inaxes in axes:
                    elev, azim = event.inaxes.elev, event.inaxes.azim
                    for ax in axes:
                        ax.view_init(elev, azim)
                    fig.canvas.draw_idle()
    
            def on_key(event):
                views = {
                    '1': (90, -90),  # XY
                    '2': (0, -90),   # XZ
                    '3': (0, 0),     # YZ
                    '4': (30, -60)   # Isométrica
                }
                if event.key in views:
                    elev, azim = views[event.key]
                    for ax in axes:
                        ax.view_init(elev, azim)
                    fig.canvas.draw_idle()
    
            fig.canvas.mpl_connect('key_press_event', on_key)
            fig.canvas.mpl_connect('motion_notify_event', on_move)
            plt.subplots_adjust(left=0.01, right=0.97, top=0.95, bottom=0.05,
                                hspace=0.15, wspace=0.05)
            plt.show()
        
        elif dim == "2d":
            fig, axes = plt.subplots(2, 4,figsize=(self._imgwidth/100, self._imgheight/100),constrained_layout=True)
            axes = axes.flatten()
            cmap_map = {
                'Presión [kPa]':           'Spectral',
                'Velocidad [m/s]':         'coolwarm',
                'Temperatura Gas [K]':     'inferno',
                'Temperatura Sólido [K]':  'inferno',
                'y CO₂':                   'RefDiff',
                'y Resto':                 'RefDiff',
                'qCO₂ [mol/kg]':          'winter',
                'qResto [mol/kg]':        'winter',
            } 
            
            for ax, (var, label) in zip(axes, variables):
                cmap = cmap_map.get(label, 'viridis')
                cf = ax.contourf(Tgrid, Zgrid, var, levels=50, cmap=cmap)
                cbar = fig.colorbar(cf, ax=ax, orientation='vertical')
                cbar.set_label(label)
                ax.set_xlabel('t [s]')
                ax.set_ylabel('z [m]')
                ax.set_title(label)
                ax.grid(True)
    
            plt.show()
            
            
        elif dim == "1d":
            fig = plt.figure(figsize=(self._imgwidth/100, self._imgheight/100), constrained_layout=True)
            # Creamos 2 filas x 4 columnas de ejes 3D
            axes = []
            for i in range(8):
                ax = fig.add_subplot(2, 4, i+1, projection='3d')
                axes.append(ax)
            
            cmap_map = {
                'Presión [kPa]':           'Spectral',
                'Velocidad [m/s]':         'coolwarm',
                'Temperatura Gas [K]':     'inferno',
                'Temperatura Sólido [K]':  'inferno',
                'y CO₂':                   'RefDiff',
                'y Resto':                 'RefDiff',
                'qCO₂ [mol/kg]':           'winter',
                'qResto [mol/kg]':         'winter',
            }
            
            for ax, (var, label) in zip(axes, variables):
                cmap = cmap_map.get(label, 'viridis')
                surf = ax.plot_surface(Tgrid, Zgrid, var,
                                       rstride=1, cstride=1,
                                       cmap=cmap,
                                       linewidth=0, antialiased=True)
                # Añadimos la barra de color
                m = plt.cm.ScalarMappable(cmap=cmap)
                m.set_array(var)
                # cbar = fig.colorbar(m, ax=ax, shrink=0.5, aspect=10)
                # cbar.set_label(label)
                
                ax.set_xlabel('t [s]')
                ax.set_ylabel('z [m]')
                # ax.set_zlabel(label)
                ax.set_title(label)
            
            def on_move(event):
                if event.inaxes in axes:
                    elev, azim = event.inaxes.elev, event.inaxes.azim
                    for ax in axes:
                        ax.view_init(elev, azim)
                    fig.canvas.draw_idle()
        
            def on_key(event):
                views = {
                    '1': (90, -90),  # XY
                    '2': (0, -90),   # XZ
                    '3': (0, 0),     # YZ
                    '4': (30, -60)   # Isométrica
                }
                if event.key in views:
                    elev, azim = views[event.key]
                    for ax in axes:
                        ax.view_init(elev, azim)
                    fig.canvas.draw_idle()
        
            fig.canvas.mpl_connect('key_press_event', on_key)
            fig.canvas.mpl_connect('motion_notify_event', on_move)
            plt.show()
            
    def dbgAniStep(self, step, dim):

        data = self.results[step]
        L = self._L
        N = self._N + 2 
        species = self._species
        
        t = np.array(data['t'])
        z = np.linspace(0, L, N)
        P    = np.stack(data['P'], axis=1)
        Tg   = np.stack(data['Tg'], axis=1)
        v    = np.stack(data['v'], axis=1)
        xarr = np.stack(data['x'], axis=0)
        qarr = np.stack(data['q'], axis=0)
        
        # CO2 index or fallback
        try:
            idx = species.index('CO2')
        except ValueError:
            idx = 0
        
        xCO2 = xarr[:, idx, :].T
        qCO2 = qarr[:, idx, :].T
        xRESTO = np.sum([xarr[:, i, :] for i in range(self._ncomp) if i != idx], axis=0).T
        qRESTO = np.sum([qarr[:, i, :] for i in range(self._ncomp) if i != idx], axis=0).T

        
        variables = [
            (P/1000.,      'Presión [kPa]'),
            (v/self._epsilon, 'Velocidad [m/s]'),
            (Tg,     'Temperatura Gas [K]'),
            (Tg,     'Temperatura Sólido [K]'),
            (xCO2,   'y_CO₂'),
            (xRESTO, 'y_Resto'),
            (qCO2,   'q_CO₂ [mol/kg]'),
            (qRESTO, 'q_Resto [mol/kg]')
        ]
        
        #3D====================================================================
        if dim == '3d':
            plt.ion()
            fig = plt.figure(figsize=(12, 8))
            axes = []
        
            for i, (_, label) in enumerate(variables):
                ax = fig.add_subplot(2, 4, i+1, projection='3d')
                ax.set_xlim(0, self._L)
                ax.set_xlabel('z [m]')
                ax.set_ylabel('t [s]')
                ax.set_zlabel(label)
                ax.set_title(label)
                ax.grid(True)
                axes.append(ax)
            fig.tight_layout()
        
            try:
                while plt.fignum_exists(fig.number):
                    for j, t_val in enumerate(t):
                        for ax, (var, label) in zip(axes, variables):
                            ax.clear()
                            ax.set_xlim(0, self._L)
                            ax.set_xlabel('z [m]')
                            ax.set_ylabel('t [s]')
                            ax.set_zlabel(label)
                            ax.set_title(label)
                            ax.plot(z, [t_val]*len(z), var[:, j], color='C0')
                            ax.scatter(z[0], t_val, var[0, j], marker='s', s=30, color='C1')
                            ax.scatter(z[-1], t_val, var[-1, j], marker='>', s=30, color='C2')
                            
                        fig.canvas.draw()
                        plt.pause(0.1)
            except KeyboardInterrupt:
                pass
            finally:
                plt.ioff()
                plt.close(fig)
        #3D====================================================================
        
        #2D====================================================================
        elif dim == "2d": 
            plt.ion()
            fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)
            axes = axes.flatten()
            lines = []
            titles = []
        
            # Inicializar subplots
            for ax, (var, label) in zip(axes, variables):
                ax.set_xlim(0, self._L)
                ax.set_ylim(np.nanmin(var), np.nanmax(var))
                ax.set_xlabel('z [m]')
                ax.set_ylabel(label)
                ax.grid(True)
                line, = ax.plot(z, var[:, 0], lw=2)
                title = ax.set_title(f"{label} | t={t[0]:.0f}s")
                lines.append(line)
                titles.append(title)

        
            fig.tight_layout()
        
            try:
                while plt.fignum_exists(fig.number):
                    for i, time_val in enumerate(t):
                        for line, title, (var, label) in zip(lines, titles, variables):
                            line.set_ydata(var[:, i])
                            title.set_text(f"{label} | t={time_val:.0f}s")
                        fig.canvas.draw()
                        plt.pause(0.1)
            except KeyboardInterrupt:
                pass
            finally:
                plt.ioff()
                plt.close(fig)
        #2D====================================================================
        
        #1D====================================================================
        elif dim == "1d":
            # … dentro de PSA.aniStep, reemplaza el bloque dim=='1d' por esto …

            ranges = [(mat.min(), mat.max()) for mat, _ in variables]
            # Defino un grosor muy pequeño para la cara
            thickness = 0.01 * self._L
        
            # Meshgrid: X = [0, thickness], Y = z
            Xgrid, Ygrid = np.meshgrid([0, thickness], z)  # shape (N+2, 2)
        
            plt.ion()
            fig, axes = plt.subplots(1, 8,
                                     figsize=(8*self._imgwidth/100, self._imgheight/100),
                                     constrained_layout=True)
            axes = axes.flatten()
            pcm_list = []
            title_list = []
        
            # Inicializar cada cara
            for ax, ((mat, label), (vmin, vmax)) in zip(axes, zip(variables, ranges)):
                # Primer frame
                V0 = np.column_stack([mat[:,0], mat[:,0]])  # (N+2, 2)
                pcm = ax.pcolormesh(
                    Xgrid, Ygrid, V0,
                    vmin=vmin, vmax=vmax,
                    shading='auto',
                    cmap='coolwarm'
                )
                # Una sola barra por columna
                cbar = fig.colorbar(pcm, ax=ax,
                                    orientation='horizontal',
                                    pad=0.05,
                                    fraction=0.07)
                cbar.set_label(label)
                # Etiquetas
                ax.set(xlabel='', ylabel='z [m]', xticks=[])
                title = ax.set_title(f"{label} | t={t[0]:.1f}s")
                pcm_list.append((pcm, mat))
                title_list.append((title, label))
        
            # Bucle de animación
            try:
                while plt.fignum_exists(fig.number):
                    for i, tv in enumerate(t):
                        for (pcm, mat), (title, label) in zip(pcm_list, title_list):
                            V = np.column_stack([mat[:,i], mat[:,i]])
                            pcm.set_array(V.ravel())
                            title.set_text(f"{label} | t={tv:.1f}s")
                        fig.canvas.draw_idle()
                        plt.pause(0.1)
            except KeyboardInterrupt:
                pass
            finally:
                plt.ioff()
                plt.close(fig)
    

        #1D====================================================================
        return 
        
    def solveCycle(self,):
        star_cycle = time.time()
        self._actualCycle +=1 
        print("\n")
        print("="*50)
        print(" Cycle = ",self._actualCycle)
        print("="*50)
        start_time = time.time()
        self._actualStep = "Pressurization"
        self._flowDir = "coCurrent"
        y0 = self.nextStep()
        tau_Actual = self._actualTime
        tau_Step = tau_Actual + self._tPre
        t_span = [tau_Actual, tau_Step]
        
        t_eva = np.linspace(tau_Actual,tau_Step,self._saveData)
        # t_eva = np.arange(tau_Actual, tau_Step, self._saveData)
        # if t_eva.size == 0 or t_eva[-1] < tau_Step:
        #     t_eva = np.arange(tau_Actual, tau_Step, int(self._saveData/5))
        
        self._yData = solve_ivp(fun    = self.solveStep,
                t_span = t_span,
                t_eval = t_eva,
                y0     = y0,
                method = 'BDF',
                rtol = 1e-12,
                atol = self._atol,
                )
        execution_time = time.time() - start_time
        real_time = time.time() - star_cycle
        
        
        if self._yData.success:
            print("="*50)
            print(f"✅ Simulación {self._actualStep} con éxito en {execution_time:.1f} segundos.")
            print("="*50)
            self.storeStep() 
            self.updateStep()
            self._actualTime=tau_Step
            print(f"{self._actualTime:.1f} {real_time:.1f}") 
        else:
            print("solveCycle error!x7",self._actualStep)         
        
        start_time = time.time()
        self._actualStep = "Adsorption"
        self._flowDir = "coCurrent"
        y0 = self.nextStep()
        tau_Actual = self._actualTime
        tau_Step = tau_Actual + self._tAds
        t_span = [tau_Actual, tau_Step]
        
        t_eva = np.linspace(tau_Actual,tau_Step,self._saveData)
        # t_eva = np.arange(tau_Actual, tau_Step, self._saveData)
        # if t_eva.size == 0 or t_eva[-1] < tau_Step:
        #     t_eva = np.arange(tau_Actual, tau_Step, int(self._saveData/5))
        
        self._yData = solve_ivp(fun    = self.solveStep,
                t_span = t_span,
                t_eval = t_eva,
                y0     = y0,
                method = 'BDF',
                rtol = 1E-12,
                atol = 1E-12
                )
        execution_time = time.time() - start_time
        real_time = time.time() - star_cycle

        if self._yData.success:
            print("="*50)
            print(f"✅ Simulación {self._actualStep} con éxito en {execution_time:.1f} segundos.")
            print("="*50)
            self.storeStep() 
            self.updateStep()
            self._actualTime=tau_Step
            print(f"{self._actualTime:.1f} {real_time:.1f}") 
        else:
            print("solveCycle error!x7",self._actualStep)         
        
        
        start_time = time.time()    
        self._actualStep = "DePressurization"
        self._flowDir = "coCurrent"
        y0 = self.nextStep()
        tau_Actual = self._actualTime
        tau_Step = tau_Actual + self._tCoDepre
        t_span = [tau_Actual, tau_Step]
        t_eva = np.linspace(tau_Actual,tau_Step,self._saveData)
        # t_eva = np.arange(tau_Actual, tau_Step, self._saveData)
        # if t_eva.size == 0 or t_eva[-1] < tau_Step:
        #     t_eva = np.append(t_eva, tau_Step)
        self._yData = solve_ivp(fun    = self.solveStep,
                t_span = t_span,
                t_eval = t_eva,
                y0     = y0,
                method = 'BDF',
                rtol = 1E-12,
                atol = 1E-12,
                )
        execution_time = time.time() - start_time
        real_time = time.time() - star_cycle

        
        if self._yData.success:
            print("="*50)
            print(f"✅ Simulación {self._actualStep} con éxito en {execution_time:.1f} segundos.")
            print("="*50)
            self.storeStep()
            self.updateStep()
            self._actualTime=tau_Step
            print(f"{self._actualTime:.1f} {real_time:.1f}") 
        else:
            print("solveCycle error!x7",self._actualStep)         
        
        
        start_time = time.time()    
        self._actualStep = "DePressurization"
        self._flowDir = "cnCurrent"
        y0 = self.nextStep()
        self.xxx=y0
        tau_Actual = self._actualTime
        tau_Step = tau_Actual + self._tCnDepre
        t_span = [tau_Actual, tau_Step]
        t_eva = np.linspace(tau_Actual,tau_Step,self._saveData)
        # t_eva = np.arange(tau_Actual, tau_Step, self._saveData)
        # if t_eva.size == 0 or t_eva[-1] < tau_Step:
        #     t_eva = np.append(t_eva, tau_Step)
        self._yData = solve_ivp(fun    = self.solveStep,
                t_span = t_span,
                t_eval = t_eva,
                y0     = y0,
                method = 'BDF',
                rtol = 1E-12,
                atol = 1E-12,
                )
        execution_time = time.time() - start_time
        real_time = time.time() - star_cycle

        
        if self._yData.success:
            print("="*50)
            print(f"✅ Simulación {self._actualStep} con éxito en {execution_time:.1f} segundos.")
            print("="*50)
            self.storeStep()
            self.updateStep()
            self._actualTime=tau_Step
            print(f"{self._actualTime:.1f} {real_time:.1f}") 
        else:
            print("solveCycle error!x7",self._actualStep)
        
            
        start_time = time.time()    
        self._actualStep = "Purge"
        self._flowDir = "cnCurrent"
        y0 = self.nextStep()
        tau_Actual = self._actualTime
        tau_Step = tau_Actual + self._tLRF
        
        t_span = [tau_Actual, tau_Step]
        t_eva = np.linspace(tau_Actual,tau_Step,self._saveData)
        # t_eva = np.arange(tau_Actual, tau_Step, self._saveData)
        # if t_eva.size == 0 or t_eva[-1] < tau_Step:
        #     t_eva = np.append(t_eva, tau_Step)
            
        self._yData = solve_ivp(fun    = self.solveStep,
                t_span = t_span,
                t_eval = t_eva,
                y0     = y0,
                method = 'BDF',
                rtol = 1E-3,
                atol = 1E-3
                )
        execution_time = time.time() - start_time
        real_time = time.time() - star_cycle

        
        if self._yData.success:
            print("="*50)
            print(f"✅ Simulación {self._actualStep} con éxito en {execution_time:.1f} segundos.")
            print("="*50)
            self.storeStep()
            self.updateStep()
            self._actualTime=tau_Step
            print(f"{self._actualTime:.1f} {real_time:.1f}") 
        else:
            print("solveCycle error!x7",self._actualStep)
            
              
        start_time = time.time()    
        self._actualStep = "Pressurization"
        self._flowDir = "cnCurrent"
        y0 = self.nextStep()
        tau_Actual = self._actualTime
        tau_Step = tau_Actual + self._tHRF
        
        t_span = [tau_Actual, tau_Step]
        t_eva = np.linspace(tau_Actual,tau_Step,self._saveData)
        # t_eva = np.arange(tau_Actual, tau_Step, self._saveData)
        # if t_eva.size == 0 or t_eva[-1] < tau_Step:
        #     t_eva = np.append(t_eva, tau_Step)
            
        self._yData = solve_ivp(fun    = self.solveStep,
                t_span = t_span,
                t_eval = t_eva,
                y0     = y0,
                method = 'BDF',
                rtol = 1E-12,
                atol = 1E-12
                )
        execution_time = time.time() - start_time
        real_time = time.time() - star_cycle

        if self._yData.success:
            print("="*50)
            print(f"✅ Simulación {self._actualStep} con éxito en {execution_time:.1f} segundos.")
            print("="*50)
            self.storeStep()
            self.updateStep()
            self._actualTime=tau_Step
            print(f"{self._actualTime:.1f} {real_time:.1f}") 
        else:
            print("solveCycle error!x7",self._actualStep) 
        
        pass
    
    

            
            
   


            
