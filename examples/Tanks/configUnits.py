# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 11:58:26 2025

@author: MiguelCamaraSanz
"""
import sys, os
import numpy as np
from pprint import pprint

ruta_libs = os.path.abspath("../libs")
if ruta_libs not in sys.path:
    sys.path.append(ruta_libs)

from materialLibs import get_properties_gas
from valveLibs import Valve
from tankLibs import Tank
from adsColumnLibs import AdsorptionColumn
from isoLibs import _unpack_isoFuncs

species = ["CO2","CO","O2","N2"]
WM, mu, sigmaLJ, epskB, cpg_molar, cpg_mass, k, H = get_properties_gas(species)
prop_gas = {
    "species": species,
    "MW": WM,
    "mu": mu,
    "sigmaLJ": sigmaLJ,
    "epskB" : epskB,
    "Cp_molar": cpg_molar,
    "Cp_mass": cpg_mass,
    "k": k,
    "H": H}

#TANKES
design_info ={
            "Longitud": 3.0,
            "Diametro": 0.61,
            "Espesor" : 0.02}


initInfo = {"P0":3.5e5,
            "T0":298.0,
            "x0":[0.55, 0.25,0.15,0.05]}


boundaryCInfo = {"Pin":9e5,
                "Tin":350.0,
                "xin":[0.2, 0.8,0,0],
                "Pout":1e5}

thermalInfo = {"adi":True,
            "kw":1e99,
            "hint":1e99,
            "hext":1e99,
            "Tamb":298.15}

tank_1 = Tank(
    Name="T1",
    design_info=design_info,
    prop_gas=prop_gas,
    init_info=initInfo,
    boundary_info=boundaryCInfo,
    thermal_info=thermalInfo)

#tank_1.initialC_info(P0=3.5e5,T0=298.0,x0=[0.5, 0.5,])
#tank_1.boundaryC_info(Pin=9e5,Tin=350.0,xin=[0.2, 0.8],Pout=1e5)
#tank_1.thermal_info(adi=True,kw=1e99,hint=1e99,hext=1e99,Tamb=298.15)

vT1bi = Valve(
    Name="vT1bi",
    Cv_max= 1.35,  
    valve_type="linear",
    a_min = 0.,
    a_max = 1.,
    logic= "linear",
    logic_params={"start": 1, "duration": 1e-6},
    opening_direction="co")

vT1to = Valve(
    Name="vT1to",
    Cv_max= 1.35,  
    valve_type="linear",
    a_min = 0.,
    a_max = 1.,
    logic= "linear",
    logic_params={"start": 99, "duration": 1e-6},
    opening_direction="co")

tank_2 = Tank(
    Name="T2",
    design_info=design_info,
    prop_gas=prop_gas,
    init_info=initInfo,
    boundary_info=boundaryCInfo,
    thermal_info=thermalInfo)

#tank_2.initialC_info(P0=3.5e5,T0=298.15,x0=[0.5, 0.5])
#tank_2.boundaryC_info(Pin=9e5,Tin=350.15,xin=[0.2, 0.8],Pout=1e5)
#tank_2.thermal_info(adi=True,kw=1e99,hint=1e99,hext=1e99,Tamb=298.15)

vT2bi = Valve(
    Name="vT2bi",
    Cv_max= 1.35,  
    valve_type="linear",
    a_min = 0.,
    a_max = 1.,
    logic= "linear",
    logic_params={"start": 1, "duration": 1e-6},
    opening_direction="co")

vT2to = Valve(
    Name="vT2to",
    Cv_max= 1.35,  
    valve_type="linear",
    a_min = 0.,
    a_max = 1.,
    logic= "linear",
    logic_params={"start": 99, "duration": 1e-6},
    opening_direction="co")

vT1bT2b = Valve(
    Name="vT1bT2b",
    Cv_max= 1.35,  
    valve_type="linear",
    a_min = 0.,
    a_max = 1.,
    logic= "linear",
    logic_params={"start": 99, "duration": 1e-6},
    opening_direction="co")

vT1tT2t = Valve(
    Name="vT1tT2t",
    Cv_max= 1.35,  
    valve_type="linear",
    a_min = 0.,
    a_max = 1.,
    logic= "linear",
    logic_params={"start": 99, "duration": 1e-6},
    opening_direction="co")