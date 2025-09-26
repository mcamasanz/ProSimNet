import sys, os
import numpy as np
from pprint import pprint
import time
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

ruta_libs = os.path.abspath("../libs")
if ruta_libs not in sys.path:
    sys.path.append(ruta_libs)

ruta_libs = os.path.abspath("../../libs")
if ruta_libs not in sys.path:
    sys.path.append(ruta_libs)

from networkLibs import *
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


#COLUMNAS DE ADSORCION
design_info ={
            "Longitud": 3.0,  #m
            "Diametro": 0.61, #m
            "Espesor" : 0.02, #m
            "Nodos"   : 11}

packed_info ={
            "Longitud": 3.,  #m
             "Porosidad": 0.33,
             "Tortuosidad": 1.5,
              }

prop_solid = {
    "Name"   : 'CMS-3K',
    "rho"    : 715,  #kg/m3
    "eps"    : 0.5,  #kg/m3
    "diam"   : 9E-4, #m
    "rp"     : 1E-5, #m
    "sphere" : 1.,
    "cp"     : 1070, # J/kg/K
    "k"      : 0.12 #W/m/k
}

x0 = np.array([0.55, 0.25,0.15,0.05])
initInfo_C1 = {"P0":1.5e5,
            "Tg0":298.0,
            "Ts0":298.0,
            "x0": x0}

x0 = np.array([0.55, 0.25,0.15,0.05])
initInfo_C2 = {"P0":6.5e5,
            "Tg0":298.0,
            "Ts0":298.0,
            "x0": x0}

x0 = np.array([0.55, 0.25,0.15,0.05])
initInfo_C3 = {"P0":9.5e5,
            "Tg0":298.0,
            "Ts0":298.0,
            "x0": x0}


xin = np.array([0.2, 0.8,0,0])
boundaryCInfo = {"Pin":6e5,
                "Tin":350.0,
                "xin": xin,
                "Pout":1e5}

thermalInfo = {"adi":True,
            "kw":1e99,
            "hint":1e99,
            "hext":1e99,
            "Tamb":298.15}

prop_kmtl= {"kmtl":[1,0.,0.,0]}

prop_isoFuns = _unpack_isoFuncs('eNrVU8FKAzEQ/Zec0zCTTDJJrwVZQVpEb2Upy3are9ht3bhexH93VisKVlDoxUOGzMub4eVN8qxyfd90lZqry7y/GPs6mydUWuVDU7dNVvO1WqysAIuVhOW0e0uL93AznRSkSq3avN/sxn7TV10zlV2P1XaoHttaKFdVf9eN7fDH7WeLL/0P1SDt12giMWGAwLJSZNZgAKNnRAKOlBiD194wBOeIgbxgZJsZ+FKvhQuOvGWikDwwRnZTA4iJfAgeMQXCZO0/5FrjY7KRogty8mENkBghrCgw25BIvCF0FixEIUPyXrwJ5dHqhwxi82mXTys5N/j9Gkdp22JSRsI3AROwZ/ZIQc9AciuDd1aGjwkJz46JWcEZFzwhsrwo646ibodmJ7Jscgb0L6KUNcOw6fL0WZbVUv+wypdXtBbbPg==')

adsColumn_1 = AdsorptionColumn(
    Name="AC1",
    design_info = design_info,
    packed_info = packed_info,
    prop_gas=prop_gas,
    prop_solid=prop_solid,
    prop_isoFuns=prop_isoFuns,
    init_info=initInfo_C1,
    prop_kmtl=prop_kmtl,
    boundary_info=boundaryCInfo,
    thermal_info=thermalInfo)

adsColumn_2 = AdsorptionColumn(
    Name="AC2",
    design_info = design_info,
    packed_info = packed_info,
    prop_gas=prop_gas,
    prop_solid=prop_solid,
    prop_isoFuns=prop_isoFuns,
    init_info=initInfo_C2,
    prop_kmtl=prop_kmtl,
    boundary_info=boundaryCInfo,
    thermal_info=thermalInfo)

adsColumn_3 = AdsorptionColumn(
    Name="AC3",
    design_info = design_info,
    packed_info = packed_info,
    prop_gas=prop_gas,
    prop_solid=prop_solid,
    prop_isoFuns=prop_isoFuns,
    init_info=initInfo_C3,
    prop_kmtl=prop_kmtl,
    boundary_info=boundaryCInfo,
    thermal_info=thermalInfo)

vAC1bi = Valve(
    Name="vAC1bi",
    Cv_max= 0.1,  
    valve_type="linear",
    a_min = 0.,
    a_max = 1.,
    logic= "linear",
    logic_params={"start": 1, "duration": 75},
    opening_direction="co")

vAC1to = Valve(
    Name="vAC1to",
    Cv_max= 0.1,  
    valve_type="linear",
    a_min = 0.,
    a_max = 1.,
    logic= "linear",
    logic_params={"start": 1e99, "duration": 1e99},
    opening_direction="co")

vAC2bi = Valve(
    Name="vAC2bi",
    Cv_max= 0.1,  
    valve_type="linear",
    a_min = 0.,
    a_max = 1.,
    logic= "linear",
    logic_params={"start": 1e99, "duration": 1e99},
    opening_direction="co")

vAC2to = Valve(
    Name="vAC2to",
    Cv_max= 0.1,  
    valve_type="linear",
    a_min = 0.,
    a_max = 1.,
    logic= "linear",
    logic_params={"start": 1, "duration": 75},
    opening_direction="co")

vAC3bi = Valve(
    Name="vAC3bi",
    Cv_max= 0.1,  
    valve_type="linear",
    a_min = 0.,
    a_max = 1.,
    logic= "linear",
    logic_params={"start": 1e99, "duration": 1e99},
    opening_direction="co")

vAC3to = Valve(
    Name="vAC3to",
    Cv_max= 0.1,  
    valve_type="linear",
    a_min = 0.,
    a_max = 1.,
    logic= "linear",
    logic_params={"start": 1e99, "duration": 1e99},
    opening_direction="co")

vAC1bAC2b = Valve(
    Name="vAC1bAC2b",
    Cv_max= 0.1,  
    valve_type="linear",
    a_min = 0.,
    a_max = 1.,
    logic= "linear",
    logic_params={"start": 1e99, "duration": 1e99},
    opening_direction="co")

vAC1bAC3b = Valve(
    Name="vAC1bAC3b",
    Cv_max= 1.5,  
    valve_type="linear",
    a_min = 0.,
    a_max = 1.,
    logic= "linear",
    logic_params={"start": 1e99, "duration": 1e99},
    opening_direction="co")

valves=[vAC1bi,vAC1to]
units=[adsColumn_1]
sim=Network(prop_gas=prop_gas,
            Units=units,
            Valves=valves)


#%%
sim._run(saveData=5,
    startTime=0,
    endTime=60,
    solver='BDF',
    atol=1e-12,
    rtol=1e-12,
    plot=True,
    logBal=False)