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


#COLUMNAS DE ADSORCION
design_info ={
            "Longitud": 3.0,  #m
            "Diametro": 0.61, #m
            "Espesor" : 0.02, #m
            "Nodos"   : 10}

packed_info ={
            "Longitud": 2.,  #m
             "Porosidad": 0.33,
             "Tortuosidad": 1.0,
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

initInfo = {"P0":3.5e5,
            "Tg0":298.0,
            "Ts0":298.0,
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

prop_kmtl= {"kmtl":[1,0.,0.,0]}

prop_isoFuns = _unpack_isoFuncs('eNrVU8FKAzEQ/Zec0zCTTDJJrwVZQVpEb2Upy3are9ht3bhexH93VisKVlDoxUOGzMub4eVN8qxyfd90lZqry7y/GPs6mydUWuVDU7dNVvO1WqysAIuVhOW0e0uL93AznRSkSq3avN/sxn7TV10zlV2P1XaoHttaKFdVf9eN7fDH7WeLL/0P1SDt12giMWGAwLJSZNZgAKNnRAKOlBiD194wBOeIgbxgZJsZ+FKvhQuOvGWikDwwRnZTA4iJfAgeMQXCZO0/5FrjY7KRogty8mENkBghrCgw25BIvCF0FixEIUPyXrwJ5dHqhwxi82mXTys5N/j9Gkdp22JSRsI3AROwZ/ZIQc9AciuDd1aGjwkJz46JWcEZFzwhsrwo646ibodmJ7Jscgb0L6KUNcOw6fL0WZbVUv+wypdXtBbbPg==')

adsColumn_1 = AdsorptionColumn(
    Name="C1",
    design_info = design_info,
    packed_info = packed_info,
    prop_gas=prop_gas,
    prop_solid=prop_solid,
    prop_isoFuns=prop_isoFuns,
    init_info=initInfo,
    prop_kmtl=prop_kmtl,
    boundary_info=boundaryCInfo,
    thermal_info=thermalInfo)

adsColumn_2 = AdsorptionColumn(
    Name="C2",
    design_info = design_info,
    packed_info = packed_info,
    prop_gas=prop_gas,
    prop_solid=prop_solid,
    prop_isoFuns=prop_isoFuns,
    init_info=initInfo,
    prop_kmtl=prop_kmtl,
    boundary_info=boundaryCInfo,
    thermal_info=thermalInfo)

adsColumn_3 = AdsorptionColumn(
    Name="C3",
    design_info = design_info,
    packed_info = packed_info,
    prop_gas=prop_gas,
    prop_solid=prop_solid,
    prop_isoFuns=prop_isoFuns,
    init_info=initInfo,
    prop_kmtl=prop_kmtl,
    boundary_info=boundaryCInfo,
    thermal_info=thermalInfo)

adsColumn_4 = AdsorptionColumn(
    Name="C4",
    design_info = design_info,
    packed_info = packed_info,
    prop_gas=prop_gas,
    prop_solid=prop_solid,
    prop_isoFuns=prop_isoFuns,
    init_info=initInfo,
    prop_kmtl=prop_kmtl,
    boundary_info=boundaryCInfo,
    thermal_info=thermalInfo)

adsColumn_5 = AdsorptionColumn(
    Name="C5",
    design_info = design_info,
    packed_info = packed_info,
    prop_gas=prop_gas,
    prop_solid=prop_solid,
    prop_isoFuns=prop_isoFuns,
    init_info=initInfo,
    prop_kmtl=prop_kmtl,
    boundary_info=boundaryCInfo,
    thermal_info=thermalInfo)

adsColumn_6 = AdsorptionColumn(
    Name="C6",
    design_info = design_info,
    packed_info = packed_info,
    prop_gas=prop_gas,
    prop_solid=prop_solid,
    prop_isoFuns=prop_isoFuns,
    init_info=initInfo,
    prop_kmtl=prop_kmtl,
    boundary_info=boundaryCInfo,
    thermal_info=thermalInfo)

adsColumn_7 = AdsorptionColumn(
    Name="C7",
    design_info = design_info,
    packed_info = packed_info,
    prop_gas=prop_gas,
    prop_solid=prop_solid,
    prop_isoFuns=prop_isoFuns,
    init_info=initInfo,
    prop_kmtl=prop_kmtl,
    boundary_info=boundaryCInfo,
    thermal_info=thermalInfo)

adsColumn_8 = AdsorptionColumn(
    Name="C8",
    design_info = design_info,
    packed_info = packed_info,
    prop_gas=prop_gas,
    prop_solid=prop_solid,
    prop_isoFuns=prop_isoFuns,
    init_info=initInfo,
    prop_kmtl=prop_kmtl,
    boundary_info=boundaryCInfo,
    thermal_info=thermalInfo)