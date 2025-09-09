# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 09:17:36 2025

@author: MiguelCamaraSanz
"""

import numpy as np
# --- NÚMEROS CON TUS DATOS ---
D = 3.8
A = np.pi * (D/2)**2
m_full_water_kg = 95600
m_empty_kg = 26490
h_bed = 4.83  
void_fraction=0.4

Volume = (m_full_water_kg - m_empty_kg)/1000                  # 69.11 m3
h_all= Volume / A                                                 # 6.09

Vg=(h_all-h_bed)*A+h_bed*A*void_fraction 

from helpLibs import *

DATA_PATH = r"C:\Users\MiguelCamaraSanz\OneDrive - Fundacion CIRCE\Escritorio\github\ProSimNet\validacion\psa_data.csv"   # <-- AJUSTA ESTO

t0 = "2025-04-20 00:00:01"   
t1 = "2025-04-20 23:59:59"   

df, meta = getPSAdata(DATA_PATH, start=t0,end=t1)

#%%
t0 = "2025-04-20 00:00:01"   
t1 = "2025-04-20 01:00:01"   
plot_raw_pressure(df, start=t0, end=t1)
plot_raw_flows(df, start=t0, end=t1)
plot_raw_temperature(df, start=t0, end=t1)
plot_raw_species(df, start=t0, end=t1)

#%%
# t0 = "2025-04-20 00:35:40"
# t1 = "2025-04-20 00:49:18"
# flows = compute_flows(df,volumes_m3 = Vg)
# 

# plot_raw_species(df, start=t0, end=t1)
# plot_steps_gantt(df, start=t0, end=t1, mode="steps")
# plot_steps_gantt(df, start=t0, end=t1, mode="substeps")
# plot_steps_gantt(df, start=t0, end=t1, mode="inout")
# plot_flows(df, flows,start=t0, end=t1,beds="all", steps=["feed","prz-feed"],units="Nm3_h", figsize=(14, 4.5),)
# plot_flows(df, flows,start=t0, end=t1,beds="all", steps=["purge","bwd-purge"],units="Nm3_h", figsize=(14, 4.5),)






