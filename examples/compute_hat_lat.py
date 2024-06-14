# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:23:48 2024

@author: veenstra
"""

import os
import pandas as pd
import hatyan
import kenmerkendewaarden as kw

dir_hatyan_data = r'c:\DATA\hatyan\tests\data_unitsystemtests'
# or online at https://github.com/Deltares/hatyan/blob/main/tests/data_unitsystemtests/HOEKVHLD_ana.txt
dir_kw_data = r'c:\DATA\kenmerkendewaarden\tests\testdata'
# or online at https://github.com/Deltares-research/kenmerkendewaarden/blob/main/tests/testdata/HOEK_KW.dia

current_station = 'HOEKVHLD'
methods = ["components", "measurements"]
hatlat = ["HAT","LAT"]
hat_lat_allstats = pd.DataFrame(index=methods, columns=hatlat)

# based on components
print(f'computing hat/lat from components for {current_station}')
file_comp = os.path.join(dir_hatyan_data, f'{current_station}_ana.txt')
comp = hatyan.read_components(filename=file_comp)
print(f'>> components for period {comp.attrs["tstart"]} to {comp.attrs["tstop"]}, SA/SM from 19y')
hat, lat = kw.calc_hat_lat_fromcomponents(comp=comp)
hat_lat_allstats.loc["components","HAT"] = hat
hat_lat_allstats.loc["components","LAT"] = lat

# based on measurements
print(f'computing hat/lat from measurements for {current_station}')
file_meas = os.path.join(dir_kw_data, 'HOEK_KW.dia')
df_meas = hatyan.read_dia(filename=file_meas)
df_meas_19y = df_meas.loc["2001":"2019"]
print(f">> measurements for period {df_meas_19y.index.min()} to {df_meas_19y.index.max()}")
hat, lat = kw.calc_hat_lat_frommeasurements(df_meas_19y)
hat_lat_allstats.loc["measurements","HAT"] = hat
hat_lat_allstats.loc["measurements","LAT"] = lat

# compute range and print
hat_lat_allstats["range"] = hat_lat_allstats["HAT"] - hat_lat_allstats["LAT"]
print(hat_lat_allstats)