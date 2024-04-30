# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:23:48 2024

@author: veenstra
"""

import os
import pandas as pd
import hatyan
import kenmerkendewaarden as kw

dir_base = r'c:\DATA\hatyan_data_acceptancetests\predictie2019'
# or online at https://github.com/Deltares/hatyan/blob/main/tests/data_unitsystemtests/HOEKVHLD_ana.txt

selected_stations = ['HOEKVHLD']

hat_vallist_allstats = pd.Series(dtype=float, index=selected_stations)
lat_vallist_allstats = pd.Series(dtype=float, index=selected_stations)

for current_station in selected_stations:
    print(f'current_station: {current_station}')
    
    file_comp = os.path.join(dir_base, f'{current_station}_ana.txt')
    
    COMP_merged = hatyan.read_components(filename=file_comp)
    
    hat, lat = kw.calc_hat_lat_fromcomponents(comp=COMP_merged)
    
    hat_vallist_allstats.loc[current_station] = hat
    lat_vallist_allstats.loc[current_station] = lat

print(f'LAT:\n{lat_vallist_allstats}\nHAT:\n{hat_vallist_allstats}')
# hat_vallist_allstats.to_csv('HAT_indication.csv')
# lat_vallist_allstats.to_csv('LAT_indication.csv')
