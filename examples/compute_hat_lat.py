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



# TODO: decide on hat/lat method
# An alternative (pragmatic) method to compute LAT is to retrieve a long astro timeseries from the DDL and to compute its minimum.
# A faster method would be to retrieve only the astro extremes, if they are all available.
# It is also not sure if all astro timeseries values are available for this entire period, which is why the nvalues column is added to the dataframe:

import ddlpy # available via pip install rws-ddlpy
import pandas as pd

# retrieve astro timeseries for >=19 years and derive minimum from it
start_date = "2000-01-01"
end_date = "2020-01-01"

locations = ddlpy.locations()

# station list from n:\Projects\11206500\11206799\C. Report - advise\3. Natuurmonitoring\2. Showcase Friesche Zeegat - Digitale Systeemrapportage\getijanalyse\TA_wadden_filtersurge.py
station_list = ['WIERMGDN','WESTTSLG','TEXNZE','TERSLNZE','SCHIERMNOG','NES','LAUWOG','HUIBGT','HOLWD','HARLGN','EEMSHVN','DENHDR','DELFZL']
station_list = ['DENHDR','WESTTSLG']

lat_list = []
nvalues_list = []
for current_station in station_list:
    
    bool_grootheid = locations['Grootheid.Code'].isin(['WATHTBRKD']) # measured waterlevels (not astro)
    bool_groepering = locations['Groepering.Code'].isin(['NVT']) # timeseries (not extremes)
    bool_hoedanigheid = locations['Hoedanigheid.Code'].isin(['NAP']) # vertical reference
    bool_station = locations.index.isin([current_station])
    locs_wathte = locations.loc[bool_grootheid & bool_groepering & bool_hoedanigheid & bool_station]
    
    # no support for multiple rows, so pass one at a time
    if len(locs_wathte) != 1:
        raise Exception(f"no or duplicate stations for wathte for {current_station}:\n{locs_wathte}")
    
    # get the measurements
    meas_wathte = ddlpy.measurements(locs_wathte.iloc[0], start_date=start_date, end_date=end_date)
    ts_astro = meas_wathte["Meetwaarde.Waarde_Numeriek"]
    lat_list.append(ts_astro.min())
    nvalues_list.append(len(ts_astro.dropna()))

lat_df = pd.DataFrame({"LAT":lat_list, "nvalues":nvalues_list}, index=station_list)
print(lat_df)
