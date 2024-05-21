# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 11:35:58 2022

@author: veenstra
"""

import os
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
plt.close('all')
import kenmerkendewaarden as kw

# set logging level to INFO to get log messages
# calling basicConfig is essential to set logging level of single module, format is optional
import logging
logging.basicConfig(format='%(message)s')
logging.getLogger("kenmerkendewaarden").setLevel(level="INFO")

try:
    import contextily as ctx # pip install contextily
    ctx_available = True
except ModuleNotFoundError:
    ctx_available = False
try:
    import dfm_tools as dfmt # pip install dfm_tools
    dfmt_available = True
except ModuleNotFoundError:
    dfmt_available = False

# TODO: overview of data improvements: https://github.com/Deltares-research/kenmerkendewaarden/issues/29
# TODO: overview of data issues in https://github.com/Deltares-research/kenmerkendewaarden/issues/4

retrieve_meas_amount = False
plot_meas_amount = False
retrieve_meas = False
plot_meas = False
plot_stations = False
create_summary = False
test = False

# TODO: add timezone to start/stop date? (and re-retrieve all data): https://github.com/Deltares-research/kenmerkendewaarden/issues/29
start_date = "1870-01-01"
end_date = "2024-01-01"
if test:
    start_date = "2021-12-01"
    end_date = "2022-02-01"
    start_date = "2010-12-01"
    end_date = "2022-02-01"

# dir_base = r'p:\11208031-010-kenmerkende-waarden-k\work'
dir_base = r"p:\11210325-005-kenmerkende-waarden\work"
dir_meas = os.path.join(dir_base, f"measurements_wl_{start_date.replace('-','')}_{end_date.replace('-','')}")
os.makedirs(dir_meas, exist_ok=True)
dir_meas_amount = os.path.join(dir_base, f"measurements_amount_wl_{start_date.replace('-','')}_{end_date.replace('-','')}")
os.makedirs(dir_meas_amount, exist_ok=True)


# all stations from TK (dataTKdia)
station_list = ['A12','AWGPFM','BAALHK','BATH','BERGSDSWT','BROUWHVSGT02','BROUWHVSGT08','GATVBSLE','BRESKVHVN','CADZD',
                'D15','DELFZL','DENHDR','EEMSHVN','EURPFM','F16','F3PFM','HARVT10','HANSWT','HARLGN','HOEKVHLD','HOLWD','HUIBGT',
                'IJMDBTHVN','IJMDSMPL','J6','K13APFM','K14PFM','KATSBTN','KORNWDZBTN','KRAMMSZWT','L9PFM','LAUWOG','LICHTELGRE',
                'MARLGT','NES','NIEUWSTZL','NORTHCMRT','DENOVBTN','OOSTSDE04','OOSTSDE11','OOSTSDE14','OUDSD','OVLVHWT','Q1',
                'ROOMPBNN','ROOMPBTN','SCHAARVDND','SCHEVNGN','SCHIERMNOG','SINTANLHVSGR','STAVNSE','STELLDBTN','TERNZN','TERSLNZE','TEXNZE',
                'VLAKTVDRN','VLIELHVN','VLISSGN','WALSODN','WESTKPLE','WESTTSLG','WIERMGDN','YERSKE']
# TODO: maybe add from Dillingh 2013: DORDT, MAASMSMPL, PETTZD, ROTTDM
# station_list = ['BATH','EURPFM','VLISSGN']

locs_meas_ts, _, _ = kw.retrieve_catalog()
for station_name in station_list:
    bool_isstation = locs_meas_ts.index == station_name
    if bool_isstation.sum()!=1:
        print(f'station name {station_name} found {bool_isstation.sum()} times, should be 1:')
        print(f'{locs_meas_ts.loc[bool_isstation,["Naam","Locatie_MessageID","Hoedanigheid.Code"]]}')
        print()


# skip duplicate code stations (hist/realtime) # TODO: avoid this https://github.com/Rijkswaterstaat/wm-ws-dl/issues/12
stations_realtime_hist_dupl = ["BATH", "D15", "J6", "NES"]
"""
# from station_list_tk
station name BATH found 2 times, should be 1:
      Naam  Locatie_MessageID Hoedanigheid.Code
Code                                           
BATH  Bath              10518               NAP
BATH  Bath              13615               NAP

station name D15 found 2 times, should be 1:
                Naam  Locatie_MessageID Hoedanigheid.Code
Code                                                     
D15     D15 platform               6876               MSL
D15   Platform D15-A              10968               MSL

station name J6 found 2 times, should be 1:
             Naam  Locatie_MessageID Hoedanigheid.Code
Code                                                  
J6    J6 platform               5377               MSL
J6    Platform J6              10982               MSL

station name NES found 2 times, should be 1:
     Naam  Locatie_MessageID Hoedanigheid.Code
Code                                          
NES   Nes               5391               NAP
NES   Nes              10309               NAP
"""

# skip MSL/NAP duplicate stations # TODO: avoid this: https://github.com/Rijkswaterstaat/wm-ws-dl/issues/17
stations_nap_mls_dupl = ["EURPFM", "LICHTELGRE"]
"""
# from station_list_tk
station name EURPFM found 2 times, should be 1:
                 Naam  Locatie_MessageID Hoedanigheid.Code
Code                                                      
EURPFM  Euro platform              10946               MSL
EURPFM  Euro platform              10946               NAP

station name LICHTELGRE found 2 times, should be 1:
                          Naam  Locatie_MessageID Hoedanigheid.Code
Code                                                               
LICHTELGRE  Lichteiland Goeree              10953               MSL
LICHTELGRE  Lichteiland Goeree              10953               NAP
"""

stations_dupl = stations_realtime_hist_dupl + stations_nap_mls_dupl


# TODO: missings/duplicates reported in https://github.com/Rijkswaterstaat/wm-ws-dl/issues/39. Some of the duplicates are not retrieved since we use clean_df in ddlpy
# TODO: some stations are now realtime instead of hist (https://github.com/Rijkswaterstaat/wm-ws-dl/issues/20), these are manually skipped in actual data retrieval/statistics

### RETRIEVE MEASUREMENTS AMOUNT
if retrieve_meas_amount:
    kw.retrieve_measurements_amount(dir_output=dir_meas_amount, station_list=station_list, 
                                    start_date=start_date, end_date=end_date)


### PLOT MEASUREMENTS AMOUNT
if plot_meas_amount:
    
    df_amount_ts, df_amount_ext = kw.read_measurements_amount(dir_output=dir_meas_amount)
    
    file_plot = os.path.join(dir_meas_amount, "data_amount")
    
    fig, ax = kw.df_amount_pcolormesh(df_amount_ts, relative=True)
    fig.savefig(file_plot + "_ts_pcolormesh_relative", dpi=200)
    fig, ax = kw.df_amount_pcolormesh(df_amount_ext, relative=True)
    fig.savefig(file_plot + "_ext_pcolormesh_relative", dpi=200)
    
    fig, ax = kw.df_amount_boxplot(df_amount_ts)
    fig.savefig(file_plot + "_ts_boxplot", dpi=200)
    fig, ax = kw.df_amount_boxplot(df_amount_ext)
    fig.savefig(file_plot + "_ext_boxplot", dpi=200)



### RETRIEVE DATA FROM DDL AND WRITE TO NETCDF
for current_station in station_list:
    if not retrieve_meas:
        continue
    
    if current_station in stations_dupl:
        continue
    
    kw.retrieve_measurements(dir_output=dir_meas, station=current_station, extremes=False,
                             start_date=start_date, end_date=end_date)
    kw.retrieve_measurements(dir_output=dir_meas, station=current_station, extremes=True,
                             start_date=start_date, end_date=end_date)
    
### PLOT TIMESERIES DATA
for current_station in station_list:
    if not plot_meas:
        continue
    
    if current_station in stations_dupl:
        continue
    
    print(f'plotting timeseries data for {current_station}')
    
    #load data
    ds_ts_meas = kw.read_measurements(dir_output=dir_meas, station=current_station, extremes=False)
    ds_ext_meas = kw.read_measurements(dir_output=dir_meas, station=current_station, extremes=True)
    
    fig,(ax1, ax2) = kw.plot_measurements(ds=ds_ts_meas, ds_ext=ds_ext_meas)
    
    # save figure
    file_wl_png = os.path.join(dir_meas,f'ts_{current_station}.png')
    ax1.set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date)) # entire period
    fig.savefig(file_wl_png.replace('.png','_alldata.png'))
    ax1.set_xlim(dt.datetime(2000,1,1),dt.datetime(2024,1,1)) # period of interest
    fig.savefig(file_wl_png.replace('.png','_2000_2024.png'))
    plt.close(fig)


if plot_stations:
    #make spatial plot of available/retrieved stations
    crs = 28992
    locs_meas_ts, locs_meas_ext, _ = kw.retrieve_catalog(crs=crs)
    locs_ts_sel = locs_meas_ts.loc[locs_meas_ts.index.isin(station_list)]
    locs_ext_sel = locs_meas_ext.loc[locs_meas_ext.index.isin(station_list)]
    
    fig_map,ax_map = plt.subplots(figsize=(8,8))
    ax_map.plot(locs_ts_sel['X'], locs_ts_sel['Y'],'xk', label="timeseries")#,alpha=0.4) #all ext stations
    ax_map.plot(locs_ext_sel['X'], locs_ext_sel['Y'],'xr', label="extremes") # selected ext stations (stat_list)
    ax_map.legend()
    
    """
    for iR, row in locs_ts_sel.iterrows():
        ax_map.text(row['X'],row['Y'],row.name)
    """
    ax_map.set_xlim(-50000,300000) # RD
    ax_map.set_ylim(350000,850000) # RD
    ax_map.set_title('overview of stations with GETETM2 data')
    ax_map.set_aspect('equal')
    ax_map.set_xlabel(f'X (EPSG:{crs})')
    ax_map.set_ylabel(f'Y (EPSG:{crs})')
    ax_map.grid(alpha=0.5)
    
    # optionally add basemap/coastlines
    if dfmt_available:
        dfmt.plot_coastlines(ax=ax_map, crs=crs)
    elif ctx_available:
        ctx.add_basemap(ax_map, source=ctx.providers.Esri.WorldImagery, crs=crs, attribution=False)
    
    fig_map.tight_layout()
    fig_map.savefig(os.path.join(dir_base,'stations_map.png'), dpi=200)



### CREATE SUMMARY
row_list_ts = []
row_list_ext = []
for current_station in station_list:
    if not create_summary:
        continue
    
    if current_station in stations_dupl:
        continue
    
    print(f'checking data for {current_station}')
    data_summary_row_ts = {}
    data_summary_row_ext = {}
    
    #load measwl data
    # file_wl_nc = os.path.join(dir_meas,f"{current_station}_measwl.nc")
    # if os.path.exists(file_wl_nc):
    ds_ts_meas = kw.read_measurements(dir_output=dir_meas, station=current_station, extremes=False)
    if ds_ts_meas is not None:
        
        meta_dict_flat_ts = kw.get_flat_meta_from_dataset(ds_ts_meas)
        data_summary_row_ts.update(meta_dict_flat_ts)
        
        ds_stats = kw.get_stats_from_dataset(ds_ts_meas)
        data_summary_row_ts.update(ds_stats)
        
        del ds_ts_meas
    

    #load measext data
    # file_ext_nc = os.path.join(dir_meas,f"{current_station}_measext.nc")
    # if os.path.exists(file_ext_nc):
    ds_ext_meas = kw.read_measurements(dir_output=dir_meas, station=current_station, extremes=True)
    if ds_ext_meas is not None:
        
        meta_dict_flat_ext = kw.get_flat_meta_from_dataset(ds_ext_meas)
        data_summary_row_ext.update(meta_dict_flat_ext)
        
        # TODO: warns about extremes being too close for BERGSDSWT, BROUWHVSGT02, BROUWHVSGT08, HOEKVHLD and more
        # TODOTODO: this is partly due to aggers so first convert to 1/2 instead of 1/2/3/4/5
        # TODO: but also due to incorrect data: https://github.com/Rijkswaterstaat/wm-ws-dl/issues/43
        ds_stats = kw.get_stats_from_dataset(ds_ext_meas)
        data_summary_row_ext.update(ds_stats)
        
    
    row_list_ts.append(pd.Series(data_summary_row_ts))
    row_list_ext.append(pd.Series(data_summary_row_ext))

if create_summary:
    data_summary_ts = pd.concat(row_list_ts, axis=1).T
    data_summary_ts = data_summary_ts.set_index('Code').sort_index()
    data_summary_ext = pd.concat(row_list_ext, axis=1).T
    data_summary_ext = data_summary_ext.set_index('Code').sort_index()
    data_summary_ts.to_csv(os.path.join(dir_meas,'data_summary_ts.csv'))
    data_summary_ext.to_csv(os.path.join(dir_meas,'data_summary_ext.csv'))
    



 


    
