# -*- coding: utf-8 -*-

import os
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
plt.close('all')
import kenmerkendewaarden as kw

# set logging level to INFO to get log messages
import logging
logging.basicConfig() # calling basicConfig is essential to set logging level for sub-modules
logging.getLogger("kenmerkendewaarden").setLevel(level="INFO")

# TODO: overview of data improvements: https://github.com/Deltares-research/kenmerkendewaarden/issues/29
# TODO: overview of data issues in https://github.com/Deltares-research/kenmerkendewaarden/issues/4
# TODO: missings/duplicates reported in https://github.com/Rijkswaterstaat/wm-ws-dl/issues/39. Some of the duplicates are not retrieved since we use clean_df in ddlpy

retrieve_meas_amount = False
plot_meas_amount = False
retrieve_meas = False
derive_stats = False
plot_meas = False
plot_stations = False
test = False

# TODO: add timezone to start/stop date? (and re-retrieve all data): https://github.com/Deltares-research/kenmerkendewaarden/issues/29
start_date = "1870-01-01"
end_date = "2024-01-01"
if test:
    start_date = "2021-12-01"
    end_date = "2022-02-01"
    # start_date = "2010-12-01"
    # end_date = "2022-02-01"

# dir_base = r'p:\11208031-010-kenmerkende-waarden-k\work'
dir_base = r"p:\11210325-005-kenmerkende-waarden\work"
dir_meas = os.path.join(dir_base, f"measurements_wl_{start_date.replace('-','')}_{end_date.replace('-','')}")
os.makedirs(dir_meas, exist_ok=True)
dir_meas_amount = os.path.join(dir_base, f"measurements_amount_wl_{start_date.replace('-','')}_{end_date.replace('-','')}")
os.makedirs(dir_meas_amount, exist_ok=True)

# all stations from TK (dataTKdia)
# TODO: maybe add from Dillingh 2013: DORDT, MAASMSMPL, PETTZD, ROTTDM
station_list = ['A12','AWGPFM','BAALHK','BATH','BERGSDSWT','BROUWHVSGT02','BROUWHVSGT08','GATVBSLE','BRESKVHVN','CADZD',
                'D15','DELFZL','DENHDR','EEMSHVN','EURPFM','F16','F3PFM','HARVT10','HANSWT','HARLGN','HOEKVHLD','HOLWD','HUIBGT',
                'IJMDBTHVN','IJMDSMPL','J6','K13APFM','K14PFM','KATSBTN','KORNWDZBTN','KRAMMSZWT','L9PFM','LAUWOG','LICHTELGRE',
                'MARLGT','NES','NIEUWSTZL','NORTHCMRT','DENOVBTN','OOSTSDE04','OOSTSDE11','OOSTSDE14','OUDSD','OVLVHWT','Q1',
                'ROOMPBNN','ROOMPBTN','SCHAARVDND','SCHEVNGN','SCHIERMNOG','SINTANLHVSGR','STAVNSE','STELLDBTN','TERNZN','TERSLNZE','TEXNZE',
                'VLAKTVDRN','VLIELHVN','VLISSGN','WALSODN','WESTKPLE','WESTTSLG','WIERMGDN','YERSKE']
# subset of 11 stations along the coast
station_list = ["VLISSGN","HOEKVHLD","IJMDBTHVN","HARLGN","DENHDR","DELFZL","SCHIERMNOG","VLIELHVN","STELLDBTN","SCHEVNGN","ROOMPBTN"]
# short list for testing
station_list = ["HOEKVHLD"]

# skip duplicate code stations from station_list_tk (hist/realtime) # TODO: avoid this https://github.com/Rijkswaterstaat/wm-ws-dl/issues/12 and https://github.com/Rijkswaterstaat/wm-ws-dl/issues/20
stations_realtime_hist_dupl = ["BATH", "D15", "J6", "NES"]
# skip MSL/NAP duplicate stations from station_list_tk # TODO: avoid this: https://github.com/Rijkswaterstaat/wm-ws-dl/issues/17
stations_nap_mls_dupl = ["EURPFM", "LICHTELGRE"]
stations_dupl = stations_realtime_hist_dupl + stations_nap_mls_dupl


### RETRIEVE MEASUREMENTS AMOUNT
if retrieve_meas_amount:
    kw.retrieve_measurements_amount(dir_output=dir_meas_amount, station_list=station_list, 
                                    start_date=start_date, end_date=end_date,
                                    extremes=False)
    kw.retrieve_measurements_amount(dir_output=dir_meas_amount, station_list=station_list, 
                                    start_date=start_date, end_date=end_date,
                                    extremes=True)


### PLOT MEASUREMENTS AMOUNT
if plot_meas_amount:
    df_amount_ts = kw.read_measurements_amount(dir_output=dir_meas_amount, extremes=False)
    df_amount_ext = kw.read_measurements_amount(dir_output=dir_meas_amount, extremes=True)
    
    file_plot = os.path.join(dir_meas_amount, "data_amount")
    
    fig, ax = kw.plot_measurements_amount(df_amount_ts, relative=True)
    fig.savefig(file_plot + "_ts_pcolormesh_relative", dpi=200)
    fig, ax = kw.plot_measurements_amount(df_amount_ext, relative=True)
    fig.savefig(file_plot + "_ext_pcolormesh_relative", dpi=200)
    


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



### CREATE SUMMARY
if derive_stats:
    stats_ts = kw.derive_statistics(dir_output=dir_meas, station_list=station_list, extremes=False)
    stats_ext = kw.derive_statistics(dir_output=dir_meas, station_list=station_list, extremes=True)
    stats_ts.to_csv(os.path.join(dir_meas,'data_summary_ts.csv'))
    stats_ext.to_csv(os.path.join(dir_meas,'data_summary_ext.csv'))



### PLOT TIMESERIES DATA
for current_station in station_list:
    if not plot_meas:
        continue
    
    if current_station in stations_dupl:
        continue
    print(f'plotting timeseries data for {current_station}')
    
    # load data
    df_ts_meas = kw.read_measurements(dir_output=dir_meas, station=current_station, extremes=False)
    df_ext_meas = kw.read_measurements(dir_output=dir_meas, station=current_station, extremes=True)
    
    # create and save figure
    fig,(ax1, ax2) = kw.plot_measurements(df=df_ts_meas, df_ext=df_ext_meas)
    file_wl_png = os.path.join(dir_meas,f'ts_{current_station}.png')
    ax1.set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date)) # entire period
    fig.savefig(file_wl_png.replace('.png','_alldata.png'))
    ax1.set_xlim(dt.datetime(2000,1,1),dt.datetime(2024,1,1)) # period of interest
    fig.savefig(file_wl_png.replace('.png','_2000_2024.png'))
    plt.close(fig)



### PLOT SELECTION OF AVAILABLE STATIONS ON MAP
if plot_stations:
    station_list_map = station_list.copy()
    if "NORTHCMRT" in station_list_map:
        northcmrt_idx = station_list_map.index("NORTHCMRT")
        station_list_map.pop(northcmrt_idx)
    
    fig, ax = kw.plot_stations(station_list=station_list_map, crs=28992, add_labels=False)
    fig.savefig(os.path.join(dir_base,'stations_map.png'), dpi=200)

