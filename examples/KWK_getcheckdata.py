# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
plt.close('all')
import kenmerkendewaarden as kw
from kenmerkendewaarden.data_retrieve import retrieve_catalog

# set logging level to INFO to get log messages
import logging
logging.basicConfig() # calling basicConfig is essential to set logging level for sub-modules
logging.getLogger("kenmerkendewaarden").setLevel(level="INFO")

# retrieve fresh DDL catalog
retrieve_catalog(overwrite=False)

# TODO: overview of data improvements: https://github.com/Deltares-research/kenmerkendewaarden/issues/29
# TODO: overview of data issues in https://github.com/Deltares-research/kenmerkendewaarden/issues/4
# TODO: missings/duplicates reported in https://github.com/Rijkswaterstaat/wm-ws-dl/issues/39. Some of the duplicates are not retrieved since we use clean_df in ddlpy

retrieve_meas_amount = False
plot_meas_amount = False
retrieve_meas = False
derive_stats = False
plot_meas = False
plot_stations = True
write_stations_table = True

start_date = pd.Timestamp(1870, 1, 1, tz="UTC+01:00")
start_date = pd.Timestamp(2024, 1, 1, tz="UTC+01:00")
end_date = pd.Timestamp(2024, 2, 1, tz="UTC+01:00")

dir_base = r"p:\11210325-005-kenmerkende-waarden\work"
dir_meas = os.path.join(dir_base, f"measurements_wl_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}")
os.makedirs(dir_meas, exist_ok=True)
dir_meas_amount = os.path.join(dir_base, f"measurements_amount_wl_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}")
os.makedirs(dir_meas_amount, exist_ok=True)

# all stations from TK (dataTKdia)
station_list = ["A12","AWGPFM","BAALHK","BATH","BERGSDSWT","BROUWHVSGT02","BROUWHVSGT08","GATVBSLE","BRESKVHVN","CADZD",
                "D15","DELFZL","DENHDR","EEMSHVN","EURPFM","F16","F3PFM","HARVT10","HANSWT","HARLGN","HOEKVHLD","HOLWD","HUIBGT",
                "IJMDBTHVN","IJMDSMPL","J6","K13APFM","K14PFM","KATSBTN","KORNWDZBTN","KRAMMSZWT","L9PFM","LAUWOG","LICHTELGRE",
                "MARLGT","NES","NIEUWSTZL","NORTHCMRT","DENOVBTN","OOSTSDE04","OOSTSDE11","OOSTSDE14","OUDSD","OVLVHWT","Q1",
                "ROOMPBNN","ROOMPBTN","SCHAARVDND","SCHEVNGN","SCHIERMNOG","SINTANLHVSGR","STAVNSE","STELLDBTN","TERNZN","TERSLNZE","TEXNZE",
                "VLAKTVDRN","VLIELHVN","VLISSGN","WALSODN","WESTKPLE","WESTTSLG","WIERMGDN","YERSKE"]
# subset of 11 stations along the coast
station_list = ["VLISSGN","HOEKVHLD","IJMDBTHVN","HARLGN","DENHDR","DELFZL","SCHIERMNOG","VLIELHVN","STELLDBTN","SCHEVNGN","ROOMPBTN"]
# short list for testing
station_list = ["hoekvanholland"]#,"vlissingen"]

stations_skip = []
# skip duplicate code stations from station_list_tk (hist/realtime)
# TODO: avoid this https://github.com/Rijkswaterstaat/wm-ws-dl/issues/12 and https://github.com/Rijkswaterstaat/wm-ws-dl/issues/20
stations_skip += ["BATH", "D15", "J6", "NES"]
# skip MSL/NAP duplicate stations from station_list_tk
# TODO: avoid this: https://github.com/Rijkswaterstaat/wm-ws-dl/issues/17
stations_skip += ["EURPFM", "LICHTELGRE", "K13APFM"]
# remove stations from station_list
for stat_remove in stations_skip:
    if stat_remove in station_list:
        print(f"removing {stat_remove} from station_list")
        station_list.remove(stat_remove)


### RETRIEVE MEASUREMENTS AMOUNT
if retrieve_meas_amount:
    kw.retrieve_measurements_amount(dir_output=dir_meas_amount, station_list=station_list, 
                                    start_date=start_date, end_date=end_date,
                                    quantity="meas_wl")
    kw.retrieve_measurements_amount(dir_output=dir_meas_amount, station_list=station_list, 
                                    start_date=start_date, end_date=end_date,
                                    quantity="meas_ext")


### PLOT MEASUREMENTS AMOUNT
if plot_meas_amount:
    df_amount_wl = kw.read_measurements_amount(dir_output=dir_meas_amount, quantity="meas_wl")
    df_amount_ext = kw.read_measurements_amount(dir_output=dir_meas_amount, quantity="meas_ext")
    
    file_plot = os.path.join(dir_meas_amount, "data_amount")
    
    fig, ax = kw.plot_measurements_amount(df_amount_wl, relative=True)
    fig.savefig(file_plot + "_wl_pcolormesh_relative", dpi=200)
    fig, ax = kw.plot_measurements_amount(df_amount_ext, relative=True)
    fig.savefig(file_plot + "_ext_pcolormesh_relative", dpi=200)
    


### RETRIEVE DATA FROM DDL AND WRITE TO NETCDF
for current_station in station_list:
    if not retrieve_meas:
        continue
    
    kw.retrieve_measurements(dir_output=dir_meas, station=current_station, quantity="meas_wl",
                             start_date=start_date, end_date=end_date)
    kw.retrieve_measurements(dir_output=dir_meas, station=current_station, quantity="meas_ext",
                             start_date=start_date, end_date=end_date)



### CREATE SUMMARY
if derive_stats:
    stats_wl = kw.derive_statistics(dir_output=dir_meas, station_list=station_list, quantity="meas_wl")
    stats_ext = kw.derive_statistics(dir_output=dir_meas, station_list=station_list, quantity="meas_ext")
    stats_wl.to_csv(os.path.join(dir_meas,'data_summary_wl.csv'))
    stats_ext.to_csv(os.path.join(dir_meas,'data_summary_ext.csv'))



### PLOT TIMESERIES DATA
for current_station in station_list:
    if not plot_meas:
        continue
    
    print(f'plotting timeseries data for {current_station}')
    
    # load data
    df_meas = kw.read_measurements(dir_output=dir_meas, station=current_station, quantity="meas_wl")
    df_ext = kw.read_measurements(dir_output=dir_meas, station=current_station, quantity="meas_ext")
    
    # create and save figure
    fig,(ax1, ax2) = kw.plot_measurements(df_meas=df_meas, df_ext=df_ext)
    file_wl_png = os.path.join(dir_meas,f'ts_{current_station}.png')
    ax1.set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date)) # entire period
    fig.savefig(file_wl_png.replace('.png','_alldata.png'))
    ax1.set_xlim(pd.Timestamp(2000,1,1),pd.Timestamp(2024,1,1)) # period of interest
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



### WRITE CSV WITH STATION CODE/X/Y/EPSG
if write_stations_table:
    # TODO: consider making retrieve_catalog public
    from kenmerkendewaarden.data_retrieve import retrieve_catalog
    locs_meas_wl_all, _, _, _ = retrieve_catalog(crs=4326)
    locs_wl = locs_meas_wl_all.loc[locs_meas_wl_all.index.isin(station_list)]
    file_csv = os.path.join(dir_base, "station_locations.csv")
    locs_wl[["Locatie_MessageID","Lon","Lat","Coordinatenstelsel","Naam"]].to_csv(file_csv)
