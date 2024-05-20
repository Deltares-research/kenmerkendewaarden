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
import hatyan # requires hatyan>=2.8.0 for hatyan.ddlpy_to_hatyan() and hatyan.convert_HWLWstr2num()
import xarray as xr
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

meas_amount = True
retrieve_data = False
create_summary = True
test = False


start_date = "1870-01-01" # TODO: add timezone to start/stop date? (and re-retrieve all data): https://github.com/Deltares-research/kenmerkendewaarden/issues/29
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



# station lists
# "KW kust en GR Dillingh 2013" en "KW getijgebied RWS 2011.0", aangevuld met 3 stations AB, aangevuld met BOI wensen, aangevuld met dialijst ABCT
station_list_kw2013 = ['TERNZN','BATH','HANSWT','VLISSGN','BERGSDSWT','KRAMMSZWT',
                       'STAVNSE','ROOMPBNN','CADZD','WESTKPLE','ROOMPBTN','BROUWHVSGT08',
                       'HARVT10','HOEKVHLD','SCHEVNGN','IJMDBTHVN','PETTZD','DENHDR','TEXNZE','TERSLNZE',
                       'WIERMGDN','HUIBGT','OUDSD','VLIELHVN','WESTTSLG','NES','SCHIERMNOG',
                       'DENOVBTN','KORNWDZBTN','HARLGN','LAUWOG','EEMSHVN','DELFZL',
                       'NIEUWSTZL','LICHTELGRE','EURPFM','K13APFM'] + ['DORDT','STELLDBTN','ROTTDM'] + ['MAASMSMPL','OOSTSDE11']
# all stations from TK (dataTKdia)
station_list_tk = ['A12','AWGPFM','BAALHK','BATH','BERGSDSWT','BROUWHVSGT02','BROUWHVSGT08','GATVBSLE','BRESKVHVN','CADZD','D15','DELFZL','DENHDR','EEMSHVN','EURPFM','F16','F3PFM','HARVT10','HANSWT','HARLGN','HOEKVHLD','HOLWD','HUIBGT','IJMDBTHVN','IJMDSMPL','J6','K13APFM','K14PFM','KATSBTN','KORNWDZBTN','KRAMMSZWT','L9PFM','LAUWOG','LICHTELGRE','MARLGT','NES','NIEUWSTZL','NORTHCMRT','DENOVBTN','OOSTSDE04','OOSTSDE11','OOSTSDE14','OUDSD','OVLVHWT','Q1','ROOMPBNN','ROOMPBTN','SCHAARVDND','SCHEVNGN','SCHIERMNOG','SINTANLHVSGR','STAVNSE','STELLDBTN','TERNZN','TERSLNZE','TEXNZE','VLAKTVDRN','VLIELHVN','VLISSGN','WALSODN','WESTKPLE','WESTTSLG','WIERMGDN','YERSKE']
# all files with valid data for 2010 to 2021
station_list_valid2011 = ['BAALHK','BATH','BERGSDSWT','BRESKVHVN','CADZD','DELFZL','DENHDR','DENOVBTN','EEMSHVN','GATVBSLE','HANSWT','HARLGN','HARVT10','HOEKVHLD','IJMDBTHVN','KATSBTN','KORNWDZBTN','KRAMMSZWT','LAUWOG','OUDSD','ROOMPBNN','ROOMPBTN','SCHAARVDND','SCHEVNGN','SCHIERMNOG','STAVNSE','STELLDBTN','TERNZN','VLAKTVDRN','VLIELHVN','VLISSGN','WALSODN','WESTKPLE','WESTTSLG','WIERMGDN']

# station_list = station_list_kw2013
station_list = station_list_tk
# station_list = station_list_valid2011
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

if meas_amount:
    # TODOTODO: introduce kw.retrieve_measurements_amount() somewhere, maybe make modular with station_list
    
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
    if not retrieve_data:
        continue
    
    if current_station in stations_dupl:
        continue
    
    kw.retrieve_measurements(dir_output=dir_meas, station=current_station, extremes=False,
                             start_date=start_date, end_date=end_date)
    kw.retrieve_measurements(dir_output=dir_meas, station=current_station, extremes=True,
                             start_date=start_date, end_date=end_date)
    





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
    
    # add location coordinates to data_summaries
    for sumrow in [data_summary_row_ts, data_summary_row_ext]:
        sumrow['Code'] = locs_meas_ts.loc[current_station].name
        sumrow['RDx'] = locs_meas_ts.loc[current_station,'RDx']
        sumrow['RDy'] = locs_meas_ts.loc[current_station,'RDy']
    
    #load measwl data
    file_wl_nc = os.path.join(dir_meas,f"{current_station}_measwl.nc")
    if not os.path.exists(file_wl_nc):
        ts_available = False
    else:
        ts_available = True
    
    if ts_available:
        ds_ts_meas = xr.open_dataset(file_wl_nc)
        
        meta_dict_flat_ts = kw.get_flat_meta_from_dataset(ds_ts_meas)
        data_summary_row_ts.update(meta_dict_flat_ts)
        
        ds_stats = kw.get_stats_from_dataset(ds_ts_meas)
        data_summary_row_ts.update(ds_stats)
        
        # calculate monthly/yearly mean for meas wl data
        # TODOTODO: use kw.calc_wltidalindicators() instead (with threshold of eg 2900 like slotgem)
        df_meas_values = ds_ts_meas['Meetwaarde.Waarde_Numeriek'].to_pandas()/100
        mean_peryearmonth_long = df_meas_values.groupby(pd.PeriodIndex(df_meas_values.index, freq="M")).mean()
        mean_peryear_long = df_meas_values.groupby(pd.PeriodIndex(df_meas_values.index, freq="Y")).mean()
        
        ts_meas_pd = kw.xarray_to_hatyan(ds_ts_meas)
        del ds_ts_meas
    

    #load measext data
    file_ext_nc = os.path.join(dir_meas,f"{current_station}_measext.nc")
    if not os.path.exists(file_ext_nc):
        ext_available = False
    else:
        ext_available = True
    
    if ext_available:
        ds_ext_meas = xr.open_dataset(file_ext_nc)
        
        meta_dict_flat_ext = kw.get_flat_meta_from_dataset(ds_ext_meas)
        data_summary_row_ext.update(meta_dict_flat_ext)
        
        # TODO: warns about extremes being too close for BERGSDSWT, BROUWHVSGT02, BROUWHVSGT08, HOEKVHLD and more
        # TODOTODO: this is partly due to aggers so first convert to 1/2 instead of 1/2/3/4/5
        # TODO: but also due to incorrect data: https://github.com/Rijkswaterstaat/wm-ws-dl/issues/43
        ds_stats = kw.get_stats_from_dataset(ds_ext_meas)
        data_summary_row_ext.update(ds_stats)
        
        #calculate monthly/yearly mean for meas ext data
        # TODOTODO: make kw function (exact or approximation?), also for timeseries
        ts_meas_ext_pd = kw.xarray_to_hatyan(ds_ext_meas)
        if len(ts_meas_ext_pd['HWLWcode'].unique()) > 2:
            data_pd_HWLW_12 = hatyan.calc_HWLW12345to12(ts_meas_ext_pd) #convert 12345 to 12 by taking minimum of 345 as 2 (laagste laagwater).
            # TODOTODO: currently, first/last values are skipped if LW
        else:
            data_pd_HWLW_12 = ts_meas_ext_pd.copy()
        data_pd_HW = data_pd_HWLW_12.loc[data_pd_HWLW_12['HWLWcode']==1]
        data_pd_LW = data_pd_HWLW_12.loc[data_pd_HWLW_12['HWLWcode']==2]
        #TODOTODO: use kw.calc_HWLWtidalindicators() instead (with threshold of eg 1400 like slotgem)
        HW_mean_peryear_long = data_pd_HW.groupby(pd.PeriodIndex(data_pd_HW.index, freq="y"))['values'].mean()
        LW_mean_peryear_long = data_pd_LW.groupby(pd.PeriodIndex(data_pd_LW.index, freq="y"))['values'].mean()
    
        # # replace 345 HWLWcode with 2, simple approximation of actual LW
        # bool_hwlw_3 = ds_ext_meas['HWLWcode'].isin([3])
        # bool_hwlw_45 = ds_ext_meas['HWLWcode'].isin([4,5])
        # ds_ext_meas_12only = ds_ext_meas.copy()
        # ds_ext_meas_12only['HWLWcode'][bool_hwlw_3] = 2
        # ds_ext_meas_12only = ds_ext_meas_12only.sel(time=~bool_hwlw_45)
        
        # #calculate monthly/yearly mean for meas ext data
        # # TODOTODO: use kw.calc_HWLWtidalindicators() instead (with threshold of eg 1400 like slotgem)
        # data_pd_HW = ds_ext_meas_12only.sel(time=ds_ext_meas_12only['HWLWcode'].isin([1])).to_pandas()['Meetwaarde.Waarde_Numeriek']/100
        # data_pd_LW = ds_ext_meas_12only.sel(time=ds_ext_meas_12only['HWLWcode'].isin([2])).to_pandas()['Meetwaarde.Waarde_Numeriek']/100
        # HW_mean_peryear_long = data_pd_HW.groupby(pd.PeriodIndex(data_pd_HW.index, freq="y")).mean()
        # LW_mean_peryear_long = data_pd_LW.groupby(pd.PeriodIndex(data_pd_LW.index, freq="y")).mean()
    
    row_list_ts.append(pd.Series(data_summary_row_ts))
    row_list_ext.append(pd.Series(data_summary_row_ext))

    #plotting
    if not os.path.exists(file_wl_nc):
        print("[NO DATA, skipping plot]")
        continue
    if os.path.exists(file_ext_nc):
        fig,(ax1,ax2) = hatyan.plot_timeseries(ts=ts_meas_pd, ts_ext=ts_meas_ext_pd)
    else:
        fig,(ax1,ax2) = hatyan.plot_timeseries(ts=ts_meas_pd)
    ax1.set_title(f'timeseries for {current_station}')
    ax1.plot(mean_peryearmonth_long,'c',linewidth=0.7, label='monthly mean')
    ax1.plot(mean_peryear_long,'m',linewidth=0.7, label='yearly mean')
    ax2.plot(mean_peryearmonth_long,'c',linewidth=0.7, label='monthly mean')
    ax2.plot(mean_peryear_long,'m',linewidth=0.7, label='yearly mean')
    if os.path.exists(file_ext_nc):
        ax1.plot(HW_mean_peryear_long,'m',linewidth=0.7, label=None) #'yearly mean HW')
        ax1.plot(LW_mean_peryear_long,'m',linewidth=0.7, label=None) #'yearly mean LW')
    ax1.set_ylim(-4,4)
    ax1.legend(loc=4)
    ax2.legend(loc=1)
    ax2.set_ylim(-0.5,0.5)
    
    # save figure
    file_wl_png = os.path.join(dir_meas,f'ts_{current_station}.png')
    ax1.set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date)) # entire period
    fig.savefig(file_wl_png.replace('.png','_alldata.png'))
    ax1.set_xlim(dt.datetime(2000,1,1),dt.datetime(2024,1,1)) # period of interest
    fig.savefig(file_wl_png.replace('.png','_2000_2024.png'))
    plt.close(fig)
    

if create_summary:
    data_summary_ts = pd.concat(row_list_ts, axis=1).T
    data_summary_ts = data_summary_ts.set_index('Code').sort_index()
    data_summary_ext = pd.concat(row_list_ext, axis=1).T
    data_summary_ext = data_summary_ext.set_index('Code').sort_index()
    data_summary_ts.to_csv(os.path.join(dir_meas,'data_summary_ts.csv'))
    data_summary_ext.to_csv(os.path.join(dir_meas,'data_summary_ext.csv'))
    
    
    #make spatial plot of available/retrieved stations
    fig_map,ax_map = plt.subplots(figsize=(8,7))
    
    ax_map.plot(data_summary_ext['RDx'], data_summary_ext['RDy'],'xk')#,alpha=0.4) #all ext stations
    ax_map.plot(data_summary_ext.loc[data_summary_ext['data_ext'],'RDx'], data_summary_ext.loc[data_summary_ext['data_ext'],'RDy'],'xr') # selected ext stations (stat_list)
    """
    for iR, row in data_summary_ts.iterrows():
        ax_map.text(row['RDx'],row['RDy'],row.name)
    """
    ax_map.set_xlim(-50000,300000)
    ax_map.set_ylim(350000,650000)
    ax_map.set_title('overview of stations with GETETM2 data')
    ax_map.set_aspect('equal')
    ax_map.set_xlabel('X')
    ax_map.set_ylabel('Y')
    ax_map.grid(alpha=0.5)
    
    # optionally add basemap/coastlines
    if dfmt_available:
        dfmt.plot_coastlines(ax=ax_map, crs=crs)
    elif ctx_available:
        ctx.add_basemap(ax_map, source=ctx.providers.Esri.WorldImagery, crs=crs, attribution=False)
    
    fig_map.tight_layout()
    fig_map.savefig(os.path.join(dir_meas,'stations_map.png'), dpi=200)
