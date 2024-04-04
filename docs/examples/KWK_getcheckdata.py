# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 11:35:58 2022

@author: veenstra
"""

import os
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
import ddlpy
#import contextily as ctx # pip install contextily
import hatyan

#TODO: convert to netcdf instead of pkl (dfm_tools ssh retrieve format is the same as DCSM, could be useful)
#TODO: visually check availability (start/stop/gaps/aggers) of wl/ext, monthmean wl, outliers (nog niet gedaan voor hele periode, wel voor 2000-2022 (listAB+HARVT10): https://github.com/Deltares-research/kenmerkendewaarden/issues/10

retrieve_data = True
create_summary = True

tstart_dt_DDL = dt.datetime(1870,1,1) #1870,1,1 for measall folder
tstart_dt_DDL = dt.datetime(2021,12,1)
tstop_dt_DDL = dt.datetime(2022,1,1)
tstart_dt = dt.datetime(2001,1,1)
tstop_dt = dt.datetime(2011,1,1)

NAP2005correction = False #True #TODO: define for all stations
if ((tstop_dt.year-tstart_dt.year)==10) & (tstop_dt.month==tstop_dt.day==tstart_dt.month==tstart_dt.day==1):
    year_slotgem = tstop_dt.year
else:
    year_slotgem = 'invalid'
print(f'year_slotgem: {year_slotgem}')

# dir_base = r'p:\11208031-010-kenmerkende-waarden-k\work'
dir_base = r"p:\11210325-005-kenmerkende-waarden\work"
# dir_meas = os.path.join(dir_base,'measurements_wl_18700101_20240101')
# dir_meas_alldata = os.path.join(dir_base,'measurements_wl_18700101_20240101')
    
dir_meas_ddl = os.path.join(dir_base,f"measurements_wl_{tstart_dt_DDL.strftime('%Y%m%d')}_{tstop_dt_DDL.strftime('%Y%m%d')}")
os.makedirs(dir_meas_ddl, exist_ok=True)

fig_alltimes_ext = [dt.datetime.strptime(x,'%Y%m%d') for x in os.path.basename(dir_meas_ddl).split('_')[2:4]]

file_catalog_pkl = os.path.join(dir_base, 'DDL_catalog.pkl')
if not os.path.exists(file_catalog_pkl): #TODO: prefferably speed up catalog retrieval
    print('retrieving DDL locations catalog with ddlpy')
    locations = ddlpy.locations()
    pd.to_pickle(locations, file_catalog_pkl)
else:
    print('loading DDL locations catalog from pickle')
    locations = pd.read_pickle(file_catalog_pkl)
print('...done')

bool_grootheid = locations["Grootheid.Code"].isin(["WATHTE"])
bool_groepering_ts = locations["Groepering.Code"].isin(["NVT"])
bool_groepering_ext = locations["Groepering.Code"].isin(["GETETM2","GETETMSL2"]) # TODO: why distinction between MSL and NAP? This is already filtered via Hoedanigheid
bool_hoedanigheid_nap = locations["Hoedanigheid.Code"].isin(["NAP"])
bool_hoedanigheid_msl = locations["Hoedanigheid.Code"].isin(["MSL"])
# TODO: we cannot subset Typering.Code==GETETTPE here (not present in dataframe), so we use Grootheid.Code==NVT: https://github.com/Rijkswaterstaat/wm-ws-dl/issues/19
bool_grootheid_exttypes = locations['Grootheid.Code'].isin(['NVT'])

# these selections correspond to cat_aquometadatalijst_sel/cat_locatielijst_sel from old hatyan ddl methods
locs_meas_ts_nap = locations.loc[bool_grootheid & bool_groepering_ts & bool_hoedanigheid_nap]
locs_meas_ext_nap = locations.loc[bool_grootheid & bool_groepering_ext & bool_hoedanigheid_nap]
locs_meas_exttype_nap = locations.loc[bool_grootheid_exttypes & bool_groepering_ext]

# find all stations in list # TODO: way too much matches so move to station code instead (still Bath will be duplicated and maybe others)
# stat_name_list = ['BATH','DELFZIJL','DEN HELDER','DORDRECHT','EEMSHAVEN','EURO PLATFORM','HANSWEERT','HARINGVLIETSLUIZEN','HARLINGEN','HOEK VAN HOLLAND','HUIBERTGAT','IJMUIDEN','KORNWERDERZAND','LAUWERSOOG','ROOMPOT BUITEN','ROTTERDAM','SCHEVENINGEN','STAVENISSE','TERNEUZEN','VLISSINGEN','WEST-TERSCHELLING'] # lijst AB
stat_name_list = ['Terneuzen','Bath','HANSWT','Vlissingen','Bergse Diepsluis west','Krammersluizen west','Stavenisse','Roompot binnen','Cadzand','Westkapelle','Roompot buiten','Brouwershavensche Gat 08','Haringvliet 10','Hoek van Holland','Scheveningen','IJmuiden buitenhaven','Petten zuid','Den Helder','Texel Noordzee','Terschelling Noordzee','Wierumergronden','Huibertgat','Oudeschild','Vlieland haven','West-Terschelling','Nes','Schiermonnikoog','Den Oever buiten','Kornwerderzand buiten','Harlingen','Lauwersoog','Eemshaven','Delfzijl','Nieuwe Statenzijl','Lichteiland Goeree','Euro platform','K13a platform'] + ['Dordrecht','Stellendam Buiten','Rotterdam'] + ['Maasmond','Oosterschelde 11'] #+ stat_list_addnonext[2:] #"KW kust en GR Dillingh 2013" en "KW getijgebied RWS 2011.0", aangevuld met 3 stations AB, aangevuld met BOI wensen, aangevuld met dialijst ABCT
stat_list = []
for stat_name in stat_name_list:
    bool_isstation = locs_meas_ts_nap['Naam'].str.contains(stat_name,case=False) | locs_meas_ts_nap.index.str.contains(stat_name,case=False)
    if bool_isstation.sum()!=1:
        print(f'station name {stat_name} found {bool_isstation.sum()} times, should be 1:')
        print(f'{locs_meas_ts_nap.loc[bool_isstation,["Naam","Locatie_MessageID"]]}')
        print()
    if bool_isstation.sum()==0: #skip if none found
        continue
    stat_list.append(locs_meas_ts_nap.loc[bool_isstation].index[0])
    #print(f'{stat_name:30s}: {bool_isstation.sum()}')
#stat_list = ['BATH','DELFZL','DENHDR','DORDT','EEMSHVN','EURPFM','HANSWT','STELLDBTN','HARLGN','HOEKVHLD','HUIBGT','IJMDBTHVN','KORNWDZBTN','LAUWOG','ROOMPBTN','ROTTDM','SCHEVNGN','STAVNSE','TERNZN','VLISSGN','WESTTSLG'] # lijst AB vertaald naar DONAR namen
stat_list = ['HOEKVHLD','HARVT10','VLISSGN']
# dataTKdia station selections
# stat_list = ['A12','AWGPFM','BAALHK','BATH','BERGSDSWT','BROUWHVSGT02','BROUWHVSGT08','GATVBSLE','BRESKVHVN','CADZD','D15','DELFZL','DENHDR','EEMSHVN','EURPFM','F16','F3PFM','HARVT10','HANSWT','HARLGN','HOEKVHLD','HOLWD','HUIBGT','IJMDBTHVN','IJMDSMPL','J6','K13APFM','K14PFM','KATSBTN','KORNWDZBTN','KRAMMSZWT','L9PFM','LAUWOG','LICHTELGRE','MARLGT','NES','NIEUWSTZL','NORTHCMRT','DENOVBTN','OOSTSDE04','OOSTSDE11','OOSTSDE14','OUDSD','OVLVHWT','Q1','ROOMPBNN','ROOMPBTN','SCHAARVDND','SCHEVNGN','SCHIERMNOG','SINTANLHVSGR','STAVNSE','STELLDBTN','TERNZN','TERSLNZE','TEXNZE','VLAKTVDRN','VLIELHVN','VLISSGN','WALSODN','WESTKPLE','WESTTSLG','WIERMGDN','YERSKE'] #all stations from TK
# stat_list = ['BAALHK','BATH','BERGSDSWT','BRESKVHVN','CADZD','DELFZL','DENHDR','DENOVBTN','EEMSHVN','GATVBSLE','HANSWT','HARLGN','HARVT10','HOEKVHLD','IJMDBTHVN','KATSBTN','KORNWDZBTN','KRAMMSZWT','LAUWOG','OUDSD','ROOMPBNN','ROOMPBTN','SCHAARVDND','SCHEVNGN','SCHIERMNOG','STAVNSE','STELLDBTN','TERNZN','VLAKTVDRN','VLIELHVN','VLISSGN','WALSODN','WESTKPLE','WESTTSLG','WIERMGDN'] #all files with valid data for 2010 to 2021
# stat_list = stat_list[stat_list.index('STELLDBTN'):]
M2_period_timedelta = pd.Timedelta(hours=hatyan.schureman.get_schureman_freqs(['M2']).loc['M2','period [hr]'])



### RETRIEVE DATA FROM DDL AND WRITE TO PICKLE
for current_station in stat_list:
    if not retrieve_data:
        continue
    
    #TODO: write to netcdf instead (including metadata)
    file_wl_pkl = os.path.join(dir_meas_ddl,f"{current_station}_measwl.pkl")
    file_ext_pkl = os.path.join(dir_meas_ddl,f"{current_station}_measext.pkl")
    
    bool_station_ts = locs_meas_ts_nap.index.isin([current_station])
    bool_station_ext = locs_meas_ext_nap.index.isin([current_station])
    bool_station_exttype = locs_meas_exttype_nap.index.isin([current_station])
    loc_meas_ts_nap_one = locs_meas_ts_nap.loc[bool_station_ts]
    loc_meas_ext_nap_one = locs_meas_ext_nap.loc[bool_station_ext]
    loc_meas_exttype_nap_one = locs_meas_exttype_nap.loc[bool_station_exttype]

    if len(loc_meas_ts_nap_one)!=1:
        raise ValueError(f"no or multiple stations present after station subsetting:\n{loc_meas_ts_nap_one}")
    if len(loc_meas_ext_nap_one)!=1:
        raise ValueError(f"no or multiple stations present after station subsetting:\n{loc_meas_ext_nap_one}")
    
    #retrieving waterlevels
    if os.path.exists(file_wl_pkl):
        print(f'measwl data for {current_station} already available in {os.path.basename(dir_meas_ddl)}')
    else:
        print(f'retrieving measwl data from DDL for {current_station} to {os.path.basename(dir_meas_ddl)}')
        measurements_ts = ddlpy.measurements(location=loc_meas_ts_nap_one.iloc[0], start_date=tstart_dt_DDL, end_date=tstop_dt_DDL)
        ts_meas_pd = hatyan.ddlpy_to_hatyan(measurements_ts)
        ts_meas_pd.to_pickle(file_wl_pkl)
    
    #retrieving measured extremes
    if os.path.exists(file_ext_pkl):
        print(f'measext data for {current_station} already available in {os.path.basename(dir_meas_ddl)}')
    else:
        print(f'retrieving measext data from DDL for {current_station} to {os.path.basename(dir_meas_ddl)}')
        measurements_ext = ddlpy.measurements(location=loc_meas_ext_nap_one.iloc[0], start_date=tstart_dt_DDL, end_date=tstop_dt_DDL)
        measurements_exttyp = ddlpy.measurements(location=loc_meas_exttype_nap_one.iloc[0], start_date=tstart_dt_DDL, end_date=tstop_dt_DDL)
        ts_meas_extval_pd = hatyan.ddlpy_to_hatyan(measurements_ext)
        ts_meas_exttype_pd = hatyan.ddlpy_to_hatyan(measurements_exttyp)
        ts_meas_ext_pd = hatyan.convert_HWLWstr2num(ts_meas_extval_pd, ts_meas_exttype_pd)
        ts_meas_ext_pd.to_pickle(file_ext_pkl)


"""
not in M2phasediff textfile (in hatyan): ['LICHTELGRE','EURPFM']
HW/LW numbers not always increasing: ['HANSWT','BROUWHVSGT08','PETTZD','DORDT']
no extremes in requested time frame: ['STELLDBTN','OOSTSDE11']
Catalog query yielded no results (no ext available like K13APFM): A12
"""
data_summary = pd.DataFrame(index=stat_list).sort_index()
#TODO: make row per station and append to list (concat after loop), probably solves warnings
# >>Value 'False' has dtype incompatible with float64, please explicitly cast to a compatible dtype first: data_summary.loc[current_station,'data_ext'] = False
for current_station in stat_list:
    if not create_summary:
        continue
    print(f'checking data for {current_station}')
    list_relevantmetadata = ['WaardeBepalingsmethode.Code','WaardeBepalingsmethode.Omschrijving','MeetApparaat.Code','MeetApparaat.Omschrijving','Hoedanigheid.Code','Grootheid.Code','Groepering.Code','Typering.Code']
    list_relevantDDLdata = ['WaardeBepalingsmethode.Code','MeetApparaat.Code','MeetApparaat.Omschrijving','Hoedanigheid.Code']
    
    # TODO: re-enable after pickle to netcdf conversion (including metadata)
    # station_dict = dict(cat_locatielijst_sel_codeidx.loc[current_station,['Naam','Code']])
    # cat_aquometadatalijst_temp, cat_locatielijst_temp = hatyan.get_DDL_stationmetasubset(catalog_dict=catalog_dict,station_dict=station_dict,meta_dict={'Grootheid.Code':'WATHTE','Groepering.Code':'NVT'})
    # for metakey in list_relevantDDLdata:
    #     data_summary.loc[current_station,f'DDL_{metakey}_wl'] = '|'.join(cat_aquometadatalijst_temp[metakey].unique())
    # if not current_station in ['K13APFM','MAASMSMPL']:# no ext available for these stations
    #     cat_aquometadatalijst_temp, cat_locatielijst_temp = hatyan.get_DDL_stationmetasubset(catalog_dict=catalog_dict,station_dict=station_dict,meta_dict={'Grootheid.Code':'WATHTE','Groepering.Code':'GETETM2'})
    #     for metakey in list_relevantDDLdata:
    #         data_summary.loc[current_station,f'DDL_{metakey}_ext'] = '|'.join(cat_aquometadatalijst_temp[metakey].unique())
    
    #add coordinates to data_summary
    # TODO: maybe sometimes not present in ts, then take from ext
    data_summary.loc[current_station,['X','Y']] = locs_meas_ts_nap.loc[current_station,['X','Y']]
    # data_summary.loc[current_station,['Coordinatenstelsel']] = locs_meas_ts_nap.loc[current_station,['Coordinatenstelsel']]
    time_interest_start = dt.datetime(2000,1,1)
    time_interest_stop = dt.datetime(2021,2,1)
    
    #load measwl data
    file_wl_pkl = os.path.join(dir_meas_ddl,f"{current_station}_measwl.pkl")
    # file_wlmeta_pkl = os.path.join(dir_meas_alldata,f"meta_{current_station}_measwl.pkl")
    if not os.path.exists(file_wl_pkl):
        data_summary.loc[current_station,'data_wl'] = False
        data_summary.loc[current_station,'data_ext'] = False
        continue
    data_summary.loc[current_station,'data_wl'] = True
    ts_meas_pd = pd.read_pickle(file_wl_pkl)
    # TODO: re-enable after pickle to netcdf conversion (including metadata)
    # metawl = pd.read_pickle(file_wlmeta_pkl)
    # for metakey in list_relevantmetadata:
    #     data_summary.loc[current_station,f'{metakey}_wl'] = '|'.join(metawl[metakey].unique())
    ts_meas_pd = ts_meas_pd[['values','QC']] # reduces the memory consumption significantly
    # TODO: re-check timezone later on
    # if str(ts_meas_pd.index[0].tz) != 'Etc/GMT-1': #this means UTC+1
    #     raise Exception(f'measwl data for {current_station} is not in expected timezone (Etc/GMT-1): {ts_meas_pd.index[0].tz}')
    ts_meas_pd.index = ts_meas_pd.index.tz_localize(None)
    bool_99 = ts_meas_pd['QC']==99
    if bool_99.any(): #ts contains invalid values
        ts_meas_pd[bool_99] = np.nan
    data_summary.loc[current_station,'tstart_wl'] = ts_meas_pd.index[0]
    data_summary.loc[current_station,'tstop_wl'] = ts_meas_pd.index[-1]
    data_summary.loc[current_station,'tstart2000_wl'] = ts_meas_pd.index[0]<=time_interest_start
    data_summary.loc[current_station,'tstop202102_wl'] = ts_meas_pd.index[-1]>=time_interest_stop
    data_summary.loc[current_station,'nvals_wl'] = len(ts_meas_pd['values'])
    data_summary.loc[current_station,'#nans_wl'] = bool_99.sum()
    data_summary.loc[current_station,'min_wl'] = ts_meas_pd['values'].min()
    data_summary.loc[current_station,'max_wl'] = ts_meas_pd['values'].max()
    data_summary.loc[current_station,'std_wl'] = ts_meas_pd['values'].std()
    data_summary.loc[current_station,'mean_wl'] = ts_meas_pd['values'].mean()
    ts_meas_dupltimes = ts_meas_pd.index.duplicated()
    data_summary.loc[current_station,'dupltimes_wl'] = ts_meas_dupltimes.sum()
    #count #nans for duplicated times, happens at HARVT10/HUIBGT/STELLDBTN
    data_summary.loc[current_station,'#nans_dupltimes_wl'] = ts_meas_pd.loc[ts_meas_pd.index.duplicated(keep=False),'values'].isnull().sum()
    
    #calc #nan-values in recent period
    ts_meas_2000to202102a = ts_meas_pd.loc[~ts_meas_dupltimes,['values']].loc[time_interest_start:min(ts_meas_pd.index[-1],time_interest_stop)]
    ts_meas_2000to202102b = pd.DataFrame({'values':ts_meas_pd.loc[~ts_meas_dupltimes,'values']},index=pd.date_range(start=time_interest_start,end=time_interest_stop,freq='10min'))
    data_summary.loc[current_station,'#nans_2000to202102a_wl'] = ts_meas_2000to202102a['values'].isnull().sum()
    data_summary.loc[current_station,'#nans_2000to202102b_wl'] = ts_meas_2000to202102b['values'].isnull().sum()
    
    #calculate monthly/yearly mean for meas wl data #TODO: use hatyan.calc_wltidalindicators() instead (with threshold of eg 2900 like slotgem)
    mean_peryearmonth_long = ts_meas_pd.groupby(pd.PeriodIndex(ts_meas_pd.index, freq="M"))['values'].mean()
    data_summary.loc[current_station,'monthmean_mean_wl'] = mean_peryearmonth_long.mean()
    data_summary.loc[current_station,'monthmean_std_wl'] = mean_peryearmonth_long.std()
    mean_peryear_long = ts_meas_pd.groupby(pd.PeriodIndex(ts_meas_pd.index, freq="Y"))['values'].mean()
    data_summary.loc[current_station,'yearmean_mean_wl'] = mean_peryear_long.mean()
    data_summary.loc[current_station,'yearmean_std_wl'] = mean_peryear_long.std()
    """#TODO: move to function. Add minimum # values to calculate monthmean? Make long2array edit simpler with pandas smart stuff?
    numvals_peryearmonth_long = ts_meas_pd.groupby(pd.PeriodIndex(ts_meas_pd.index, freq="M"))['values'].count()
    mean_peryearmonth_array = pd.DataFrame(index=range(1,13))
    for year in mean_peryearmonth_long.index.year.unique():
        bool_year = mean_peryearmonth_long.index.year==year
        mean_peryearmonth_long_oneyr = mean_peryearmonth_long.loc[bool_year]
        mean_peryearmonth_array.loc[mean_peryearmonth_long_oneyr.index.month,year] = mean_peryearmonth_long_oneyr.values
    mean_permonth = mean_peryearmonth_array.mean(axis=1)
    """

    #load measext data
    file_ext_pkl = os.path.join(dir_meas_ddl,f"{current_station}_measext.pkl")
    file_extmeta_pkl = os.path.join(dir_meas_ddl,f"meta_{current_station}_measext.pkl")
    if not os.path.exists(file_ext_pkl):
        data_summary.loc[current_station,'data_ext'] = False
    else:
        data_summary.loc[current_station,'data_ext'] = True
        ts_meas_ext_pd = pd.read_pickle(file_ext_pkl)
        timediff_ext = ts_meas_ext_pd.index[1:]-ts_meas_ext_pd.index[:-1]
        if timediff_ext.min() < dt.timedelta(hours=4): #TODO: min timediff for e.g. BROUWHVSGT08 is 3 minutes: ts_meas_ext_pd.loc[dt.datetime(2015,1,1):dt.datetime(2015,1,2),['values', 'QC', 'Status']]. This should not happen and with new dataset should be converted to an error
            print(f'WARNING: extreme data contains values that are too close ({timediff_ext.min()}), should be at least 4 hours difference')
        # TODO: re-enable after pickle to netcdf conversion (including metadata)
        # metaext = pd.read_pickle(file_extmeta_pkl)
        # for metakey in list_relevantmetadata:
        #     data_summary.loc[current_station,f'{metakey}_ext'] = '|'.join(metaext[metakey].unique())
        # TODO: re-check timezone later on
        # if str(ts_meas_ext_pd.index[0].tz) != 'Etc/GMT-1': #this means UTC+1
        #     raise Exception(f'measext data for {current_station} is not in expected timezone (Etc/GMT-1): {ts_meas_ext_pd.index[0].tz}')
        ts_meas_ext_pd.index = ts_meas_ext_pd.index.tz_localize(None)
        ts_meas_ext_dupltimes = ts_meas_ext_pd.index.duplicated()
        data_summary.loc[current_station,'mintimediff_ext'] = timediff_ext.min()
        data_summary.loc[current_station,'dupltimes_ext'] = ts_meas_ext_dupltimes.sum()
        data_summary.loc[current_station,'tstart_ext'] = ts_meas_ext_pd.index[0]
        data_summary.loc[current_station,'tstop_ext'] = ts_meas_ext_pd.index[-1]
        data_summary.loc[current_station,'tstart2000_ext'] = ts_meas_ext_pd.index[0]<=(time_interest_start+M2_period_timedelta)
        data_summary.loc[current_station,'tstop202102_ext'] = ts_meas_ext_pd.index[-1]>=(time_interest_stop-M2_period_timedelta)
        data_summary.loc[current_station,'nvals_ext'] = len(ts_meas_ext_pd['values'])
        data_summary.loc[current_station,'min_ext'] = ts_meas_ext_pd['values'].min()
        data_summary.loc[current_station,'max_ext'] = ts_meas_ext_pd['values'].max()
        data_summary.loc[current_station,'std_ext'] = ts_meas_ext_pd['values'].std()
        data_summary.loc[current_station,'mean_ext'] = ts_meas_ext_pd['values'].mean()
        if len(ts_meas_ext_pd['HWLWcode'].unique()) > 2:
            data_summary.loc[current_station,'aggers_ext'] = True
        else:
            data_summary.loc[current_station,'aggers_ext'] = False
        try:
            ts_meas_ext_2000to202102 = ts_meas_ext_pd.loc[(ts_meas_ext_pd.index>=time_interest_start) & (ts_meas_ext_pd.index<=time_interest_stop)]
            ts_meas_ext_pd.loc[dt.datetime(2015,1,1):dt.datetime(2015,1,2)]
            ts_meas_ext_2000to202102 = hatyan.calc_HWLWnumbering(ts_meas_ext_2000to202102)
            HWmissings = (ts_meas_ext_2000to202102.loc[ts_meas_ext_pd['HWLWcode']==1,'HWLWno'].diff().dropna()!=1).sum()
            data_summary.loc[current_station,'#HWgaps_2000to202102_ext'] = HWmissings
        except Exception as e: #"tidal wave numbering: HW/LW numbers not always increasing" and "zero-size array to reduction operation minimum which has no identity" #TODO: fix by calulate and providing station or corr_tideperiods argument? Or fix otherwise in hatyan (maybe under different project)
            print(f'ERROR: {e}')
        
        #calculate monthly/yearly mean for meas ext data
        if len(ts_meas_ext_pd['HWLWcode'].unique()) > 2:
            data_pd_HWLW_12 = hatyan.calc_HWLW12345to12(ts_meas_ext_pd) #convert 12345 to 12 by taking minimum of 345 as 2 (laagste laagwater). TODO: currently, first/last values are skipped if LW
        else:
            data_pd_HWLW_12 = ts_meas_ext_pd.copy()
        data_pd_HW = data_pd_HWLW_12.loc[data_pd_HWLW_12['HWLWcode']==1]
        data_pd_LW = data_pd_HWLW_12.loc[data_pd_HWLW_12['HWLWcode']==2]
        HW_mean_peryear_long = data_pd_HW.groupby(pd.PeriodIndex(data_pd_HW.index, freq="y"))['values'].mean() #TODO: use hatyan.calc_HWLWtidalindicators() instead (with threshold of eg 1400 like slotgem)
        LW_mean_peryear_long = data_pd_LW.groupby(pd.PeriodIndex(data_pd_LW.index, freq="y"))['values'].mean()
        
    if data_summary['data_ext'].isnull().sum() == 0: #if all stat_list stations were processed (only True/False in this array, no nans)
        #print and save data_summary
        print(data_summary[['data_wl','tstart_wl','tstop_wl','nvals_wl','dupltimes_wl','#nans_wl','#nans_2000to202102a_wl']])
        try:
            print(data_summary[['data_ext','dupltimes_ext','#HWgaps_2000to202102_ext']])
        except KeyError:
            print(data_summary[['data_ext','dupltimes_ext']])            
        data_summary.to_csv(os.path.join(dir_meas_ddl,'data_summary.csv'))
        
        #make spatial plot of available/retrieved stations
        fig_map,ax_map = plt.subplots(figsize=(8,7))
        file_ldb = r'p:\11206813-006-kpp2021_rmm-2d\C_Work\31_RMM_FMmodel\computations\model_setup\run_205\20101209-06.ldb' #TODO: make ldb available in code or at least KWK project drive
        if os.path.exists(file_ldb):
            ldb_pd = pd.read_csv(file_ldb, delim_whitespace=True,skiprows=4,names=['RDx','RDy'],na_values=[999.999])
            ax_map.plot(ldb_pd['RDx'],ldb_pd['RDy'],'-k',linewidth=0.4)
        ax_map.plot(data_summary['X'],data_summary['Y'],'xk')#,alpha=0.4) #all ext stations
        ax_map.plot(data_summary.loc[stat_list,'X'],data_summary.loc[stat_list,'Y'],'xr') # selected ext stations (stat_list)
        ax_map.plot(data_summary.loc[data_summary['data_ext'],'X'],data_summary.loc[data_summary['data_ext'],'Y'],'xm') # data retrieved
        """
        for iR, row in cat_locatielijst_sel.iterrows():
            ax_map.text(row['RDx'],row['RDy'],row['Code'])
        """
        # TODO: convert coords to RD and apply axis limits again
        # ax_map.set_xlim(-50000,300000)
        # ax_map.set_ylim(350000,650000)
        ax_map.set_title('overview of stations with GETETM2 data')
        ax_map.set_aspect('equal')
        # def div1000(x,pos): return f'{int(x//1000)}'
        # ax_map.xaxis.set_major_formatter(ticker.FuncFormatter(div1000))
        # ax_map.yaxis.set_major_formatter(ticker.FuncFormatter(div1000))
        ax_map.set_xlabel('X')
        ax_map.set_ylabel('Y')
        ax_map.grid(alpha=0.5)
        fig_map.tight_layout()
        #ctx.add_basemap(ax_map, source=ctx.providers.Esri.WorldImagery, crs="EPSG:28992", attribution=False)
        fig_map.savefig(os.path.join(dir_meas_ddl,'stations_map.png'))
    
    #plotting
    file_wl_png = os.path.join(dir_meas_ddl,f'ts_{current_station}.png')
    if 0:#os.path.exists(file_wl_png):
        continue #skip the plotting if there is already a png available
    if os.path.exists(file_ext_pkl):
        fig,(ax1,ax2) = hatyan.plot_timeseries(ts=ts_meas_pd, ts_ext=ts_meas_ext_pd)
    else:
        fig,(ax1,ax2) = hatyan.plot_timeseries(ts=ts_meas_pd)
    ax1.set_title(f'timeseries for {current_station}')
    ax1_legendlabels = ax1.get_legend_handles_labels()[1]
    ax2_legendlabels = ['zero']
    ax1_legendlabels.insert(1,'zero') #legend for zero line was not displayed but will be now so it needs to be added
    ax1_legendlabels[0] = 'measured waterlevels'
    ax1_legendlabels[2] = 'mean'
    ax1.plot(mean_peryearmonth_long,'c',linewidth=0.7); ax1_legendlabels.append('monthly mean')
    ax1.plot(mean_peryear_long,'m',linewidth=0.7); ax1_legendlabels.append('yearly mean')
    ax2.plot(mean_peryearmonth_long,'c',linewidth=0.7); ax2_legendlabels.append('monthly mean')
    ax2.plot(mean_peryear_long,'m',linewidth=0.7); ax2_legendlabels.append('yearly mean')
    ax1.set_ylim(-4,4)
    ax1.legend(ax1_legendlabels,loc=4)
    ax2.legend(ax2_legendlabels,loc=1)
    if os.path.exists(file_ext_pkl): #plot after legend creation, so these entries are not included
        ax1.plot(HW_mean_peryear_long,'m',linewidth=0.7)#; ax1_legendlabels.append('yearly mean')
        ax1.plot(LW_mean_peryear_long,'m',linewidth=0.7)#; ax1_legendlabels.append('yearly mean')
    ax2.set_ylim(-0.5,0.5)
    ax1.set_xlim(fig_alltimes_ext) # entire period
    fig.savefig(file_wl_png.replace('.png','_alldata.png'))
    ax1.set_xlim(dt.datetime(2000,1,1),dt.datetime(2022,1,1)) # period of interest
    fig.savefig(file_wl_png)
    plt.close(fig)
