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
import ddlpy
import hatyan
import xarray as xr
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
    

#TODO: visually check availability (start/stop/gaps/aggers) of wl/ext, monthmean wl, outliers (nog niet gedaan voor hele periode, wel voor 2000-2022 (listAB+HARVT10). create new issues if needed: https://github.com/Deltares-research/kenmerkendewaarden/issues/4
#TODO: DATA: extremes not available for HOEKVHLD/HARVT10 for 2021
#TODO: check prints, some should be errors (or converted to issues)

# TODO: move to functions
def xarray_to_hatyan(ds):
    df = pd.DataFrame({"values":ds["Meetwaarde.Waarde_Numeriek"].to_pandas()/100,
                       "QC": ds["WaarnemingMetadata.KwaliteitswaardecodeLijst"].to_pandas(),
                       })
    if "HWLWcode" in ds.data_vars:
        df["HWLWcode"] = ds["HWLWcode"]
    
    # convert timezone back to UTC+1 # TODO: add testcase
    df.index = df.index.tz_localize("UTC").tz_convert("Etc/GMT-1")
    # remove timezone label (timestamps are still UTC+1 in fact)
    df.index = df.index.tz_localize(None)
    return df

def get_flat_meta_from_dataset(ds):
    meta_dict_flat = {}
    for key in list_relevantmetadata:
        if key in ds.data_vars:
            vals_unique = ds[key].to_pandas().drop_duplicates()
            meta_dict_flat[key] = '|'.join(vals_unique)
        else:
            meta_dict_flat[key] = ds.attrs[key]
    return meta_dict_flat


retrieve_data = True
create_summary = True

tstart_dt_DDL = dt.datetime(1870,1,1) #1870,1,1 for measall folder
tstart_dt_DDL = dt.datetime(2021,12,1)
tstop_dt_DDL = dt.datetime(2022,2,1)
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
if not os.path.exists(file_catalog_pkl): #TODO: speed up catalog retrieval (or reading)
    print('retrieving DDL locations catalog with ddlpy')
    locations_full = ddlpy.locations()
    drop_columns = [x for x in locations_full.columns if x.endswith(".Omschrijving")]
    # TODO: remove additional columns if not required anymore (PWO required by ddlpy.ddlpy._combine_waarnemingenlijst, the rest by this script)
    # drop_columns += ["Parameter_Wat_Omschrijving"]#, 'Coordinatenstelsel','Locatie_MessageID','Naam']
    locations = locations_full.drop(columns=drop_columns)
    pd.to_pickle(locations, file_catalog_pkl)
else:
    print('loading DDL locations catalog from pickle')
    locations = pd.read_pickle(file_catalog_pkl)
print('...done')

bool_grootheid = locations["Grootheid.Code"].isin(["WATHTE"])
bool_groepering_ts = locations["Groepering.Code"].isin(["NVT"])
bool_groepering_ext = locations["Groepering.Code"].isin(["GETETM2","GETETMSL2"]) # TODO: why distinction between MSL and NAP? This is already filtered via Hoedanigheid
# bool_hoedanigheid_nap = locations["Hoedanigheid.Code"].isin(["NAP"])
# bool_hoedanigheid_msl = locations["Hoedanigheid.Code"].isin(["MSL"])
# TODO: we cannot subset Typering.Code==GETETTPE here (not present in dataframe), so we use Grootheid.Code==NVT: https://github.com/Rijkswaterstaat/wm-ws-dl/issues/19
bool_grootheid_exttypes = locations['Grootheid.Code'].isin(['NVT'])

# these selections correspond to cat_aquometadatalijst_sel/cat_locatielijst_sel from old hatyan ddl methods
#TODO: make generic for NAP/MSL
locs_meas_ts = locations.loc[bool_grootheid & bool_groepering_ts]
locs_meas_ext = locations.loc[bool_grootheid & bool_groepering_ext]
locs_meas_exttype = locations.loc[bool_grootheid_exttypes & bool_groepering_ext]

# station_list
# TODO: fix multiple hits (first one is retrieved, although two rows probably raises error in code below)
"""
station name BATH found 2 times, should be 1:
      Naam  Locatie_MessageID Hoedanigheid.Code
Code                                           
BATH  Bath              10518               NAP
BATH  Bath              13615               NAP

station name NES found 2 times, should be 1:
     Naam  Locatie_MessageID Hoedanigheid.Code
Code                                          
NES   Nes               5391               NAP
NES   Nes              10309               NAP

station name LICHTELGRE found 2 times, should be 1:
                          Naam  Locatie_MessageID Hoedanigheid.Code
Code                                                               
LICHTELGRE  Lichteiland Goeree              10953               MSL
LICHTELGRE  Lichteiland Goeree              10953               NAP

station name EURPFM found 2 times, should be 1:
                 Naam  Locatie_MessageID Hoedanigheid.Code
Code                                                      
EURPFM  Euro platform              10946               MSL
EURPFM  Euro platform              10946               NAP
"""
stat_name_list = ['TERNZN','BATH','HANSWT','VLISSGN','BERGSDSWT','KRAMMSZWT',
                  'STAVNSE','ROOMPBNN','CADZD','WESTKPLE','ROOMPBTN','BROUWHVSGT08',
                  'HARVT10','HOEKVHLD','SCHEVNGN','IJMDBTHVN','PETTZD','DENHDR','TEXNZE','TERSLNZE',
                  'WIERMGDN','HUIBGT','OUDSD','VLIELHVN','WESTTSLG','NES','SCHIERMNOG',
                  'DENOVBTN','KORNWDZBTN','HARLGN','LAUWOG','EEMSHVN','DELFZL',
                  'NIEUWSTZL','LICHTELGRE','EURPFM','K13APFM'] + ['DORDT','STELLDBTN','ROTTDM'] + ['MAASMSMPL','OOSTSDE11'] #+ stat_list_addnonext[2:] #"KW kust en GR Dillingh 2013" en "KW getijgebied RWS 2011.0", aangevuld met 3 stations AB, aangevuld met BOI wensen, aangevuld met dialijst ABCT
stat_list = []
for stat_name in stat_name_list:
    bool_isstation = locs_meas_ts.index == stat_name
    if bool_isstation.sum()!=1:
        print(f'station name {stat_name} found {bool_isstation.sum()} times, should be 1:')
        print(f'{locs_meas_ts.loc[bool_isstation,["Naam","Locatie_MessageID","Hoedanigheid.Code"]]}')
        print()
    if bool_isstation.sum()==0: #skip if none found
        continue
    stat_list.append(locs_meas_ts.loc[bool_isstation].index[0])
stat_list = ['HOEKVHLD','HARVT10','VLISSGN']

# dataTKdia station selections
# stat_list = ['A12','AWGPFM','BAALHK','BATH','BERGSDSWT','BROUWHVSGT02','BROUWHVSGT08','GATVBSLE','BRESKVHVN','CADZD','D15','DELFZL','DENHDR','EEMSHVN','EURPFM','F16','F3PFM','HARVT10','HANSWT','HARLGN','HOEKVHLD','HOLWD','HUIBGT','IJMDBTHVN','IJMDSMPL','J6','K13APFM','K14PFM','KATSBTN','KORNWDZBTN','KRAMMSZWT','L9PFM','LAUWOG','LICHTELGRE','MARLGT','NES','NIEUWSTZL','NORTHCMRT','DENOVBTN','OOSTSDE04','OOSTSDE11','OOSTSDE14','OUDSD','OVLVHWT','Q1','ROOMPBNN','ROOMPBTN','SCHAARVDND','SCHEVNGN','SCHIERMNOG','SINTANLHVSGR','STAVNSE','STELLDBTN','TERNZN','TERSLNZE','TEXNZE','VLAKTVDRN','VLIELHVN','VLISSGN','WALSODN','WESTKPLE','WESTTSLG','WIERMGDN','YERSKE'] #all stations from TK
# stat_list = ['BAALHK','BATH','BERGSDSWT','BRESKVHVN','CADZD','DELFZL','DENHDR','DENOVBTN','EEMSHVN','GATVBSLE','HANSWT','HARLGN','HARVT10','HOEKVHLD','IJMDBTHVN','KATSBTN','KORNWDZBTN','KRAMMSZWT','LAUWOG','OUDSD','ROOMPBNN','ROOMPBTN','SCHAARVDND','SCHEVNGN','SCHIERMNOG','STAVNSE','STELLDBTN','TERNZN','VLAKTVDRN','VLIELHVN','VLISSGN','WALSODN','WESTKPLE','WESTTSLG','WIERMGDN'] #all files with valid data for 2010 to 2021


M2_period_timedelta = pd.Timedelta(hours=hatyan.schureman.get_schureman_freqs(['M2']).loc['M2','period [hr]'])



### RETRIEVE DATA FROM DDL AND WRITE TO PICKLE
drop_if_constant = ["WaarnemingMetadata.OpdrachtgevendeInstantieLijst",
                    "WaarnemingMetadata.BemonsteringshoogteLijst",
                    "WaarnemingMetadata.ReferentievlakLijst",
                    "AquoMetadata_MessageID", 
                    "BioTaxonType", 
                    "BemonsteringsSoort.Code", 
                    "Compartiment.Code", "Eenheid.Code", "Grootheid.Code", "Hoedanigheid.Code",
                    "WaardeBepalingsmethode.Code", "MeetApparaat.Code",
                    ]

for current_station in stat_list:
    if not retrieve_data:
        continue
    
    # write to netcdf instead (including metadata)
    file_wl_nc = os.path.join(dir_meas_ddl,f"{current_station}_measwl.nc")
    file_ext_nc = os.path.join(dir_meas_ddl,f"{current_station}_measext.nc")
    
    bool_station_ts = locs_meas_ts.index.isin([current_station])
    bool_station_ext = locs_meas_ext.index.isin([current_station])
    bool_station_exttype = locs_meas_exttype.index.isin([current_station])
    loc_meas_ts_one = locs_meas_ts.loc[bool_station_ts]
    loc_meas_ext_one = locs_meas_ext.loc[bool_station_ext]
    loc_meas_exttype_one = locs_meas_exttype.loc[bool_station_exttype]

    if len(loc_meas_ts_one)!=1:
        raise ValueError(f"no or multiple stations present after station subsetting:\n{loc_meas_ts_one}")
    if len(loc_meas_ext_one)!=1:
        raise ValueError(f"no or multiple stations present after station subsetting:\n{loc_meas_ext_one}")
    
    #retrieving waterlevels
    if os.path.exists(file_wl_nc):
        print(f'measwl data for {current_station} already available in {os.path.basename(dir_meas_ddl)}')
    else:
        print(f'retrieving measwl data from DDL for {current_station} to {os.path.basename(dir_meas_ddl)}')
        measurements_ts = ddlpy.measurements(location=loc_meas_ts_one.iloc[0], start_date=tstart_dt_DDL, end_date=tstop_dt_DDL)
        meas_ts_ds = ddlpy.dataframe_to_xarray(measurements_ts, drop_if_constant)
        meas_ts_ds.to_netcdf(file_wl_nc)
    
    #retrieving measured extremes
    if os.path.exists(file_ext_nc):
        print(f'measext data for {current_station} already available in {os.path.basename(dir_meas_ddl)}')
    else:
        print(f'retrieving measext data from DDL for {current_station} to {os.path.basename(dir_meas_ddl)}')
        measurements_ext = ddlpy.measurements(location=loc_meas_ext_one.iloc[0], start_date=tstart_dt_DDL, end_date=tstop_dt_DDL)
        if measurements_ext.empty:
            raise ValueError("[NO DATA]")
        measurements_exttyp = ddlpy.measurements(location=loc_meas_exttype_one.iloc[0], start_date=tstart_dt_DDL, end_date=tstop_dt_DDL)
        meas_ext_ds = ddlpy.dataframe_to_xarray(measurements_ext, drop_if_constant)
        
        #convert extreme type to HWLWcode add extreme type and HWLcode as dataset variables
        # TODO: simplify by retrieving the extreme value and type from ddl in a single request (not supported yet)
        ts_meas_extval_pd = hatyan.ddlpy_to_hatyan(measurements_ext)
        ts_meas_exttype_pd = hatyan.ddlpy_to_hatyan(measurements_exttyp)
        ts_meas_ext_pd = hatyan.convert_HWLWstr2num(ts_meas_extval_pd, ts_meas_exttype_pd)
        meas_ext_ds["extreme_type"] = xr.DataArray(ts_meas_exttype_pd['values'].values, dims="time")
        meas_ext_ds["HWLWcode"] = xr.DataArray(ts_meas_ext_pd['HWLWcode'].values, dims="time")
        meas_ext_ds.to_netcdf(file_ext_nc)



"""
not in M2phasediff textfile (in hatyan): ['LICHTELGRE','EURPFM']
HW/LW numbers not always increasing: ['HANSWT','BROUWHVSGT08','PETTZD','DORDT']
no extremes in requested time frame: ['STELLDBTN','OOSTSDE11']
Catalog query yielded no results (no ext available like K13APFM): A12
"""
list_relevantmetadata = ['WaarnemingMetadata.StatuswaardeLijst', 
                         'WaarnemingMetadata.KwaliteitswaardecodeLijst', 
                         'WaardeBepalingsmethode.Code',
                         'MeetApparaat.Code',
                         'Hoedanigheid.Code',
                         'WaardeBepalingsmethode.Code',
                         'MeetApparaat.Code',
                         'Hoedanigheid.Code',
                         'Grootheid.Code',
                         'Groepering.Code',
                         'Typering.Code'
                         ]
row_list_ts = []
row_list_ext = []
for current_station in stat_list:
    if not create_summary:
        continue
    print(f'checking data for {current_station}')
    data_summary_row_ts = {}
    data_summary_row_ext = {}
    
    # add location coordinates to data_summaries
    # TODO: convert to RD
    for sumrow in [data_summary_row_ts, data_summary_row_ext]:
        sumrow['Code'] = locs_meas_ts.loc[current_station].name
        sumrow['X'] = locs_meas_ts.loc[current_station,'X']
        sumrow['Y'] = locs_meas_ts.loc[current_station,'Y']
        sumrow['Coordinatenstelsel'] = locs_meas_ts.loc[current_station,'Coordinatenstelsel']
    time_interest_start = dt.datetime(2000,1,1)
    time_interest_stop = dt.datetime(2021,2,1)
    
    #load measwl data
    file_wl_nc = os.path.join(dir_meas_ddl,f"{current_station}_measwl.nc")
    if not os.path.exists(file_wl_nc):
        ts_available = False
    else:
        ts_available = True
    data_summary_row_ts['data_wl'] = ts_available
    
    if ts_available:
        ds_ts_meas = xr.open_dataset(file_wl_nc)
        ts_meas_pd = xarray_to_hatyan(ds_ts_meas) #TODO: maybe base summary on netcdf only (but beware on timezones)
        
        meta_dict_flat_ts = get_flat_meta_from_dataset(ds_ts_meas)
        data_summary_row_ts.update(meta_dict_flat_ts)
        
        # TODO: generalize statistics generation (apply same function to ts and ext)
        data_summary_row_ts['tstart_wl'] = ts_meas_pd.index[0]
        data_summary_row_ts['tstop_wl'] = ts_meas_pd.index[-1]
        data_summary_row_ts['tstart2000_wl'] = ts_meas_pd.index[0]<=time_interest_start
        data_summary_row_ts['tstop202102_wl'] = ts_meas_pd.index[-1]>=time_interest_stop
        data_summary_row_ts['nvals_wl'] = len(ts_meas_pd['values'])
        data_summary_row_ts['#nans_wl'] = ts_meas_pd['values'].isnull().sum()
        data_summary_row_ts['min_wl'] = ts_meas_pd['values'].min()
        data_summary_row_ts['max_wl'] = ts_meas_pd['values'].max()
        data_summary_row_ts['std_wl'] = ts_meas_pd['values'].std()
        data_summary_row_ts['mean_wl'] = ts_meas_pd['values'].mean()
        ts_meas_dupltimes = ts_meas_pd.index.duplicated()
        data_summary_row_ts['dupltimes_wl'] = ts_meas_dupltimes.sum()
        #count #nans for duplicated times, happens at HARVT10/HUIBGT/STELLDBTN
        data_summary_row_ts['#nans_dupltimes_wl'] = ts_meas_pd.loc[ts_meas_pd.index.duplicated(keep=False),'values'].isnull().sum()
        
        #calc #nan-values in recent period
        ts_meas_2000to202102a = ts_meas_pd.loc[~ts_meas_dupltimes,['values']].loc[time_interest_start:min(ts_meas_pd.index[-1],time_interest_stop)]
        ts_meas_2000to202102b = pd.DataFrame({'values':ts_meas_pd.loc[~ts_meas_dupltimes,'values']},index=pd.date_range(start=time_interest_start,end=time_interest_stop,freq='10min'))
        data_summary_row_ts['#nans_2000to202102a_wl'] = ts_meas_2000to202102a['values'].isnull().sum()
        data_summary_row_ts['#nans_2000to202102b_wl'] = ts_meas_2000to202102b['values'].isnull().sum()
        
        #calculate monthly/yearly mean for meas wl data #TODO: use kw.calc_wltidalindicators() instead (with threshold of eg 2900 like slotgem)
        mean_peryearmonth_long = ts_meas_pd.groupby(pd.PeriodIndex(ts_meas_pd.index, freq="M"))['values'].mean()
        data_summary_row_ts['monthmean_mean_wl'] = mean_peryearmonth_long.mean()
        data_summary_row_ts['monthmean_std_wl'] = mean_peryearmonth_long.std()
        mean_peryear_long = ts_meas_pd.groupby(pd.PeriodIndex(ts_meas_pd.index, freq="Y"))['values'].mean()
        data_summary_row_ts['yearmean_mean_wl'] = mean_peryear_long.mean()
        data_summary_row_ts['yearmean_std_wl'] = mean_peryear_long.std()
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
    file_ext_nc = os.path.join(dir_meas_ddl,f"{current_station}_measext.nc")
    if not os.path.exists(file_ext_nc):
        ext_available = False
    else:
        ext_available = True
    data_summary_row_ext['data_ext'] = ext_available
    
    if ext_available:
        data_summary_row_ext['data_ext'] = True
        ds_ext_meas = xr.open_dataset(file_ext_nc)
        ts_meas_ext_pd = xarray_to_hatyan(ds_ext_meas) #TODO: maybe base summary on netcdf only (but beware on timezones)

        timediff_ext = ts_meas_ext_pd.index[1:]-ts_meas_ext_pd.index[:-1]
        if timediff_ext.min() < dt.timedelta(hours=4): #TODO: min timediff for e.g. BROUWHVSGT08 is 3 minutes: ts_meas_ext_pd.loc[dt.datetime(2015,1,1):dt.datetime(2015,1,2),['values', 'QC', 'Status']]. This should not happen and with new dataset should be converted to an error
            print(f'WARNING: extreme data contains values that are too close ({timediff_ext.min()}), should be at least 4 hours difference')
        meta_dict_flat_ext = get_flat_meta_from_dataset(ds_ext_meas)
        data_summary_row_ext.update(meta_dict_flat_ext)
        
        ts_meas_ext_dupltimes = ts_meas_ext_pd.index.duplicated()
        data_summary_row_ext['mintimediff_ext'] = timediff_ext.min()
        data_summary_row_ext['dupltimes_ext'] = ts_meas_ext_dupltimes.sum()
        data_summary_row_ext['tstart_ext'] = ts_meas_ext_pd.index[0]
        data_summary_row_ext['tstop_ext'] = ts_meas_ext_pd.index[-1]
        data_summary_row_ext['tstart2000_ext'] = ts_meas_ext_pd.index[0]<=(time_interest_start+M2_period_timedelta)
        data_summary_row_ext['tstop202102_ext'] = ts_meas_ext_pd.index[-1]>=(time_interest_stop-M2_period_timedelta)
        data_summary_row_ext['nvals_ext'] = len(ts_meas_ext_pd['values'])
        data_summary_row_ext['min_ext'] = ts_meas_ext_pd['values'].min()
        data_summary_row_ext['max_ext'] = ts_meas_ext_pd['values'].max()
        data_summary_row_ext['std_ext'] = ts_meas_ext_pd['values'].std()
        data_summary_row_ext['mean_ext'] = ts_meas_ext_pd['values'].mean()
        if len(ts_meas_ext_pd['HWLWcode'].unique()) > 2:
            data_summary_row_ext['aggers_ext'] = True
        else:
            data_summary_row_ext['aggers_ext'] = False
        try:
            ts_meas_ext_2000to202102 = ts_meas_ext_pd.loc[(ts_meas_ext_pd.index>=time_interest_start) & (ts_meas_ext_pd.index<=time_interest_stop)]
            ts_meas_ext_pd.loc[dt.datetime(2015,1,1):dt.datetime(2015,1,2)]
            ts_meas_ext_2000to202102 = hatyan.calc_HWLWnumbering(ts_meas_ext_2000to202102)
            HWmissings = (ts_meas_ext_2000to202102.loc[ts_meas_ext_pd['HWLWcode']==1,'HWLWno'].diff().dropna()!=1).sum()
            data_summary_row_ext['#HWgaps_2000to202102_ext'] = HWmissings
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
    
    row_list_ts.append(pd.Series(data_summary_row_ts))
    row_list_ext.append(pd.Series(data_summary_row_ext))

    #plotting
    file_wl_png = os.path.join(dir_meas_ddl,f'ts_{current_station}.png')
    if os.path.exists(file_wl_png):
        continue #skip the plotting if there is already a png available
    if os.path.exists(file_ext_nc):
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
    if os.path.exists(file_ext_nc): #plot after legend creation, so these entries are not included
        ax1.plot(HW_mean_peryear_long,'m',linewidth=0.7)#; ax1_legendlabels.append('yearly mean')
        ax1.plot(LW_mean_peryear_long,'m',linewidth=0.7)#; ax1_legendlabels.append('yearly mean')
    ax2.set_ylim(-0.5,0.5)
    ax1.set_xlim(fig_alltimes_ext) # entire period
    fig.savefig(file_wl_png.replace('.png','_alldata.png'))
    ax1.set_xlim(dt.datetime(2000,1,1),dt.datetime(2022,1,1)) # period of interest
    fig.savefig(file_wl_png)
    plt.close(fig)
    

if create_summary:
    data_summary_ts = pd.concat(row_list_ts, axis=1).T
    data_summary_ts = data_summary_ts.set_index('Code').sort_index()
    data_summary_ext = pd.concat(row_list_ext, axis=1).T
    data_summary_ext = data_summary_ext.set_index('Code').sort_index()
    data_summary_ts.to_csv(os.path.join(dir_meas_ddl,'data_summary_ts.csv'))
    data_summary_ext.to_csv(os.path.join(dir_meas_ddl,'data_summary_ext.csv'))
    
    #print and save data_summary
    print(data_summary_ts[['data_wl','tstart_wl','tstop_wl','nvals_wl','dupltimes_wl','#nans_wl','#nans_2000to202102a_wl']])
    try:
        print(data_summary_ext[['data_ext','dupltimes_ext','#HWgaps_2000to202102_ext']])
    except KeyError:
        print(data_summary_ext[['data_ext','dupltimes_ext']])            
    
    #make spatial plot of available/retrieved stations
    fig_map,ax_map = plt.subplots(figsize=(8,7))
    
    ax_map.plot(data_summary_ts['X'], data_summary_ts['Y'],'xk')#,alpha=0.4) #all ext stations
    ax_map.plot(data_summary_ts.loc[stat_list,'X'], data_summary_ts.loc[stat_list,'Y'],'xr') # selected ext stations (stat_list)
    ax_map.plot(data_summary_ext.loc[data_summary_ext['data_ext'],'X'], data_summary_ext.loc[data_summary_ext['data_ext'],'Y'],'xm') # data retrieved
    """
    for iR, row in data_summary_ts.iterrows():
        ax_map.text(row['X'],row['Y'],row.name)
    """
    # TODO: convert coords to RD and apply axis limits again
    # ax_map.set_xlim(-50000,300000)
    # ax_map.set_ylim(350000,650000)
    ax_map.set_title('overview of stations with GETETM2 data')
    ax_map.set_aspect('equal')
    ax_map.set_xlabel('X')
    ax_map.set_ylabel('Y')
    ax_map.grid(alpha=0.5)
    
    # add basemap/coastlines
    #TODO: update crs after conversion to RD
    crs = data_summary_ext["Coordinatenstelsel"].iloc[0]
    if dfmt_available:
        dfmt.plot_coastlines(ax=ax_map, crs=crs)
    elif ctx_available:
        ctx.add_basemap(ax_map, source=ctx.providers.Esri.WorldImagery, crs=int(crs), attribution=False)
    
    fig_map.tight_layout()
    fig_map.savefig(os.path.join(dir_meas_ddl,'stations_map.png'), dpi=200)
