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
import hatyan # requires hatyan>=2.8.0 for hatyan.ddlpy_to_hatyan() and hatyan.convert_HWLWstr2num() # TODO: not released yet
import xarray as xr
from pyproj import Transformer # dependency of hatyan
import kenmerkendewaarden as kw

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

# TODO: report wl/ext missings/duplicates/outliers in recent period 2000-2021 (based on data_summary csv's)
# TODO: visually check availability (start/stop/gaps/aggers) of wl/ext, monthmean wl, outliers. Create new issues if needed: https://github.com/Deltares-research/kenmerkendewaarden/issues/4
# TODO: all TODOS in this script

retrieve_meas_amount = False
plot_meas_amount = True
retrieve_data = False
create_summary = False
test = False


start_date = "1870-01-01"
end_date = "2024-01-01"
if test:
    start_date = "2021-12-01"
    end_date = "2022-02-01"
    start_date = "2010-12-01"
    end_date = "2022-02-01"
fig_alltimes_xlim = [dt.datetime.strptime(start_date,'%Y-%m-%d'), dt.datetime.strptime(end_date,'%Y-%m-%d')]

# dir_base = r'p:\11208031-010-kenmerkende-waarden-k\work'
dir_base = r"p:\11210325-005-kenmerkende-waarden\work"
dir_meas = os.path.join(dir_base,f"measurements_wl_{start_date.replace('-','')}_{end_date.replace('-','')}")
dir_meas_amount = os.path.join(dir_meas, "measurements_amount")
os.makedirs(dir_meas, exist_ok=True)
os.makedirs(dir_meas_amount, exist_ok=True)


file_catalog_pkl = os.path.join(dir_base, 'DDL_catalog.pkl')
if not os.path.exists(file_catalog_pkl):
    print('retrieving DDL locations catalog with ddlpy')
    locations_full = ddlpy.locations()
    drop_columns = [x for x in locations_full.columns if x.endswith(".Omschrijving")]
    locations = locations_full.drop(columns=drop_columns)
    pd.to_pickle(locations, file_catalog_pkl)
else:
    print('loading DDL locations catalog from pickle')
    locations = pd.read_pickle(file_catalog_pkl)
print('...done')

# convert coordinates to RD
crs = 28992
assert len(locations["Coordinatenstelsel"].drop_duplicates()) == 1
epsg_in = locations["Coordinatenstelsel"].iloc[0]
transformer = Transformer.from_crs(f'epsg:{epsg_in}', f'epsg:{crs}', always_xy=True)
locations["RDx"], locations["RDy"] = transformer.transform(locations["X"], locations["Y"])

bool_grootheid = locations["Grootheid.Code"].isin(["WATHTE"])
bool_groepering_ts = locations["Groepering.Code"].isin(["NVT"])
bool_groepering_ext = locations["Groepering.Code"].isin(["GETETM2","GETETMSL2"]) # TODO: why distinction between MSL and NAP? This is already filtered via Hoedanigheid
# bool_hoedanigheid_nap = locations["Hoedanigheid.Code"].isin(["NAP"])
# bool_hoedanigheid_msl = locations["Hoedanigheid.Code"].isin(["MSL"])
# TODO: we cannot subset Typering.Code==GETETTPE here (not present in dataframe), so we use Grootheid.Code==NVT: https://github.com/Rijkswaterstaat/wm-ws-dl/issues/19
bool_grootheid_exttypes = locations['Grootheid.Code'].isin(['NVT'])

# select locations on grootheid/groepering/exttypes
locs_meas_ts = locations.loc[bool_grootheid & bool_groepering_ts]
locs_meas_ext = locations.loc[bool_grootheid & bool_groepering_ext]
locs_meas_exttype = locations.loc[bool_grootheid_exttypes & bool_groepering_ext]

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

for station_name in station_list:
    bool_isstation = locs_meas_ts.index == station_name
    if bool_isstation.sum()!=1:
        print(f'station name {station_name} found {bool_isstation.sum()} times, should be 1:')
        print(f'{locs_meas_ts.loc[bool_isstation,["Naam","Locatie_MessageID","Hoedanigheid.Code"]]}')
        print()

# TODO: fix multiple hits (two rows probably raises error in retrieve_data code below): https://github.com/Rijkswaterstaat/wm-ws-dl/issues/12
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

station name EURPFM found 2 times, should be 1:
                 Naam  Locatie_MessageID Hoedanigheid.Code
Code                                                      
EURPFM  Euro platform              10946               MSL
EURPFM  Euro platform              10946               NAP

station name J6 found 2 times, should be 1:
             Naam  Locatie_MessageID Hoedanigheid.Code
Code                                                  
J6    J6 platform               5377               MSL
J6    Platform J6              10982               MSL

station name LICHTELGRE found 2 times, should be 1:
                          Naam  Locatie_MessageID Hoedanigheid.Code
Code                                                               
LICHTELGRE  Lichteiland Goeree              10953               MSL
LICHTELGRE  Lichteiland Goeree              10953               NAP

station name NES found 2 times, should be 1:
     Naam  Locatie_MessageID Hoedanigheid.Code
Code                                          
NES   Nes               5391               NAP
NES   Nes              10309               NAP
"""

# TODO: missing/duplicate stations reported in https://github.com/Rijkswaterstaat/wm-ws-dl/issues/39. Many of these are not retrieved since we use clean_df for ddlpy
# TODO: some stations are now realtime instead of hist
### RETRIEVE MEASUREMENTS AMOUNT
ts_amount_list = []
ext_amount_list = []
for current_station in station_list:
    if not retrieve_meas_amount:
        continue
    print(f'retrieving measurement amount from DDL for {current_station}')
    
    bool_station_ts = locs_meas_ts.index.isin([current_station])
    bool_station_ext = locs_meas_ext.index.isin([current_station])
    loc_meas_ts_one = locs_meas_ts.loc[bool_station_ts]
    loc_meas_ext_one = locs_meas_ext.loc[bool_station_ext]
    
    amount_ts = ddlpy.measurements_amount(location=loc_meas_ts_one.iloc[0], start_date=start_date, end_date=end_date)
    amount_ts_clean = amount_ts.set_index("Groeperingsperiode").rename(columns={"AantalMetingen":current_station})
    if amount_ts_clean.index.duplicated().any():
        # TODO: multiple 1993 in dataframe for BATH, because of multiple waardebepalingsmethoden/meetapparaten: https://github.com/Deltares/ddlpy/issues/92
        amount_ts_clean = amount_ts_clean.groupby(amount_ts_clean.index).sum()
    ts_amount_list.append(amount_ts_clean)
    
    try:
        amount_ext = ddlpy.measurements_amount(location=loc_meas_ext_one.iloc[0], start_date=start_date, end_date=end_date)
        amount_ext_clean = amount_ext.set_index("Groeperingsperiode").rename(columns={"AantalMetingen":current_station})
    except IndexError: # IndexError: single positional indexer is out-of-bounds
        print("ext no station available")
        # TODO: no ext station available for ["A12","AWGPFM","BAALHK","GATVBSLE","D15","F16","F3PFM","J6","K14PFM",
        #                                     "L9PFM","MAASMSMPL","NORTHCMRT","OVLVHWT","Q1","SINTANLHVSGR","WALSODN"]
        amount_ext_clean = pd.DataFrame({current_station:[]})
        amount_ext_clean.index.name = "Groeperingsperiode"
    if amount_ext_clean.index.duplicated().any():
        # TODO: see ts above
        amount_ext_clean = amount_ext_clean.groupby(amount_ext_clean.index).sum()
    ext_amount_list.append(amount_ext_clean)

file_csv_amount_ts = os.path.join(dir_meas_amount, "data_amount_ts.csv")
file_csv_amount_ext = os.path.join(dir_meas_amount, "data_amount_ext.csv")
if retrieve_meas_amount:
    print(f'write measurement amount csvs to {os.path.basename(dir_meas)}')
    df_amount_ts = pd.concat(ts_amount_list, axis=1).sort_index()
    df_amount_ts = df_amount_ts.fillna(0).astype(int)
    df_amount_ext = pd.concat(ext_amount_list, axis=1).sort_index()
    df_amount_ext = df_amount_ext.fillna(0).astype(int)
    df_amount_ts.to_csv(file_csv_amount_ts.replace(".csv","_PREVENTOVERWRITE.csv"))
    df_amount_ext.to_csv(file_csv_amount_ext.replace(".csv","_PREVENTOVERWRITE.csv"))


if plot_meas_amount:
    df_amount_ts = pd.read_csv(file_csv_amount_ts)
    df_amount_ts = df_amount_ts.set_index("Groeperingsperiode")
    df_amount_ext = pd.read_csv(file_csv_amount_ext)
    df_amount_ext = df_amount_ext.set_index("Groeperingsperiode")
    
    fig, ax = kw.df_amount_pcolormesh(df_amount_ts, relative=False)
    fig.savefig(file_csv_amount_ts.replace(".csv","_pcolormesh"), dpi=200)
    fig, ax = kw.df_amount_pcolormesh(df_amount_ext, relative=False)
    fig.savefig(file_csv_amount_ext.replace(".csv","_pcolormesh"), dpi=200)

    fig, ax = kw.df_amount_pcolormesh(df_amount_ts, relative=True)
    fig.savefig(file_csv_amount_ts.replace(".csv","_pcolormesh_relative"), dpi=200)
    fig, ax = kw.df_amount_pcolormesh(df_amount_ext, relative=True)
    fig.savefig(file_csv_amount_ext.replace(".csv","_pcolormesh_relative"), dpi=200)

    fig, ax = kw.df_amount_boxplot(df_amount_ts)
    fig.savefig(file_csv_amount_ts.replace(".csv","_boxplot"), dpi=200)
    fig, ax = kw.df_amount_boxplot(df_amount_ext)
    fig.savefig(file_csv_amount_ext.replace(".csv","_boxplot"), dpi=200)



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


for current_station in station_list:
    if not retrieve_data:
        continue
    
    # skip duplicate code stations (hist/realtime) # TODO: avoid this
    if current_station in ["BATH", "D15", "J6", "NES"]:
        continue
    # skip MSL/NAP duplicate stations # TODO: avoid this
    if current_station in ["EURPFM", "LICHTELGRE"]:
        continue
    
    # write to netcdf instead (including metadata)
    file_wl_nc = os.path.join(dir_meas,f"{current_station}_measwl.nc")
    file_ext_nc = os.path.join(dir_meas,f"{current_station}_measext.nc")
    
    bool_station_ts = locs_meas_ts.index.isin([current_station])
    bool_station_ext = locs_meas_ext.index.isin([current_station])
    bool_station_exttype = locs_meas_exttype.index.isin([current_station])
    loc_meas_ts_one = locs_meas_ts.loc[bool_station_ts]
    loc_meas_ext_one = locs_meas_ext.loc[bool_station_ext]
    loc_meas_exttype_one = locs_meas_exttype.loc[bool_station_exttype]

    if len(loc_meas_ts_one)!=1:
        raise ValueError(f"no or multiple stations present after station subsetting for {current_station} (ts):\n{loc_meas_ts_one}")
    
    ext_available = True
    if len(loc_meas_ext_one)==0:
        ext_available = False
    elif len(loc_meas_ext_one)!=1:
        raise ValueError(f"no or multiple stations present after station subsetting for {current_station} (ext):\n{loc_meas_ext_one}")
    
    #retrieving waterlevels
    if os.path.exists(file_wl_nc):
        print(f'measwl data for {current_station} already available in {os.path.basename(dir_meas)}')
    else:
        print(f'retrieving measwl data from DDL for {current_station} to {os.path.basename(dir_meas)}')
        measurements_ts = ddlpy.measurements(location=loc_meas_ts_one.iloc[0], start_date=start_date, end_date=end_date)
        meas_ts_ds = ddlpy.dataframe_to_xarray(measurements_ts, drop_if_constant)
        meas_ts_ds.to_netcdf(file_wl_nc)
    
    #retrieving measured extremes
    if os.path.exists(file_ext_nc):
        print(f'measext data for {current_station} already available in {os.path.basename(dir_meas)}')
    elif not ext_available:
        print(f'no measext data for {current_station}, skipping downloading')
    else:
        print(f'retrieving measext data from DDL for {current_station} to {os.path.basename(dir_meas)}')
        measurements_ext = ddlpy.measurements(location=loc_meas_ext_one.iloc[0], start_date=start_date, end_date=end_date)
        if measurements_ext.empty:
            raise ValueError("[NO DATA]")
        measurements_exttyp = ddlpy.measurements(location=loc_meas_exttype_one.iloc[0], start_date=start_date, end_date=end_date)
        meas_ext_ds = ddlpy.dataframe_to_xarray(measurements_ext, drop_if_constant)
        
        #convert extreme type to HWLWcode add extreme type and HWLcode as dataset variables
        # TODO: simplify by retrieving the extreme value and type from ddl in a single request (not supported yet)
        ts_meas_extval_pd = hatyan.ddlpy_to_hatyan(measurements_ext)
        ts_meas_exttype_pd = hatyan.ddlpy_to_hatyan(measurements_exttyp)
        ts_meas_ext_pd = hatyan.convert_HWLWstr2num(ts_meas_extval_pd, ts_meas_exttype_pd)
        meas_ext_ds["extreme_type"] = xr.DataArray(ts_meas_exttype_pd['values'].values, dims="time")
        meas_ext_ds["HWLWcode"] = xr.DataArray(ts_meas_ext_pd['HWLWcode'].values, dims="time")
        meas_ext_ds.to_netcdf(file_ext_nc)



row_list_ts = []
row_list_ext = []
for current_station in station_list:
    if not create_summary:
        continue
    print(f'checking data for {current_station}')
    data_summary_row_ts = {}
    data_summary_row_ext = {}
    
    # add location coordinates to data_summaries
    for sumrow in [data_summary_row_ts, data_summary_row_ext]:
        sumrow['Code'] = locs_meas_ts.loc[current_station].name
        sumrow['RDx'] = locs_meas_ts.loc[current_station,'RDx']
        sumrow['RDy'] = locs_meas_ts.loc[current_station,'RDy']
    time_interest_start = dt.datetime(2000,1,1)
    time_interest_stop = dt.datetime(2021,2,1)
    
    #load measwl data
    file_wl_nc = os.path.join(dir_meas,f"{current_station}_measwl.nc")
    if not os.path.exists(file_wl_nc):
        ts_available = False
    else:
        ts_available = True
    data_summary_row_ts['data_wl'] = ts_available
    
    if ts_available:
        ds_ts_meas = xr.open_dataset(file_wl_nc)
        
        meta_dict_flat_ts = kw.get_flat_meta_from_dataset(ds_ts_meas)
        data_summary_row_ts.update(meta_dict_flat_ts)
        
        ds_stats = kw.get_stats_from_dataset(ds_ts_meas, time_interest_start=time_interest_start, time_interest_stop=time_interest_stop)
        data_summary_row_ts.update(ds_stats)
        
        # calculate monthly/yearly mean for meas wl data
        # TODO: use kw.calc_wltidalindicators() instead (with threshold of eg 2900 like slotgem)
        df_meas_values = ds_ts_meas['Meetwaarde.Waarde_Numeriek'].to_pandas()/100
        mean_peryearmonth_long = df_meas_values.groupby(pd.PeriodIndex(df_meas_values.index, freq="M")).mean()
        data_summary_row_ts['monthmean_mean'] = mean_peryearmonth_long.mean()
        data_summary_row_ts['monthmean_std'] = mean_peryearmonth_long.std()
        mean_peryear_long = df_meas_values.groupby(pd.PeriodIndex(df_meas_values.index, freq="Y")).mean()
        data_summary_row_ts['yearmean_mean'] = mean_peryear_long.mean()
        data_summary_row_ts['yearmean_std'] = mean_peryear_long.std()
        
        ts_meas_pd = kw.xarray_to_hatyan(ds_ts_meas)
        

    #load measext data
    file_ext_nc = os.path.join(dir_meas,f"{current_station}_measext.nc")
    if not os.path.exists(file_ext_nc):
        ext_available = False
    else:
        ext_available = True
    data_summary_row_ext['data_ext'] = ext_available
    
    if ext_available:
        data_summary_row_ext['data_ext'] = True
        ds_ext_meas = xr.open_dataset(file_ext_nc)

        meta_dict_flat_ext = kw.get_flat_meta_from_dataset(ds_ext_meas)
        data_summary_row_ext.update(meta_dict_flat_ext)
        
        ds_stats = kw.get_stats_from_dataset(ds_ext_meas, time_interest_start=time_interest_start, time_interest_stop=time_interest_stop)
        data_summary_row_ext.update(ds_stats)
        
        #calculate monthly/yearly mean for meas ext data
        #TODO: make kw function (exact or approximation?)
        ts_meas_ext_pd = kw.xarray_to_hatyan(ds_ext_meas)
        if len(ts_meas_ext_pd['HWLWcode'].unique()) > 2:
            data_pd_HWLW_12 = hatyan.calc_HWLW12345to12(ts_meas_ext_pd) #convert 12345 to 12 by taking minimum of 345 as 2 (laagste laagwater). TODO: currently, first/last values are skipped if LW
        else:
            data_pd_HWLW_12 = ts_meas_ext_pd.copy()
        data_pd_HW = data_pd_HWLW_12.loc[data_pd_HWLW_12['HWLWcode']==1]
        data_pd_LW = data_pd_HWLW_12.loc[data_pd_HWLW_12['HWLWcode']==2]
        HW_mean_peryear_long = data_pd_HW.groupby(pd.PeriodIndex(data_pd_HW.index, freq="y"))['values'].mean() #TODO: use kw.calc_HWLWtidalindicators() instead (with threshold of eg 1400 like slotgem)
        LW_mean_peryear_long = data_pd_LW.groupby(pd.PeriodIndex(data_pd_LW.index, freq="y"))['values'].mean()
    
        # # replace 345 HWLWcode with 2, simple approximation of actual LW
        # bool_hwlw_3 = ds_ext_meas['HWLWcode'].isin([3])
        # bool_hwlw_45 = ds_ext_meas['HWLWcode'].isin([4,5])
        # ds_ext_meas_12only = ds_ext_meas.copy()
        # ds_ext_meas_12only['HWLWcode'][bool_hwlw_3] = 2
        # ds_ext_meas_12only = ds_ext_meas_12only.sel(time=~bool_hwlw_45)
        
        # #calculate monthly/yearly mean for meas ext data
        # #TODO: use kw.calc_HWLWtidalindicators() instead (with threshold of eg 1400 like slotgem)
        # data_pd_HW = ds_ext_meas_12only.sel(time=ds_ext_meas_12only['HWLWcode'].isin([1])).to_pandas()['Meetwaarde.Waarde_Numeriek']/100
        # data_pd_LW = ds_ext_meas_12only.sel(time=ds_ext_meas_12only['HWLWcode'].isin([2])).to_pandas()['Meetwaarde.Waarde_Numeriek']/100
        # HW_mean_peryear_long = data_pd_HW.groupby(pd.PeriodIndex(data_pd_HW.index, freq="y")).mean()
        # LW_mean_peryear_long = data_pd_LW.groupby(pd.PeriodIndex(data_pd_LW.index, freq="y")).mean()
    
    row_list_ts.append(pd.Series(data_summary_row_ts))
    row_list_ext.append(pd.Series(data_summary_row_ext))

    #plotting
    file_wl_png = os.path.join(dir_meas,f'ts_{current_station}.png')
    # if os.path.exists(file_wl_png):
    #     continue #skip the plotting if there is already a png available
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
    ax1.set_xlim(fig_alltimes_xlim) # entire period
    fig.savefig(file_wl_png.replace('.png','_alldata.png'))
    ax1.set_xlim(dt.datetime(2000,1,1),dt.datetime(2022,1,1)) # period of interest
    fig.savefig(file_wl_png)
    plt.close(fig)
    

if create_summary:
    data_summary_ts = pd.concat(row_list_ts, axis=1).T
    data_summary_ts = data_summary_ts.set_index('Code').sort_index()
    data_summary_ext = pd.concat(row_list_ext, axis=1).T
    data_summary_ext = data_summary_ext.set_index('Code').sort_index()
    data_summary_ts.to_csv(os.path.join(dir_meas,'data_summary_ts.csv'))
    data_summary_ext.to_csv(os.path.join(dir_meas,'data_summary_ext.csv'))
    
    #print and save data_summary
    print(data_summary_ts[['data_wl','tstart','tstop','nvals','dupltimes','#nans','#nans_2000to202102']])
    try:
        print(data_summary_ext[['data_ext','dupltimes','#HWgaps_2000to202102']])
    except KeyError:
        print(data_summary_ext[['data_ext','dupltimes']])
    
    #make spatial plot of available/retrieved stations
    fig_map,ax_map = plt.subplots(figsize=(8,7))
    
    ax_map.plot(data_summary_ts['RDx'], data_summary_ts['RDy'],'xk')#,alpha=0.4) #all ext stations
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
