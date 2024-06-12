# -*- coding: utf-8 -*-
"""
Retrieve data from ddlpy and write to netcdf files including all metadata
"""

import os
import pandas as pd
import ddlpy
from pyproj import Transformer
import pooch
import logging
import dateutil
import hatyan
import xarray as xr

__all__ = [
    "retrieve_measurements_amount",
    "read_measurements_amount",
    "retrieve_measurements",
    "read_measurements",
    ]

logger = logging.getLogger(__name__)

DICT_FNAMES = {"meas_ts":"{station}_measwl.nc",
               "meas_ext":"{station}_measext.nc",
               "amount_ts":"data_amount_ts.csv",
               "amount_ext":"data_amount_ext.csv",
               }


def retrieve_catalog(overwrite=False, crs:int = None):
    # create cache dir %USERPROFILE%/AppData/Local/kenmerkendewaarden/kenmerkendewaarden/Cache
    dir_cache = str(pooch.os_cache('kenmerkendewaarden'))
    os.makedirs(dir_cache, exist_ok=True)
    
    file_catalog_pkl = os.path.join(dir_cache, 'DDL_catalog.pkl')
    if os.path.exists(file_catalog_pkl) and not overwrite:
        logger.info('loading DDL locations catalog from pickle')
        locations = pd.read_pickle(file_catalog_pkl)
    else:
        logger.info('retrieving DDL locations catalog with ddlpy')
        # include Typeringen in locations catalog
        catalog_filter = ['Eenheden','Grootheden','Hoedanigheden','Groeperingen','Parameters','Compartimenten','Typeringen']
        locations_full = ddlpy.locations(catalog_filter=catalog_filter)
        drop_columns = [x for x in locations_full.columns if x.endswith(".Omschrijving")]
        # drop_columns.append("Parameter_Wat_Omschrijving") # TODO: uncomment after ddlpy 0.6.0 is released: https://github.com/Deltares/ddlpy/pull/104
        locations = locations_full.drop(columns=drop_columns)
        pd.to_pickle(locations, file_catalog_pkl)
    
    # convert coordinates to new crs
    if crs is not None:
        assert len(locations["Coordinatenstelsel"].drop_duplicates()) == 1
        epsg_in = locations["Coordinatenstelsel"].iloc[0]
        transformer = Transformer.from_crs(f'epsg:{epsg_in}', f'epsg:{crs}', always_xy=True)
        locations["X"], locations["Y"] = transformer.transform(locations["X"], locations["Y"])
        locations["Coordinatenstelsel"] = str(crs)
    
    bool_grootheid = locations["Grootheid.Code"].isin(["WATHTE"])
    bool_groepering_ts = locations["Groepering.Code"].isin(["NVT"])
    bool_groepering_ext = locations["Groepering.Code"].isin(["GETETM2","GETETMSL2"])
    # TODO: for now we do not separately retrieve NAP and MSL for EURPFM/LICHELGRE which have both sets (https://github.com/Rijkswaterstaat/wm-ws-dl/issues/17), these stations are skipped
    # bool_hoedanigheid_nap = locations["Hoedanigheid.Code"].isin(["NAP"])
    # bool_hoedanigheid_msl = locations["Hoedanigheid.Code"].isin(["MSL"])
    
    # filtering locations dataframe on Typering is possible because "Typeringen" was in catalog_filter for ddlpy.locations
    bool_typering_exttypes = locations['Typering.Code'].isin(['GETETTPE'])
    
    # select locations on grootheid/groepering/exttypes
    locs_meas_ts = locations.loc[bool_grootheid & bool_groepering_ts]
    locs_meas_ext = locations.loc[bool_grootheid & bool_groepering_ext]
    locs_meas_exttype = locations.loc[bool_typering_exttypes & bool_groepering_ext]
    return locs_meas_ts, locs_meas_ext, locs_meas_exttype


def check_locations_amount(locations):
    """
    checks the amount of rows in a ddlpy.locations dataframe.
    It allows for zero stations, since this regularly happens for extremes, in that case the station is skipped
    It raises an error in case of multiple stations, stricter station selection is required.
    """
    if len(locations)==0:
        logger.info(f"no stations present after station subsetting, skipping station:\n{locations}")
        return
    elif len(locations)!=1:
        raise ValueError(f"multiple stations present after station subsetting:\n{locations}")


def retrieve_measurements_amount(dir_output, station_list, extremes:bool, start_date, end_date):
    locs_meas_ts, locs_meas_ext, locs_meas_exttype = retrieve_catalog()
    
    if extremes:
        fname = DICT_FNAMES['amount_ext']
        locs_meas = locs_meas_ext
    else:
        fname = DICT_FNAMES['amount_ts']
        locs_meas = locs_meas_ts
    file_csv_amount = os.path.join(dir_output,fname)
    
    if os.path.exists(file_csv_amount):
        raise FileExistsError(f"{file_csv_amount} already exists, delete file or change dir_output")
    
    # if csv file(s) do not exist, get the measurement amount from the DDL
    amount_list = []
    for station in station_list:
        logger.info(f'retrieving measurement amount from DDL for {station}')
        
        bool_station = locs_meas.index.isin([station])
        loc_meas_one = locs_meas.loc[bool_station]
        
        check_locations_amount(loc_meas_one)
        
        if len(loc_meas_one) == 0:
            logger.info(f"no station available (extremes={extremes})")
            # TODO: no ext station available for ["A12","AWGPFM","BAALHK","GATVBSLE","D15","F16","F3PFM","J6","K14PFM",
            #                                     "L9PFM","MAASMSMPL","NORTHCMRT","OVLVHWT","Q1","SINTANLHVSGR","WALSODN"]
            # https://github.com/Rijkswaterstaat/wm-ws-dl/issues/39
            amount_meas = pd.DataFrame({station:[]}, dtype='int64')
            amount_meas.index.name = "Groeperingsperiode"
        else:
            amount_meas = ddlpy.measurements_amount(location=loc_meas_one.iloc[0], start_date=start_date, end_date=end_date)
            amount_meas = amount_meas.rename(columns={"AantalMetingen":station})
        
        amount_list.append(amount_meas)
    
    logger.info(f'write measurement amount csvs to {os.path.basename(dir_output)}')
    df_amount = pd.concat(amount_list, axis=1).sort_index()
    df_amount = df_amount.fillna(0).astype(int)
    
    df_amount.to_csv(file_csv_amount)
    

def read_measurements_amount(dir_output, extremes:bool):
    if extremes:
        fname = DICT_FNAMES['amount_ext']
    else:
        fname = DICT_FNAMES['amount_ts']
    file_csv_amount = os.path.join(dir_output,fname)
    
    if not os.path.exists(file_csv_amount):
        raise FileNotFoundError(f"{file_csv_amount} does not exist")
        
    logger.info("found existing data amount csv files, loading with pandas")
    df_amount = pd.read_csv(file_csv_amount)
    df_amount = df_amount.set_index("Groeperingsperiode")
    return df_amount


def retrieve_measurements(dir_output:str, station:str, extremes:bool, start_date, end_date, drop_if_constant=None):
    
    locs_meas_ts, locs_meas_ext, locs_meas_exttype = retrieve_catalog()
    
    if drop_if_constant is None:
        drop_if_constant = ["WaarnemingMetadata.OpdrachtgevendeInstantieLijst",
                            "WaarnemingMetadata.BemonsteringshoogteLijst",
                            "WaarnemingMetadata.ReferentievlakLijst",
                            "AquoMetadata_MessageID", 
                            "BioTaxonType", 
                            "BemonsteringsSoort.Code", 
                            "Compartiment.Code", "Eenheid.Code", "Grootheid.Code", "Hoedanigheid.Code",
                            "WaardeBepalingsmethode.Code", "MeetApparaat.Code",
                            ]
            
    bool_station_ts = locs_meas_ts.index.isin([station])
    bool_station_ext = locs_meas_ext.index.isin([station])
    bool_station_exttype = locs_meas_exttype.index.isin([station])
    loc_meas_ts_one = locs_meas_ts.loc[bool_station_ts]
    loc_meas_ext_one = locs_meas_ext.loc[bool_station_ext]
    loc_meas_exttype_one = locs_meas_exttype.loc[bool_station_exttype]

    if extremes:
        fname = DICT_FNAMES["meas_ext"].format(station=station)
        loc_meas_one = loc_meas_ext_one
        freq = dateutil.rrule.YEARLY
    else:
        fname = DICT_FNAMES["meas_ts"].format(station=station)
        loc_meas_one = loc_meas_ts_one
        freq = dateutil.rrule.MONTHLY
    file_nc = os.path.join(dir_output,fname)
    
    #retrieving waterlevel extremes or timeseries
    if os.path.exists(file_nc):
        logger.info(f'meas data (extremes={extremes}) for {station} already available in {os.path.basename(dir_output)}, skipping station')
        return
    
    check_locations_amount(loc_meas_one)
    
    logger.info(f'retrieving meas data (extremes={extremes}) from DDL for {station} to {os.path.basename(dir_output)}')
    measurements = ddlpy.measurements(location=loc_meas_one.iloc[0], start_date=start_date, end_date=end_date, freq=freq)
    if measurements.empty:
        raise ValueError("[NO DATA]")
    ds_meas = ddlpy.dataframe_to_xarray(measurements, drop_if_constant)
    if extremes:
        #convert extreme type to HWLWcode add extreme type and HWLcode as dataset variables
        # TODO: simplify by retrieving the extreme value and type from ddl in a single request: https://github.com/Rijkswaterstaat/wm-ws-dl/issues/19
        measurements_exttyp = ddlpy.measurements(location=loc_meas_exttype_one.iloc[0], start_date=start_date, end_date=end_date, freq=dateutil.rrule.YEARLY)
        ts_meas_ext_pd = hatyan.ddlpy_to_hatyan(measurements, measurements_exttyp)
        ds_meas["extreme_type"] = xr.DataArray(ts_meas_ext_pd['values'].values, dims="time")
        ds_meas["HWLWcode"] = xr.DataArray(ts_meas_ext_pd['HWLWcode'].values, dims="time")

    # write to netcdf (including metadata)
    ds_meas.to_netcdf(file_nc)
    ds_meas.close()


def xarray_to_hatyan(ds):
    """
    converting the xarray dataset in the format of the 
    kenmerkendewaarden netcdf files to a hatyan dataframe.
    This saves memory and prevents converting it multiple times 
    in the kenmerkendewaarden code when passing it to hatyan.
    """
    df = pd.DataFrame({"values":ds["Meetwaarde.Waarde_Numeriek"].to_pandas()/100,
                       "qualitycode": ds["WaarnemingMetadata.KwaliteitswaardecodeLijst"].to_pandas(),
                       "status": ds["WaarnemingMetadata.StatuswaardeLijst"].to_pandas(),
                       })
    if "HWLWcode" in ds.data_vars:
        df["HWLWcode"] = ds["HWLWcode"]
    
    # convert timezone back to UTC+1 # TODO: add testcase
    df.index = df.index.tz_localize("UTC").tz_convert("Etc/GMT-1")
    
    # add attrs
    df.attrs["station"] = ds.attrs["Code"]
    return df


def read_measurements(dir_output:str, station:str, extremes:bool, return_xarray=False, nap_correction=False):

    if extremes:
        fname = DICT_FNAMES["meas_ext"].format(station=station)
    else:
        fname = DICT_FNAMES["meas_ts"].format(station=station)
    file_nc = os.path.join(dir_output,fname)

    if not os.path.exists(file_nc):
        # return None if file does not exist
        logger.info(f"file {fname} not found, returning None")
        return
    
    logger.info(f"loading {fname}")
    ds_meas = xr.open_dataset(file_nc)
    if return_xarray:
        return ds_meas
    
    df_meas = xarray_to_hatyan(ds_meas)
    
    if nap_correction:
        # TODO: not available for all stations
        df_meas = nap2005_correction(df_meas)
    return df_meas


def clip_timeseries_physical_break(df_meas):
    # TODO: move to csv file and add as package data
    #physical_break_dict for slotgemiddelden and overschrijdingsfrequenties TODO: maybe use everywhere to crop data?
    physical_break_dict = {'DENOVBTN':'1933', #laatste sluitgat afsluitdijk in 1932 
                           'HARLGN':'1933', #laatste sluitgat afsluitdijk in 1932
                           'VLIELHVN':'1933', #laatste sluitgat afsluitdijk in 1932
                           } #TODO: add physical_break for STAVNSE and KATSBTN? (Oosterscheldekering)
    
    station = df_meas.attrs["station"]
    if station not in physical_break_dict.keys():
        logger.info(f'no physical_break defined for {station}, returning input timeseries')
        return df_meas
    
    physical_break = physical_break_dict[station]
    assert isinstance(physical_break, str)
    logger.info(f'clipping timeseries for {station} before physical_break={physical_break}')
    df_meas = df_meas.loc[physical_break:]
    
    return df_meas


def nap2005_correction(df_meas):
    #NAP correction for dates before 1-1-2005
    # TODO: check if ths make a difference (for havengetallen it makes a slight difference so yes. For gemgetijkromme it only makes a difference for spring/doodtij. (now only applied at gemgetij en havengetallen)). If so, make this flexible per station, where to get the data or is the RWS data already corrected for it?
    #herdefinitie van NAP (~20mm voor HvH in fig2, relevant?): https://puc.overheid.nl/PUC/Handlers/DownloadDocument.ashx?identifier=PUC_113484_31&versienummer=1
    #Dit is de rapportage waar het gebruik voor PSMSL data voor het eerst beschreven is: https://puc.overheid.nl/PUC/Handlers/DownloadDocument.ashx?identifier=PUC_137204_31&versienummer=1
    # TODO: maybe move dict to csv file and add as package data
    dict_correct_nap2005 = {'HOEKVHLD':-0.0277,
                            'HARVT10':-0.0210,
                            'VLISSGN':-0.0297}
    
    station = df_meas.attrs["station"]
    if station not in dict_correct_nap2005.keys():
        raise KeyError(f'NAP2005 correction not defined for {station}')

    logger.info(f'applying NAP2005 correction for {station}')
    correct_value = dict_correct_nap2005[station]
    df_meas_corr = df_meas.copy(deep=True) # make copy to avoid altering the original dataframe
    before2005bool = df_meas_corr.index < pd.Timestamp("2005-01-01 00:00:00 +01:00")
    df_meas_corr.loc[before2005bool,'values'] = df_meas_corr.loc[before2005bool,'values'] + correct_value
    
    return df_meas_corr
