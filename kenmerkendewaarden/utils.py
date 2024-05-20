# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 09:39:32 2024

@author: veenstra
"""

import pandas as pd


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
    
    meta_dict_flat = {}
    for key in list_relevantmetadata:
        if key in ds.data_vars:
            vals_unique = ds[key].to_pandas().drop_duplicates()
            meta_dict_flat[key] = '|'.join(vals_unique)
        else:
            meta_dict_flat[key] = ds.attrs[key]
    return meta_dict_flat


def get_stats_from_dataset(ds):
    ds_stats = {}
    
    # TODO: beware on timezones
    ds_times = ds.time.to_pandas().index.tz_localize("UTC").tz_convert("Etc/GMT-1")
    ts_dupltimes = ds_times.duplicated()
    ts_timediff = ds_times.diff()[1:]
    
    ds_stats['tstart'] = ds_times.min()
    ds_stats['tstop'] = ds_times.max()
    ds_stats['timediff_min'] = ts_timediff.min()
    ds_stats['timediff_max'] = ts_timediff.max()
    ds_stats['nvals'] = len(ds['Meetwaarde.Waarde_Numeriek'])
    ds_stats['#nans'] = ds['Meetwaarde.Waarde_Numeriek'].isnull().values.sum()
    ds_stats['min'] = ds['Meetwaarde.Waarde_Numeriek'].min().values
    ds_stats['max'] = ds['Meetwaarde.Waarde_Numeriek'].max().values
    ds_stats['std'] = ds['Meetwaarde.Waarde_Numeriek'].std().values
    ds_stats['mean'] = ds['Meetwaarde.Waarde_Numeriek'].mean().values
    ds_stats['dupltimes'] = ts_dupltimes.sum()
    #count #nans for duplicated times, happens at HARVT10/HUIBGT/STELLDBTN
    ds_stats['dupltimes_#nans'] = ds.sel(time=ds_times.duplicated(keep=False))['Meetwaarde.Waarde_Numeriek'].isnull().values.sum()
    
    if '' in ds['WaarnemingMetadata.KwaliteitswaardecodeLijst']:
        qc_none = True
    else:
        qc_none = False
    ds_stats['qc_none'] = qc_none
            
    if "HWLWcode" in ds.data_vars:
        #TODO: should be based on 12 only, not 345 (HOEKVHLD now gives warning)
        if ts_timediff.min() < pd.Timedelta(hours=4): #TODO: min timediff for e.g. BROUWHVSGT08 is 3 minutes: ts_meas_ext_pd.loc[dt.datetime(2015,1,1):dt.datetime(2015,1,2),['values', 'QC', 'Status']]. This should not happen and with new dataset should be converted to an error
            print(f'WARNING: extreme data contains values that are too close ({ts_timediff.min()}), should be at least 4 hours difference')
        
        if len(ds['HWLWcode'].to_pandas().unique()) > 2:
            ds_stats['aggers'] = True
        else:
            ds_stats['aggers'] = False
        
    return ds_stats


