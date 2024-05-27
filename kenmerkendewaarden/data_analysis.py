# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:23:46 2024

@author: veenstra
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kenmerkendewaarden.data_retrieve import read_measurements, xarray_to_hatyan
import hatyan # requires hatyan>=2.8.0 for hatyan.ddlpy_to_hatyan() and hatyan.convert_HWLWstr2num()
import logging

__all__ = [
    "df_amount_boxplot",
    "df_amount_pcolormesh",
    "plot_measurements",
    "derive_statistics",
    ]

logger = logging.getLogger(__name__)


def df_amount_boxplot(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df[df==0] = np.nan
    
    fig, ax = plt.subplots(figsize=(14,8))
    df.plot.box(ax=ax, rot=90, grid=True)
    ax.set_ylabel("measurements per year (0 excluded) [-]")
    fig.tight_layout()
    return fig, ax


def df_amount_pcolormesh(df, relative=False):
    df = df.copy()
    df[df==0] = np.nan
    
    if relative:
        # this is useful for ts, because the frequency was changed from hourly to 10-minute
        df_relative = df.div(df.median(axis=1), axis=0) * 100
        df_relative = df_relative.clip(upper=200)
        df = df_relative
        
    fig, ax = plt.subplots(figsize=(14,8))
    pc = ax.pcolormesh(df.columns, df.index, df.values, cmap="turbo")
    cbar = fig.colorbar(pc, ax=ax)
    ax.tick_params(axis='x', labelrotation=90)
    ax.grid()
    if relative:
        cbar.set_label("measurements per year w.r.t. year median (0 excluded) [%]")
    else:
        cbar.set_label("measurements per year (0 excluded) [-]")
    fig.tight_layout()
    return fig, ax


def plot_measurements(ds, ds_ext=None):
    station_ds = ds.attrs["Code"]
    # TODO: beware on timezones
    ts_meas_pd = xarray_to_hatyan(ds)
    if ds_ext is not None:
        station_ds_ext = ds_ext.attrs["Code"]
        assert station_ds == station_ds_ext
        ts_meas_ext_pd = xarray_to_hatyan(ds_ext)
        fig,(ax1,ax2) = hatyan.plot_timeseries(ts=ts_meas_pd, ts_ext=ts_meas_ext_pd)
    else:
        fig,(ax1,ax2) = hatyan.plot_timeseries(ts=ts_meas_pd)
    ax1.set_title(f'timeseries for {station_ds}')
    
    # calculate monthly/yearly mean for meas wl data
    df_meas_values = ds['Meetwaarde.Waarde_Numeriek'].to_pandas()/100
    mean_peryearmonth_long = df_meas_values.groupby(pd.PeriodIndex(df_meas_values.index, freq="M")).mean()
    mean_peryear_long = df_meas_values.groupby(pd.PeriodIndex(df_meas_values.index, freq="Y")).mean()
    
    ax1.plot(mean_peryearmonth_long,'c',linewidth=0.7, label='monthly mean')
    ax1.plot(mean_peryear_long,'m',linewidth=0.7, label='yearly mean')
    ax2.plot(mean_peryearmonth_long,'c',linewidth=0.7, label='monthly mean')
    ax2.plot(mean_peryear_long,'m',linewidth=0.7, label='yearly mean')
    if ds_ext is not None:
        # select all hoogwater
        data_pd_HW = ts_meas_ext_pd.loc[ts_meas_ext_pd['HWLWcode'].isin([1])]
        # select all laagwater, laagwater1, laagwater2 (so approximation in case of aggers)
        data_pd_LW = ts_meas_ext_pd.loc[ts_meas_ext_pd['HWLWcode'].isin([2,3,5])]
        
        # calculate monthly/yearly mean for meas ext data
        HW_mean_peryear_long = data_pd_HW.groupby(pd.PeriodIndex(data_pd_HW.index, freq="y"))['values'].mean()
        LW_mean_peryear_long = data_pd_LW.groupby(pd.PeriodIndex(data_pd_LW.index, freq="y"))['values'].mean()
        
        ax1.plot(HW_mean_peryear_long,'m',linewidth=0.7, label=None)
        ax1.plot(LW_mean_peryear_long,'m',linewidth=0.7, label=None)
    ax1.set_ylim(-4,4)
    ax1.legend(loc=4)
    ax2.legend(loc=1)
    ax2.set_ylim(-0.5,0.5)
    return fig, (ax1,ax2)


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
        # count the number of too small time differences (<4hr), sometimes happens because of aggers
        mintimediff_hr = 4
        bool_timediff_toosmall = ts_timediff < pd.Timedelta(hours=mintimediff_hr)
        if bool_timediff_toosmall.sum() > 0:
            ds_stats[f'timediff<{mintimediff_hr}hr'] = bool_timediff_toosmall.sum()
        
        # check whether there are aggers present
        if len(ds['HWLWcode'].to_pandas().unique()) > 2:
            ds_stats['aggers'] = True
        else:
            ds_stats['aggers'] = False
        
    return ds_stats


def derive_statistics(dir_output, station_list, extremes):
    row_list = []
    for current_station in station_list:
        logger.info(f'deriving statistics for {current_station} (extremes={extremes})')
        data_summary_row = {}
        
        # load measwl data
        ds_meas = read_measurements(dir_output=dir_output, station=current_station, extremes=extremes)
        if ds_meas is not None:
            meta_dict_flat_ts = get_flat_meta_from_dataset(ds_meas)
            data_summary_row.update(meta_dict_flat_ts)
            ds_stats = get_stats_from_dataset(ds_meas)
            data_summary_row.update(ds_stats)
            del ds_meas
        data_summary_row["Code"] = current_station
        row_list.append(pd.Series(data_summary_row))
    
    logger.info("writing statistics to csv file")
    data_summary = pd.concat(row_list, axis=1).T
    data_summary = data_summary.set_index('Code').sort_index()
    return data_summary
    
