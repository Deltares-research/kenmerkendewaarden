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


def plot_measurements(df, df_ext=None):
    station_df = df.attrs["station"]
    if df_ext is not None:
        station_df_ext = df_ext.attrs["station"]
        assert station_df == station_df_ext
        fig,(ax1,ax2) = hatyan.plot_timeseries(ts=df, ts_ext=df_ext)
    else:
        fig,(ax1,ax2) = hatyan.plot_timeseries(ts=df)
    ax1.set_title(f'timeseries for {station_df}')
    
    # calculate monthly/yearly mean for meas wl data
    df_meas_values = df['values']
    mean_peryearmonth_long = df_meas_values.groupby(pd.PeriodIndex(df_meas_values.index, freq="M")).mean()
    mean_peryear_long = df_meas_values.groupby(pd.PeriodIndex(df_meas_values.index, freq="Y")).mean()
    
    ax1.plot(mean_peryearmonth_long,'c',linewidth=0.7, label='monthly mean')
    ax1.plot(mean_peryear_long,'m',linewidth=0.7, label='yearly mean')
    ax2.plot(mean_peryearmonth_long,'c',linewidth=0.7, label='monthly mean')
    ax2.plot(mean_peryear_long,'m',linewidth=0.7, label='yearly mean')
    if df_ext is not None:
        # select all hoogwater
        data_pd_HW = df_ext.loc[df_ext['HWLWcode'].isin([1])]
        # select all laagwater, laagwater1, laagwater2 (so approximation in case of aggers)
        data_pd_LW = df_ext.loc[df_ext['HWLWcode'].isin([2,3,5])]
        
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


def get_stats_from_dataframe(df):
    df_times = df.index
    ts_dupltimes = df_times.duplicated()
    ts_timediff = df_times.diff()[1:]
    
    ds_stats = {}
    ds_stats['tstart'] = df_times.min()
    ds_stats['tstop'] = df_times.max()
    ds_stats['timediff_min'] = ts_timediff.min()
    ds_stats['timediff_max'] = ts_timediff.max()
    ds_stats['nvals'] = len(df['values'])
    ds_stats['#nans'] = df['values'].isnull().sum()
    ds_stats['min'] = df['values'].min()
    ds_stats['max'] = df['values'].max()
    ds_stats['std'] = df['values'].std()
    ds_stats['mean'] = df['values'].mean()
    ds_stats['dupltimes'] = ts_dupltimes.sum()
    #count #nans for duplicated times, happens at HARVT10/HUIBGT/STELLDBTN
    ds_stats['dupltimes_#nans'] = df.loc[df_times.duplicated(keep=False)]['values'].isnull().sum()
    
    # None in kwaliteitswaardecodelijst: https://github.com/Rijkswaterstaat/wm-ws-dl/issues/14
    # TODO: add test to see if '' is indeed the missing value (or None or np.nan)
    if '' in df['qualitycode'].values:
        ds_stats['qc_none'] = True
    else:
        ds_stats['qc_none'] = False
    
    if "HWLWcode" in df.columns:
        # count the number of too small time differences (<4hr), sometimes happens because of aggers
        mintimediff_hr = 4
        bool_timediff_toosmall = ts_timediff < pd.Timedelta(hours=mintimediff_hr)
        ds_stats[f'timediff<{mintimediff_hr}hr'] = bool_timediff_toosmall.sum()
        
        # check whether there are aggers present
        if len(df['HWLWcode'].unique()) > 2:
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
        ds_meas = read_measurements(dir_output=dir_output, station=current_station, extremes=extremes, return_xarray=True)
        if ds_meas is not None:
            meta_dict_flat_ts = get_flat_meta_from_dataset(ds_meas)
            data_summary_row.update(meta_dict_flat_ts)
            
            df_meas = xarray_to_hatyan(ds_meas)
            df_stats = get_stats_from_dataframe(df_meas)
            data_summary_row.update(df_stats)
            del ds_meas
        data_summary_row["Code"] = current_station
        row_list.append(pd.Series(data_summary_row))
    
    logger.info("writing statistics to csv file")
    data_summary = pd.concat(row_list, axis=1).T
    data_summary = data_summary.set_index('Code').sort_index()
    return data_summary
    
