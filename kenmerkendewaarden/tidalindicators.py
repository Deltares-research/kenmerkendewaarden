# -*- coding: utf-8 -*-
"""
Computation of tidal indicators from waterlevel extremes or timeseries
"""

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import hatyan
import logging
from kenmerkendewaarden.utils import raise_extremes_with_aggers


__all__ = ["calc_wltidalindicators",
           "calc_HWLWtidalindicators",
           "plot_tidalindicators",
           "calc_HWLWtidalrange",
           "calc_hat_lat_fromcomponents",
           "calc_hat_lat_frommeasurements",
           ]

logger = logging.getLogger(__name__)


def calc_HWLWtidalindicators(df_ext, min_coverage:float = None):
    """
    computes several tidal extreme indicators from tidal extreme dataset

    Parameters
    ----------
    data_pd_HWLW_all : TYPE
        DESCRIPTION.
     min_coverage : float, optional
         The minimum percentage (from 0 to 1) of timeseries coverage to consider the statistics to be valid. The default is None.

    Returns
    -------
    dict_tidalindicators : TYPE
        DESCRIPTION.

    """
    # dropping the timezone makes the code below much faster and gives equal results: https://github.com/pandas-dev/pandas/issues/58956
    if df_ext.index.tz is not None:
        df_ext = df_ext.tz_localize(None)
    
    raise_extremes_with_aggers(df_ext)
    
    #split to HW and LW separately, also groupby year
    data_pd_HW = df_ext.loc[df_ext['HWLWcode']==1]
    data_pd_LW = df_ext.loc[df_ext['HWLWcode']==2]
    
    #yearmean HWLW from HWLW values #maybe also add *_mean_permonth
    HW_mean_peryear = data_pd_HW.groupby(pd.PeriodIndex(data_pd_HW.index, freq="Y"))[['values']].mean()
    LW_mean_peryear = data_pd_LW.groupby(pd.PeriodIndex(data_pd_LW.index, freq="Y"))[['values']].mean()
    
    #derive GHHW/GHWS (gemiddeld hoogwater springtij) per month
    HW_monthmax_permonth = data_pd_HW.groupby(pd.PeriodIndex(data_pd_HW.index, freq="M"))[['values']].max() #proxy for HW at spring tide
    LW_monthmin_permonth = data_pd_LW.groupby(pd.PeriodIndex(data_pd_LW.index, freq="M"))[['values']].min() #proxy for LW at spring tide
    HW_monthmin_permonth = data_pd_HW.groupby(pd.PeriodIndex(data_pd_HW.index, freq="M"))[['values']].min() #proxy for HW at neap tide
    LW_monthmax_permonth = data_pd_LW.groupby(pd.PeriodIndex(data_pd_LW.index, freq="M"))[['values']].max() #proxy for LW at neap tide
    
    #replace invalids with nan (in case of too less values per month or year)
    if min_coverage is not None:
        assert 0 <= min_coverage <= 1
        # count timeseries values per year/month (first drop nans)
        df_ext_nonan = df_ext.loc[~df_ext["values"].isnull()]
        ext_count_peryear = df_ext_nonan.groupby(pd.PeriodIndex(df_ext_nonan.index, freq="Y"))['values'].count()
        ext_count_permonth = df_ext_nonan.groupby(pd.PeriodIndex(df_ext_nonan.index, freq="M"))['values'].count()
        
        # compute expected counts and multiply with min_coverage to get minimal counts
        min_count_peryear = compute_expected_counts(df_ext, freq="Y") * min_coverage
        min_count_permonth = compute_expected_counts(df_ext, freq="M") * min_coverage
        
        # set all statistics that were based on too little values to nan
        HW_mean_peryear.loc[ext_count_peryear<min_count_peryear] = np.nan
        LW_mean_peryear.loc[ext_count_peryear<min_count_peryear] = np.nan
        HW_monthmax_permonth.loc[ext_count_permonth<min_count_permonth] = np.nan
        LW_monthmin_permonth.loc[ext_count_permonth<min_count_permonth] = np.nan
        HW_monthmin_permonth.loc[ext_count_permonth<min_count_permonth] = np.nan
        LW_monthmax_permonth.loc[ext_count_permonth<min_count_permonth] = np.nan
    
    # make periodindex in all dataframes/series contiguous
    HW_mean_peryear = make_periodindex_contiguous(HW_mean_peryear)
    LW_mean_peryear = make_periodindex_contiguous(LW_mean_peryear)
    HW_monthmax_permonth = make_periodindex_contiguous(HW_monthmax_permonth)
    LW_monthmin_permonth = make_periodindex_contiguous(LW_monthmin_permonth)
    HW_monthmin_permonth = make_periodindex_contiguous(HW_monthmin_permonth)
    LW_monthmax_permonth = make_periodindex_contiguous(LW_monthmax_permonth)
    
    #derive GHHW/GHWS (gemiddeld hoogwater springtij)
    HW_monthmax_mean_peryear = HW_monthmax_permonth.groupby(pd.PeriodIndex(HW_monthmax_permonth.index, freq="Y"))[['values']].mean()
    LW_monthmin_mean_peryear = LW_monthmin_permonth.groupby(pd.PeriodIndex(LW_monthmin_permonth.index, freq="Y"))[['values']].mean()
    HW_monthmin_mean_peryear = HW_monthmin_permonth.groupby(pd.PeriodIndex(HW_monthmin_permonth.index, freq="Y"))[['values']].mean()
    LW_monthmax_mean_peryear = LW_monthmax_permonth.groupby(pd.PeriodIndex(LW_monthmax_permonth.index, freq="Y"))[['values']].mean()
    
    dict_HWLWtidalindicators = {'HW_mean':data_pd_HW['values'].mean(), #GHW
                                'LW_mean':data_pd_LW['values'].mean(), #GLW
                                'HW_mean_peryear':HW_mean_peryear['values'], #GHW peryear
                                'LW_mean_peryear':LW_mean_peryear['values'], #GLW peryear
                                'HW_monthmax_permonth':HW_monthmax_permonth['values'], #GHHW/GHWS permonth
                                'LW_monthmin_permonth':LW_monthmin_permonth['values'], #GLLW/GLWS permonth
                                'HW_monthmax_mean_peryear':HW_monthmax_mean_peryear['values'], #GHHW/GHWS peryear
                                'LW_monthmin_mean_peryear':LW_monthmin_mean_peryear['values'], #GLLW/GLWS peryear
                                'HW_monthmin_mean_peryear':HW_monthmin_mean_peryear['values'], #GLHW/GHWN peryear
                                'LW_monthmax_mean_peryear':LW_monthmax_mean_peryear['values'], #GHLW/GLWN peryear
                                }

    return dict_HWLWtidalindicators


def calc_wltidalindicators(data_wl_pd, min_coverage:float = None):
    """
    computes monthly and yearly means from waterlevel timeseries

    Parameters
    ----------
    data_wl_pd : TYPE
        DESCRIPTION.
    min_coverage : float, optional
        The minimum percentage (from 0 to 1) of timeseries coverage to consider the statistics to be valid. The default is None.

    Returns
    -------
    dict_wltidalindicators : TYPE
        DESCRIPTION.

    """
    
    # dropping the timezone makes the code below much faster and gives equal results: https://github.com/pandas-dev/pandas/issues/58956
    if data_wl_pd.index.tz is not None:
        data_wl_pd = data_wl_pd.tz_localize(None)
    
    # yearmean wl from wl values
    wl_mean_peryear = data_wl_pd.groupby(pd.PeriodIndex(data_wl_pd.index, freq="Y"))[['values']].mean()
    wl_mean_permonth = data_wl_pd.groupby(pd.PeriodIndex(data_wl_pd.index, freq="M"))[['values']].mean()
    
    # replace invalids with nan (in case of too less values per month or year)
    if min_coverage is not None:
        assert 0 <= min_coverage <= 1
        # count timeseries values per year/month (first drop nans)
        data_wl_pd_nonan = data_wl_pd.loc[~data_wl_pd["values"].isnull()]
        wl_count_peryear = data_wl_pd_nonan.groupby(pd.PeriodIndex(data_wl_pd_nonan.index, freq="Y"))['values'].count()
        wl_count_permonth = data_wl_pd_nonan.groupby(pd.PeriodIndex(data_wl_pd_nonan.index, freq="M"))['values'].count()
        
        # compute expected counts and multiply with min_coverage to get minimal counts
        min_count_peryear = compute_expected_counts(data_wl_pd, freq="Y") * min_coverage
        min_count_permonth = compute_expected_counts(data_wl_pd, freq="M") * min_coverage
        
        # set all statistics that were based on too little values to nan
        wl_mean_peryear.loc[wl_count_peryear<min_count_peryear] = np.nan
        wl_mean_permonth.loc[wl_count_permonth<min_count_permonth] = np.nan
    
    # make periodindex in all dataframes/series contiguous
    wl_mean_peryear = make_periodindex_contiguous(wl_mean_peryear)
    wl_mean_permonth = make_periodindex_contiguous(wl_mean_permonth)

    dict_wltidalindicators = {'wl_mean':data_wl_pd['values'].mean(),
                              'wl_mean_peryear':wl_mean_peryear['values'], #yearly mean wl
                              'wl_mean_permonth':wl_mean_permonth['values'], #monthly mean wl
                              }

    return dict_wltidalindicators


def make_periodindex_contiguous(df):
    assert isinstance(df.index, pd.PeriodIndex)
    period_index_full = pd.period_range(df.index.min(), df.index.max(), freq=df.index.freq)
    if isinstance(df, pd.Series):
        df_full = pd.Series(df, index=period_index_full)
    elif isinstance(df, pd.DataFrame):
        df_full = pd.DataFrame(df, index=period_index_full)
    
    # add attrs from input dataframe
    df_full.attrs = df.attrs
    return df_full


def compute_expected_counts(df_meas, freq):
    """
    Compute the expected number of values for all years/months in a timeseries index,
    by taking the number of days for each year/month and dividing it by the median frequency in that period.
    """
    # TODO: beware of series with e.g. only first and last value of month, this will result in freq=30days and then expected count of 1, it will pass even if there is almost no data
    df_meas = df_meas.copy()
    df_meas["timediff"] = df_meas.index.diff() # TODO: not supported by pandas<2.2.0: https://github.com/Deltares-research/kenmerkendewaarden/blob/d7f8f5f3f915dd897e9aa037fad67e1920ff5cbf/kenmerkendewaarden/data_analysis.py#L152
    period_index = pd.PeriodIndex(df_meas.index, freq=freq)
    # compute median freq, the mean could be skewed in case of large gaps
    median_freq = df_meas.groupby(period_index)['timediff'].median()
    if freq=="Y":
        days_inperiod = median_freq.index.dayofyear
    elif freq=="M":
        days_inperiod = median_freq.index.daysinmonth
    else:
        raise ValueError(f"invalid freq: '{freq}'")
    days_inperiod_td = pd.to_timedelta(days_inperiod, unit='D')
    expected_count = days_inperiod_td / median_freq
    return expected_count


def plot_pd_series(indicators_dict, ax):
    for key in indicators_dict.keys():
        value = indicators_dict[key]
        if not isinstance(value, pd.Series):
            continue
        if key.endswith("peryear"):
            linestyle = "-"
        elif key.endswith("permonth"):
            linestyle = "--"
        value.plot(ax=ax, label=key, linestyle=linestyle, marker='.')
        xmin = value.index.min()
        xmax = value.index.max()
    
    # separate loop for floats to make sure the xlim is already correct
    for key in indicators_dict.keys():
        value = indicators_dict[key]
        if not isinstance(value, float):
            continue
        ax.hlines(value, xmin, xmax, linestyle="--", color="k", label=key, zorder=1)


def plot_tidalindicators(indicators_wl:dict = None, indicators_ext = None):
    """
    plot tidalindicators

    Parameters
    ----------
    indicators_wl : dict, optional
        Dictionary as returned from `kw.calc_wltidalindicators()`. The default is None.
    indicators_ext : TYPE, optional
        Dictionary as returned from `kw.calc_HWLWtidalindicators()`. The default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.
    ax : matplotlib.axes._axes.Axes
        Figure axis handle.

    """
        
    fig, ax = plt.subplots(figsize=(12,6))
        
    if indicators_wl is not None:
        # TODO: maybe add an escape for if the station attr is not present
        station = indicators_wl['wl_mean_peryear'].attrs["station"]
        plot_pd_series(indicators_wl, ax)
    if indicators_ext is not None:
        # TODO: maybe add an escape for if the station attr is not present
        station = indicators_ext['HW_mean_peryear'].attrs["station"]
        plot_pd_series(indicators_ext, ax)
    
    ax.grid()
    ax.legend(loc=1)
    ax.set_ylabel("water level [m]")
    ax.set_title(f"tidal indicators for {station}")
    fig.tight_layout()
    return fig, ax


def calc_HWLWtidalrange(ts_ext):
    """
    creates column 'tidalrange' in ts_ext dataframe
    """
    raise_extremes_with_aggers(ts_ext)
    
    ts_ext = hatyan.calc_HWLWnumbering(ts_ext=ts_ext)
    ts_ext['times_backup'] = ts_ext.index
    ts_ext_idxHWLWno = ts_ext.set_index('HWLWno',drop=False)
    ts_ext_idxHWLWno['tidalrange'] = ts_ext_idxHWLWno.loc[ts_ext_idxHWLWno['HWLWcode']==1,'values'] - ts_ext_idxHWLWno.loc[ts_ext_idxHWLWno['HWLWcode']==2,'values']
    ts_ext = ts_ext_idxHWLWno.set_index('times_backup')
    ts_ext.index.name = 'times'
    return ts_ext


def calc_hat_lat_fromcomponents(comp: pd.DataFrame) -> tuple:
    """
    Derive highest and lowest astronomical tide (HAT/LAT) from a component set.
    The component set is used to make a tidal prediction for an arbitrary period of 19 years with a 10 minute interval. 
    The max/min values of the predictions of all years are the HAT/LAT values.
    The HAT/LAT is very dependent on the A0 of the component set. Therefore, the HAT/LAT values are 
    relevant for the same year as the slotgemiddelde that is used to replace A0 in the component set. 
    For instance, if the slotgemiddelde is valid for 2021.0, HAT and LAT are also relevant for that year.
    It is important to use the same tidal prediction settings as used to derive the tidal components.
    
    Parameters
    ----------
    comp : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    tuple
        hat and lat values.

    """
    
    min_vallist_allyears = pd.Series(dtype=float)
    max_vallist_allyears = pd.Series(dtype=float)
    logger.info("generating prediction for 19 years")
    for year in range(2020,2039): # 19 arbitrary consequtive years to capture entire nodal cycle
        times_pred_all = pd.date_range(start=dt.datetime(year,1,1), end=dt.datetime(year+1,1,1), freq="10min")
        ts_prediction = hatyan.prediction(comp=comp, times=times_pred_all)
        
        min_vallist_allyears.loc[year] = ts_prediction['values'].min()
        max_vallist_allyears.loc[year] = ts_prediction['values'].max()
    
    logger.info("deriving hat/lat")
    hat = max_vallist_allyears.max()
    lat = min_vallist_allyears.min()
    return hat, lat


def calc_hat_lat_frommeasurements(df_meas_19y: pd.DataFrame) -> tuple:
    """
    Derive highest and lowest astronomical tide (HAT/LAT) from a measurement timeseries of 19 years.
    Tidal components are derived for each year of the measurement timeseries.
    The resulting component sets are used to make a tidal prediction each year of the measurement timeseries with a 10 minute interval. 
    The max/min values of the predictions of all years are the HAT/LAT values.
    The HAT/LAT is very dependent on the A0 of the component sets. Therefore, the HAT/LAT values are 
    relevant for the same period as the measurement timeseries. 

    Parameters
    ----------
    df_meas_19y : pd.DataFrame
        Measurements timeseries of 19 years.

    Returns
    -------
    tuple
        hat and lat values.

    """
    
    num_years = len(df_meas_19y.index.year.unique())
    if num_years != 19:
        raise ValueError(f"please provide a timeseries of 19 years instead of {num_years} years")
    
    # TODO: fu_alltimes=False makes the process significantly faster (default is True)
    # TODO: xfac actually varies between stations (default is False), but different xfac has only very limited impact on the resulting hat/lat values
    _, comp_all = hatyan.analysis(df_meas_19y, const_list="year", analysis_perperiod="Y", return_allperiods=True, fu_alltimes=False)
    
    # TODO: a frequency of 1min is better in theory, but 10min is faster and hat/lat values differ only 2mm for HOEKVHLD
    df_pred = hatyan.prediction(comp_all, timestep="10min")
    
    lat = df_pred["values"].min()
    hat = df_pred["values"].max()
    return hat, lat

