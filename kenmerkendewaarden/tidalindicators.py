# -*- coding: utf-8 -*-
"""
Computation of tidal indicators from waterlevel extremes or timeseries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hatyan
import logging
from kenmerkendewaarden.utils import raise_extremes_with_aggers, raise_empty_df

__all__ = [
    "calc_wltidalindicators",
    "calc_HWLWtidalindicators",
    "plot_tidalindicators",
    "calc_HWLWtidalrange",
]

logger = logging.getLogger(__name__)


def calc_HWLWtidalindicators(df_ext: pd.DataFrame, min_coverage: float = None):
    """
    Computes several tidal extreme indicators from tidal extreme dataset.

    Parameters
    ----------
    df_ext : pd.DataFrame
        Dataframe with extremes timeseries.
    min_coverage : float, optional
        The minimal required coverage (between 0 to 1) of the df_ext timeseries to
        consider the statistics to be valid. It is the factor between the actual amount
        and the expected amount of high waters in the series. Note that the expected
        amount is not an exact extimate, so min_coverage=1 will probably result in nans
        even though all extremes are present. The default is None.

    Returns
    -------
    dict_tidalindicators : dict
        Dictionary with several tidal indicators like yearly/monthly means.

    """
    raise_empty_df(df_ext)
    # dropping the timezone makes the code below much faster and gives equal results
    # https://github.com/pandas-dev/pandas/issues/58956
    if df_ext.index.tz is not None:
        df_ext = df_ext.tz_localize(None)

    raise_extremes_with_aggers(df_ext)

    # split to HW and LW separately, also groupby year
    data_pd_hw = df_ext.loc[df_ext["HWLWcode"] == 1]["values"]
    data_pd_lw = df_ext.loc[df_ext["HWLWcode"] == 2]["values"]

    # yearmean HWLW from HWLW values #maybe also add *_mean_permonth
    pi_hw_y = pd.PeriodIndex(data_pd_hw.index, freq="Y")
    pi_lw_y = pd.PeriodIndex(data_pd_lw.index, freq="Y")
    HW_mean_peryear = data_pd_hw.groupby(pi_hw_y).mean()
    LW_mean_peryear = data_pd_lw.groupby(pi_lw_y).mean()

    # derive GHHW/GHWS (gemiddeld hoogwater springtij) per month
    pi_hw_m = pd.PeriodIndex(data_pd_hw.index, freq="M")
    pi_lw_m = pd.PeriodIndex(data_pd_lw.index, freq="M")
    # proxy for HW at spring tide
    HW_monthmax_permonth = data_pd_hw.groupby(pi_hw_m).max()
    # proxy for LW at spring tide
    LW_monthmin_permonth = data_pd_lw.groupby(pi_lw_m).min()
    # proxy for HW at neap tide
    HW_monthmin_permonth = data_pd_hw.groupby(pi_hw_m).min()
    # proxy for LW at neap tide
    LW_monthmax_permonth = data_pd_lw.groupby(pi_lw_m).max()

    ser_list = [
        HW_mean_peryear,
        LW_mean_peryear,
        HW_monthmax_permonth,
        LW_monthmin_permonth,
        HW_monthmin_permonth,
        LW_monthmax_permonth,
    ]
    for ser_one in ser_list:
        ser_one.index.name = "period"

    # replace invalids with nan (in case of too less values per month or year)
    if min_coverage is not None:
        assert 0 <= min_coverage <= 1
        # count timeseries values per year/month
        ext_count_peryear = compute_actual_counts(data_pd_hw, freq="Y")
        ext_count_permonth = compute_actual_counts(data_pd_hw, freq="M")

        # compute expected counts and multiply with min_coverage to get minimal counts
        min_count_peryear = compute_expected_counts(data_pd_hw, freq="Y")
        min_count_permonth = compute_expected_counts(data_pd_hw, freq="M")
        # floor expected counts to avoid rounding issues
        min_count_peryear = min_count_peryear.apply(np.floor) * min_coverage
        min_count_permonth = min_count_permonth.apply(np.floor) * min_coverage

        # set all statistics that were based on too little values to nan
        HW_mean_peryear.loc[ext_count_peryear < min_count_peryear] = np.nan
        LW_mean_peryear.loc[ext_count_peryear < min_count_peryear] = np.nan
        HW_monthmax_permonth.loc[ext_count_permonth < min_count_permonth] = np.nan
        LW_monthmin_permonth.loc[ext_count_permonth < min_count_permonth] = np.nan
        HW_monthmin_permonth.loc[ext_count_permonth < min_count_permonth] = np.nan
        LW_monthmax_permonth.loc[ext_count_permonth < min_count_permonth] = np.nan

    # make periodindex in all dataframes/series contiguous
    HW_mean_peryear = make_periodindex_contiguous(HW_mean_peryear)
    LW_mean_peryear = make_periodindex_contiguous(LW_mean_peryear)
    HW_monthmax_permonth = make_periodindex_contiguous(HW_monthmax_permonth)
    LW_monthmin_permonth = make_periodindex_contiguous(LW_monthmin_permonth)
    HW_monthmin_permonth = make_periodindex_contiguous(HW_monthmin_permonth)
    LW_monthmax_permonth = make_periodindex_contiguous(LW_monthmax_permonth)

    # derive GHHW/GHWS (proxy for gemiddeld hoogwater/laagwater springtij/doodtij)
    pi_hw_mmax_pm_y = pd.PeriodIndex(HW_monthmax_permonth.index, freq="Y")
    pi_lw_mmax_pm_y = pd.PeriodIndex(LW_monthmax_permonth.index, freq="Y")
    pi_hw_mmin_pm_y = pd.PeriodIndex(HW_monthmin_permonth.index, freq="Y")
    pi_lw_mmin_pm_y = pd.PeriodIndex(LW_monthmin_permonth.index, freq="Y")
    HW_monthmax_mean_peryear = HW_monthmax_permonth.groupby(pi_hw_mmax_pm_y).mean()
    LW_monthmax_mean_peryear = LW_monthmax_permonth.groupby(pi_lw_mmax_pm_y).mean()
    HW_monthmin_mean_peryear = HW_monthmin_permonth.groupby(pi_hw_mmin_pm_y).mean()
    LW_monthmin_mean_peryear = LW_monthmin_permonth.groupby(pi_lw_mmin_pm_y).mean()

    dict_tidalindicators = {
        "HW_mean": data_pd_hw.mean(),  # GHW
        "LW_mean": data_pd_lw.mean(),  # GLW
        "HW_mean_peryear": HW_mean_peryear,  # GHW peryear
        "LW_mean_peryear": LW_mean_peryear,  # GLW peryear
        "HW_monthmax_permonth": HW_monthmax_permonth,  # GHHW/GHWS permonth
        "LW_monthmin_permonth": LW_monthmin_permonth,  # GLLW/GLWS permonth
        "HW_monthmax_mean_peryear": HW_monthmax_mean_peryear,  # GHHW/GHWS peryear
        "LW_monthmax_mean_peryear": LW_monthmax_mean_peryear,  # GHLW/GLWN peryear
        "HW_monthmin_mean_peryear": HW_monthmin_mean_peryear,  # GLHW/GHWN peryear
        "LW_monthmin_mean_peryear": LW_monthmin_mean_peryear,  # GLLW/GLWS peryear
    }

    return dict_tidalindicators


def calc_wltidalindicators(df_meas: pd.DataFrame, min_coverage: float = None):
    """
    Computes monthly and yearly means from waterlevel timeseries.

    Parameters
    ----------
    df_meas : pd.DataFrame
        Dataframe with waterlevel timeseries.
    min_coverage : float, optional
        The minimum percentage (from 0 to 1) of timeseries coverage to consider the
        statistics to be valid. The default is None.

    Returns
    -------
    dict_tidalindicators : dict
        Dictionary with several tidal indicators like yearly/monthly means.

    """
    raise_empty_df(df_meas)

    # dropping the timezone makes the code below much faster and gives equal results
    # https://github.com/pandas-dev/pandas/issues/58956
    if df_meas.index.tz is not None:
        df_meas = df_meas.tz_localize(None)

    # series from dataframe
    ser_meas = df_meas["values"]

    # yearmean wl from wl values
    pi_wl_y = pd.PeriodIndex(ser_meas.index, freq="Y")
    pi_wl_m = pd.PeriodIndex(ser_meas.index, freq="M")
    wl_mean_peryear = ser_meas.groupby(pi_wl_y).mean()
    wl_mean_peryear.index.name = "period"
    wl_mean_permonth = ser_meas.groupby(pi_wl_m).mean()
    wl_mean_permonth.index.name = "period"

    # replace invalids with nan (in case of too less values per month or year)
    if min_coverage is not None:
        assert 0 <= min_coverage <= 1
        # count timeseries values per year/month
        wl_count_peryear = compute_actual_counts(ser_meas, freq="Y")
        wl_count_permonth = compute_actual_counts(ser_meas, freq="M")

        # compute expected counts and multiply with min_coverage to get minimal counts
        min_count_peryear = compute_expected_counts(ser_meas, freq="Y") * min_coverage
        min_count_permonth = compute_expected_counts(ser_meas, freq="M") * min_coverage

        # set all statistics that were based on too little values to nan
        wl_mean_peryear.loc[wl_count_peryear < min_count_peryear] = np.nan
        wl_mean_permonth.loc[wl_count_permonth < min_count_permonth] = np.nan

    # make periodindex in all dataframes/series contiguous
    wl_mean_peryear = make_periodindex_contiguous(wl_mean_peryear)
    wl_mean_permonth = make_periodindex_contiguous(wl_mean_permonth)

    dict_tidalindicators = {
        "wl_mean": ser_meas.mean(),
        "wl_mean_peryear": wl_mean_peryear,  # yearly mean wl
        "wl_mean_permonth": wl_mean_permonth,  # monthly mean wl
    }

    return dict_tidalindicators


def compute_actual_counts(ser_meas, freq):
    """
    Compute the number of non-nan values in a column for all years/months in a
    timeseries index.
    """
    ser_meas_isnotnull = ~ser_meas.isnull()
    period_index = pd.PeriodIndex(ser_meas_isnotnull.index, freq=freq)
    ser_actual_counts = ser_meas_isnotnull.groupby(period_index).sum()
    return ser_actual_counts


def compute_expected_counts(ser_meas, freq):
    """
    Compute the expected number of values for all years/months in a timeseries index,
    by taking the number of days for each year/month and dividing it by the median
    frequency in that period.
    This functionhas will result in incorrect values for months/years with only a value
    on the first and last timestep, since the derived frequency will be 15/183 days days
    and this will result in 2 expected counts.
    Furthermore, if applied to extremes, it is advisable to compute the expected counts
    on high waters only since these have a more stable period than alternating high/low
    waters. Even then, the expected counts are an indication for extremes.
    """
    # TODO: beware of series with e.g. only first and last value of month/year, this
    # will result in freq=30days and then expected count of 2, it will pass even if
    # there is almost no data
    df_meas = pd.DataFrame(ser_meas)
    df_meas["timediff"] = df_meas.index.diff()
    period_index = pd.PeriodIndex(df_meas.index, freq=freq)
    # compute median freq, the mean could be skewed in case of large gaps
    median_freq = df_meas.groupby(period_index)["timediff"].median()
    if freq == "Y":
        days_inperiod = median_freq.index.dayofyear
    elif freq == "M":
        days_inperiod = median_freq.index.daysinmonth
    else:
        raise ValueError(f"invalid freq: '{freq}'")
    days_inperiod_td = pd.to_timedelta(days_inperiod, unit="D")
    ser_expected_counts = days_inperiod_td / median_freq
    return ser_expected_counts


def make_periodindex_contiguous(df):
    assert isinstance(df.index, pd.PeriodIndex)
    period_index_full = pd.period_range(
        df.index.min(), df.index.max(), freq=df.index.freq
    )
    if isinstance(df, pd.Series):
        df_full = pd.Series(df, index=period_index_full)
    elif isinstance(df, pd.DataFrame):
        df_full = pd.DataFrame(df, index=period_index_full)

    # add attrs from input dataframe
    df_full.attrs = df.attrs
    df_full.index.name = df.index.name
    return df_full


def plot_pd_series(indicators_dict, ax):
    for key in indicators_dict.keys():
        value = indicators_dict[key]
        if not isinstance(value, pd.Series):
            continue
        if key.endswith("peryear"):
            linestyle = "-"
        elif key.endswith("permonth"):
            linestyle = "--"
        value.plot(ax=ax, label=key, linestyle=linestyle, marker=".")
        xmin = value.index.min()
        xmax = value.index.max()

    # separate loop for floats to make sure the xlim is already correct
    for key in indicators_dict.keys():
        value = indicators_dict[key]
        if not isinstance(value, float):
            continue
        ax.hlines(value, xmin, xmax, linestyle="--", color="k", label=key, zorder=1)


def plot_tidalindicators(dict_indicators: dict):
    """
    Plot tidalindicators.

    Parameters
    ----------
    dict_indicators : dict, optional
        Dictionary as returned from `kw.calc_wltidalindicators()` and/or
        `kw.calc_HWLWtidalindicators()`. The default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.
    ax : matplotlib.axes._axes.Axes
        Figure axis handle.

    """

    # get and compare station attributes
    station_attrs = [
        v.attrs["station"] for k, v in dict_indicators.items() if hasattr(v, "attrs")
    ]
    assert all(x == station_attrs[0] for x in station_attrs)
    station = station_attrs[0]

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_pd_series(dict_indicators, ax)
    ax.grid()
    ax.legend(loc=1)
    ax.set_ylabel("water level [cm]")
    ax.set_title(f"tidal indicators for {station}")
    fig.tight_layout()
    return fig, ax


def calc_HWLWtidalrange(df_ext: pd.DataFrame):
    """
    Compute the difference between a high water and the following low water.
    This tidal range is added as a column to the df_ext dataframe.

    Parameters
    ----------
    df_ext : pd.DataFrame
        Dataframe with extremes timeseries.

    Returns
    -------
    df_ext : pd.DataFrame
        Input dataframe enriched with 'tidalindicators' and 'HWLWno' columns.

    """
    raise_empty_df(df_ext)
    raise_extremes_with_aggers(df_ext)

    df_ext = hatyan.calc_HWLWnumbering(ts_ext=df_ext)
    df_ext["times_backup"] = df_ext.index
    df_ext_idxHWLWno = df_ext.set_index("HWLWno", drop=False)
    df_ext_idxHWLWno["tidalrange"] = (
        df_ext_idxHWLWno.loc[df_ext_idxHWLWno["HWLWcode"] == 1, "values"]
        - df_ext_idxHWLWno.loc[df_ext_idxHWLWno["HWLWcode"] == 2, "values"]
    )
    df_ext = df_ext_idxHWLWno.set_index("times_backup")
    df_ext.index.name = "times"
    return df_ext


def calc_getijcomponenten(
    df_meas,
    const_list=None,
    analysis_perperiod="Y",
    return_allperiods=False,
):
    raise_empty_df(df_meas)
    # RWS-default settings
    if const_list is None:
        const_list = hatyan.get_const_list_hatyan("year")
    # nodalfactors=True is RWS and hatyan default
    nodalfactors = True
    # fu_alltimes=False makes the hat/lat process significantly faster (default is True)
    fu_alltimes = False
    # xfac actually varies between stations (default is False), but different xfac
    # has very limited impact on the resulting hat/lat values
    xfac = True

    # analysis
    comp_avg = hatyan.analysis(
        ts=df_meas,
        const_list=const_list,
        nodalfactors=nodalfactors,
        fu_alltimes=fu_alltimes,
        xfac=xfac,
        analysis_perperiod=analysis_perperiod,
        return_allperiods=return_allperiods,
    )
    return comp_avg
