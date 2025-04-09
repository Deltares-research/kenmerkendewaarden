# -*- coding: utf-8 -*-
"""
Computation of havengetallen
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import logging
from hatyan.astrog import astrog_culminations
from hatyan.timeseries import calc_HWLWnumbering
from kenmerkendewaarden.tidalindicators import (
    calc_HWLWtidalrange,
    compute_actual_counts,
    compute_expected_counts,
)
from kenmerkendewaarden.utils import (
    raise_extremes_with_aggers,
    crop_timeseries_last_nyears,
    TimeSeries_TimedeltaFormatter_improved,
)
from matplotlib.ticker import MultipleLocator

__all__ = [
    "calc_havengetallen",
    "calc_HWLW_springneap",
    "plot_HWLW_pertimeclass",
    "plot_aardappelgrafiek",
]

logger = logging.getLogger(__name__)


def calc_havengetallen(
    df_ext: pd.DataFrame,
    return_df_ext=False,
    min_coverage=None,
    moonculm_offset: int = 4,
):
    """
    havengetallen consist of the extreme (high and low) median values and the
    extreme median time delays with respect to the moonculmination.
    Besides that it computes the tide difference for each cycle and the tidal period.
    All these indicators are derived by dividing the extremes in hour-classes
    with respect to the moonculminination.

    Parameters
    ----------
    df_ext : pd.DataFrame
        DataFrame with extremes (highs and lows, no aggers). The last 10 years of this 
        timeseries are used to compute the havengetallen.
    return_df : bool
        Whether to return the enriched input dataframe. Default is False.
    min_coverage : float, optional
        The minimal required coverage (between 0 to 1) of the df_ext timeseries to
        consider the statistics to be valid. It is the factor between the actual amount
        and the expected amount of high waters in the series. Note that the expected
        amount is not an exact extimate, so min_coverage=1 will probably result in nans
        even though all extremes are present. The default is None.
    moonculm_offset : int, optional
        Offset between moonculmination and extremes. Passed on to
        `calc_HWLW_moonculm_combi`. The default is 4, which corresponds to a 2-day
        offset, which is applicable to the Dutch coast.

    Returns
    -------
    df_havengetallen : pd.DataFrame
        DataFrame with havengetallen for all hour-classes.
        0 corresponds to spring, 6 corresponds to neap, mean is mean.
    df_ext_culm : pd.DataFrame
        An enriched copy of the input DataFrame including a 'culm_hr' column.

    """
    raise_extremes_with_aggers(df_ext)
    df_ext_10y = crop_timeseries_last_nyears(df=df_ext, nyears=10)

    # check if coverage is high enough
    if min_coverage is not None:
        check_min_coverage_extremes(df_ext=df_ext_10y, min_coverage=min_coverage)
    
    current_station = df_ext_10y.attrs["station"]
    logger.info(f"computing havengetallen for {current_station}")
    df_ext_culm = calc_hwlw_moonculm_combi(
        df_ext=df_ext_10y,
        moonculm_offset=moonculm_offset,
        )
    df_havengetallen = calc_HWLW_culmhr_summary(df_ext_culm)
    logger.info("computing havengetallen done")
    if return_df_ext:
        return df_havengetallen, df_ext_culm
    else:
        return df_havengetallen
    

def calc_HWLW_springneap(
        df_ext: pd.DataFrame,
        min_coverage=None,
        moonculm_offset: int = 4):
    raise_extremes_with_aggers(df_ext)
    
    # TODO: moonculminations cannot be computed before 1900
    # https://github.com/Deltares-research/kenmerkendewaarden/issues/184
    if df_ext.index.min().year < 1901:
        logger.warning("calc_HWLW_springneap() only supports timestamps after 1900 "
                       "all older data will be ignored")
        df_ext = df_ext.loc["1901":]

    # check if coverage is high enough
    if min_coverage is not None:
        check_min_coverage_extremes(df_ext=df_ext, min_coverage=min_coverage)
    
    current_station = df_ext.attrs["station"]
    logger.info(f"computing HWLW for spring/neap tide for {current_station}")
    df_ext_culm = calc_hwlw_moonculm_combi(
        df_ext=df_ext,
        moonculm_offset=moonculm_offset,
        )

    # all HW/LW at spring/neaptide
    bool_hw = df_ext_culm["HWLWcode"] == 1
    bool_lw = df_ext_culm["HWLWcode"] == 2
    bool_spring = df_ext_culm["culm_hr"] == 0
    bool_neap = df_ext_culm["culm_hr"] == 6
    hw_spring = df_ext_culm.loc[bool_hw & bool_spring]['values']
    lw_spring = df_ext_culm.loc[bool_lw & bool_spring]['values']
    hw_neap = df_ext_culm.loc[bool_hw & bool_neap]['values']
    lw_neap = df_ext_culm.loc[bool_lw & bool_neap]['values']
    
    # mean HW/LW at spring/neap tide
    pi_hw_sp_y = pd.PeriodIndex(hw_spring.index, freq="Y")
    pi_lw_sp_y = pd.PeriodIndex(lw_spring.index, freq="Y")
    pi_hw_np_y = pd.PeriodIndex(hw_neap.index, freq="Y")
    pi_lw_np_y = pd.PeriodIndex(lw_neap.index, freq="Y")
    hw_spring_peryear = hw_spring.groupby(pi_hw_sp_y).mean()
    lw_spring_peryear = lw_spring.groupby(pi_lw_sp_y).mean()
    hw_neap_peryear = hw_neap.groupby(pi_hw_np_y).mean()
    lw_neap_peryear = lw_neap.groupby(pi_lw_np_y).mean()
    
    # merge in dict
    dict_hwlw_springneap = {
        "HW_spring_mean_peryear": hw_spring_peryear,
        "LW_spring_mean_peryear": lw_spring_peryear,
        "HW_neap_mean_peryear": hw_neap_peryear,
        "LW_neap_mean_peryear": lw_neap_peryear,
    }
    return dict_hwlw_springneap


def check_min_coverage_extremes(df_ext, min_coverage):
    # TODO: compute_actual_counts only returns years for which there are no nans, so
    # will have different length than expected counts if there is an all-nan year
    # TODO: if we supply 4 years of complete data instead of 10 years, no error raised
    data_pd_hw = df_ext.loc[df_ext["HWLWcode"] == 1]["values"]
    df_actual_counts_peryear = compute_actual_counts(data_pd_hw, freq="Y")
    df_expected_counts_peryear = compute_expected_counts(data_pd_hw, freq="Y")
    # floor expected counts to avoid rounding issues
    df_expected_counts_peryear = df_expected_counts_peryear.apply(np.floor)
    df_min_counts_peryear = df_expected_counts_peryear * min_coverage
    bool_coverage_toolow = df_actual_counts_peryear < df_min_counts_peryear
    df_debug = pd.DataFrame(
        {
            "#required": df_min_counts_peryear.loc[bool_coverage_toolow],
            "#actual": df_actual_counts_peryear.loc[bool_coverage_toolow],
        }
    )
    if bool_coverage_toolow.any():
        raise ValueError(
            f"coverage of some years is lower than "
            f"min_coverage={min_coverage}:\n{df_debug}"
        )


def get_moonculm_idxHWLWno(tstart, tstop):
    # in UTC, which is important since data_pd_HWLW['culm_hr']=range(12) hourvalues 
    # should be in UTC since that relates to the relation dateline/sun
    data_pd_moonculm = astrog_culminations(tFirst=tstart, tLast=tstop)
    data_pd_moonculm = data_pd_moonculm.tz_convert("UTC")  # convert to UTC (is already)
    data_pd_moonculm["datetime"] = data_pd_moonculm.index
    # dummy values for TA in hatyan.calc_HWLWnumbering()
    data_pd_moonculm["values"] = data_pd_moonculm["type"]
    data_pd_moonculm["HWLWcode"] = 1  # all HW values since one every ~12h25m
    # TODO: currently w.r.t. cadzd, is that an issue? With DELFZL the matched
    # culmination is incorrect (since far away), but that might not be a big issue
    data_pd_moonculm = calc_HWLWnumbering(data_pd_moonculm)
    moonculm_idxHWLWno = data_pd_moonculm.set_index("HWLWno")
    return moonculm_idxHWLWno


def calc_hwlw_moonculm_combi(df_ext: pd.DataFrame, moonculm_offset: int = 4):
    """
    Links moonculminations to each extreme. All low waters correspond to the same
    moonculmination as the preceding high water. Computes the time differences between
    moonculminations and extreme and several other times and durations.

    Parameters
    ----------
    df_ext : pd.DataFrame
        DataFrame with extremes (highs and lows, no aggers).
    moonculm_offset : int, optional
        The extremes of a Dutch station are related to the moonculmination two days
        before, so the fourth extreme after a certain moonculmination is related to that
        moonculmination. For more northward stations, one could consider using the 5th
        extreme after a certain moonculmination. This number rotates the
        aardappelgrafiek, and impacts its shape. The default is 4.

    Returns
    -------
    df_ext_moon : pd.DataFrame
        Copy of the input dataframe enriched with several columns related to the
        moonculminations.

    """
    
    moonculm_idxHWLWno = get_moonculm_idxHWLWno(
        tstart=df_ext.index.min() - dt.timedelta(days=3),
        tstop=df_ext.index.max(),
    )
    # correlate HWLW to moonculmination 2 days before.
    moonculm_idxHWLWno.index = moonculm_idxHWLWno.index + moonculm_offset
    
    df_ext_idxHWLWno = calc_HWLWnumbering(df_ext)
    df_ext_idxHWLWno["times"] = df_ext_idxHWLWno.index
    df_ext_idxHWLWno = df_ext_idxHWLWno.set_index("HWLWno", drop=False)

    hw_bool = df_ext_idxHWLWno["HWLWcode"] == 1
    # computing getijperiod like this works properly since index is HWLW
    getijperiod = df_ext_idxHWLWno.loc[hw_bool, "times"].diff()
    df_ext_idxHWLWno.loc[hw_bool, "getijperiod"] = getijperiod
    # compute duurdaling
    df_ext_idxHWLWno.loc[hw_bool, "duurdaling"] = (
        df_ext_idxHWLWno.loc[~hw_bool, "times"]
        - df_ext_idxHWLWno.loc[hw_bool, "times"]
    )
    # couple HWLW to moonculminations two days earlier (works because of HWLWno index)
    tz_hwlw = df_ext.index.tz
    culm_time_utc = moonculm_idxHWLWno["datetime"]
    culm_hr = culm_time_utc.dt.round("h").dt.hour % 12
    df_ext_idxHWLWno["culm_time"] = culm_time_utc.dt.tz_convert(tz_hwlw)
    df_ext_idxHWLWno["culm_hr"] = culm_hr
    # compute time of hwlw after moonculmination
    hwlw_delay = df_ext_idxHWLWno["times"] - df_ext_idxHWLWno["culm_time"]
    df_ext_idxHWLWno["HWLW_delay"] = hwlw_delay

    # culm_addtime was an 2d and 2u20min correction, this shifts the x-axis of
    # aardappelgrafiek. We now only do 2d and 1u20m now since we already account for the
    # timezone when computing the timediffs. More information about the effects of
    # moonculm_offset and general time-offsets are documented in
    # https://github.com/Deltares-research/kenmerkendewaarden/issues/164
    # HW is 2 days after culmination (so 4 x 12h25min difference between length of avg
    # moonculm and length of 2 days)
    # 20 minutes (0 to 5 meridian)
    culm_addtime = (
        moonculm_offset * dt.timedelta(hours=12, minutes=25)
        - dt.timedelta(minutes=20)
    )
    df_ext_idxHWLWno["HWLW_delay"] -= culm_addtime

    df_ext_moon = df_ext_idxHWLWno.set_index("times")
    return df_ext_moon


def calc_HWLW_culmhr_summary(data_pd_HWLW):
    logger.info("calculate median per hour group for LW and HW")
    data_pd_HW = data_pd_HWLW.loc[data_pd_HWLW["HWLWcode"] == 1]
    data_pd_LW = data_pd_HWLW.loc[data_pd_HWLW["HWLWcode"] == 2]

    hw_per_culmhr = data_pd_HW.groupby(data_pd_HW["culm_hr"])
    lw_per_culmhr = data_pd_LW.groupby(data_pd_LW["culm_hr"])

    HWLW_culmhr_summary = pd.DataFrame()
    HWLW_culmhr_summary["HW_values_median"] = hw_per_culmhr["values"].median()
    HWLW_culmhr_summary["HW_delay_median"] = hw_per_culmhr["HWLW_delay"].median()
    HWLW_culmhr_summary["LW_values_median"] = lw_per_culmhr["values"].median()
    HWLW_culmhr_summary["LW_delay_median"] = lw_per_culmhr["HWLW_delay"].median()
    HWLW_culmhr_summary["tijverschil"] = (
        HWLW_culmhr_summary["HW_values_median"]
        - HWLW_culmhr_summary["LW_values_median"]
    )
    HWLW_culmhr_summary["getijperiod_median"] = hw_per_culmhr["getijperiod"].median()
    HWLW_culmhr_summary["duurdaling_median"] = hw_per_culmhr["duurdaling"].median()

    # add mean row to dataframe
    HWLW_culmhr_summary.loc["mean", :] = HWLW_culmhr_summary.mean()

    # round all timedeltas to seconds to make outputformat nicer
    for colname in HWLW_culmhr_summary.columns:
        if HWLW_culmhr_summary[colname].dtype == "timedelta64[ns]":
            HWLW_culmhr_summary[colname] = HWLW_culmhr_summary[colname].dt.round("s")

    return HWLW_culmhr_summary


def calc_HWLW_culmhr_summary_tidalcoeff(df_ext):
    # TODO: use tidal coefficient instead?: The tidal coefficient is the size of the
    # tide in relation to its mean. It usually varies between 20 and 120. The higher the
    # tidal coefficient, the larger the tidal range – i.e. the difference in water
    # height between high and low tide. This means that the sea level rises and falls
    # back a long way. The mean value is 70. We talk of strong tides – called spring
    # tides – from coefficient 95.  Conversely, weak tides are called neap tides.
    # https://escales.ponant.com/en/high-low-tide/ en
    # https://www.manche-toerisme.com/springtij
    # for HOEKVHLD, sp=0 is approx tc=1.2, np=6 is approx tc=0.8, av=mean is approx
    # tc=1.0 (for HW, for LW it is different)
    raise_extremes_with_aggers(df_ext)

    data_pd_HWLW = df_ext.copy()
    data_pd_HWLW = calc_HWLWtidalrange(data_pd_HWLW)
    data_pd_HWLW["tidalcoeff"] = (
        data_pd_HWLW["tidalrange"] / data_pd_HWLW["tidalrange"].mean()
    )
    data_pd_HWLW["tidalcoeff_round"] = data_pd_HWLW["tidalcoeff"].round(1)
    TR_groupby_median = data_pd_HWLW.groupby("tidalcoeff_round")["tidalrange"].median()
    HW_groupby_median = (
        data_pd_HWLW.loc[data_pd_HWLW["HWLWcode"] == 1]
        .groupby("tidalcoeff_round")["values"]
        .median()
    )
    LW_groupby_median = (
        data_pd_HWLW.loc[data_pd_HWLW["HWLWcode"] == 2]
        .groupby("tidalcoeff_round")["values"]
        .median()
    )

    HWLW_culmhr_summary = pd.DataFrame()
    HWLW_culmhr_summary["HW_values_median"] = HW_groupby_median
    HWLW_culmhr_summary["LW_values_median"] = LW_groupby_median
    HWLW_culmhr_summary["tidalrange_median"] = TR_groupby_median
    HWLW_culmhr_summary = HWLW_culmhr_summary.loc[
        [0.8, 1.0, 1.2]
    ]  # select neap/mean/springtide
    HWLW_culmhr_summary.index = ["neap", "mean", "spring"]

    return HWLW_culmhr_summary


def plot_HWLW_pertimeclass(df_ext: pd.DataFrame, df_havengetallen: pd.DataFrame):
    """
    Plot the extremes for each hour-class, including a median line.

    Parameters
    ----------
    df_ext : pd.DataFrame
        DataFrame with measurement extremes, as provided by `kw.calc_havengetallen()`.
    df_havengetallen : pd.DataFrame
        DataFrame with havengetallen for all hour-classes, as provided by
        `kw.calc_havengetallen()`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.
    ax : matplotlib.axes._axes.Axes
        Figure axis handle.

    """

    assert "culm_hr" in df_ext.columns

    station = df_ext.attrs["station"]

    HWLW_culmhr_summary = df_havengetallen.loc[:11].copy()  # remove mean column

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 8), sharex=True)
    data_pd_HW = df_ext.loc[df_ext["HWLWcode"] == 1]
    data_pd_LW = df_ext.loc[df_ext["HWLWcode"] == 2]
    ax1.set_title(f"HW values {station}")
    ax1.plot(data_pd_HW["culm_hr"], data_pd_HW["values"], ".")
    ax1.plot(HWLW_culmhr_summary["HW_values_median"], ".-")
    ax2.set_title(f"LW values {station}")
    ax2.plot(data_pd_LW["culm_hr"], data_pd_LW["values"], ".")
    ax2.plot(HWLW_culmhr_summary["LW_values_median"], ".-")
    ax3.set_title(f"HW time delays {station}")
    ax3.plot(
        data_pd_HW["culm_hr"], data_pd_HW["HWLW_delay"].dt.total_seconds() / 3600, "."
    )
    ax3.plot(HWLW_culmhr_summary["HW_delay_median"].dt.total_seconds() / 3600, ".-")
    ax4.set_title(f"LW time delays {station}")
    ax4.plot(
        data_pd_LW["culm_hr"], data_pd_LW["HWLW_delay"].dt.total_seconds() / 3600, "."
    )
    ax4.plot(HWLW_culmhr_summary["LW_delay_median"].dt.total_seconds() / 3600, ".-")
    ax4.set_xlim([0 - 0.5, 12 - 0.5])
    
    ax1.set_ylabel("water level [cm]")
    ax2.set_ylabel("water level [cm]")
    ax3.set_ylabel("time delay [hours]")
    ax4.set_ylabel("time delay [hours]")
    ax3.set_xlabel("culmination time class [hours]")
    ax4.set_xlabel("culmination time class [hours]")
    fig.tight_layout()
    axs = np.array(((ax1, ax2), (ax3, ax4)))

    return fig, axs


def plot_aardappelgrafiek(df_havengetallen: pd.DataFrame):
    """
    Plot the median values of each hour-class in a aardappelgrafiek.

    Parameters
    ----------
    df_havengetallen : pd.DataFrame
        DataFrame with havengetallen for all hour-classes, as provided by
        `kw.calc_havengetallen()`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.
    ax : matplotlib.axes._axes.Axes
        Figure axis handle.

    """
    # remove mean column
    HWLW_culmhr_summary = df_havengetallen.loc[:11].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 4), sharex=False)
    ax1.set_title("HW")
    ax1.set_xlabel("maansverloop [uu:mm:ss]")
    ax1.set_ylabel("waterstand [cm]")
    ax1.plot(
        HWLW_culmhr_summary["HW_delay_median"],
        HWLW_culmhr_summary["HW_values_median"],
        ".-",
    )
    ax1.xaxis.set_major_formatter(TimeSeries_TimedeltaFormatter_improved())
    ax1.grid()
    ax2.set_title("LW")
    ax2.set_xlabel("maansverloop [uu:mm:ss]")
    ax2.set_ylabel("waterstand [cm]")
    ax2.plot(
        HWLW_culmhr_summary["LW_delay_median"],
        HWLW_culmhr_summary["LW_values_median"],
        ".-",
    )
    ax2.xaxis.set_major_formatter(TimeSeries_TimedeltaFormatter_improved())
    ax2.grid()
    for iH, row in HWLW_culmhr_summary.iterrows():
        # plt.text only supports values as nanoseconds
        ax1.text(
            row["HW_delay_median"].total_seconds() * 1e9,
            row["HW_values_median"],
            str(int(iH)),
        )
        ax2.text(
            row["LW_delay_median"].total_seconds() * 1e9,
            row["LW_values_median"],
            str(int(iH)),
        )
    # set equal ylims
    ax1_xlimmean = np.mean(ax1.get_xlim())
    ax2_xlimmean = np.mean(ax2.get_xlim())
    ax1_ylimmean = np.mean(ax1.get_ylim())
    ax2_ylimmean = np.mean(ax2.get_ylim())
    xlimrange = 2 * 3600e9  # in nanoseconds
    ylimrange = 100  # in centimeters
    ax1.set_xlim([ax1_xlimmean - xlimrange / 2, ax1_xlimmean + xlimrange / 2])
    ax2.set_xlim([ax2_xlimmean - xlimrange / 2, ax2_xlimmean + xlimrange / 2])
    ax1.set_ylim([ax1_ylimmean - ylimrange / 2, ax1_ylimmean + ylimrange / 2])
    ax2.set_ylim([ax2_ylimmean - ylimrange / 2, ax2_ylimmean + ylimrange / 2])
    # set nice xtick interval
    ax1.xaxis.set_major_locator(MultipleLocator(base=0.5 * 3600e9))
    ax2.xaxis.set_major_locator(MultipleLocator(base=0.5 * 3600e9))

    # plot gemtij dotted lines
    ax1.hlines(
        HWLW_culmhr_summary["HW_values_median"].mean(),
        ax1.get_xlim()[0],
        ax1.get_xlim()[1],
        color="k",
        linestyle="--",
    )
    ax1.vlines(
        HWLW_culmhr_summary["HW_delay_median"].mean().total_seconds() * 1e9,
        ax1.get_ylim()[0],
        ax1.get_ylim()[1],
        color="k",
        linestyle="--",
    )
    ax2.hlines(
        HWLW_culmhr_summary["LW_values_median"].mean(),
        ax2.get_xlim()[0],
        ax2.get_xlim()[1],
        color="k",
        linestyle="--",
    )
    ax2.vlines(
        HWLW_culmhr_summary["LW_delay_median"].mean().total_seconds() * 1e9,
        ax2.get_ylim()[0],
        ax2.get_ylim()[1],
        color="k",
        linestyle="--",
    )
    fig.tight_layout()

    axs = (ax1, ax2)
    return fig, axs
