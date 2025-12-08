# -*- coding: utf-8 -*-
"""
Computation of slotgemiddelden of waterlevels and extremes
"""

import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from kenmerkendewaarden.data_retrieve import clip_timeseries_physical_break
from kenmerkendewaarden.tidalindicators import (
    calc_wltidalindicators,
    calc_HWLWtidalindicators,
)
from kenmerkendewaarden.utils import clip_timeseries_last_newyearsday, raise_empty_df
import logging

__all__ = [
    "calc_slotgemiddelden",
    "plot_slotgemiddelden",
]

logger = logging.getLogger(__name__)


def calc_slotgemiddelden(
    df_meas: pd.DataFrame = None,
    df_ext: pd.DataFrame = None,
    min_coverage: float = None,
    clip_physical_break: bool = False,
):
    """
    Compute slotgemiddelden from measurement timeseries and optionally also from
    extremes timeseries.
    A simple linear trend is used to avoid all pretend-accuracy. However, when fitting
    a linear trend on a limited amount of data, the nodal cycle and wind effects will
    cause the model fit to be inaccurate. It is wise to use at least 30 years of data
    for a valid fit, this is >1.5 times the nodal cycle.

    Parameters
    ----------
    df_meas : pd.DataFrame, optional
        the timeseries of measured waterlevels. The default is None.
    df_ext : pd.DataFrame, optional
        the timeseries of extremes (high and low waters). The default is None.
    min_coverage : float, optional
        Set yearly means to nans for years that do not have sufficient data coverage.
        The default is None.
    clip_physical_break : bool, optional
        Whether to exclude the part of the timeseries before physical breaks like
        estuary closures. The default is False.

    Returns
    -------
    slotgemiddelden_dict : dict
        dictionary with yearly means and model fits, optionally also for extremes
        and corresponding tidal range.

    """
    # initialize dict
    slotgemiddelden_dict = {}

    if df_meas is None and df_ext is None:
        raise ValueError("At least one of df_meas or df_ext should be provided")

    if df_meas is not None and df_ext is not None:
        # compare station attributes
        _ = compare_get_station_from_dataframes([df_meas, df_ext])

    if df_meas is not None:
        raise_empty_df(df_meas)

        # clip last value of the timeseries if this is exactly newyearsday
        df_meas = clip_timeseries_last_newyearsday(df_meas)

        # calculate yearly means
        dict_wltidalindicators = calc_wltidalindicators(
            df_meas, min_coverage=min_coverage
        )
        wl_mean_peryear = dict_wltidalindicators["wl_mean_peryear"]
        slotgemiddelden_dict["wl_mean_peryear"] = wl_mean_peryear

        # clip part of mean timeseries before physical break to supply to model
        if clip_physical_break:
            wl_mean_peryear = clip_timeseries_physical_break(wl_mean_peryear)

        # fit linear models over yearly mean values
        pred_pd_wl = predict_linear_model(wl_mean_peryear)
        slotgemiddelden_dict["wl_model_fit"] = pred_pd_wl

    if df_ext is not None:
        raise_empty_df(df_ext)

        # clip last value of the timeseries if this is exactly newyearsday
        df_ext = clip_timeseries_last_newyearsday(df_ext)

        # calculate yearly means
        dict_HWLWtidalindicators = calc_HWLWtidalindicators(
            df_ext, min_coverage=min_coverage
        )
        HW_mean_peryear = dict_HWLWtidalindicators["HW_mean_peryear"]
        LW_mean_peryear = dict_HWLWtidalindicators["LW_mean_peryear"]
        tidalrange_mean_peryear = HW_mean_peryear - LW_mean_peryear
        slotgemiddelden_dict["HW_mean_peryear"] = HW_mean_peryear
        slotgemiddelden_dict["LW_mean_peryear"] = LW_mean_peryear
        slotgemiddelden_dict["tidalrange_mean_peryear"] = tidalrange_mean_peryear

        # clip part of mean timeseries before physical break to supply to model
        if clip_physical_break:
            HW_mean_peryear = clip_timeseries_physical_break(HW_mean_peryear)
            LW_mean_peryear = clip_timeseries_physical_break(LW_mean_peryear)
            tidalrange_mean_peryear = clip_timeseries_physical_break(
                tidalrange_mean_peryear
            )

        # fit linear models over yearly mean values
        pred_pd_HW = predict_linear_model(HW_mean_peryear)
        pred_pd_LW = predict_linear_model(LW_mean_peryear)
        pred_pd_tidalrange = predict_linear_model(tidalrange_mean_peryear)
        slotgemiddelden_dict["HW_model_fit"] = pred_pd_HW
        slotgemiddelden_dict["LW_model_fit"] = pred_pd_LW
        slotgemiddelden_dict["tidalrange_model_fit"] = pred_pd_tidalrange

    return slotgemiddelden_dict


def plot_slotgemiddelden(
    slotgemiddelden_dict: dict, slotgemiddelden_dict_all: dict = None
):
    """
    Plot timeseries of yearly mean waterlevels and corresponding model fits.

    Parameters
    ----------
    slotgemiddelden_dict : dict
        Output from `kw.calc_slotgemiddelden` containing timeseries
        of yearly mean waterlevels and corresponding model fits.
    slotgemiddelden_dict_all : dict, optional
        Optionally provide another dictionary with unfiltered mean waterlevels.
        Only used to plot the mean waterlevels (in grey). The default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.
    ax : matplotlib.axes._axes.Axes
        Figure axis handle.

    """

    # get station attribute
    station = compare_get_station_from_dataframes(slotgemiddelden_dict.values())

    # convert to timeindex for plotting (first make deep copy)
    slotgemiddelden_dict = dict_indexes_to_timestamp(slotgemiddelden_dict)
    if slotgemiddelden_dict_all is not None:
        slotgemiddelden_dict_all = dict_indexes_to_timestamp(slotgemiddelden_dict_all)

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("tab10")

    # plot timeseries of average waterlevels
    if slotgemiddelden_dict_all is not None:
        if "wl_mean_peryear" in slotgemiddelden_dict_all.keys():
            wl_mean_peryear_all = slotgemiddelden_dict_all["wl_mean_peryear"]
            ax.plot(
                wl_mean_peryear_all,
                "x",
                color="grey",
                label="yearly means incl invalid",
            )
    if "wl_mean_peryear" in slotgemiddelden_dict.keys():
        wl_mean_peryear = slotgemiddelden_dict["wl_mean_peryear"]
        ax.plot(wl_mean_peryear, "xr", label="yearly means")

    if "wl_model_fit" in slotgemiddelden_dict.keys():
        # plot model fits of average waterlevels
        wl_model_fit = slotgemiddelden_dict["wl_model_fit"]
        ax.plot(wl_model_fit, ".-", color=cmap(0), label="model fit")
        # add single dot for slotgemiddelde value
        slotgem_time_value = slotgemiddelden_dict["wl_model_fit"].iloc[[-1]]
        ax.plot(
            slotgem_time_value,
            ".k",
            label=f"slotgemiddelde for {slotgem_time_value.index.year[0]}",
        )

    # plot timeseries of average extremes
    if slotgemiddelden_dict_all is not None:
        # compare station attributes
        station_all = compare_get_station_from_dataframes(
            slotgemiddelden_dict_all.values()
        )
        assert station == station_all

        if "HW_mean_peryear" in slotgemiddelden_dict_all.keys():
            HW_mean_peryear_all = slotgemiddelden_dict_all["HW_mean_peryear"]
            LW_mean_peryear_all = slotgemiddelden_dict_all["LW_mean_peryear"]
            ax.plot(HW_mean_peryear_all, "x", color="grey")
            ax.plot(LW_mean_peryear_all, "x", color="grey")

    if "HW_mean_peryear" in slotgemiddelden_dict.keys():
        HW_mean_peryear = slotgemiddelden_dict["HW_mean_peryear"]
        LW_mean_peryear = slotgemiddelden_dict["LW_mean_peryear"]
        ax.plot(HW_mean_peryear, "xr")
        ax.plot(LW_mean_peryear, "xr")

    if "HW_model_fit" in slotgemiddelden_dict.keys():
        # plot model fits of average extremes
        HW_model_fit = slotgemiddelden_dict["HW_model_fit"]
        LW_model_fit = slotgemiddelden_dict["LW_model_fit"]
        ax.plot(HW_model_fit, ".-", color=cmap(0), label=None)
        ax.plot(LW_model_fit, ".-", color=cmap(0), label=None)
        ax.plot(slotgemiddelden_dict["HW_model_fit"].iloc[[-1]], ".k")
        ax.plot(slotgemiddelden_dict["LW_model_fit"].iloc[[-1]], ".k")

    ax.set_ylabel("water level [cm]")
    ax.set_title(f"yearly mean HW/wl/LW for {station}")
    ax.grid()
    ax.legend(loc=2)
    fig.tight_layout()
    return fig, ax


def compare_get_station_from_dataframes(df_list):
    station_list = []
    for df in df_list:
        station_list.append(df.attrs["station"])
    if len(set(station_list)) != 1:
        raise ValueError(
            f"station attributes are not equal for all dataframes: {station_list}"
        )
    station = station_list[0]
    return station


def dict_indexes_to_timestamp(dict_in):
    # convert to timeindex for plotting (first make deep copy)
    dict_out = {k: v.copy() for k, v in dict_in.items()}
    for k, v in dict_out.items():
        v.index = v.index.to_timestamp()
    return dict_out


def predict_linear_model(ser: pd.Series, with_nodal=False) -> pd.DataFrame:
    """
    Fit linear model over yearly means in mean_array_todate, including five years in the
    future.

    Parameters
    ----------
    ser : pd.Series
        DESCRIPTION.

    Returns
    -------
    pred_pd : TYPE
        DESCRIPTION.

    """

    start = ser.index.min()
    end = ser.index.max() + 1

    logger.info(f"fit linear model for {start} to {end}")

    # generate contiguous timeseries including gaps and including slotgemiddelde year
    allyears_dt = pd.period_range(start=start, end=end)
    ser_allyears = pd.Series(ser, index=allyears_dt)

    ser_nonans = ser_allyears.loc[~ser_allyears.isnull()]
    if len(ser_nonans) < 2:
        raise ValueError(
            "nan-filtered timeseries has only one timestep, cannot perform model "
            f"fit:\n{ser_nonans}"
        )

    # get model fit
    fit, _, X = fit_linear_model(ser_allyears, with_nodal=with_nodal)

    # predict
    pred_arr = fit.predict(X)
    pred_pd = pd.Series(pred_arr, index=allyears_dt, name="values")
    pred_pd.index.name = ser.index.name
    pred_pd.attrs = ser.attrs
    return pred_pd


def fit_linear_model(df, with_nodal=False):
    """
    Define the linear model with constant and trend, optionally with nodal.
    simplified from https://github.com/openearth/sealevel/blob/master/slr/slr/models.py
    """

    # just use the years. This assumes that annual waterlevels are used that are
    # stored left-padded, the mean waterlevel for 2020 is stored as 2020-1-1
    # This is not logical, but common practice.
    years = df.index.year
    y = df.values
    X = np.c_[years - 1970,]
    names = ["Constant", "Trend"]
    if with_nodal:
        X = np.c_[
            X,
            np.cos(2 * np.pi * (years - 1970) / 18.613),
            np.sin(2 * np.pi * (years - 1970) / 18.613),
        ]
        names.extend(["Nodal U", "Nodal V"])
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing="drop")
    fit = model.fit(cov_type="HC0")
    return fit, names, X
