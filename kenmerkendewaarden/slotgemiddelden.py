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
import logging

__all__ = [
    "calc_slotgemiddelden",
    "plot_slotgemiddelden",
]

logger = logging.getLogger(__name__)


def calc_slotgemiddelden(
    df_meas: pd.DataFrame,
    df_ext: pd.DataFrame = None,
    min_coverage: float = None,
    clip_physical_break: bool = False,
    with_nodal: bool = True,
):
    """
    Compute slotgemiddelden from measurement timeseries and optionally also from extremes timeseries.

    Parameters
    ----------
    df_meas : pd.DataFrame
        the timeseries of measured waterlevels.
    df_ext : pd.DataFrame, optional
        the timeseries of extremes (high and low waters). The default is None.
    min_coverage : float, optional
        Set yearly means to nans for years that do not have sufficient data coverage. The default is None.
    clip_physical_break : bool, optional
        Whether to exclude the part of the timeseries before physical breaks like estuary closures. The default is False.
    with_nodal : bool, optional
        Whether to include a nodal cycle in the linear trend model. The default is True.

    Returns
    -------
    slotgemiddelden_dict : dict
        dictionary with yearly means and model fits, optionally also for extremes.

    """
    # initialize dict
    slotgemiddelden_dict = {}

    # clip last value of the timeseries if this is exactly newyearsday
    if df_meas.index[-1] == pd.Timestamp(
        df_meas.index[-1].year, 1, 1, tz=df_meas.index.tz
    ):
        df_meas = df_meas.iloc[:-1]

    # calculate yearly means
    dict_wltidalindicators = calc_wltidalindicators(df_meas, min_coverage=min_coverage)
    wl_mean_peryear = dict_wltidalindicators["wl_mean_peryear"]
    # convert periodindex to datetimeindex
    # TODO: alternatively let fit_models support periodindex
    # wl_mean_peryear.index = wl_mean_peryear.index.to_timestamp()
    slotgemiddelden_dict["wl_mean_peryear"] = wl_mean_peryear

    # clip part of mean timeseries before physical break to supply to model
    if clip_physical_break:
        wl_mean_peryear = clip_timeseries_physical_break(wl_mean_peryear)

    # fit linear models over yearly mean values
    pred_pd_wl = fit_models(wl_mean_peryear, with_nodal=with_nodal)
    slotgemiddelden_dict["wl_model_fit"] = pred_pd_wl

    if df_ext is not None:
        # compare station attributes
        station_attrs = [df.attrs["station"] for df in [df_meas, df_ext]]
        assert all(x == station_attrs[0] for x in station_attrs)

        # clip last value of the timeseries if this is exactly newyearsday
        if df_ext.index[-1] == pd.Timestamp(
            df_ext.index[-1].year, 1, 1, tz=df_ext.index.tz
        ):
            df_ext = df_ext.iloc[:-1]

        # calculate yearly means
        dict_HWLWtidalindicators = calc_HWLWtidalindicators(
            df_ext, min_coverage=min_coverage
        )
        HW_mean_peryear = dict_HWLWtidalindicators["HW_mean_peryear"]
        LW_mean_peryear = dict_HWLWtidalindicators["LW_mean_peryear"]
        # HW_mean_peryear.index = HW_mean_peryear.index.to_timestamp()
        # LW_mean_peryear.index = LW_mean_peryear.index.to_timestamp()
        slotgemiddelden_dict["HW_mean_peryear"] = HW_mean_peryear
        slotgemiddelden_dict["LW_mean_peryear"] = LW_mean_peryear

        # clip part of mean timeseries before physical break to supply to model
        if clip_physical_break:
            HW_mean_peryear = clip_timeseries_physical_break(HW_mean_peryear)
            LW_mean_peryear = clip_timeseries_physical_break(LW_mean_peryear)

        # fit linear models over yearly mean values
        pred_pd_HW = fit_models(HW_mean_peryear, with_nodal=with_nodal)
        pred_pd_LW = fit_models(LW_mean_peryear, with_nodal=with_nodal)
        slotgemiddelden_dict["HW_model_fit"] = pred_pd_HW
        slotgemiddelden_dict["LW_model_fit"] = pred_pd_LW

    return slotgemiddelden_dict


def plot_slotgemiddelden(
    slotgemiddelden_dict: dict, slotgemiddelden_dict_all: dict = None
):
    """
    plot timeseries of yearly mean waterlevels and corresponding model fits.

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
    station = slotgemiddelden_dict["wl_mean_peryear"].attrs["station"]

    # convert to timeindex for plotting (first make deep copy)
    slotgemiddelden_dict = {k: v.copy() for k, v in slotgemiddelden_dict.items()}
    for k, v in slotgemiddelden_dict.items():
        v.index = v.index.to_timestamp()
    if slotgemiddelden_dict_all is not None:
        slotgemiddelden_dict_all = {
            k: v.copy() for k, v in slotgemiddelden_dict_all.items()
        }
        for k, v in slotgemiddelden_dict_all.items():
            v.index = v.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("tab10")

    # plot timeseries of average waterlevels
    if slotgemiddelden_dict_all is not None:
        wl_mean_peryear_all = slotgemiddelden_dict_all["wl_mean_peryear"]
        ax.plot(
            wl_mean_peryear_all, "x", color="grey", label="yearly means incl invalid"
        )
    wl_mean_peryear = slotgemiddelden_dict["wl_mean_peryear"]
    ax.plot(wl_mean_peryear, "xr", label="yearly means")

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
        station_attrs = [
            dic["wl_mean_peryear"].attrs["station"]
            for dic in [slotgemiddelden_dict, slotgemiddelden_dict_all]
        ]
        assert all(x == station_attrs[0] for x in station_attrs)

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

    ax.set_ylabel("waterstand [m]")
    ax.set_title(f"yearly mean HW/wl/LW for {station}")
    ax.grid()
    ax.legend(loc=2)
    fig.tight_layout()
    return fig, ax


def fit_models(mean_array_todate: pd.Series, with_nodal=True) -> pd.DataFrame:
    """
    Fit linear model over yearly means in mean_array_todate, including five years in the future.

    Parameters
    ----------
    mean_array_todate : pd.Series
        DESCRIPTION.

    Returns
    -------
    pred_pd : TYPE
        DESCRIPTION.

    """

    start = mean_array_todate.index.min()
    end = mean_array_todate.index.max() + 1

    logger.info(f"fit linear model (with_nodal={with_nodal}) for {start} to {end}")

    # We'll just use the years. This assumes that annual waterlevels are used that are stored left-padded, the mean waterlevel for 2020 is stored as 2020-1-1. This is not logical, but common practice.
    allyears_dt = pd.period_range(start=start, end=end)
    mean_array_allyears = pd.Series(mean_array_todate, index=allyears_dt)
    
    mean_array_allyears_nonans = mean_array_allyears.loc[~mean_array_allyears.isnull()]
    if len(mean_array_allyears_nonans) < 2:
        raise ValueError(
            f"nan-filtered timeseries has only one timestep, cannot perform model fit:\n{mean_array_allyears_nonans}"
        )

    # convert to dataframe of expected format
    # TODO: make functions accept mean_array instead of df as argument?
    df = pd.DataFrame(
        {"year": mean_array_allyears.index.year, "height": mean_array_allyears.values}
    )

    # below methods are copied from https://github.com/openearth/sealevel/blob/master/slr/slr/models.py
    # TODO: install slr package as dependency or keep separate?
    fit, _, X = linear_model(df, with_wind=False, with_nodal=with_nodal)
    pred_linear = fit.predict(X)

    linear_fit = pd.Series(pred_linear, index=allyears_dt, name="values")
    linear_fit.index.name = mean_array_todate.index.name
    return linear_fit


# copied from https://github.com/openearth/sealevel/blob/master/slr/slr/models.py
def linear_model(df, with_wind=True, with_ar=True, with_nodal=True, quantity="height"):
    """Define the linear model with optional wind and autoregression.
    See the latest report for a detailed description.
    """

    y = df[quantity]
    X = np.c_[df["year"] - 1970,]
    # month = np.mod(df['year'], 1) * 12.0
    names = ["Constant", "Trend"]
    if with_nodal:
        X = np.c_[
            X,
            np.cos(2 * np.pi * (df["year"] - 1970) / 18.613),
            np.sin(2 * np.pi * (df["year"] - 1970) / 18.613),
        ]
        names.extend(["Nodal U", "Nodal V"])
    if with_wind:
        X = np.c_[X, df["u2"], df["v2"]]
        names.extend(["Wind $u^2$", "Wind $v^2$"])
    X = sm.add_constant(X)
    if with_ar:
        model = sm.GLSAR(y, X, missing="drop", rho=1)
    else:
        model = sm.OLS(y, X, missing="drop")
    fit = model.fit(cov_type="HC0")
    return fit, names, X
