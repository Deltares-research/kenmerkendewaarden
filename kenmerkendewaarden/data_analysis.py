# -*- coding: utf-8 -*-
"""
Data analysis like missings, duplicates, outliers and several other statistics
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kenmerkendewaarden.data_retrieve import (
    read_measurements,
    xarray_to_hatyan,
    retrieve_catalog,
)
import hatyan
import logging

__all__ = [
    "plot_measurements_amount",
    "plot_measurements",
    "plot_stations",
    "derive_statistics",
]

logger = logging.getLogger(__name__)


def plot_measurements_amount(df: pd.DataFrame, relative: bool = False):
    """
    Read the measurements amount csv and generate a pcolormesh figure of all years and stations.
    The colors indicate the absolute or relative number of measurements per year.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with the amount of measurements for several years per station.
    relative : bool, optional
        Whether to scale the amount of measurements with the median of all measurement amounts for the same year. The default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.
    ax : matplotlib.axes._axes.Axes
        Figure axis handle.

    """
    df = df.copy()
    df[df == 0] = np.nan

    if relative:
        # this is useful for ts, because the frequency was changed from hourly to 10-minute
        df_relative = df.div(df.median(axis=1), axis=0) * 100
        df_relative = df_relative.clip(upper=200)
        df = df_relative

    fig, ax = plt.subplots(figsize=(14, 8))
    pc = ax.pcolormesh(df.columns, df.index, df.values, cmap="turbo")
    cbar = fig.colorbar(pc, ax=ax)
    ax.tick_params(axis="x", labelrotation=90)
    ax.grid()
    if relative:
        cbar.set_label("measurements per year w.r.t. year median (0 excluded) [%]")
    else:
        cbar.set_label("measurements per year (0 excluded) [-]")
    fig.tight_layout()
    return fig, ax


def plot_measurements(df_meas: pd.DataFrame, df_ext: pd.DataFrame = None):
    """
    Generate a timeseries figure for the measurement timeseries (and extremes) of this station.

    Parameters
    ----------
    df_meas : pd.DataFrame
        Dataframe with the measurement timeseries for a particular station.
    df_ext : pd.DataFrame, optional
        Dataframe with the measurement extremes for a particular station.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.
    ax : matplotlib.axes._axes.Axes
        Figure axis handle.

    """
    station_df = df_meas.attrs["station"]
    if df_ext is not None:
        station_df_ext = df_ext.attrs["station"]
        assert station_df == station_df_ext
        fig, (ax1, ax2) = hatyan.plot_timeseries(ts=df_meas, ts_ext=df_ext)
    else:
        fig, (ax1, ax2) = hatyan.plot_timeseries(ts=df_meas)
    ax1.set_title(f"timeseries for {station_df}")

    # calculate monthly/yearly mean for meas wl data
    df_meas_values = df_meas["values"]
    mean_peryearmonth_long = df_meas_values.groupby(
        pd.PeriodIndex(df_meas_values.index, freq="M")
    ).mean()
    mean_peryear_long = df_meas_values.groupby(
        pd.PeriodIndex(df_meas_values.index, freq="Y")
    ).mean()

    ax1.plot(mean_peryearmonth_long, "c", linewidth=0.7, label="monthly mean")
    ax1.plot(mean_peryear_long, "m", linewidth=0.7, label="yearly mean")
    ax2.plot(mean_peryearmonth_long, "c", linewidth=0.7, label="monthly mean")
    ax2.plot(mean_peryear_long, "m", linewidth=0.7, label="yearly mean")
    if df_ext is not None:
        # select all hoogwater
        data_pd_HW = df_ext.loc[df_ext["HWLWcode"].isin([1])]
        # select all laagwater, laagwater1, laagwater2 (so approximation in case of aggers)
        data_pd_LW = df_ext.loc[df_ext["HWLWcode"].isin([2, 3, 5])]

        # calculate monthly/yearly mean for meas ext data
        HW_mean_peryear_long = data_pd_HW.groupby(
            pd.PeriodIndex(data_pd_HW.index, freq="Y")
        )["values"].mean()
        LW_mean_peryear_long = data_pd_LW.groupby(
            pd.PeriodIndex(data_pd_LW.index, freq="Y")
        )["values"].mean()

        ax1.plot(HW_mean_peryear_long, "m", linewidth=0.7, label=None)
        ax1.plot(LW_mean_peryear_long, "m", linewidth=0.7, label=None)
    ax1.set_ylim(-4, 4)
    ax1.legend(loc=4)
    ax2.legend(loc=1)
    ax2.set_ylim(-0.5, 0.5)
    return fig, (ax1, ax2)


def plot_stations(station_list: list, crs: int = None, add_labels: bool = False):
    """
    Plot the stations by subsetting a ddlpy catalog with the provided list of stations.

    Parameters
    ----------
    station_list : list
        List of stations to plot the locations from.
    crs : int, optional
        Coordinate reference system, for instance 28992. The coordinates retrieved from the DDL will be converted to this EPSG. The default is None.
    add_labels : bool, optional
        Whether to add station code labels in the figure, useful for debugging. The default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.
    ax : matplotlib.axes._axes.Axes
        Figure axis handle.

    """
    locs_meas_ts_all, locs_meas_ext_all, _ = retrieve_catalog(crs=crs)
    locs_ts = locs_meas_ts_all.loc[locs_meas_ts_all.index.isin(station_list)]
    locs_ext = locs_meas_ext_all.loc[locs_meas_ext_all.index.isin(station_list)]
    if crs is None:
        crs = int(locs_ts["Coordinatenstelsel"].iloc[0])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(locs_ts["X"], locs_ts["Y"], "xk", label="timeseries")
    ax.plot(locs_ext["X"], locs_ext["Y"], "+r", label="extremes")
    ax.legend()

    ax.set_title("stations with timeseries/extremes data")
    ax.set_aspect("equal")
    ax.set_xlabel(f"X (EPSG:{crs})")
    ax.set_ylabel(f"Y (EPSG:{crs})")
    ax.grid(alpha=0.5)

    # optionally add basemap/coastlines
    try:
        import dfm_tools as dfmt  # pip install dfm_tools

        dfmt.plot_coastlines(ax=ax, crs=crs)
        dfmt.plot_borders(ax=ax, crs=crs)
    except ModuleNotFoundError:
        try:
            import contextily as ctx  # pip install contextily

            ctx.add_basemap(
                ax, source=ctx.providers.Esri.WorldImagery, crs=crs, attribution=False
            )
        except ModuleNotFoundError:
            pass

    fig.tight_layout()

    if add_labels:
        for irow, row in locs_ts.iterrows():
            ax.text(row["X"], row["Y"], row.name)

    return fig, ax


def get_flat_meta_from_dataset(ds):
    list_relevantmetadata = [
        "WaarnemingMetadata.StatuswaardeLijst",
        "WaarnemingMetadata.KwaliteitswaardecodeLijst",
        "WaardeBepalingsmethode.Code",
        "MeetApparaat.Code",
        "Hoedanigheid.Code",
        "WaardeBepalingsmethode.Code",
        "MeetApparaat.Code",
        "Hoedanigheid.Code",
        "Grootheid.Code",
        "Groepering.Code",
        "Typering.Code",
    ]

    meta_dict_flat = {}
    for key in list_relevantmetadata:
        if key in ds.data_vars:
            vals_unique = ds[key].to_pandas().drop_duplicates()
            meta_dict_flat[key] = "|".join(vals_unique)
        else:
            meta_dict_flat[key] = ds.attrs[key]
    return meta_dict_flat


def get_stats_from_dataframe(df):
    df_times = df.index
    ts_dupltimes = df_times.duplicated()
    ts_timediff = (
        df_times[1:] - df_times[:-1]
    )  # TODO: from pandas 2.1.4 the following also works: df_times.diff()[1:]

    ds_stats = {}
    ds_stats["tstart"] = df_times.min()
    ds_stats["tstop"] = df_times.max()
    ds_stats["timediff_min"] = ts_timediff.min()
    ds_stats["timediff_max"] = ts_timediff.max()
    ds_stats["nvals"] = len(df["values"])
    ds_stats["#nans"] = df["values"].isnull().sum()
    ds_stats["min"] = df["values"].min()
    ds_stats["max"] = df["values"].max()
    ds_stats["std"] = df["values"].std()
    ds_stats["mean"] = df["values"].mean()
    ds_stats["dupltimes"] = ts_dupltimes.sum()
    # count #nans for duplicated times, happens at HARVT10/HUIBGT/STELLDBTN
    ds_stats["dupltimes_#nans"] = (
        df.loc[df_times.duplicated(keep=False)]["values"].isnull().sum()
    )

    # None in kwaliteitswaardecodelijst: https://github.com/Rijkswaterstaat/wm-ws-dl/issues/14
    # TODO: add test to see if '' is indeed the missing value (or None or np.nan)
    if "" in df["qualitycode"].values:
        ds_stats["qc_none"] = True
    else:
        ds_stats["qc_none"] = False

    if "HWLWcode" in df.columns:
        # count the number of too small time differences (<4hr), sometimes happens because of aggers
        # but sometimes due to incorrect data: https://github.com/Rijkswaterstaat/wm-ws-dl/issues/43
        mintimediff_hr = 4
        bool_timediff_toosmall = ts_timediff < pd.Timedelta(hours=mintimediff_hr)
        ds_stats[f"timediff<{mintimediff_hr}hr"] = bool_timediff_toosmall.sum()

        # check whether there are aggers present
        if len(df["HWLWcode"].unique()) > 2:
            ds_stats["aggers"] = True
        else:
            ds_stats["aggers"] = False

    return ds_stats


def derive_statistics(dir_output: str, station_list: list, extremes: bool):
    """
    Derive several statistics for the measurements of each station in the list.

    Parameters
    ----------
    dir_output : str
        Path where the measurement netcdf file will be stored.
    station : list
        list of station names to derive statistics for, for instance ["HOEKVHLD"].
    extremes : bool
        Whether to derive statistics from waterlevel timeseries or extremes.

    Returns
    -------
    data_summary : pd.DataFrame
        A dataframe with several statistics for each station from the provided list.

    """
    row_list = []
    for current_station in station_list:
        logger.info(f"deriving statistics for {current_station} (extremes={extremes})")
        data_summary_row = {}

        # load measwl data
        ds_meas = read_measurements(
            dir_output=dir_output,
            station=current_station,
            extremes=extremes,
            return_xarray=True,
        )
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
    data_summary = data_summary.set_index("Code").sort_index()
    return data_summary
