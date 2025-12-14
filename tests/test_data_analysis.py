# -*- coding: utf-8 -*-

import pytest
import kenmerkendewaarden as kw
import numpy as np
import pandas as pd


@pytest.mark.timeout(60)  # useful in case of ddl failure
@pytest.mark.systemtest
@pytest.mark.parametrize("quantity", ["meas_wl", "meas_ext"])
def test_derive_statistics(dir_meas, quantity):
    current_station = "hoekvanholland"
    station_list = [current_station]

    if quantity == "meas_ext":
        cols_stats = [
            "WaarnemingMetadata.Statuswaarde",
            "WaarnemingMetadata.Kwaliteitswaardecode",
            "WaardeBepalingsMethode.Code",
            "MeetApparaat.Code",
            "Hoedanigheid.Code",
            "Grootheid.Code",
            "Groepering.Code",
            "Typering.Code",
            "tstart",
            "tstop",
            "timediff_min",
            "timediff_max",
            "nvals",
            "#nans",
            "min",
            "max",
            "std",
            "mean",
            "dupltimes",
            "dupltimes_#nans",
            "qc_none",
            "timediff<4hr",
            "aggers",
        ]
        stats_expected = np.array([7.922705314009662, -133, 211])
        timedif_min = pd.Timedelta("0 days 00:34:00")
        timedif_max = pd.Timedelta("0 days 08:57:00")
    elif quantity == "meas_wl":
        cols_stats = [
            "WaarnemingMetadata.Statuswaarde",
            "WaarnemingMetadata.Kwaliteitswaardecode",
            "WaardeBepalingsMethode.Code",
            "MeetApparaat.Code",
            "Hoedanigheid.Code",
            "Grootheid.Code",
            "Groepering.Code",
            "Typering.Code",
            "tstart",
            "tstop",
            "timediff_min",
            "timediff_max",
            "nvals",
            "#nans",
            "min",
            "max",
            "std",
            "mean",
            "dupltimes",
            "dupltimes_#nans",
            "qc_none",
        ]
        stats_expected = np.array([7.962614866536023, -133, 211])
        timedif_min = pd.Timedelta("0 days 00:10:00")
        timedif_max = pd.Timedelta("0 days 00:10:00")

    stats = kw.derive_statistics(
        dir_output=dir_meas, station_list=station_list, quantity=quantity
    )

    # assert statistics columns
    assert set(stats.columns) == set(cols_stats)

    # assert statistics values, this might change if ddl data is updated
    stats_vals = stats.loc[current_station, ["mean", "min", "max"]].values.astype(float)
    assert np.allclose(stats_vals, stats_expected)

    assert stats.loc[current_station, "timediff_min"] == timedif_min
    assert stats.loc[current_station, "timediff_max"] == timedif_max


@pytest.mark.timeout(120)  # useful in case of ddl failure
@pytest.mark.unittest
@pytest.mark.parametrize("quantity", ["meas_wl", "meas_ext"])
def test_plot_measurements_amount(dir_meas_amount, quantity):
    df_amount = kw.read_measurements_amount(
        dir_output=dir_meas_amount, quantity=quantity
    )
    kw.plot_measurements_amount(df=df_amount, relative=True)


@pytest.mark.unittest
def test_plot_measurements(df_meas_2010, df_ext_2010):
    kw.plot_measurements(df_meas=df_meas_2010, df_ext=df_ext_2010)


@pytest.mark.unittest
def test_plot_stations():
    kw.plot_stations(station_list=["hoekvanholland"], crs=None, add_labels=True)
