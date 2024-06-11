# -*- coding: utf-8 -*-

import pytest
import kenmerkendewaarden as kw
import numpy as np
import pandas as pd



@pytest.mark.timeout(60) # useful in case of ddl failure
@pytest.mark.unittest
def test_retrieve_catalog():
    crs = 28992
    locs_meas_ts, locs_meas_ext, _ = kw.data_retrieve.retrieve_catalog(crs=crs)
    
    assert np.isclose(locs_meas_ts.loc["HOEKVHLD"]["X"], 67930.00003341127)
    assert np.isclose(locs_meas_ts.loc["HOEKVHLD"]["Y"], 444000.0027572268)
    assert np.isclose(locs_meas_ext.loc["HOEKVHLD"]["X"], 67930.00003341127)
    assert np.isclose(locs_meas_ext.loc["HOEKVHLD"]["Y"], 444000.0027572268)
    df_crs = locs_meas_ext["Coordinatenstelsel"].drop_duplicates().tolist()
    assert len(df_crs) == 1
    assert isinstance(df_crs[0], str)
    assert int(df_crs[0]) == crs


@pytest.mark.timeout(60) # useful in case of ddl failure
@pytest.mark.systemtest
@pytest.mark.parametrize("extremes", [False,True], ids=["timeseries", "extremes"])
def test_retrieve_read_measurements_amount(tmp_path, extremes):
    start_date = pd.Timestamp(2010,11,1, tz="UTC+01:00")
    end_date = pd.Timestamp(2011,2,1, tz="UTC+01:00")
    station_list = ["HOEKVHLD"]
    
    kw.retrieve_measurements_amount(dir_output=tmp_path, station_list=station_list, 
                                    start_date=start_date, end_date=end_date,
                                    extremes=extremes)

    df_amount = kw.read_measurements_amount(dir_output=tmp_path, extremes=extremes)
    
    # assert amounts, this might change if ddl data is updated
    assert df_amount.columns.tolist() == ["HOEKVHLD"]
    assert df_amount.index.tolist() == [2010,2011]
    if extremes:
        df_vals = np.array([312, 157])
    else:
        df_vals = np.array([8784, 4465])
    assert len(df_amount) == 2
    assert np.allclose(df_amount["HOEKVHLD"].values, df_vals)


@pytest.mark.timeout(60) # useful in case of ddl failure
@pytest.mark.systemtest
@pytest.mark.parametrize("extremes", [False,True], ids=["timeseries", "extremes"])
def test_retrieve_measurements_derive_statistics(tmp_path, extremes):
    start_date = pd.Timestamp(2010,1,1, tz="UTC+01:00")
    end_date = pd.Timestamp(2011,1,1, tz="UTC+01:00")
    current_station = "HOEKVHLD"
    station_list = [current_station]
    
    if extremes:
        cols_stats = ['WaarnemingMetadata.StatuswaardeLijst',
               'WaarnemingMetadata.KwaliteitswaardecodeLijst',
               'WaardeBepalingsmethode.Code', 'MeetApparaat.Code', 'Hoedanigheid.Code',
               'Grootheid.Code', 'Groepering.Code', 'Typering.Code', 'tstart', 'tstop',
               'timediff_min', 'timediff_max', 'nvals', '#nans', 'min', 'max', 'std',
               'mean', 'dupltimes', 'dupltimes_#nans', 'qc_none', 'timediff<4hr',
               'aggers']
        stats_expected = np.array([0.07922705314009662, -1.33, 2.11])
        timedif_min = pd.Timedelta('0 days 00:34:00')
        timedif_max = pd.Timedelta('0 days 08:57:00')
    else:
        cols_stats = ['WaarnemingMetadata.StatuswaardeLijst',
               'WaarnemingMetadata.KwaliteitswaardecodeLijst',
               'WaardeBepalingsmethode.Code', 'MeetApparaat.Code', 'Hoedanigheid.Code',
               'Grootheid.Code', 'Groepering.Code', 'Typering.Code', 'tstart', 'tstop',
               'timediff_min', 'timediff_max', 'nvals', '#nans', 'min', 'max', 'std',
               'mean', 'dupltimes', 'dupltimes_#nans', 'qc_none']
        stats_expected = np.array([0.07962614866536023, -1.33, 2.11])
        timedif_min = pd.Timedelta('0 days 00:10:00')
        timedif_max = pd.Timedelta('0 days 00:10:00')
    
    # retrieve measurements
    kw.retrieve_measurements(dir_output=tmp_path, station=current_station, extremes=extremes,
                             start_date=start_date, end_date=end_date)

    
    stats = kw.derive_statistics(dir_output=tmp_path, station_list=station_list, extremes=extremes)
    
    # assert statistics columns
    assert set(stats.columns) == set(cols_stats)
    
    # assert statistics values, this might change if ddl data is updated
    stats_vals = stats.loc[current_station, ["mean","min","max"]].values.astype(float)
    assert np.allclose(stats_vals, stats_expected)
    
    assert stats.loc[current_station, "timediff_min"] == timedif_min
    assert stats.loc[current_station, "timediff_max"] == timedif_max
