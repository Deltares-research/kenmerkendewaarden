# -*- coding: utf-8 -*-

import pytest
import kenmerkendewaarden as kw
import numpy as np
import pandas as pd


@pytest.mark.timeout(60) # useful in case of ddl failure
@pytest.mark.systemtest
def test_retrieve_read_measurements_amount(tmp_path):
    start_date = pd.Timestamp(2010,1,1, tz="UTC+01:00")
    end_date = pd.Timestamp(2011,1,1, tz="UTC+01:00")
    station_list = ["HOEKVHLD"]
    
    kw.retrieve_measurements_amount(dir_output=tmp_path, station_list=station_list, 
                                    start_date=start_date, end_date=end_date,
                                    extremes=False)
    kw.retrieve_measurements_amount(dir_output=tmp_path, station_list=station_list, 
                                    start_date=start_date, end_date=end_date,
                                    extremes=True)

    df_amount_ts = kw.read_measurements_amount(dir_output=tmp_path, extremes=False)
    df_amount_ext = kw.read_measurements_amount(dir_output=tmp_path, extremes=True)
    
    assert len(df_amount_ts) == 2
    assert len(df_amount_ext) == 1
    assert np.allclose(df_amount_ts["HOEKVHLD"].values, np.array([52560,     1]))
    assert np.allclose(df_amount_ext["HOEKVHLD"].values, np.array([1863]))


@pytest.mark.timeout(60) # useful in case of ddl failure
@pytest.mark.systemtest
def test_retrieve_read_measurements_derive_statistics(tmp_path):
    start_date = pd.Timestamp(2010,1,1, tz="UTC+01:00")
    end_date = pd.Timestamp(2011,1,1, tz="UTC+01:00")
    station_list = ["HOEKVHLD"]
    current_station = station_list[0]
    
    # retrieve meas
    kw.retrieve_measurements(dir_output=tmp_path, station=current_station, extremes=False,
                             start_date=start_date, end_date=end_date)
    kw.retrieve_measurements(dir_output=tmp_path, station=current_station, extremes=True,
                             start_date=start_date, end_date=end_date)

    # read meas
    df_ts_meas = kw.read_measurements(dir_output=tmp_path, station=current_station, extremes=False)
    df_ext_meas = kw.read_measurements(dir_output=tmp_path, station=current_station, extremes=True)

    # assert amount of measurements, this might change if ddl data is updated
    assert len(df_ts_meas) == 52561
    assert len(df_ext_meas) == 1863
    
    stats_ts = kw.derive_statistics(dir_output=tmp_path, station_list=station_list, extremes=False)
    stats_ext = kw.derive_statistics(dir_output=tmp_path, station_list=station_list, extremes=True)
    
    # assert statistics columns
    cols_stats_ts = ['WaarnemingMetadata.StatuswaardeLijst',
           'WaarnemingMetadata.KwaliteitswaardecodeLijst',
           'WaardeBepalingsmethode.Code', 'MeetApparaat.Code', 'Hoedanigheid.Code',
           'Grootheid.Code', 'Groepering.Code', 'Typering.Code', 'tstart', 'tstop',
           'timediff_min', 'timediff_max', 'nvals', '#nans', 'min', 'max', 'std',
           'mean', 'dupltimes', 'dupltimes_#nans', 'qc_none']
    assert set(stats_ts.columns) == set(cols_stats_ts)
    cols_stats_ext = ['WaarnemingMetadata.StatuswaardeLijst',
           'WaarnemingMetadata.KwaliteitswaardecodeLijst',
           'WaardeBepalingsmethode.Code', 'MeetApparaat.Code', 'Hoedanigheid.Code',
           'Grootheid.Code', 'Groepering.Code', 'Typering.Code', 'tstart', 'tstop',
           'timediff_min', 'timediff_max', 'nvals', '#nans', 'min', 'max', 'std',
           'mean', 'dupltimes', 'dupltimes_#nans', 'qc_none', 'timediff<4hr',
           'aggers']
    assert set(stats_ext.columns) == set(cols_stats_ext)
    
    # assert statistics values, this might change if ddl data is updated
    stats_ts_vals = stats_ts.loc[current_station, ["mean","min","max"]].values.astype(float)
    stats_ts_expected = np.array([0.07962614866536023, -1.33, 2.11])
    assert np.allclose(stats_ts_vals, stats_ts_expected)
    
    stats_ext_vals = stats_ext.loc[current_station, ["mean","min","max"]].values.astype(float)
    stats_ext_expected = np.array([0.07922705314009662, -1.33, 2.11])
    assert np.allclose(stats_ext_vals, stats_ext_expected)
