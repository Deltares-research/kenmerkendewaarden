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
@pytest.mark.unittest
def test_retrieve_measurements(dir_meas):
    df_meas = kw.read_measurements(dir_output=dir_meas, station="HOEKVHLD", extremes=False)
    df_ext = kw.read_measurements(dir_output=dir_meas, station="HOEKVHLD", extremes=True)
    assert df_meas.index.tz.zone == 'Etc/GMT-1'
    assert df_ext.index.tz.zone == 'Etc/GMT-1'
    assert df_meas.index[0] == pd.Timestamp('2010-01-01 00:00:00+0100', tz='Etc/GMT-1')
    assert df_meas.index[-1] == pd.Timestamp('2011-01-01 00:00:00+0100', tz='Etc/GMT-1')
    assert df_ext.index[0] == pd.Timestamp('2010-01-01 02:35:00+0100', tz='Etc/GMT-1')
    assert df_ext.index[-1] == pd.Timestamp('2010-12-31 23:50:00+0100', tz='Etc/GMT-1')
