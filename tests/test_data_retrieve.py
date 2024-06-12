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


@pytest.mark.timeout(120) # useful in case of ddl failure
@pytest.mark.systemtest
@pytest.mark.parametrize("extremes", [False,True], ids=["timeseries", "extremes"])
def test_retrieve_read_measurements_amount(dir_meas_amount, extremes):
    df_amount = kw.read_measurements_amount(dir_output=dir_meas_amount, extremes=extremes)
    
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
def test_retrieve_read_measurements(dir_meas):
    df_meas = kw.read_measurements(dir_output=dir_meas, station="HOEKVHLD", extremes=False)
    df_ext = kw.read_measurements(dir_output=dir_meas, station="HOEKVHLD", extremes=True)
    assert df_meas.index.tz.zone == 'Etc/GMT-1'
    assert df_ext.index.tz.zone == 'Etc/GMT-1'
    assert df_meas.index[0] == pd.Timestamp('2010-01-01 00:00:00+0100', tz='Etc/GMT-1')
    assert df_meas.index[-1] == pd.Timestamp('2011-01-01 00:00:00+0100', tz='Etc/GMT-1')
    assert df_ext.index[0] == pd.Timestamp('2010-01-01 02:35:00+0100', tz='Etc/GMT-1')
    assert df_ext.index[-1] == pd.Timestamp('2010-12-31 23:50:00+0100', tz='Etc/GMT-1')


@pytest.mark.timeout(60) # useful in case of ddl failure
@pytest.mark.unittest
def test_check_locations_amount_toomuch():
    locs_meas_ts, _, _ = kw.data_retrieve.retrieve_catalog()
    bool_stations = locs_meas_ts.index.isin(["BATH"])
    locs_sel = locs_meas_ts.loc[bool_stations]
    with pytest.raises(ValueError) as e:
        kw.data_retrieve.check_locations_amount(locs_sel)
    assert "multiple stations present after station subsetting" in str(e.value)


@pytest.mark.timeout(60) # useful in case of ddl failure
@pytest.mark.unittest
def test_check_locations_amount_toolittle():
    locs_meas_ts, _, _ = kw.data_retrieve.retrieve_catalog()
    bool_stations = locs_meas_ts.index.isin(["NONEXISTENTSTATION"])
    locs_sel = locs_meas_ts.loc[bool_stations]
    # this will silently continue the process, returing None
    returned_value = kw.data_retrieve.check_locations_amount(locs_sel)
    assert returned_value is None
