# -*- coding: utf-8 -*-

import pytest
import kenmerkendewaarden as kw
import numpy as np
import pandas as pd
from kenmerkendewaarden.data_retrieve import drop_duplicate_times


@pytest.mark.timeout(60)  # useful in case of ddl failure
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


@pytest.mark.unittest
def test_drop_duplicate_times(df_meas_2010, caplog):
    # create dataframe with many duplicated time-value-combinations
    meas_duplicated = pd.concat([df_meas_2010, df_meas_2010], axis=0)
    # convert 30 rows to only-times-duplicated by setting arbitrary value
    meas_duplicated.iloc[:30] = 1
    meas_clean = drop_duplicate_times(meas_duplicated)

    assert len(meas_duplicated) == 105120
    assert len(meas_clean) == 52560
    
    # assert logging messages
    assert '52530 rows with duplicated time-value-combinations dropped' in caplog.text
    assert '30 rows with duplicated times dropped' in caplog.text


@pytest.mark.unittest
def test_drop_duplicate_times_noaction(df_meas_2010, caplog):
    meas_clean = drop_duplicate_times(df_meas_2010)

    assert len(df_meas_2010) == 52560
    assert len(meas_clean) == 52560
    
    # assert that there is no logging messages
    assert caplog.text == ""


@pytest.mark.timeout(120)  # useful in case of ddl failure
@pytest.mark.systemtest
@pytest.mark.parametrize("extremes", [False, True], ids=["timeseries", "extremes"])
def test_retrieve_read_measurements_amount(dir_meas_amount, extremes):
    df_amount = kw.read_measurements_amount(
        dir_output=dir_meas_amount, extremes=extremes
    )

    # assert amounts, this might change if ddl data is updated
    assert df_amount.columns.tolist() == ["HOEKVHLD"]
    assert df_amount.index.tolist() == [2010, 2011]
    if extremes:
        df_vals = np.array([312, 157])
    else:
        df_vals = np.array([8784, 4465])
    assert len(df_amount) == 2
    assert np.allclose(df_amount["HOEKVHLD"].values, df_vals)


@pytest.mark.timeout(60)  # useful in case of ddl failure
@pytest.mark.unittest
def test_retrieve_read_measurements(dir_meas):
    df_meas = kw.read_measurements(
        dir_output=dir_meas, station="HOEKVHLD", extremes=False
    )
    df_ext = kw.read_measurements(
        dir_output=dir_meas, station="HOEKVHLD", extremes=True
    )
    assert df_meas.index.tz.zone == "Etc/GMT-1"
    assert df_ext.index.tz.zone == "Etc/GMT-1"
    assert df_meas.index[0] == pd.Timestamp("2010-01-01 00:00:00+0100", tz="Etc/GMT-1")
    assert df_meas.index[-1] == pd.Timestamp("2011-01-01 00:00:00+0100", tz="Etc/GMT-1")
    assert df_ext.index[0] == pd.Timestamp("2010-01-01 02:35:00+0100", tz="Etc/GMT-1")
    assert df_ext.index[-1] == pd.Timestamp("2010-12-31 23:50:00+0100", tz="Etc/GMT-1")


@pytest.mark.timeout(60)  # useful in case of ddl failure
@pytest.mark.unittest
def test_read_measurements_amount_notfound(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        kw.read_measurements_amount(dir_output=tmp_path, extremes=False)
    assert "data_amount_ts.csv does not exist" in str(e.value)


@pytest.mark.timeout(60)  # useful in case of ddl failure
@pytest.mark.unittest
def test_read_measurements_notfound(tmp_path):
    # this will silently continue the process, returing None
    df_meas = kw.read_measurements(
        dir_output=tmp_path, station="HOEKVHLD", extremes=False
    )
    assert df_meas is None


@pytest.mark.timeout(60)  # useful in case of ddl failure
@pytest.mark.unittest
def test_retrieve_measurements_wrongperiod():
    dir_meas = "."
    start_date = pd.Timestamp(3010, 1, 1, tz="UTC+01:00")
    end_date = pd.Timestamp(3010, 1, 2, tz="UTC+01:00")
    current_station = "HOEKVHLD"

    # retrieve measurements
    with pytest.raises(ValueError) as e:
        kw.retrieve_measurements(
            dir_output=dir_meas,
            station=current_station,
            extremes=False,
            start_date=start_date,
            end_date=end_date,
        )
    assert "[NO DATA]" in str(e.value)


@pytest.mark.timeout(60)  # useful in case of ddl failure
@pytest.mark.unittest
def test_raise_multiple_locations_toomuch():
    locs_meas_ts, _, _ = kw.data_retrieve.retrieve_catalog()
    bool_stations = locs_meas_ts.index.isin(["BATH"])
    locs_sel = locs_meas_ts.loc[bool_stations]
    with pytest.raises(ValueError) as e:
        kw.data_retrieve.raise_multiple_locations(locs_sel)
    assert "multiple stations present after station subsetting" in str(e.value)


@pytest.mark.timeout(60)  # useful in case of ddl failure
@pytest.mark.unittest
def test_raise_multiple_locations_toolittle():
    locs_meas_ts, _, _ = kw.data_retrieve.retrieve_catalog()
    bool_stations = locs_meas_ts.index.isin(["NONEXISTENTSTATION"])
    locs_sel = locs_meas_ts.loc[bool_stations]
    # this will silently continue the process, returing None
    returned_value = kw.data_retrieve.raise_multiple_locations(locs_sel)
    assert returned_value is None


@pytest.mark.timeout(60)  # useful in case of ddl failure
@pytest.mark.unittest
def test_read_measurements_napcorrection(dir_meas):
    """
    the necessary assertions are done in non-ddl tests below
    """
    kw.read_measurements(
        dir_output=dir_meas, station="HOEKVHLD", extremes=True, nap_correction=True
    )


@pytest.mark.unittest
def test_napcorrection(df_meas):
    df_meas_sel = df_meas.loc["2004":"2005"]
    df_meas_sel_nap = kw.data_retrieve.nap2005_correction(df_meas=df_meas_sel)
    assert (df_meas_sel.index == df_meas_sel_nap.index).all()
    assert np.isclose(
        df_meas_sel["values"].iloc[0] - df_meas_sel_nap["values"].iloc[0], 0.0277
    )
    assert np.isclose(
        df_meas_sel["values"].iloc[-1] - df_meas_sel_nap["values"].iloc[-1], 0
    )


@pytest.mark.unittest
def test_napcorrection_notdefined(df_meas_2010):
    df_meas_nonexistentstation = (
        df_meas_2010.copy()
    )  # only change attributes on a copy of the dataframe
    df_meas_nonexistentstation.attrs["station"] = "NONEXISTENTSTATION"
    with pytest.raises(KeyError) as e:
        kw.data_retrieve.nap2005_correction(df_meas=df_meas_nonexistentstation)
    assert "NAP2005 correction not defined for NONEXISTENTSTATION" in str(e.value)


@pytest.mark.unittest
def test_clip_timeseries_physical_break(df_ext):
    df_ext_vlie = df_ext.copy()  # only change attributes on a copy of the dataframe
    df_ext_vlie.attrs["station"] = "VLIELHVN"
    df_ext_vlie_clipped = kw.data_retrieve.clip_timeseries_physical_break(
        df_meas=df_ext_vlie
    )
    assert len(df_ext_vlie_clipped) != len(df_ext_vlie)
    assert df_ext_vlie_clipped.index[0] >= pd.Timestamp("1933-01-01 00:00:00 +01:00")


@pytest.mark.unittest
def test_clip_timeseries_physical_break_notdefined(df_ext):
    df_ext_clipped = kw.data_retrieve.clip_timeseries_physical_break(df_meas=df_ext)
    assert len(df_ext_clipped) == len(df_ext)
    assert df_ext_clipped.index[0] == df_ext.index[0]
