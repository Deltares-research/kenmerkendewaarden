# -*- coding: utf-8 -*-

import pytest
import kenmerkendewaarden as kw
import numpy as np
import pandas as pd
from kenmerkendewaarden.data_retrieve import (
    drop_duplicate_times,
    raise_incorrect_quantity,
)
import logging


@pytest.mark.timeout(60)  # useful in case of ddl failure
@pytest.mark.unittest
def test_retrieve_catalog():
    crs = 28992
    locs_meas_wl, locs_meas_ext, _, locs_meas_q = kw.data_retrieve.retrieve_catalog(
        crs=crs
    )

    assert np.isclose(locs_meas_wl.loc["hoekvanholland"]["Lon"], 67924.98523360591)
    assert np.isclose(locs_meas_wl.loc["hoekvanholland"]["Lat"], 443924.9530462769)
    assert np.isclose(locs_meas_ext.loc["hoekvanholland"]["Lon"], 67924.98523360591)
    assert np.isclose(locs_meas_ext.loc["hoekvanholland"]["Lat"], 443924.9530462769)
    assert np.isclose(
        locs_meas_q.loc["schijndel.sluis.kleplinks"]["Lon"], 158894.00869222777
    )
    assert np.isclose(
        locs_meas_q.loc["schijndel.sluis.kleplinks"]["Lat"], 406610.09644023067
    )
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
    assert "52530 rows with duplicated time-value-combinations dropped" in caplog.text
    assert "30 rows with duplicated times dropped" in caplog.text


@pytest.mark.unittest
def test_drop_duplicate_times_noaction(df_meas_2010, caplog):
    meas_clean = drop_duplicate_times(df_meas_2010)

    assert len(df_meas_2010) == 52560
    assert len(meas_clean) == 52560

    # assert that there is no logging messages
    assert caplog.text == ""


@pytest.mark.timeout(120)  # useful in case of ddl failure
@pytest.mark.systemtest
@pytest.mark.parametrize("quantity", ["meas_wl", "meas_ext", "meas_q"])
def test_retrieve_read_measurements_amount(dir_meas_amount, quantity):
    df_amount = kw.read_measurements_amount(
        dir_output=dir_meas_amount, quantity=quantity
    )

    # assert amounts, this might change if ddl data is updated
    assert df_amount.index.tolist() == [2010, 2011]
    if quantity == "meas_wl":
        df_vals = np.array([8784, 4465])
        expected_station = "hoekvanholland"
    elif quantity == "meas_ext":
        df_vals = np.array([312, 157])
        expected_station = "hoekvanholland"
    elif quantity == "meas_q":
        df_vals = np.array([1525, 777])
        expected_station = "hagestein.boven"
    assert df_amount.columns.tolist() == [expected_station]
    assert len(df_amount) == 2
    assert np.allclose(df_amount[expected_station].values, df_vals)


@pytest.mark.timeout(60)  # useful in case of ddl failure
@pytest.mark.unittest
def test_retrieve_read_measurements(dir_meas):
    df_meas = kw.read_measurements(
        dir_output=dir_meas,
        station="hoekvanholland",
        quantity="meas_wl",
    )
    df_ext = kw.read_measurements(
        dir_output=dir_meas,
        station="hoekvanholland",
        quantity="meas_ext",
    )
    df_q = kw.read_measurements(
        dir_output=dir_meas,
        station="hagestein.boven",
        quantity="meas_q",
    )
    assert df_meas.index.tz.zone == "Etc/GMT-1"
    assert df_meas.index[0] == pd.Timestamp("2010-01-01 00:00:00+0100", tz="Etc/GMT-1")
    assert df_meas.index[-1] == pd.Timestamp("2011-01-01 00:00:00+0100", tz="Etc/GMT-1")
    assert df_ext.index.tz.zone == "Etc/GMT-1"
    assert df_ext.index[0] == pd.Timestamp("2010-01-01 02:35:00+0100", tz="Etc/GMT-1")
    assert df_ext.index[-1] == pd.Timestamp("2010-12-31 23:50:00+0100", tz="Etc/GMT-1")
    assert df_q.index.tz.zone == "Etc/GMT-1"
    assert df_q.index[0] == pd.Timestamp("2010-01-01 00:00:00+0100", tz="Etc/GMT-1")
    assert df_q.index[-1] == pd.Timestamp("2011-01-01 00:00:00+0100", tz="Etc/GMT-1")


@pytest.mark.timeout(60)  # useful in case of ddl failure
@pytest.mark.unittest
def test_read_measurements_amount_notfound(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        kw.read_measurements_amount(dir_output=tmp_path, quantity="meas_wl")
    assert "data_amount_wl.csv does not exist" in str(e.value)


@pytest.mark.timeout(60)  # useful in case of ddl failure
@pytest.mark.unittest
def test_read_measurements_notfound(tmp_path):
    # this will silently continue the process, returing None
    df_meas = kw.read_measurements(
        dir_output=tmp_path,
        station="hoekvanholland",
        quantity="meas_wl",
    )
    assert df_meas is None


@pytest.mark.unittest
def test_retrieve_measurements_already_exists(tmp_path, caplog):
    # create dummy file that would be created by kw.retrieve_measurements()
    expected_file = tmp_path / "hoekvanholland_meas_wl.nc"
    with open(expected_file, "w") as f:
        f.write("")

    start_date = pd.Timestamp(2010, 1, 1, tz="UTC+01:00")
    end_date = pd.Timestamp(2010, 1, 2, tz="UTC+01:00")
    current_station = "hoekvanholland"

    # retrieve measurements
    with caplog.at_level(logging.INFO):
        meas = kw.retrieve_measurements(
            dir_output=tmp_path,
            station=current_station,
            quantity="meas_wl",
            start_date=start_date,
            end_date=end_date,
        )
    assert (
        "meas data (quantity=meas_wl) for hoekvanholland already available"
        in caplog.text
    )
    assert meas is None


@pytest.mark.unittest
def test_retrieve_measurements_no_station(caplog):
    start_date = pd.Timestamp(2010, 1, 1, tz="UTC+01:00")
    end_date = pd.Timestamp(2010, 1, 2, tz="UTC+01:00")
    current_station = "NON-EXISTENT-STATION"

    # retrieve measurements
    with caplog.at_level(logging.INFO):
        meas = kw.retrieve_measurements(
            dir_output=".",
            station=current_station,
            quantity="meas_wl",
            start_date=start_date,
            end_date=end_date,
        )
    assert "no station available (quantity=meas_wl), skipping station" in caplog.text
    assert meas is None


@pytest.mark.timeout(60)  # useful in case of ddl failure
@pytest.mark.unittest
def test_retrieve_measurements_wrongperiod(caplog):
    start_date = pd.Timestamp(3010, 1, 1, tz="UTC+01:00")
    end_date = pd.Timestamp(3010, 1, 2, tz="UTC+01:00")
    current_station = "hoekvanholland"

    # retrieve measurements
    with caplog.at_level(logging.INFO):
        kw.retrieve_measurements(
            dir_output=".",
            station=current_station,
            quantity="meas_wl",
            start_date=start_date,
            end_date=end_date,
        )
    assert "no data found for the requested period" in caplog.text


@pytest.mark.timeout(60)  # useful in case of ddl failure
@pytest.mark.unittest
def test_retrieve_measurements_amount_periodwithoutdata(tmp_path, caplog):
    start_date = pd.Timestamp(2050, 1, 1, tz="UTC+01:00")
    end_date = pd.Timestamp(2050, 1, 2, tz="UTC+01:00")
    current_station = "kloosterzande.baalhoek"

    # retrieve measurements
    with caplog.at_level(logging.INFO):
        kw.retrieve_measurements_amount(
            dir_output=tmp_path,
            station_list=[current_station],
            quantity="meas_ext",
            start_date=start_date,
            end_date=end_date,
        )
    assert "no measurements available in this period" in caplog.text


@pytest.mark.timeout(60)  # useful in case of ddl failure
@pytest.mark.unittest
def test_retrieve_measurements_amount_emptylocslist(tmp_path, caplog):
    start_date = pd.Timestamp(2020, 1, 1, tz="UTC+01:00")
    end_date = pd.Timestamp(2021, 1, 2, tz="UTC+01:00")
    current_station = "a12"

    # retrieve measurements
    with caplog.at_level(logging.INFO):
        kw.retrieve_measurements_amount(
            dir_output=tmp_path,
            station_list=[current_station],
            quantity="meas_ext",
            start_date=start_date,
            end_date=end_date,
        )
    assert "no station available" in caplog.text


@pytest.mark.timeout(60)  # useful in case of ddl failure
@pytest.mark.unittest
def test_raise_multiple_locations_toomuch():
    locs_meas_wl, _, _, _ = kw.data_retrieve.retrieve_catalog()
    bool_stations = locs_meas_wl.index.isin(["europlatform"])
    locs_sel = locs_meas_wl.loc[bool_stations]
    with pytest.raises(ValueError) as e:
        kw.data_retrieve.raise_multiple_locations(locs_sel)
    assert "multiple stations present after station subsetting" in str(e.value)


@pytest.mark.timeout(60)  # useful in case of ddl failure
@pytest.mark.unittest
def test_raise_multiple_locations_toolittle():
    locs_meas_wl, _, _, _ = kw.data_retrieve.retrieve_catalog()
    bool_stations = locs_meas_wl.index.isin(["NONEXISTENTSTATION"])
    locs_sel = locs_meas_wl.loc[bool_stations]
    # this will silently continue the process, returing None
    returned_value = kw.data_retrieve.raise_multiple_locations(locs_sel)
    assert returned_value is None


@pytest.mark.unittest
def test_raise_incorrect_quantity():
    with pytest.raises(ValueError) as e:
        raise_incorrect_quantity("incorrect")
    assert "quantity 'incorrect' is not allowed, choose from" in str(e.value)


@pytest.mark.timeout(60)  # useful in case of ddl failure
@pytest.mark.unittest
def test_read_measurements_napcorrection(dir_meas):
    """
    the necessary assertions are done in non-ddl tests below
    """
    kw.read_measurements(
        dir_output=dir_meas,
        station="hoekvanholland",
        quantity="meas_ext",
        nap_correction=True,
    )


@pytest.mark.unittest
def test_napcorrection(df_meas):
    df_meas_sel = df_meas.loc["2004":"2005"]
    df_meas_sel.attrs["station"] = "hoekvanholland"
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
    df_ext_vlie.attrs["station"] = "vlieland.haven"
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
