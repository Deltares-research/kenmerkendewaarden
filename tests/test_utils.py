# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 11:33:11 2024

@author: veenstra
"""
import pytest
from kenmerkendewaarden.utils import (raise_extremes_with_aggers,
                                      crop_timeseries_last_nyears)
import pandas as pd
import numpy as np


@pytest.mark.unittest
def test_raise_extremes_with_aggers_raise_12345df(df_ext):
    with pytest.raises(ValueError) as e:
        raise_extremes_with_aggers(df_ext)
    expected_error = (
        "df_ext should only contain extremes (HWLWcode 1/2), "
        "but it also contains aggers (HWLWcode 3/4/5"
    )
    assert expected_error in str(e.value)


@pytest.mark.unittest
def test_raise_extremes_with_aggers_pass_12df(df_ext_12_2010):
    raise_extremes_with_aggers(df_ext_12_2010)


@pytest.mark.unittest
def test_crop_timeseries_last_nyears(df_meas):
    assert df_meas.index[0] == pd.Timestamp("1987-01-01 00:00:00+01:00 ")
    assert df_meas.index[-1] == pd.Timestamp("2022-01-01 00:00:00+01:00")
    assert len(df_meas) == 1840897
    
    df_meas_10y = crop_timeseries_last_nyears(df_meas, nyears=10)
    # assert number of years
    num_years = len(df_meas_10y.index.year.unique())
    assert num_years == 10
    # assert start/end timestamps and length
    assert df_meas_10y.index[0] == pd.Timestamp("2012-01-01 00:00:00+01:00 ")
    assert df_meas_10y.index[-1] == pd.Timestamp("2021-12-31 23:50:00+01:00")
    assert len(df_meas_10y) == 526032
    # assert on years
    actual_years = df_meas_10y.index.year.drop_duplicates().to_numpy()
    expected_years = np.arange(2012, 2021+1)
    assert (actual_years == expected_years).all()


@pytest.mark.unittest
def test_crop_timeseries_last_nyears_warning_tooshort(df_meas_2010_2014, caplog):
    crop_timeseries_last_nyears(df_meas_2010_2014, nyears=10)
    assert "requested 10 years but resulted in 5" in caplog.text


@pytest.mark.unittest
def test_raise_extremes_with_aggers_emptydf():
    import pandas as pd

    time_index = pd.DatetimeIndex(
        [], dtype="datetime64[ns, Etc/GMT-1]", name="time", freq=None
    )
    df_ext = pd.DataFrame({"HWLWcode": []}, index=time_index)
    df_ext.attrs["station"] = "dummy"
    raise_extremes_with_aggers(df_ext=df_ext)
