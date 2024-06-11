# -*- coding: utf-8 -*-

import pytest
import pandas as pd
import kenmerkendewaarden as kw
import hatyan
import logging
logging.basicConfig(format='%(message)s')
logging.getLogger("kenmerkendewaarden").setLevel(level="INFO")


@pytest.fixture
def dir_meas_timeseries(tmp_path):
    dir_meas_timeseries = tmp_path
    start_date = pd.Timestamp(2010,1,1, tz="UTC+01:00")
    end_date = pd.Timestamp(2011,1,1, tz="UTC+01:00")
    current_station = "HOEKVHLD"
    
    # retrieve meas
    kw.retrieve_measurements(dir_output=dir_meas_timeseries, station=current_station, extremes=False,
                             start_date=start_date, end_date=end_date)
    return dir_meas_timeseries


@pytest.fixture
def dir_meas_extremes(tmp_path):
    dir_meas_extremes = tmp_path
    start_date = pd.Timestamp(2010,1,1, tz="UTC+01:00")
    end_date = pd.Timestamp(2011,1,1, tz="UTC+01:00")
    current_station = "HOEKVHLD"
    
    # retrieve meas
    kw.retrieve_measurements(dir_output=dir_meas_extremes, station=current_station, extremes=True,
                             start_date=start_date, end_date=end_date)
    return dir_meas_extremes


@pytest.fixture
def df_meas_2010(dir_meas_timeseries):
    df_meas_2010 = kw.read_measurements(dir_output=dir_meas_timeseries, station="HOEKVHLD", extremes=False)
    df_meas_2010 = df_meas_2010.loc["2010":"2010"]
    return df_meas_2010


@pytest.fixture
def df_ext_2010(dir_meas_extremes):
    df_ext_2010 = kw.read_measurements(dir_output=dir_meas_timeseries, station="HOEKVHLD", extremes=False)
    df_ext_2010 = df_ext_2010.loc["2010":"2010"]
    return df_ext_2010


@pytest.fixture
def df_ext_2010_12(df_ext_2010):
    df_ext_2010_12 = hatyan.calc_HWLW12345to12(df_ext_2010)
    return df_ext_2010_12
