# -*- coding: utf-8 -*-

import os
import pytest
import hatyan
import pandas as pd
import kenmerkendewaarden as kw
import logging
logging.getLogger("kenmerkendewaarden").setLevel(level="INFO")

dir_tests = os.path.dirname(__file__) #F9 doesnt work, only F5 (F5 also only method to reload external definition scripts)
dir_testdata = os.path.join(dir_tests,'testdata')


@pytest.fixture(scope="session")
def df_meas():
    file_dia_ts = os.path.join(dir_testdata, "HOEK_KW.dia")
    df_meas = hatyan.read_dia(file_dia_ts)
    return df_meas


@pytest.fixture(scope="session")
def df_meas_2010(df_meas):
    df_meas_2010 = df_meas.loc["2010":"2010"]
    assert len(df_meas_2010) == 52560
    return df_meas_2010


@pytest.fixture(scope="session")
def df_meas_2010_2014(df_meas):
    df_meas_2010_2014 = df_meas.loc["2010":"2014"]
    assert len(df_meas_2010_2014) == 262944
    return df_meas_2010_2014


@pytest.fixture(scope="session")
def df_ext():
    file_dia_ext = os.path.join(dir_testdata, "HOEKVHLD_ext.dia")
    df_ext = hatyan.read_dia(file_dia_ext, station="HOEKVHLD", block_ids="allstation")
    return df_ext


@pytest.fixture(scope="session")
def df_ext_2010(df_ext):
    df_ext_2010 = df_ext.loc["2010":"2010"]
    assert len(df_ext_2010) == 1863
    return df_ext_2010


@pytest.fixture(scope="session")
def df_ext_2010_2014(df_ext):
    df_ext_2010_2014 = df_ext.loc["2010":"2014"]
    assert len(df_ext_2010_2014) == 9881
    return df_ext_2010_2014


@pytest.fixture(scope="session")
def df_ext_12_2010(df_ext):
    df_ext_sel = df_ext.loc["2009-12-28":"2011-01-03"]
    df_ext_12 = hatyan.calc_HWLW12345to12(df_ext_sel)
    df_ext_12_2010 = df_ext_12.loc["2010":"2010"]
    assert len(df_ext_12_2010) == 1411
    return df_ext_12_2010


@pytest.fixture(scope="session")
def df_ext_12_2010_2014(df_ext):
    df_ext_sel = df_ext.loc["2009-12-28":"2015-01-03"]
    df_ext_12 = hatyan.calc_HWLW12345to12(df_ext_sel)
    df_ext_12_2010_2014 = df_ext_12.loc["2010":"2014"]
    assert len(df_ext_12_2010_2014) == 7057
    return df_ext_12_2010_2014


@pytest.fixture(scope="session")
def df_components_2010(df_meas_2010):
    df_components_2010 = hatyan.analysis(df_meas_2010, const_list="year")
    return df_components_2010


# adding scope will raise "Failed: ScopeMismatch: You tried to access the function scoped fixture tmp_path with a session scoped request objec"
@pytest.fixture
def dir_meas(tmp_path):
    dir_meas = tmp_path
    start_date = pd.Timestamp(2010,1,1, tz="UTC+01:00")
    end_date = pd.Timestamp(2011,1,1, tz="UTC+01:00")
    current_station = "HOEKVHLD"

    # retrieve measurements
    kw.retrieve_measurements(dir_output=dir_meas, station=current_station, extremes=False,
                             start_date=start_date, end_date=end_date)
    kw.retrieve_measurements(dir_output=dir_meas, station=current_station, extremes=True,
                             start_date=start_date, end_date=end_date)
    return dir_meas


# adding scope will raise "Failed: ScopeMismatch: You tried to access the function scoped fixture tmp_path with a session scoped request objec"
@pytest.fixture
def dir_meas_amount(tmp_path):
    dir_meas_amount = tmp_path
    start_date = pd.Timestamp(2010,11,1, tz="UTC+01:00")
    end_date = pd.Timestamp(2011,2,1, tz="UTC+01:00")
    station_list = ["HOEKVHLD"]
    
    kw.retrieve_measurements_amount(dir_output=dir_meas_amount, station_list=station_list, 
                                    start_date=start_date, end_date=end_date,
                                    extremes=False)
    kw.retrieve_measurements_amount(dir_output=dir_meas_amount, station_list=station_list, 
                                    start_date=start_date, end_date=end_date,
                                    extremes=True)
    return dir_meas_amount
