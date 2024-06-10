# -*- coding: utf-8 -*-

import pytest
import pandas as pd
import kenmerkendewaarden as kw
import logging
logging.basicConfig(format='%(message)s')
logging.getLogger("kenmerkendewaarden").setLevel(level="INFO")


@pytest.mark.timeout(60) # useful in case of ddl failure
@pytest.fixture(scope="session")
def dir_meas_timeseries(tmp_path):
    dir_meas_timeseries = tmp_path
    start_date = pd.Timestamp(2010,1,1, tz="UTC+01:00")
    end_date = pd.Timestamp(2011,1,1, tz="UTC+01:00")
    current_station = "HOEKVHLD"
    
    # retrieve meas
    kw.retrieve_measurements(dir_output=dir_meas_timeseries, station=current_station, extremes=False,
                             start_date=start_date, end_date=end_date)
    return dir_meas_timeseries


@pytest.mark.timeout(60) # useful in case of ddl failure
@pytest.fixture(scope="session")
def dir_meas_extremes(tmp_path):
    dir_meas_extremes = tmp_path
    start_date = pd.Timestamp(2010,1,1, tz="UTC+01:00")
    end_date = pd.Timestamp(2011,1,1, tz="UTC+01:00")
    current_station = "HOEKVHLD"
    
    # retrieve meas
    kw.retrieve_measurements(dir_output=dir_meas_extremes, station=current_station, extremes=True,
                             start_date=start_date, end_date=end_date)
    return dir_meas_extremes
