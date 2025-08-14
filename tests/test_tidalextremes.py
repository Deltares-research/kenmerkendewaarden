# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 20:44:29 2025

@author: veenstra
"""

import pytest
import kenmerkendewaarden as kw
import numpy as np
import logging


@pytest.mark.unittest
def test_calc_hat_lat_frommeasurements(df_meas):
    df_meas_19y = df_meas.loc["2001":"2019"]
    hat, lat = kw.calc_highest_lowest_astronomical_tide(df_meas_19y)
    assert np.isclose(hat, 1.6383413178426192)
    assert np.isclose(lat, -0.926607606107066)


@pytest.mark.unittest
def test_calc_hat_lat_frommeasurements_tooshortperiod(df_meas_2010_2014, caplog):
    with caplog.at_level(logging.WARNING):
        kw.calc_highest_lowest_astronomical_tide(df_meas_2010_2014)
    assert "requested 19 years but resulted in 5" in caplog.text
