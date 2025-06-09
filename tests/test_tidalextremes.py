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
def test_calc_hat_lat_fromcomponents(df_components_2010):
    # use subset to speed up test
    df_components_2010_sel = df_components_2010.loc[["M2", "M4", "S2"]]
    # generate prediction and derive hat/lat
    hat, lat = kw.calc_hat_lat_fromcomponents(df_components_2010_sel)
    assert np.isclose(hat, 1.2259179749801052)
    assert np.isclose(lat, -0.8368954797393148)


@pytest.mark.unittest
def test_calc_hat_lat_frommeasurements(df_meas):
    df_meas_19y = df_meas.loc["2001":"2019"]
    hat, lat = kw.calc_hat_lat_frommeasurements(df_meas_19y)
    assert np.isclose(hat, 1.6513812158333379)
    assert np.isclose(lat, -0.9135677081163474)


@pytest.mark.unittest
def test_calc_hat_lat_frommeasurements_tooshortperiod(df_meas_2010_2014, caplog):
    with caplog.at_level(logging.WARNING):
        kw.calc_hat_lat_frommeasurements(df_meas_2010_2014)
    assert "requested 19 years but resulted in 5" in caplog.text


