# -*- coding: utf-8 -*-

import pytest
import kenmerkendewaarden as kw
import hatyan
import numpy as np


@pytest.mark.timeout(60) # useful in case of ddl failure
@pytest.mark.systemtest
def test_gemiddeld_getijkromme_av_sp_np_raw(dir_meas_timeseries, dir_meas_extremes):
    pred_freq = "60s"
    df_meas = kw.read_measurements(dir_output=dir_meas_timeseries, station="HOEKVHLD", extremes=False)
    df_meas = df_meas.loc["2010":"2010"]
    
    prediction_av_raw, prediction_sp_raw, prediction_np_raw = kw.gemiddeld_getijkromme_av_sp_np(
                                    df_meas=df_meas, df_ext=None,
                                    freq=pred_freq, nb=0, nf=0, 
                                    scale_extremes=False, scale_period=False)
    
    assert len(prediction_av_raw) == 746
    assert np.isclose(prediction_av_raw["values"].min(), -0.567608885905236)
    assert np.isclose(prediction_av_raw["values"].max(), 1.1665100442336331)

    assert len(prediction_sp_raw) == 741
    assert np.isclose(prediction_sp_raw["values"].min(), -0.5305575695904736)
    assert np.isclose(prediction_sp_raw["values"].max(), 1.313379801706846)

    assert len(prediction_np_raw) == 755
    assert np.isclose(prediction_np_raw["values"].min(), -0.579937620616439)
    assert np.isclose(prediction_np_raw["values"].max(), 0.8044625899304197)


@pytest.mark.timeout(60) # useful in case of ddl failure
@pytest.mark.systemtest
def test_gemiddeld_getijkromme_av_sp_np_corr(dir_meas_timeseries, dir_meas_extremes):
    pred_freq = "60s"
    df_meas = kw.read_measurements(dir_output=dir_meas_timeseries, station="HOEKVHLD", extremes=False)
    df_meas = df_meas.loc["2010":"2010"]
    df_ext = kw.read_measurements(dir_output=dir_meas_extremes, station="HOEKVHLD", extremes=True)
    df_ext_12 = hatyan.calc_HWLW12345to12(df_ext)
    
    prediction_av_corr, prediction_sp_corr, prediction_np_corr = kw.gemiddeld_getijkromme_av_sp_np(
                                    df_meas=df_meas, df_ext=df_ext_12,
                                    freq=pred_freq, nb=2, nf=2, 
                                    scale_extremes=True, scale_period=False)
    
    assert len(prediction_av_corr) == 3726
    assert np.isclose(prediction_av_corr["values"].min(), -0.6095833333333333)
    assert np.isclose(prediction_av_corr["values"].max(), 1.1300000000000003)

    assert len(prediction_sp_corr) == 3701
    assert np.isclose(prediction_sp_corr["values"].min(), -0.5700000000000001)
    assert np.isclose(prediction_sp_corr["values"].max(), 1.3450000000000002)

    assert len(prediction_np_corr) == 3771
    assert np.isclose(prediction_np_corr["values"].min(), -0.61)
    assert np.isclose(prediction_np_corr["values"].max(), 0.8650000000000001)


@pytest.mark.timeout(60) # useful in case of ddl failure
@pytest.mark.systemtest
def test_gemiddeld_getijkromme_av_sp_np_corr_boi(dir_meas_timeseries, dir_meas_extremes):
    pred_freq = "60s"
    df_meas = kw.read_measurements(dir_output=dir_meas_timeseries, station="HOEKVHLD", extremes=False)
    df_meas = df_meas.loc["2010":"2010"]
    df_ext = kw.read_measurements(dir_output=dir_meas_extremes, station="HOEKVHLD", extremes=True)
    df_ext_12 = hatyan.calc_HWLW12345to12(df_ext)
    
    prediction_av_corr_boi, prediction_sp_corr_boi, prediction_np_corr_boi = kw.gemiddeld_getijkromme_av_sp_np(
                                    df_meas=df_meas, df_ext=df_ext_12,
                                    freq=pred_freq, nb=0, nf=10, 
                                    scale_extremes=True, scale_period=True)
    
    assert len(prediction_av_corr_boi) == 8196
    assert np.isclose(prediction_av_corr_boi["values"].min(), -0.6095833333333333)
    assert np.isclose(prediction_av_corr_boi["values"].max(), 1.1300000000000003)

    assert len(prediction_sp_corr_boi) == 8196
    assert np.isclose(prediction_sp_corr_boi["values"].min(), -0.5700000000000001)
    assert np.isclose(prediction_sp_corr_boi["values"].max(), 1.3450000000000002)

    assert len(prediction_np_corr_boi) == 8196
    assert np.isclose(prediction_np_corr_boi["values"].min(), -0.61)
    assert np.isclose(prediction_np_corr_boi["values"].max(), 0.8650000000000001)

