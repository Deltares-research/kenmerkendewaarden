# -*- coding: utf-8 -*-

import pytest
import kenmerkendewaarden as kw
import numpy as np


@pytest.mark.unittest
def test_gemiddeld_getijkromme_av_sp_np_raw(df_meas_2010):
    pred_freq = "60s"
    
    prediction_av_raw, prediction_sp_raw, prediction_np_raw = kw.gemiddeld_getijkromme_av_sp_np(
                                    df_meas=df_meas_2010, df_ext=None,
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


@pytest.mark.unittest
def test_gemiddeld_getijkromme_av_sp_np_corr(df_meas_2010, df_ext_12_2010):
    pred_freq = "60s"
    
    prediction_av_corr, prediction_sp_corr, prediction_np_corr = kw.gemiddeld_getijkromme_av_sp_np(
                                    df_meas=df_meas_2010, df_ext=df_ext_12_2010,
                                    freq=pred_freq, nb=2, nf=2, 
                                    scale_extremes=True, scale_period=False)
    
    assert len(prediction_av_corr) == 3726
    assert np.isclose(prediction_av_corr["values"].min(), -0.6095833333333333)
    assert np.isclose(prediction_av_corr["values"].max(), 1.1300000000000003) # 1.1229166666666668 in pandas>=2.2

    assert len(prediction_sp_corr) == 3701
    assert np.isclose(prediction_sp_corr["values"].min(), -0.5700000000000001) # -0.635 in pandas>=2.2
    assert np.isclose(prediction_sp_corr["values"].max(), 1.3450000000000002) # 1.32 in pandas>=2.2

    assert len(prediction_np_corr) == 3771
    assert np.isclose(prediction_np_corr["values"].min(), -0.61)
    assert np.isclose(prediction_np_corr["values"].max(), 0.8650000000000001) # 0.89 in pandas>=2.2


@pytest.mark.unittest
def test_gemiddeld_getijkromme_av_sp_np_corr_boi(df_meas_2010, df_ext_12_2010):
    pred_freq = "60s"
    
    prediction_av_corr_boi, prediction_sp_corr_boi, prediction_np_corr_boi = kw.gemiddeld_getijkromme_av_sp_np(
                                    df_meas=df_meas_2010, df_ext=df_ext_12_2010,
                                    freq=pred_freq, nb=0, nf=10, 
                                    scale_extremes=True, scale_period=True)
    
    assert len(prediction_av_corr_boi) == 8196
    assert np.isclose(prediction_av_corr_boi["values"].min(), -0.6095833333333333)
    assert np.isclose(prediction_av_corr_boi["values"].max(),  1.1300000000000003) # 1.1229166666666668 in pandas>=2.2

    assert len(prediction_sp_corr_boi) == 8196
    assert np.isclose(prediction_sp_corr_boi["values"].min(), -0.5700000000000001) # -0.635 in pandas>=2.2
    assert np.isclose(prediction_sp_corr_boi["values"].max(), 1.3450000000000002) # 1.32 in pandas>=2.2

    assert len(prediction_np_corr_boi) == 8196
    assert np.isclose(prediction_np_corr_boi["values"].min(), -0.61)
    assert np.isclose(prediction_np_corr_boi["values"].max(), 0.8650000000000001) # 0.89 in pandas>=2.2
