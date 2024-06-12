# -*- coding: utf-8 -*-

import pytest
import numpy as np
import kenmerkendewaarden as kw


@pytest.mark.unittest
def test_fit_models(df_meas_2010_2014):
    dict_wltidalindicators_valid = kw.calc_wltidalindicators(df_meas_2010_2014, min_count=2900) #24*365=8760 (hourly interval), 24/3*365=2920 (3-hourly interval)
    wl_mean_peryear_valid = dict_wltidalindicators_valid['wl_mean_peryear']
    pred_pd_wl = kw.fit_models(wl_mean_peryear_valid)
    
    for key in ['pred_linear_nonodal', 'pred_linear_winodal']:
        assert key in pred_pd_wl.columns
        
    nonodal_expected = np.array([0.07851414, 0.0813139 , 0.08411366, 0.08691342, 0.08971318,
           0.09251294, 0.0953127 , 0.09811246, 0.10091222])
    winodal_expected = np.array([0.0141927 , 0.08612119, 0.0853051 , 0.07010864, 0.10051922,
           0.23137634, 0.50618156, 0.95218827, 1.57732542])
    assert np.allclose(pred_pd_wl['pred_linear_nonodal'].values, nonodal_expected)
    assert np.allclose(pred_pd_wl['pred_linear_winodal'].values, winodal_expected)
