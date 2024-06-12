# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd
import kenmerkendewaarden as kw


@pytest.mark.unittest
def test_fit_models(df_meas_2010_2014):
    dict_wltidalindicators_valid = kw.calc_wltidalindicators(df_meas_2010_2014, min_count=2900) #24*365=8760 (hourly interval), 24/3*365=2920 (3-hourly interval)
    wl_mean_peryear_valid = dict_wltidalindicators_valid['wl_mean_peryear']
    
    wl_model_fit_nodal = kw.slotgemiddelden.fit_models(wl_mean_peryear_valid, with_nodal=True)
    nodal_expected = np.array([0.0141927 , 0.08612119, 0.0853051 , 0.07010864, 0.10051922, 0.23137634])
    assert np.allclose(wl_model_fit_nodal.values, nodal_expected)
    
    wl_model_fit_linear = kw.slotgemiddelden.fit_models(wl_mean_peryear_valid, with_nodal=False)
    linear_expected = np.array([0.07851414, 0.0813139 , 0.08411366, 0.08691342, 0.08971318, 0.09251294])
    assert np.allclose(wl_model_fit_linear.values, linear_expected)


@pytest.mark.unittest
def test_calc_slotgemiddelden(df_meas_2010_2014, df_ext_12_2010_2014):
    slotgemiddelden_dict_inclext = kw.calc_slotgemiddelden(df_meas=df_meas_2010_2014, df_ext=df_ext_12_2010_2014)
    slotgemiddelden_dict_noext = kw.calc_slotgemiddelden(df_meas=df_meas_2010_2014, df_ext=None)
    
    # assert present keys
    expected_keys_inclext = ['wl_mean_peryear', 'wl_model_fit', 'HW_mean_peryear', 'LW_mean_peryear', 'HW_model_fit', 'LW_model_fit']
    expected_keys_noext = ['wl_mean_peryear', 'wl_model_fit']
    assert set(slotgemiddelden_dict_inclext.keys()) == set(expected_keys_inclext)
    assert set(slotgemiddelden_dict_noext.keys()) == set(expected_keys_noext)

    # assert dtypes of dictionary contents
    for key in expected_keys_inclext:
        assert isinstance(slotgemiddelden_dict_inclext[key], pd.Series)
    
    # assertion of values
    wl_mean_peryear_expected = np.array([0.07960731, 0.08612119, 0.0853051 , 0.07010864, 0.10051922])
    hw_mean_peryear_expected = np.array([1.13968839, 1.12875177, 1.13988685, 1.1415461 , 1.18998584])
    lw_mean_peryear_expected = np.array([-0.60561702, -0.59089362, -0.59342291, -0.61334278, -0.58024113])
    assert np.allclose(slotgemiddelden_dict_inclext['wl_mean_peryear'].values, wl_mean_peryear_expected)
    assert np.allclose(slotgemiddelden_dict_inclext['HW_mean_peryear'].values, hw_mean_peryear_expected)
    assert np.allclose(slotgemiddelden_dict_inclext['LW_mean_peryear'].values, lw_mean_peryear_expected)
    
    wl_model_fit_expected = np.array([0.0141927 , 0.08612119, 0.0853051 , 0.07010864, 0.10051922,
           0.23137634])
    hw_model_fit_expected = np.array([1.05295416, 1.12875177, 1.13988685, 1.1415461 , 1.18998584,
           1.336182  ])
    lw_model_fit_expected = np.array([-0.67420399, -0.59089362, -0.59342291, -0.61334278, -0.58024113,
           -0.42969074])
    assert np.allclose(slotgemiddelden_dict_inclext['wl_model_fit'].values, wl_model_fit_expected)
    assert np.allclose(slotgemiddelden_dict_inclext['HW_model_fit'].values, hw_model_fit_expected)
    assert np.allclose(slotgemiddelden_dict_inclext['LW_model_fit'].values, lw_model_fit_expected)


@pytest.mark.unittest
def test_calc_slotgemiddelden_correct_tstop(df_meas_2010_2014):
    df_meas_upto_2013 = df_meas_2010_2014.loc[:"2013"]
    slotgemiddelden_upto_2013 = kw.calc_slotgemiddelden(df_meas=df_meas_upto_2013, df_ext=None)
    
    df_meas_incl_2014 = df_meas_2010_2014.loc[:"2014-01-01 00:00:00"]
    slotgemiddelden_incl_2014 = kw.calc_slotgemiddelden(df_meas=df_meas_incl_2014, df_ext=None)
    
    # check if we get 2021 as tstop if we supply up to 2020-12-31 23:50:00 and also if we supply up to 2021-01-01 00:00:00
    assert slotgemiddelden_upto_2013["wl_model_fit"].index[-1] == pd.Timestamp('2014-01-01')
    assert slotgemiddelden_incl_2014["wl_model_fit"].index[-1] == pd.Timestamp('2014-01-01')


@pytest.mark.unittest
def test_calc_slotgemiddelden_with_gap(df_meas_2010_2014):
    # TODO: setting to nan does not work, since min_count counts these as valid: https://github.com/Deltares-research/kenmerkendewaarden/issues/58
    # df_meas_withgap = df_meas.copy() # copy to prevent altering the original dataset
    # df_meas_withgap.loc["2012-01-01":"2012-08-01", "values"] = np.nan
    # df_meas_withgap.loc["2012-01-01":"2012-08-01", "qualitycode"] = 99
    
    # create dataset with a gap
    # TODO: This requires now much to little values in a year for it to fail (to support 3hr timeseries): https://github.com/Deltares-research/kenmerkendewaarden/issues/58
    df_meas_withgap = pd.concat([df_meas_2010_2014.loc[:"2012-01-01"], df_meas_2010_2014.loc["2012-12-15":]])
    slotgemiddelden_dict = kw.calc_slotgemiddelden(df_meas=df_meas_withgap, df_ext=None, only_valid=True)
    
    # TODO: value to be updated, but should contain at least one nan value
    assert slotgemiddelden_dict["wl_mean_peryear"].isnull().sum() == 1

