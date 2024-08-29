# -*- coding: utf-8 -*-

import pytest
import kenmerkendewaarden as kw
import numpy as np
from kenmerkendewaarden.tidalindicators import (
    compute_actual_counts,
    compute_expected_counts,
)
import pandas as pd


@pytest.mark.unittest
def test_calc_tidalindicators_outputtype(df_meas_2010_2014, df_ext_12_2010):
    wl_dict = kw.calc_wltidalindicators(df_meas_2010_2014)
    wl_dict_min = kw.calc_wltidalindicators(df_meas_2010_2014, min_coverage=1)
    hwlw_dict = kw.calc_HWLWtidalindicators(df_ext_12_2010)
    hwlw_dict_min = kw.calc_HWLWtidalindicators(df_ext_12_2010, min_coverage=1)

    for indicators_dict in [wl_dict, wl_dict_min, hwlw_dict, hwlw_dict_min]:
        assert isinstance(indicators_dict, dict)
        for k, v in indicators_dict.items():
            assert isinstance(v, (pd.Series, float))
            if not isinstance(v, pd.Series):
                continue
            assert v.name == "values"
            assert isinstance(v.index, pd.PeriodIndex)
            assert v.index.name == "period"


@pytest.mark.unittest
def test_calc_HWLWtidalrange(df_ext_12_2010):
    df_ext_range = kw.calc_HWLWtidalrange(df_ext_12_2010)

    ranges = df_ext_range["tidalrange"].values
    vals_expected = np.array(
        [1.89, 1.89, 1.87, 1.87, 1.97, 1.97, 2.05, 2.05, 2.05, 2.05]
    )
    assert len(ranges) == 1411
    assert np.allclose(ranges[:10], vals_expected)


@pytest.mark.unittest
def test_calc_HWLWtidalindicators(df_ext_12_2010_2014):
    ext_stats_notimezone = kw.calc_HWLWtidalindicators(df_ext_12_2010_2014.tz_localize(None))
    ext_stats = kw.calc_HWLWtidalindicators(df_ext_12_2010_2014)
    ext_stats_min = kw.calc_HWLWtidalindicators(df_ext_12_2010_2014, min_coverage=1)
    
    expected_keys = ['HW_mean', 'LW_mean', 
                     'HW_mean_peryear', 'LW_mean_peryear', 
                     'HW_monthmax_permonth', 'LW_monthmin_permonth', 
                     'HW_monthmax_mean_peryear', 'LW_monthmax_mean_peryear', 
                     'HW_monthmin_mean_peryear', 'LW_monthmin_mean_peryear']
    for key in expected_keys:
        assert key in ext_stats.keys()
        assert (ext_stats[key] == ext_stats_notimezone[key]).all()
        
        
    assert ext_stats_notimezone['HW_monthmax_permonth'].isnull().sum() == 0
    assert ext_stats['HW_monthmax_permonth'].isnull().sum() == 0
    assert ext_stats_min['HW_monthmax_permonth'].isnull().sum() == 13


@pytest.mark.unittest
def test_calc_wltidalindicators(df_meas_2010_2014):
    wl_stats_notimezone = kw.calc_wltidalindicators(df_meas_2010_2014.tz_localize(None))
    wl_stats = kw.calc_wltidalindicators(df_meas_2010_2014)

    expected_keys = ["wl_mean_peryear", "wl_mean_permonth"]
    for key in expected_keys:
        assert key in wl_stats.keys()
        assert (wl_stats[key] == wl_stats_notimezone[key]).all()

    wl_mean_peryear_expected = np.array(
        [0.07960731, 0.08612119, 0.0853051, 0.07010864, 0.10051922]
    )
    wl_mean_permonth_expected = np.array([
        -0.00227151,  0.089313  ,  0.04443996, -0.03440509, -0.00206317,
         0.04431481,  0.03877688,  0.18267697,  0.13494907,  0.18367832,
         0.15928009,  0.11707661,  0.1087836 ,  0.02535962, -0.09558468,
        -0.0255162 , -0.00076165,  0.05667361,  0.11056228,  0.13890681,
         0.1495    ,  0.11866711,  0.07253009,  0.36550851,  0.22046819,
        -0.10208094, -0.07221102,  0.02279167,  0.02424507,  0.05409954,
         0.09238127,  0.08972894,  0.15472222,  0.16913082,  0.19712963,
         0.1639897 ,  0.05744176, -0.0134375 , -0.10685036, -0.00822222,
         0.05911066,  0.019875  ,  0.02540995,  0.07570565,  0.12776389,
         0.17321909,  0.23108102,  0.19502688,  0.06281138,  0.08588046,
        -0.00553763,  0.03490278,  0.03113575,  0.03134954,  0.10553763,
         0.16540771,  0.12535648,  0.20802195,  0.10014352,  0.25624552])
    assert np.allclose(wl_stats["wl_mean_peryear"].values, wl_mean_peryear_expected)
    assert np.allclose(wl_stats["wl_mean_permonth"].values, wl_mean_permonth_expected)


@pytest.mark.unittest
def test_calc_wltidalindicators_mincount(df_meas_2010_2014):
    # create dataset with a gap
    df_meas_withgap = (
        df_meas_2010_2014.copy()
    )  # copy to prevent altering the original dataset
    df_meas_withgap.loc["2012-01-01":"2012-01-15", "values"] = np.nan
    df_meas_withgap.loc["2012-01-01":"2012-01-15", "qualitycode"] = 99

    wltidalindicators_dict_nogap = kw.calc_wltidalindicators(
        df_meas_2010_2014, min_coverage=1
    )
    wltidalindicators_dict_withgap = kw.calc_wltidalindicators(
        df_meas_withgap, min_coverage=1
    )
    wltidalindicators_dict_withgap_lower_threshold = kw.calc_wltidalindicators(
        df_meas_withgap, min_coverage=0.95
    )

    # check if too large gap results in nan
    assert wltidalindicators_dict_nogap["wl_mean_peryear"].isnull().sum() == 0
    assert wltidalindicators_dict_withgap["wl_mean_peryear"].isnull().sum() == 1
    assert (
        wltidalindicators_dict_withgap_lower_threshold["wl_mean_peryear"].isnull().sum()
        == 0
    )


@pytest.mark.unittest
def test_compute_expected_actual_counts_samelenght(df_meas_2010_2014):
    """
    because of nan-dropping, the lenghts were not the same before
    this test makes sure this does not happen again
    https://github.com/Deltares-research/kenmerkendewaarden/issues/83
    """
    # create dataset with a gap
    df_meas_withgap = (
        df_meas_2010_2014.copy()
    )  # copy to prevent altering the original dataset
    df_meas_withgap.loc["2012", "values"] = np.nan
    df_meas_withgap.loc["2012", "qualitycode"] = 99

    # compute actual and expected counts
    actual_count_peryear = compute_actual_counts(df_meas_withgap, freq="Y")
    actual_count_permonth = compute_actual_counts(df_meas_withgap, freq="M")
    expected_count_peryear = compute_expected_counts(df_meas_withgap, freq="Y")
    expected_count_permonth = compute_expected_counts(df_meas_withgap, freq="M")

    assert len(actual_count_peryear) == len(expected_count_peryear)
    assert len(actual_count_permonth) == len(expected_count_permonth)


@pytest.mark.unittest
def test_compute_expected_counts_twotimesteps(df_meas_2010_2014):
    """
    this testcase shows that compute_expected_counts succeeds for a year with only three timesteps
    and it fails for a year with two timesteps.
    """
    # TODO: the expected count for a year with three timesteps is incorrect. How to catch this?

    # create datasets with a gap
    df_meas_withgap_success = pd.concat(
        [
            df_meas_2010_2014.loc[:"2012-01-01 00:10:00 +01:00"],
            df_meas_2010_2014.loc["2012-12-31 23:50:00 +01:00":],
        ],
        axis=0,
    )
    df_meas_withgap_fails = pd.concat(
        [
            df_meas_2010_2014.loc[:"2012-01-01 00:00:00 +01:00"],
            df_meas_2010_2014.loc["2012-12-31 23:50:00 +01:00":],
        ],
        axis=0,
    )
    assert len(df_meas_withgap_success.loc["2012"]) == 3
    assert len(df_meas_withgap_fails.loc["2012"]) == 2

    # compute expected counts
    expected_count_peryear_success = compute_expected_counts(
        df_meas_withgap_success, freq="Y"
    )
    expected_count_peryear_fails = compute_expected_counts(
        df_meas_withgap_fails, freq="Y"
    )

    count_peryear_success = np.array([52560.0, 52560.0, 52704.0, 52560.0, 52560.0])
    count_peryear_failed = np.array([52560.0, 52560.0, 2.0, 52560.0, 52560.0])

    assert np.allclose(expected_count_peryear_success.values, count_peryear_success)
    assert np.allclose(expected_count_peryear_fails.values, count_peryear_failed)


@pytest.mark.unittest
def test_calc_wltidalindicators(df_ext_12_2010_2014):
    ext_stats_notimezone = kw.calc_HWLWtidalindicators(
        df_ext_12_2010_2014.tz_localize(None)
    )
    ext_stats = kw.calc_HWLWtidalindicators(df_ext_12_2010_2014)
    expected_keys = [
        "HW_mean",
        "LW_mean",
        "HW_mean_peryear",
        "LW_mean_peryear",
        "HW_monthmax_permonth",
        "LW_monthmin_permonth",
        "HW_monthmax_mean_peryear",
        "LW_monthmin_mean_peryear",
        "HW_monthmin_mean_peryear",
        "LW_monthmax_mean_peryear",
    ]
    for key in expected_keys:
        assert key in ext_stats.keys()
        assert (ext_stats[key] == ext_stats_notimezone[key]).all()

    assert np.isclose(ext_stats["HW_mean"], 1.147976763955795)
    assert np.isclose(ext_stats["LW_mean"], -0.5967063492063492)

    hw_mean_peryear_expected = np.array(
        [1.13968839, 1.12875177, 1.13988685, 1.1415461, 1.18998584]
    )
    lw_mean_peryear_expected = np.array(
        [-0.60561702, -0.59089362, -0.59342291, -0.61334278, -0.58024113]
    )
    assert np.allclose(ext_stats["HW_mean_peryear"].values, hw_mean_peryear_expected)
    assert np.allclose(ext_stats["LW_mean_peryear"].values, lw_mean_peryear_expected)

    hw_monthmax_permonth_expected = np.array([
        1.94, 1.89, 1.86, 1.55, 1.74, 1.58, 1.54, 2.07, 2.11, 2.06, 1.9 ,
        1.75, 1.69, 1.82, 1.49, 1.39, 1.4 , 1.71, 1.72, 1.66, 1.69, 1.59,
        2.03, 2.47, 2.31, 1.63, 1.64, 1.61, 1.44, 1.51, 1.52, 1.87, 1.71,
        1.72, 1.86, 2.07, 1.87, 1.83, 1.53, 1.51, 1.62, 1.53, 1.52, 1.41,
        2.08, 1.98, 2.07, 3.03, 1.76, 1.82, 1.61, 1.73, 1.48, 1.48, 1.62,
        1.71, 1.58, 2.77, 1.6 , 1.92])
    lw_monthmin_permonth_expected = np.array([
        -1.33, -1.05, -1.05, -1.06, -1.12, -1.11, -1.07, -0.92, -0.96,
        -0.99, -1.01, -1.08, -1.16, -1.17, -1.21, -0.98, -1.1 , -0.98,
        -0.97, -0.94, -1.04, -1.22, -0.94, -1.21, -1.22, -1.32, -1.22,
        -1.04, -1.18, -0.95, -1.05, -1.  , -0.9 , -0.81, -1.03, -1.21,
        -1.11, -1.65, -1.37, -1.11, -1.11, -1.05, -0.98, -1.07, -0.88,
        -1.05, -1.15, -1.07, -1.32, -1.31, -1.21, -1.08, -1.  , -1.03,
        -1.07, -0.83, -0.98, -0.97, -0.99, -1.3 ])
    assert np.allclose(
        ext_stats["HW_monthmax_permonth"].values, hw_monthmax_permonth_expected
    )
    assert np.allclose(
        ext_stats["LW_monthmin_permonth"].values, lw_monthmin_permonth_expected
    )

    hw_monthmax_mean_peryear_expected = np.array(
        [1.8325, 1.72166667, 1.74083333, 1.83166667, 1.75666667]
    )
    lw_monthmin_mean_peryear_expected = np.array(
        [-1.0625, -1.07666667, -1.0775, -1.13333333, -1.09083333]
    )
    hw_monthmin_mean_peryear_expected = np.array(
        [0.55416667, 0.60166667, 0.5975, 0.53833333, 0.66416667]
    )
    lw_monthmax_mean_peryear_expected = np.array(
        [-0.04916667, 0.05583333, -0.02416667, 0.03, 0.00416667]
    )
    assert np.allclose(
        ext_stats["HW_monthmax_mean_peryear"].values, hw_monthmax_mean_peryear_expected
    )
    assert np.allclose(
        ext_stats["LW_monthmin_mean_peryear"].values, lw_monthmin_mean_peryear_expected
    )
    assert np.allclose(
        ext_stats["HW_monthmin_mean_peryear"].values, hw_monthmin_mean_peryear_expected
    )
    assert np.allclose(
        ext_stats["LW_monthmax_mean_peryear"].values, lw_monthmax_mean_peryear_expected
    )


@pytest.mark.unittest
def test_plot_wltidalindicators(df_meas_2010_2014, df_ext_12_2010_2014):
    wl_stats = kw.calc_wltidalindicators(df_meas_2010_2014)
    ext_stats = kw.calc_HWLWtidalindicators(df_ext_12_2010_2014)
    wl_stats.update(ext_stats)
    kw.plot_tidalindicators(wl_stats)


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
    assert np.isclose(hat, 1.6856114961274238)
    assert np.isclose(lat, -1.0395726747948162)


@pytest.mark.unittest
def test_calc_hat_lat_frommeasurements_tooshortperiod(df_meas_2010_2014):
    with pytest.raises(ValueError) as e:
        kw.calc_hat_lat_frommeasurements(df_meas_2010_2014)
    assert "please provide a timeseries of 19 years instead of 5 years" in str(e.value)


@pytest.mark.unittest
def test_calc_HWLWtidalrange_aggers_input(df_ext_2010):
    with pytest.raises(ValueError) as e:
        kw.calc_HWLWtidalrange(df_ext=df_ext_2010)
    assert "contains aggers" in str(e.value)


@pytest.mark.unittest
def test_calc_HWLWtidalindicators_aggers_input(df_ext_2010):
    with pytest.raises(ValueError) as e:
        kw.calc_HWLWtidalindicators(df_ext=df_ext_2010)
    assert "contains aggers" in str(e.value)
