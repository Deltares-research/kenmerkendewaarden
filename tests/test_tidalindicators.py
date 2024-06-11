# -*- coding: utf-8 -*-

import pytest
import hatyan
import kenmerkendewaarden as kw
import numpy as np
import pandas as pd


@pytest.fixture(scope="session")
def components():
    components = pd.DataFrame({"A": [1, 0.5, 0.2],
                         "phi_deg": [10,15,20]}, 
                        index=["M2","M4","S2"])
    components.attrs["nodalfactors"] = True
    components.attrs["fu_alltimes"] = True
    components.attrs["xfac"] = False
    components.attrs["source"] = "schureman"
    components.attrs["tzone"] = None # TODO: required but this is a bug: https://github.com/Deltares/hatyan/issues/317
    return components


@pytest.fixture(scope="session")
def prediction(components):
    dtindex = pd.date_range("2020-01-01","2024-01-01", freq="10min")
    prediction = hatyan.prediction(components, times=dtindex)
    return prediction


@pytest.fixture(scope="session")
def prediction_extremes(prediction):
    prediction_extremes = hatyan.calc_HWLW(prediction)
    return prediction_extremes


@pytest.mark.unittest
def test_calc_HWLWtidalrange(df_ext_12_2010):
    ts_ext_range = kw.calc_HWLWtidalrange(df_ext_12_2010)
    
    ranges = ts_ext_range["tidalrange"].values
    vals_expected = np.array([1.89, 1.89, 1.87, 1.87, 1.97, 1.97, 2.05, 2.05, 2.05, 2.05])
    assert len(ranges) == 1411
    assert np.allclose(ranges[:10], vals_expected)


@pytest.mark.unittest
def test_calc_HWLWtidalindicators(prediction):
    wl_stats = kw.calc_wltidalindicators(prediction)
    wl_stats_tzone = kw.calc_wltidalindicators(prediction.tz_localize("UTC+01:00"))
    
    expected_keys = ['wl_mean_peryear', 'wl_mean_permonth']
    for key in expected_keys:
        assert key in wl_stats.keys()
        assert (wl_stats[key] == wl_stats_tzone[key]).all()
    
    wl_mean_peryear_expected = np.array([-1.74201105e-04,  4.19964454e-04, -6.46875301e-05, -2.36333457e-04,
           -5.63995348e-01])
    wl_mean_permonth_expected = np.array([ 9.62018771e-04, -3.31675667e-04,  8.53396688e-04,  3.16123043e-04,
            1.13262094e-03,  3.56346658e-04,  4.53580440e-04, -1.14071322e-03,
           -7.03490275e-04, -2.39556946e-03, -6.59549426e-04, -9.43460292e-04,
            6.51602796e-04, -7.09547154e-04,  5.96494198e-04,  3.99839817e-04,
            1.22594977e-03,  3.34308253e-04,  8.58938332e-04,  1.00068126e-03,
            3.79093532e-04,  9.16718937e-04,  8.28879097e-05, -8.22305918e-04,
           -2.17133133e-03,  2.47645095e-03, -2.13131823e-03, -7.85535462e-04,
           -1.76888980e-03, -2.41317388e-04,  3.29503094e-04,  1.19024147e-03,
            3.92011136e-04,  9.62169025e-04,  2.72527819e-04,  9.41814213e-04,
            1.09355549e-03, -1.26423276e-03,  1.09647701e-03,  2.82808337e-04,
            6.38054338e-05, -3.27907803e-04, -1.91306815e-03, -2.22220727e-03,
           -5.20148544e-04, -4.85500468e-04,  1.82340166e-04,  1.09674573e-03,
           -5.63995348e-01])
    assert np.allclose(wl_stats['wl_mean_peryear'].values, wl_mean_peryear_expected)
    assert np.allclose(wl_stats['wl_mean_permonth'].values, wl_mean_permonth_expected)


@pytest.mark.unittest
def test_calc_wltidalindicators(prediction_extremes):
    ext_stats = kw.calc_HWLWtidalindicators(prediction_extremes)
    ext_stats_tzone = kw.calc_HWLWtidalindicators(prediction_extremes.tz_localize("UTC+01:00"))
    expected_keys = ['HW_mean', 'LW_mean', 
                     'HW_mean_peryear', 'LW_mean_peryear', 
                     'HW_monthmax_permonth', 'LW_monthmin_permonth', 
                     'HW_monthmax_mean_peryear', 'LW_monthmin_mean_peryear', 
                     'HW_monthmin_mean_peryear', 'LW_monthmax_mean_peryear']
    for key in expected_keys:
        assert key in ext_stats.keys()
        assert (ext_stats[key] == ext_stats_tzone[key]).all()
    
    assert np.isclose(ext_stats['HW_mean'], 1.4680606974545858)
    assert np.isclose(ext_stats['LW_mean'],-0.8534702751276967)
    
    hw_mean_peryear_expected = np.array([1.50039216, 1.47449849, 1.45577267, 1.44152448])
    lw_mean_peryear_expected = np.array([-0.86997608, -0.85709345, -0.84652208, -0.84024267])
    assert np.allclose(ext_stats['HW_mean_peryear'].values, hw_mean_peryear_expected)
    assert np.allclose(ext_stats['LW_mean_peryear'].values, lw_mean_peryear_expected)

    hw_monthmax_permonth_expected = np.array([1.70887357, 1.70693041, 1.70448206, 1.70152976, 1.69961557,
       1.69838487, 1.69720287, 1.69551568, 1.69332416, 1.69062942,
       1.68743283, 1.68434737, 1.6826634 , 1.6817192 , 1.68041369,
       1.67860467, 1.67629303, 1.67347995, 1.67016685, 1.6691834 ,
       1.66816725, 1.66664809, 1.66471394, 1.66280594, 1.66131054,
       1.65951663, 1.65722138, 1.6549959 , 1.65389291, 1.65228867,
       1.65079986, 1.64923816, 1.64786607, 1.64599259, 1.6437117 ,
       1.6430369 , 1.64242114, 1.64186457, 1.6408059 , 1.63924574,
       1.63718491, 1.63493972, 1.63476655, 1.63452784, 1.63386586,
       1.63324442, 1.63212122, 1.63049693])
    lw_monthmin_permonth_expected = np.array([-0.99242254, -0.99061194, -0.9894141 , -0.98883749, -0.98797913,
           -0.98711698, -0.98601125, -0.98473314, -0.98292192, -0.98120697,
           -0.9807385 , -0.97999275, -0.97917249, -0.97818554, -0.97691195,
           -0.97590032, -0.97466971, -0.97314856, -0.972554  , -0.97224475,
           -0.97166419, -0.97080684, -0.96966737, -0.96824058, -0.96617301,
           -0.9655175 , -0.96534086, -0.96489778, -0.96418269, -0.96319016,
           -0.96191492, -0.96035182, -0.95979141, -0.95945245, -0.95913155,
           -0.95857315, -0.95774217, -0.95682775, -0.95604764, -0.95499088,
           -0.95463796, -0.95469814, -0.95457444, -0.95353554, -0.95271051,
           -0.95211143, -0.95124064, -0.95116319])
    assert np.allclose(ext_stats['HW_monthmax_permonth'].values, hw_monthmax_permonth_expected)
    assert np.allclose(ext_stats['LW_monthmin_permonth'].values, lw_monthmin_permonth_expected)
    
    hw_monthmax_mean_peryear_expected = np.array([1.69735572, 1.67290495, 1.65165594, 1.6362904 ])
    lw_monthmin_mean_peryear_expected = np.array([-0.98599889, -0.97359719, -0.96237644, -0.95419002])
    hw_monthmin_mean_peryear_expected = np.array([1.29661114, 1.27231033, 1.2516437 , 1.23561901])
    lw_monthmax_mean_peryear_expected = np.array([-0.67207461, -0.6593444 , -0.64799771, -0.6390193 ])
    assert np.allclose(ext_stats['HW_monthmax_mean_peryear'].values, hw_monthmax_mean_peryear_expected)
    assert np.allclose(ext_stats['LW_monthmin_mean_peryear'].values, lw_monthmin_mean_peryear_expected)
    assert np.allclose(ext_stats['HW_monthmin_mean_peryear'].values, hw_monthmin_mean_peryear_expected)
    assert np.allclose(ext_stats['LW_monthmax_mean_peryear'].values, lw_monthmax_mean_peryear_expected)


@pytest.mark.unittest
def test_calc_hat_lat_fromcomponents(components):
    hat, lat = kw.calc_hat_lat_fromcomponents(components)
    assert np.isclose(hat, 1.7756676846720225)
    assert np.isclose(lat, -1.027867748556516)
