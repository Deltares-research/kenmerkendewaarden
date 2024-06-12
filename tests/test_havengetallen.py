# -*- coding: utf-8 -*-

import pytest
import kenmerkendewaarden as kw
import numpy as np


@pytest.mark.unittest
def test_havengetallen(df_ext_12_2010):
    df_havengetallen, data_pd_hwlw = kw.calc_havengetallen(df_ext=df_ext_12_2010, return_df_ext=True)
    df_columns = ['HW_values_median', 'HW_delay_median', 'LW_values_median',
           'LW_delay_median', 'tijverschil', 'getijperiod_median',
           'duurdaling_median']
    assert set(df_havengetallen.columns) == set(df_columns)
    
    # assert the havengetallen values
    hw_values_median = df_havengetallen["HW_values_median"].values
    hw_values_median_expected = np.array([1.345, 1.31 , 1.225, 1.17 , 1.04 , 0.925, 0.865, 0.9  , 1.045,
           1.135, 1.25 , 1.35 , 1.13 ])
    assert np.allclose(hw_values_median, hw_values_median_expected)


@pytest.mark.unittest
def test_calc_HWLW_culmhr_summary_tidalcoeff(df_ext_12_2010):
    """
    this function is not used, so might be removed in the future
    still good to add test for now
    """
    df_tidalcoeff = kw.havengetallen.calc_HWLW_culmhr_summary_tidalcoeff(df_ext_12_2010)
    expected = np.array([[ 0.87 , -0.575,  1.42 ],
       [ 1.16 , -0.6  ,  1.75 ],
       [ 1.34 , -0.775,  2.07 ]])
    assert np.allclose(df_tidalcoeff, expected)


@pytest.mark.unittest
def test_plot_HWLW_pertimeclass(df_ext_12_2010):
    df_havengetallen, data_pd_hwlw = kw.calc_havengetallen(df_ext=df_ext_12_2010, return_df_ext=True)
    kw.plot_HWLW_pertimeclass(df_ext=data_pd_hwlw, df_havengetallen=df_havengetallen)


@pytest.mark.unittest
def test_plot_aardappelgrafiek(df_ext_12_2010):
    df_havengetallen = kw.calc_havengetallen(df_ext=df_ext_12_2010, return_df_ext=False)
    kw.plot_aardappelgrafiek(df_havengetallen=df_havengetallen)


@pytest.mark.unittest
def test_havengetallen_aggers_input(df_ext_2010):
    with pytest.raises(ValueError) as e:
        kw.calc_havengetallen(df_ext=df_ext_2010)
    assert "contains aggers" in str(e.value)
