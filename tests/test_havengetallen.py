# -*- coding: utf-8 -*-

import pytest
import kenmerkendewaarden as kw
import hatyan
import numpy as np


@pytest.mark.timeout(60) # useful in case of ddl failure
@pytest.mark.unittest
def test_havengetallen(dir_meas_extremes):
    df_ext = kw.read_measurements(dir_output=dir_meas_extremes, station="HOEKVHLD", extremes=True)
    df_ext_12 = hatyan.calc_HWLW12345to12(df_ext)
    
    df_havengetallen, data_pd_hwlw = kw.havengetallen(df_ext=df_ext_12, return_df_ext=True)
    
    df_columns = ['HW_values_median', 'HW_delay_median', 'LW_values_median',
           'LW_delay_median', 'tijverschil', 'getijperiod_median',
           'duurdaling_median']
    assert set(df_havengetallen.columns) == set(df_columns)
    
    # assert the havengetallen values, this might change if ddl data is updated
    hw_values_median = df_havengetallen["HW_values_median"].values
    hw_values_median_expected = np.array([1.345, 1.31 , 1.225, 1.17 , 1.04 , 0.925, 0.865, 0.9  , 1.045,
           1.135, 1.25 , 1.35 , 1.13 ])
    assert np.allclose(hw_values_median, hw_values_median_expected)
    
    # assert the enriched df_ext length
    assert len(data_pd_hwlw) == len(df_ext_12)
