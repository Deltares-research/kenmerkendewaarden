# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:47:43 2024

@author: veenstra
"""
import os
import pytest
import hatyan
import kenmerkendewaarden as kw
import numpy as np

# TODO: retrieve testdata from internet instead of hardcoded local
dir_testdata = r"c:\DATA\hatyan\tests\data_unitsystemtests"

@pytest.mark.unittest
def test_calc_HWLWtidalrange():
    file_ext = os.path.join(dir_testdata, "VLISSGN_ext.txt")
    ts_ext = hatyan.readts_dia(file_ext)
    ts_ext_range = kw.calc_HWLWtidalrange(ts_ext)
    
    ranges = ts_ext_range["tidalrange"].values
    vals_expected = np.array([4.  , 4.  , 4.1 , 4.1 , 3.77, 3.77, 3.89, 3.89, 3.5 , 3.5 ])
    assert len(ranges) == 1411
    assert np.allclose(ranges[-10:], vals_expected)
