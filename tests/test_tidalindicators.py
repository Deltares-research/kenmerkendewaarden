# -*- coding: utf-8 -*-

import os
import pytest
import hatyan
import kenmerkendewaarden as kw
import numpy as np

dir_tests = os.path.dirname(__file__) #F9 doesnt work, only F5 (F5 also only method to reload external definition scripts)
dir_testdata = os.path.join(dir_tests,'testdata')


@pytest.mark.unittest
def test_calc_HWLWtidalrange():
    file_ext = os.path.join(dir_testdata, "VLISSGN_ext.txt")
    ts_ext = hatyan.read_dia(file_ext)
    ts_ext_range = kw.calc_HWLWtidalrange(ts_ext)
    
    ranges = ts_ext_range["tidalrange"].values
    vals_expected = np.array([4.  , 4.  , 4.1 , 4.1 , 3.77, 3.77, 3.89, 3.89, 3.5 , 3.5 ])
    assert len(ranges) == 1411
    assert np.allclose(ranges[-10:], vals_expected)
