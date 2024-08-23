# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 11:33:11 2024

@author: veenstra
"""
import pytest
from kenmerkendewaarden.utils import raise_extremes_with_aggers


@pytest.mark.unittest
def test_raise_extremes_with_aggers_raise_12345df(df_ext):
    with pytest.raises(ValueError) as e:
        raise_extremes_with_aggers(df_ext)
    assert ("df_ext should only contain extremes (HWLWcode 1/2), "
            "but it also contains aggers (HWLWcode 3/4/5") in str(e.value)


@pytest.mark.unittest
def test_raise_extremes_with_aggers_pass_12df(df_ext_12_2010):
    raise_extremes_with_aggers(df_ext_12_2010)


@pytest.mark.unittest
def test_raise_extremes_with_aggers_emptydf():
    import pandas as pd
    time_index = pd.DatetimeIndex([], dtype='datetime64[ns, Etc/GMT-1]', name='time', freq=None)
    df_ext = pd.DataFrame({"HWLWcode":[]},index=time_index)
    df_ext.attrs["station"] = "dummy"
    raise_extremes_with_aggers(df_ext=df_ext)
