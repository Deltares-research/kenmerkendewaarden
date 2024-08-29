# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 11:29:41 2024

@author: veenstra
"""

import pytest
import kenmerkendewaarden as kw
import numpy as np
import pandas as pd


@pytest.mark.unittest
def test_calc_overschrijding_outputtype(df_ext_12_2010_2014):
    Tfreqs_interested = [
        5,
        2,
        1,
        1 / 2,
        1 / 5,
        1 / 10,
        1 / 20,
        1 / 50,
        1 / 100,
        1 / 200,
    ]
    dist = kw.calc_overschrijding(
        df_ext=df_ext_12_2010_2014, interp_freqs=Tfreqs_interested
    )

    assert isinstance(dist, dict)
    for k, v in dist.items():
        assert isinstance(v, pd.Series)
        assert v.name == "values"
        assert isinstance(v.index, pd.Index)
        assert str(v.index.dtype) == "float64"
        assert v.index.name == "frequency"


@pytest.mark.unittest
def test_calc_overschrijding(df_ext_12_2010_2014):
    Tfreqs_interested = [
        5,
        2,
        1,
        1 / 2,
        1 / 5,
        1 / 10,
        1 / 20,
        1 / 50,
        1 / 100,
        1 / 200,
    ]
    dist = kw.calc_overschrijding(
        df_ext=df_ext_12_2010_2014, interp_freqs=Tfreqs_interested
    )

    expected_keys = [
        "Ongefilterd",
        "Trendanalyse",
        "Weibull",
        "Gecombineerd",
        "Geinterpoleerd",
    ]
    assert set(dist.keys()) == set(expected_keys)
    assert np.allclose(dist["Geinterpoleerd"].index, Tfreqs_interested)
    expected_values = np.array(
        [
            1.93,
            2.09327434,
            2.26311592,
            2.44480348,
            2.70434509,
            2.91627091,
            3.14247786,
            3.46480369,
            3.72735283,
            4.00701551,
        ]
    )
    assert np.allclose(dist["Geinterpoleerd"].values, expected_values)


@pytest.mark.unittest
def test_calc_overschrijding_with_hydra(df_ext_12_2010_2014):
    Tfreqs_interested = [
        5,
        2,
        1,
        1 / 2,
        1 / 5,
        1 / 10,
        1 / 20,
        1 / 50,
        1 / 100,
        1 / 200,
    ]
    hydra_values = np.array([2.473, 3.18 , 4.043, 4.164, 4.358, 4.696, 5.056, 5.468, 5.865,
           6.328, 7.207])
    hydra_index = np.array([1.00000000e+00, 1.00000000e-01, 2.00000000e-02, 1.00000000e-02,
           3.33333333e-03, 1.00000000e-03, 3.33333333e-04, 1.00000000e-04,
           3.33333333e-05, 1.00000000e-05, 1.00000000e-06])
    ser_hydra = pd.Series(hydra_values, index=hydra_index)
    ser_hydra.attrs = df_ext_12_2010_2014.attrs
    dist_hydra = {"Hydra-NL": ser_hydra}
    dist = kw.calc_overschrijding(
        df_ext=df_ext_12_2010_2014, interp_freqs=Tfreqs_interested, dist=dist_hydra
    )

    expected_keys = [
        "Hydra-NL",
        "Ongefilterd",
        "Trendanalyse",
        "Weibull",
        "Gecombineerd",
        "Geinterpoleerd",
    ]
    assert set(dist.keys()) == set(expected_keys)
    assert np.allclose(dist["Geinterpoleerd"].index, Tfreqs_interested)
    expected_values = np.array(
        [
            1.93,
            2.09327434,
            2.26311587,
            2.46299612,
            2.79965222,
            3.08436295,
            3.4987347,
            4.043,
            4.164,
            4.3095,
        ]
    )
    assert np.allclose(dist["Geinterpoleerd"].values, expected_values)


@pytest.mark.unittest
def test_calc_overschrijding_rule_type_break(df_ext_12_2010_2014):
    Tfreqs_interested = [
        5,
        2,
        1,
        1 / 2,
        1 / 5,
        1 / 10,
        1 / 20,
        1 / 50,
        1 / 100,
        1 / 200,
    ]
    dist = kw.calc_overschrijding(
        df_ext=df_ext_12_2010_2014,
        interp_freqs=Tfreqs_interested,
        rule_type="break",
        rule_value="2012",
    )

    expected_keys = [
        "Ongefilterd",
        "Trendanalyse",
        "Weibull",
        "Gecombineerd",
        "Geinterpoleerd",
    ]
    assert set(dist.keys()) == set(expected_keys)
    assert np.allclose(dist["Geinterpoleerd"].index, Tfreqs_interested)
    expected_values = np.array(
        [
            1.93,
            2.11353631,
            2.37418889,
            2.61405772,
            2.90685818,
            3.11376817,
            3.31039598,
            3.55705172,
            3.73504553,
            3.90661043,
        ]
    )
    assert np.allclose(dist["Geinterpoleerd"].values, expected_values)


@pytest.mark.unittest
def test_calc_overschrijding_clip_physical_break(df_ext_12_2010_2014):
    # construct fake timeseries for VLIELHVN around physical break
    tstart_2010 = df_ext_12_2010_2014.index[0]
    tstart_1931 = pd.Timestamp(
        1931,
        tstart_2010.month,
        tstart_2010.day,
        tstart_2010.hour,
        tstart_2010.minute,
        tstart_2010.second,
        tz=tstart_2010.tz,
    )
    tdiff = tstart_2010 - tstart_1931

    df_ext_12_1931_1935 = df_ext_12_2010_2014.copy()
    df_ext_12_1931_1935.index = df_ext_12_1931_1935.index - tdiff
    df_ext_12_1931_1935.attrs["station"] = "VLIELHVN"

    Tfreqs_interested = [
        5,
        2,
        1,
        1 / 2,
        1 / 5,
        1 / 10,
        1 / 20,
        1 / 50,
        1 / 100,
        1 / 200,
    ]
    dist_normal = kw.calc_overschrijding(
        df_ext=df_ext_12_1931_1935,
        interp_freqs=Tfreqs_interested,
        clip_physical_break=False,
    )
    dist_clip = kw.calc_overschrijding(
        df_ext=df_ext_12_1931_1935,
        interp_freqs=Tfreqs_interested,
        clip_physical_break=True,
    )

    expected_values_normal = np.array(
        [
            1.93,
            2.09327434,
            2.26311592,
            2.44480348,
            2.70434509,
            2.91627091,
            3.14247786,
            3.46480369,
            3.72735283,
            4.00701551,
        ]
    )
    assert np.allclose(
        dist_normal["Geinterpoleerd"].values, expected_values_normal
    )
    expected_values_clip = np.array(
        [
            1.93,
            2.11390828,
            2.37452851,
            2.61437251,
            2.90714768,
            3.11404269,
            3.3106574,
            3.5572988,
            3.73528341,
            3.90683996,
        ]
    )
    assert np.allclose(
        dist_clip["Geinterpoleerd"].values, expected_values_clip
    )


@pytest.mark.unittest
def test_calc_overschrijding_aggers(df_ext_2010_2014):
    with pytest.raises(ValueError) as e:
        kw.calc_overschrijding(df_ext=df_ext_2010_2014)
    assert "contains aggers" in str(e.value)


@pytest.mark.unittest
def test_plot_overschrijding(df_ext_12_2010_2014):
    dist = kw.calc_overschrijding(df_ext=df_ext_12_2010_2014)
    kw.plot_overschrijding(dist)
