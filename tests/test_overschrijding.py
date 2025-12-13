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
        "ongefilterd",
        "trendanalyse",
        "weibull",
        "gecombineerd",
        "geinterpoleerd",
    ]
    assert set(dist.keys()) == set(expected_keys)
    assert np.allclose(dist["geinterpoleerd"].index, Tfreqs_interested)
    expected_values = np.array(
        [
            1.93,
            2.09356726,
            2.2637632,
            2.44533302,
            2.70383299,
            2.91416492,
            3.13795447,
            3.45560027,
            3.71330277,
            3.98682045,
        ]
    )
    assert np.allclose(dist["geinterpoleerd"].values, expected_values)


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
    hydra_values = np.array(
        [2.473, 3.18, 4.043, 4.164, 4.358, 4.696, 5.056, 5.468, 5.865, 6.328, 7.207]
    )
    hydra_index = np.array(
        [
            1.00000000e00,
            1.00000000e-01,
            2.00000000e-02,
            1.00000000e-02,
            3.33333333e-03,
            1.00000000e-03,
            3.33333333e-04,
            1.00000000e-04,
            3.33333333e-05,
            1.00000000e-05,
            1.00000000e-06,
        ]
    )
    ser_hydra = pd.Series(hydra_values, index=hydra_index)
    ser_hydra.attrs = df_ext_12_2010_2014.attrs
    dist_hydra = {"Hydra-NL": ser_hydra}
    dist = kw.calc_overschrijding(
        df_ext=df_ext_12_2010_2014, interp_freqs=Tfreqs_interested, dist=dist_hydra
    )

    expected_keys = [
        "Hydra-NL",
        "ongefilterd",
        "trendanalyse",
        "weibull",
        "gecombineerd",
        "geinterpoleerd",
    ]
    assert set(dist.keys()) == set(expected_keys)
    assert np.allclose(dist["geinterpoleerd"].index, Tfreqs_interested)
    expected_values = np.array(
        [
            1.93,
            2.09356726,
            2.26376316,
            2.46348569,
            2.79932582,
            3.08359924,
            3.49814949,
            4.043,
            4.164,
            4.3095,
        ]
    )
    assert np.allclose(dist["geinterpoleerd"].values, expected_values)


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
        "ongefilterd",
        "trendanalyse",
        "weibull",
        "gecombineerd",
        "geinterpoleerd",
    ]
    assert set(dist.keys()) == set(expected_keys)
    assert np.allclose(dist["geinterpoleerd"].index, Tfreqs_interested)
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
    assert np.allclose(dist["geinterpoleerd"].values, expected_values)


@pytest.mark.unittest
def test_calc_overschrijding_clip_physical_break(df_ext_12_2010_2014):
    # construct fake timeseries for HARLGN around physical break 1933
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
    df_ext_12_1931_1935.attrs["station"] = "harlingen.waddenzee"

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
            2.09356726,
            2.2637632,
            2.44533302,
            2.70383299,
            2.91416492,
            3.13795447,
            3.45560027,
            3.71330277,
            3.98682045,
        ]
    )
    assert np.allclose(dist_normal["geinterpoleerd"].values, expected_values_normal)
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
    assert np.allclose(dist_clip["geinterpoleerd"].values, expected_values_clip)


@pytest.mark.unittest
def test_calc_overschrijding_rule_type_linear(df_ext_12_2010_2014):
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
        rule_type="linear",
        rule_value=0.00708459,  # same value as the automatic linear trend detection
    )

    expected_keys = [
        "ongefilterd",
        "trendanalyse",
        "weibull",
        "gecombineerd",
        "geinterpoleerd",
    ]
    assert set(dist.keys()) == set(expected_keys)
    assert np.allclose(dist["geinterpoleerd"].index, Tfreqs_interested)
    expected_values = np.array(
        [
            1.94463639,
            2.11407511,
            2.28284247,
            2.46198465,
            2.71549223,
            2.9204936,
            3.13742326,
            3.4433362,
            3.68988915,
            3.95005982,
        ]
    )
    assert np.allclose(dist["geinterpoleerd"].values, expected_values)


@pytest.mark.unittest
def test_calc_overschrijding_correct_trend(df_ext_12_2010_2014):
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
        correct_trend=True,
        min_coverage=0.9,
    )

    expected_keys = [
        "ongefilterd",
        "trendanalyse",
        "weibull",
        "gecombineerd",
        "geinterpoleerd",
    ]
    assert set(dist.keys()) == set(expected_keys)
    assert np.allclose(dist["geinterpoleerd"].index, Tfreqs_interested)
    expected_values = np.array(
        [
            1.94463639,
            2.11407512,
            2.28284249,
            2.4619847,
            2.71549229,
            2.92049369,
            3.13742337,
            3.44333634,
            3.68988932,
            3.95006002,
        ]
    )
    assert np.allclose(dist["geinterpoleerd"].values, expected_values)


@pytest.mark.unittest
def test_calc_overschrijding_aggers(df_ext_2010_2014):
    with pytest.raises(ValueError) as e:
        kw.calc_overschrijding(df_ext=df_ext_2010_2014)
    assert "contains aggers" in str(e.value)


@pytest.mark.unittest
def test_plot_overschrijding(df_ext_12_2010_2014):
    dist = kw.calc_overschrijding(df_ext=df_ext_12_2010_2014)
    kw.plot_overschrijding(dist)


@pytest.mark.unittest
def test_calc_highest_extremes_aggers(df_ext_2010_2014):
    with pytest.raises(ValueError) as e:
        kw.calc_highest_extremes(df_ext=df_ext_2010_2014)
    assert "contains aggers" in str(e.value)


@pytest.mark.unittest
def test_calc_highest_extremes(df_ext_12_2010_2014):
    df_ext_highest = kw.calc_highest_extremes(df_ext=df_ext_12_2010_2014)
    expected_times = pd.DatetimeIndex(
        [
            "2013-12-06 04:55:00+01:00",
            "2014-10-22 01:58:00+01:00",
            "2011-12-09 14:02:00+01:00",
            "2013-12-06 17:21:00+01:00",
            "2012-01-06 00:29:00+01:00",
        ]
    )
    expected_values = np.array([3.03, 2.77, 2.47, 2.44, 2.31])
    assert (df_ext_highest.index == expected_times).all()
    assert np.allclose(df_ext_highest.values, expected_values)
