# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd
import kenmerkendewaarden as kw


@pytest.mark.unittest
def test_calc_slotgemiddelden_outputtype(df_meas_2010_2014, df_ext_12_2010_2014):
    slotgemiddelden_dict_inclext = kw.calc_slotgemiddelden(
        df_meas=df_meas_2010_2014, df_ext=df_ext_12_2010_2014
    )

    assert isinstance(slotgemiddelden_dict_inclext, dict)
    for k, v in slotgemiddelden_dict_inclext.items():
        assert isinstance(v, pd.Series)
        assert v.name == "values"
        assert isinstance(v.index, pd.PeriodIndex)
        assert v.index.name == "period"


@pytest.mark.unittest
def test_fit_models(df_meas_2010_2014):
    dict_wltidalindicators_valid = kw.calc_wltidalindicators(
        df_meas_2010_2014
    )  # 24*365=8760 (hourly interval), 24/3*365=2920 (3-hourly interval)
    wl_mean_peryear_valid = dict_wltidalindicators_valid["wl_mean_peryear"]

    wl_model_fit_nodal = kw.slotgemiddelden.fit_models(
        wl_mean_peryear_valid, with_nodal=True
    )
    nodal_expected = np.array(
        [0.0141927, 0.08612119, 0.0853051, 0.07010864, 0.10051922, 0.23137634]
    )
    assert np.allclose(wl_model_fit_nodal.values, nodal_expected)

    wl_model_fit_linear = kw.slotgemiddelden.fit_models(
        wl_mean_peryear_valid, with_nodal=False
    )
    linear_expected = np.array(
        [0.07851414, 0.0813139, 0.08411366, 0.08691342, 0.08971318, 0.09251294]
    )
    assert np.allclose(wl_model_fit_linear.values, linear_expected)


@pytest.mark.unittest
def test_calc_slotgemiddelden(df_meas_2010_2014, df_ext_12_2010_2014):
    slotgemiddelden_dict_inclext = kw.calc_slotgemiddelden(
        df_meas=df_meas_2010_2014, df_ext=df_ext_12_2010_2014
    )
    slotgemiddelden_dict_noext = kw.calc_slotgemiddelden(
        df_meas=df_meas_2010_2014, df_ext=None
    )

    # assert present keys
    expected_keys_inclext = [
        "wl_mean_peryear",
        "wl_model_fit",
        "HW_mean_peryear",
        "LW_mean_peryear",
        "HW_model_fit",
        "LW_model_fit",
    ]
    expected_keys_noext = ["wl_mean_peryear", "wl_model_fit"]
    assert set(slotgemiddelden_dict_inclext.keys()) == set(expected_keys_inclext)
    assert set(slotgemiddelden_dict_noext.keys()) == set(expected_keys_noext)

    # assertion of values
    wl_mean_peryear_expected = np.array(
        [0.07960731, 0.08612119, 0.0853051, 0.07010864, 0.10051922]
    )
    hw_mean_peryear_expected = np.array(
        [1.13968839, 1.12875177, 1.13988685, 1.1415461, 1.18998584]
    )
    lw_mean_peryear_expected = np.array(
        [-0.60561702, -0.59089362, -0.59342291, -0.61334278, -0.58024113]
    )
    assert np.allclose(
        slotgemiddelden_dict_inclext["wl_mean_peryear"].values, wl_mean_peryear_expected
    )
    assert np.allclose(
        slotgemiddelden_dict_inclext["HW_mean_peryear"].values, hw_mean_peryear_expected
    )
    assert np.allclose(
        slotgemiddelden_dict_inclext["LW_mean_peryear"].values, lw_mean_peryear_expected
    )

    wl_model_fit_expected = np.array(
        [0.0141927, 0.08612119, 0.0853051, 0.07010864, 0.10051922, 0.23137634]
    )
    hw_model_fit_expected = np.array(
        [1.05295416, 1.12875177, 1.13988685, 1.1415461, 1.18998584, 1.336182]
    )
    lw_model_fit_expected = np.array(
        [-0.67420399, -0.59089362, -0.59342291, -0.61334278, -0.58024113, -0.42969074]
    )
    assert np.allclose(
        slotgemiddelden_dict_inclext["wl_model_fit"].values, wl_model_fit_expected
    )
    assert np.allclose(
        slotgemiddelden_dict_inclext["HW_model_fit"].values, hw_model_fit_expected
    )
    assert np.allclose(
        slotgemiddelden_dict_inclext["LW_model_fit"].values, lw_model_fit_expected
    )


@pytest.mark.unittest
def test_calc_slotgemiddelden_oneyear_fails(df_meas_2010_2014):
    df_meas_onevalidyear = df_meas_2010_2014.copy()
    df_meas_onevalidyear.loc["2011":] = np.nan
    with pytest.raises(ValueError) as e:
        kw.calc_slotgemiddelden(df_meas=df_meas_onevalidyear, df_ext=None)
    assert "nan-filtered timeseries has only one timestep" in str(e.value)


@pytest.mark.unittest
def test_plot_slotgemiddelden(df_meas_2010_2014, df_ext_12_2010_2014):
    slotgemiddelden_dict_inclext = kw.calc_slotgemiddelden(
        df_meas=df_meas_2010_2014, df_ext=df_ext_12_2010_2014
    )
    slotgemiddelden_dict_noext = kw.calc_slotgemiddelden(
        df_meas=df_meas_2010_2014, df_ext=None
    )
    kw.plot_slotgemiddelden(slotgemiddelden_dict_inclext)
    kw.plot_slotgemiddelden(slotgemiddelden_dict_noext)
    kw.plot_slotgemiddelden(slotgemiddelden_dict_inclext, slotgemiddelden_dict_inclext)
    kw.plot_slotgemiddelden(slotgemiddelden_dict_noext, slotgemiddelden_dict_noext)
    # assert dtypes of dictionary index, to check if plot_slotgemiddelden made a proper copy before converting the index to datetimes
    for key in slotgemiddelden_dict_inclext.keys():
        assert isinstance(slotgemiddelden_dict_inclext[key].index, pd.PeriodIndex)


@pytest.mark.unittest
def test_calc_slotgemiddelden_correct_tstop(df_meas_2010_2014):
    df_meas_upto_2013 = df_meas_2010_2014.loc[:"2013"]
    slotgemiddelden_upto_2013 = kw.calc_slotgemiddelden(
        df_meas=df_meas_upto_2013, df_ext=None
    )

    df_meas_incl_2014 = df_meas_2010_2014.loc[:"2014-01-01 00:00:00"]
    slotgemiddelden_incl_2014 = kw.calc_slotgemiddelden(
        df_meas=df_meas_incl_2014, df_ext=None
    )

    # check if we get 2021 as tstop if we supply up to 2020-12-31 23:50:00 and also if we supply up to 2021-01-01 00:00:00
    assert slotgemiddelden_upto_2013["wl_model_fit"].index[-1] == pd.Period("2014")
    assert slotgemiddelden_incl_2014["wl_model_fit"].index[-1] == pd.Period("2014")


@pytest.mark.unittest
def test_calc_slotgemiddelden_physical_break(df_meas_2010_2014, df_ext_12_2010_2014):
    # construct fake timeseries for VLIELHVN around physical break
    tstart_2010 = df_meas_2010_2014.index[0]
    tstart_1931 = pd.Timestamp(1931, 1, 1, tz=tstart_2010.tz)
    tdiff = tstart_2010 - tstart_1931

    df_meas_1931_1935 = df_meas_2010_2014.copy()
    df_meas_1931_1935.index = df_meas_1931_1935.index - tdiff
    df_meas_1931_1935.attrs["station"] = "VLIELHVN"

    df_ext_12_1931_1935 = df_ext_12_2010_2014.copy()
    df_ext_12_1931_1935.index = df_ext_12_1931_1935.index - tdiff
    df_ext_12_1931_1935.attrs["station"] = "VLIELHVN"

    # check if the timeseries do not extend over the expected slotgemiddelden value
    assert df_meas_1931_1935.index[-1] <= pd.Timestamp("1936-01-01 00:00:00 +01:00")
    assert df_ext_12_1931_1935.index[-1] <= pd.Timestamp("1936-01-01 00:00:00 +01:00")

    # compute slotgemiddelden
    slotgemiddelden_no_clip = kw.calc_slotgemiddelden(
        df_meas=df_meas_1931_1935, df_ext=df_ext_12_1931_1935, clip_physical_break=False
    )
    slotgemiddelden_with_clip = kw.calc_slotgemiddelden(
        df_meas=df_meas_1931_1935, df_ext=df_ext_12_1931_1935, clip_physical_break=True
    )

    # assert if yearly means are original lengths and modelfits are shorter with clip_physical_break=True
    assert len(slotgemiddelden_no_clip["wl_mean_peryear"]) == 5
    assert len(slotgemiddelden_with_clip["wl_mean_peryear"]) == 5
    assert len(slotgemiddelden_no_clip["wl_model_fit"]) == 6
    assert len(slotgemiddelden_with_clip["wl_model_fit"]) == 4

    assert len(slotgemiddelden_no_clip["HW_mean_peryear"]) == 5
    assert len(slotgemiddelden_with_clip["HW_mean_peryear"]) == 5
    assert len(slotgemiddelden_no_clip["HW_model_fit"]) == 6
    assert len(slotgemiddelden_with_clip["HW_model_fit"]) == 4

    # assert tstart/tstop of model fits
    assert slotgemiddelden_with_clip["wl_model_fit"].index[0] == pd.Period("1933")
    assert slotgemiddelden_with_clip["wl_model_fit"].index[-1] == pd.Period("1936")
    assert slotgemiddelden_with_clip["HW_model_fit"].index[0] == pd.Period("1933")
    assert slotgemiddelden_with_clip["HW_model_fit"].index[-1] == pd.Period("1936")


@pytest.mark.unittest
def test_calc_slotgemiddelden_with_gap(df_meas_2010_2014):
    df_meas_withgap = (
        df_meas_2010_2014.copy()
    )  # copy to prevent altering the original dataset
    df_meas_withgap.loc["2012-01-01":"2012-01-15", "values"] = np.nan
    df_meas_withgap.loc["2012-01-01":"2012-01-15", "qualitycode"] = 99

    # create dataset with a gap
    slotgemiddelden_dict_nogap = kw.calc_slotgemiddelden(
        df_meas=df_meas_2010_2014, df_ext=None, min_coverage=1
    )
    slotgemiddelden_dict_withgap = kw.calc_slotgemiddelden(
        df_meas=df_meas_withgap, df_ext=None, min_coverage=1
    )
    slotgemiddelden_dict_withgap_lower_threshold = kw.calc_slotgemiddelden(
        df_meas=df_meas_withgap, df_ext=None, min_coverage=0.95
    )

    # check if too large gap results in nan
    assert slotgemiddelden_dict_nogap["wl_mean_peryear"].isnull().sum() == 0
    assert slotgemiddelden_dict_withgap["wl_mean_peryear"].isnull().sum() == 1
    assert (
        slotgemiddelden_dict_withgap_lower_threshold["wl_mean_peryear"].isnull().sum()
        == 0
    )
