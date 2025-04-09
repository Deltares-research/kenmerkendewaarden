# -*- coding: utf-8 -*-

import pytest
import kenmerkendewaarden as kw
import numpy as np
import pandas as pd


@pytest.mark.unittest
def test_calc_havengetallen_outputtype(df_ext_12_2010):
    df_havengetallen = kw.calc_havengetallen(df_ext=df_ext_12_2010)

    assert isinstance(df_havengetallen, pd.DataFrame)
    for k, v in df_havengetallen.items():
        assert isinstance(v, pd.Series)
        assert v.name == k
        assert isinstance(v.index, pd.Index)
        assert str(v.index.dtype) == "object"
        assert v.index.name == "culm_hr"


@pytest.mark.unittest
def test_calc_havengetallen(df_ext_12_2010):
    df_havengetallen, data_pd_hwlw = kw.calc_havengetallen(
        df_ext=df_ext_12_2010, return_df_ext=True
    )

    # check if all expected columns are present
    df_columns = [
        "HW_values_median",
        "HW_delay_median",
        "LW_values_median",
        "LW_delay_median",
        "tijverschil",
        "getijperiod_median",
        "duurdaling_median",
    ]
    assert set(df_havengetallen.columns) == set(df_columns)

    # check if mean row is present
    assert len(df_havengetallen.index) == 13
    assert "mean" in df_havengetallen.index

    # check if extremes dataframe length has not changed
    assert len(data_pd_hwlw) == len(df_ext_12_2010)

    # assert the havengetallen values
    hw_values_median = df_havengetallen["HW_values_median"].values
    hw_values_median_expected = np.array(
        [
            1.345,
            1.31,
            1.225,
            1.17,
            1.04,
            0.925,
            0.865,
            0.9,
            1.045,
            1.135,
            1.25,
            1.35,
            1.13,
        ]
    )
    assert np.allclose(hw_values_median, hw_values_median_expected)

    # test time delays
    hw_delay_median = df_havengetallen["HW_delay_median"].values.astype(float)
    hw_delay_median_expected = np.array(
        [
            5697000000000,
            4763000000000,
            3792000000000,
            3230000000000,
            2985000000000,
            3729000000000,
            5722000000000,
            7830000000000,
            8335000000000,
            7995000000000,
            7501000000000,
            6628000000000,
            5684000000000,
        ]
    )  # nanoseconds representation
    assert np.allclose(hw_delay_median, hw_delay_median_expected)

    # test time rounding to seconds
    for (
        colname
    ) in df_havengetallen.columns:  # round timedelta to make outputformat nicer
        if df_havengetallen[colname].dtype == "timedelta64[ns]":
            assert (df_havengetallen[colname].dt.nanoseconds == 0).all()


@pytest.mark.unittest
def test_calc_havengetallen_moonculm_offset(df_ext_12_2010_2014):
    df_havengetallen = kw.calc_havengetallen(df_ext_12_2010_2014, moonculm_offset=0)

    # assert the havengetallen values
    hw_values_median = df_havengetallen["HW_values_median"].values
    hw_values_median_expected = np.array(
        [
            1.25,
            1.31,
            1.3,
            1.285,
            1.22,
            1.11,
            1.04,
            0.94,
            0.92,
            0.98,
            1.09,
            1.19,
            1.13625,
        ]
    )
    assert np.allclose(hw_values_median, hw_values_median_expected)

    # test time delays
    hw_delay_median = df_havengetallen["HW_delay_median"].values.astype(float)
    hw_delay_median_expected = np.array(
        [
            7024000000000,
            6156000000000,
            5274000000000,
            4410000000000,
            3586000000000,
            3138000000000,
            3146000000000,
            4241000000000,
            6406000000000,
            7936000000000,
            8170000000000,
            7799000000000,
            5607000000000,
        ]
    )  # nanoseconds representation
    assert np.allclose(hw_delay_median, hw_delay_median_expected)


@pytest.mark.unittest
def test_calc_havengetallen_toolittle_data(df_ext_12_2010_2014):
    df_ext = df_ext_12_2010_2014.copy()  # copy to prevent altering the original dataset
    # set 25% of one year to nan, so 75% of valid data remains
    df_ext.loc["2013-01":"2013-03", "values"] = np.nan
    with pytest.raises(ValueError) as e:
        # require a minimal coverage of 95% for all years, so this will fail
        kw.calc_havengetallen(df_ext, min_coverage=0.95)
    assert "coverage of some years is lower than min_coverage" in str(e.value)


@pytest.mark.unittest
def test_calc_HWLW_springneap(df_ext_12_2010_2014):
    dict_spnp = kw.calc_HWLW_springneap(df_ext_12_2010_2014)
    df_columns = ['HW_spring_mean_peryear', 'LW_spring_mean_peryear', 
                  'HW_neap_mean_peryear', 'LW_neap_mean_peryear']
    assert set(dict_spnp.keys()) == set(df_columns)
    
    for key in df_columns:
        years_act = dict_spnp[key].index.year.tolist()
        years_exp = [2010, 2011, 2012, 2013, 2014]
        assert years_act == years_exp
    
    vals_act = dict_spnp['HW_spring_mean_peryear'].values
    vals_exp = np.array([1.33551724, 1.28111111, 1.29563636, 1.33185185, 1.37745455])
    assert np.allclose(vals_act, vals_exp)
    
    vals_act = dict_spnp['LW_spring_mean_peryear'].values
    vals_exp = np.array([-0.59724138, -0.6212963, -0.61872727, -0.60407407, -0.60145455])
    assert np.allclose(vals_act, vals_exp)
    
    vals_act = dict_spnp['HW_neap_mean_peryear'].values
    vals_exp = np.array([0.83887097, 0.95854839, 0.86225806, 0.87903226, 0.9696875])
    assert np.allclose(vals_act, vals_exp)
    
    vals_act = dict_spnp['LW_neap_mean_peryear'].values
    vals_exp = np.array([-0.62919355, -0.46758065, -0.5633871 , -0.60193548, -0.51421875])
    assert np.allclose(vals_act, vals_exp)


@pytest.mark.unittest
def test_calc_HWLW_springneap_min_coverage(df_ext_12_2010_2014):
    # create df_ext with a large gap, deliberately including 2011-01-01 to 2011-01-10
    # since this causes the indexes between the dicts to be different (which is
    # accounted for in calc_HWLW_springneap since years_invalid is filtered per df)
    pre2011 = df_ext_12_2010_2014.index < pd.Timestamp("2011-01-10 00:00:00 +01:00")
    post2011 = df_ext_12_2010_2014.index > pd.Timestamp("2012-01-01 00:00:00 +01:00")
    df_ext = df_ext_12_2010_2014.loc[pre2011 | post2011]
    
    dict_spnp = kw.calc_HWLW_springneap(df_ext, min_coverage=0.9)
    
    years_act = dict_spnp['HW_spring_mean_peryear'].index.year.tolist()
    assert years_act == [2010, 2011, 2012, 2013, 2014]
    years_act = dict_spnp['LW_spring_mean_peryear'].index.year.tolist()
    assert years_act == [2010, 2011, 2012, 2013, 2014]
    years_act = dict_spnp['HW_neap_mean_peryear'].index.year.tolist()
    assert years_act == [2010, 2012, 2013, 2014]
    years_act = dict_spnp['LW_neap_mean_peryear'].index.year.tolist()
    assert years_act == [2010, 2012, 2013, 2014]
    
    vals_act = dict_spnp['HW_spring_mean_peryear'].values
    vals_exp = np.array([1.33551724, np.nan, 1.29563636, 1.33185185, 1.37745455])
    assert np.allclose(vals_act, vals_exp, equal_nan=True)
    
    vals_act = dict_spnp['LW_spring_mean_peryear'].values
    vals_exp = np.array([-0.59724138, np.nan, -0.61872727, -0.60407407, -0.60145455])
    assert np.allclose(vals_act, vals_exp, equal_nan=True)
    
    vals_act = dict_spnp['HW_neap_mean_peryear'].values
    vals_exp = np.array([0.83887097, 0.86225806, 0.87903226, 0.9696875])
    assert np.allclose(vals_act, vals_exp)
    
    vals_act = dict_spnp['LW_neap_mean_peryear'].values
    vals_exp = np.array([-0.62919355, -0.5633871 , -0.60193548, -0.51421875])
    assert np.allclose(vals_act, vals_exp)


@pytest.mark.unittest
def test_plot_HWLW_pertimeclass(df_ext_12_2010):
    df_havengetallen, data_pd_hwlw = kw.calc_havengetallen(
        df_ext=df_ext_12_2010, return_df_ext=True
    )
    kw.plot_HWLW_pertimeclass(df_ext=data_pd_hwlw, df_havengetallen=df_havengetallen)


@pytest.mark.unittest
def test_plot_aardappelgrafiek(df_ext_12_2010):
    df_havengetallen = kw.calc_havengetallen(df_ext=df_ext_12_2010, return_df_ext=False)
    kw.plot_aardappelgrafiek(df_havengetallen=df_havengetallen)


@pytest.mark.unittest
def test_calc_havengetallen_aggers_input(df_ext_2010):
    with pytest.raises(ValueError) as e:
        kw.calc_havengetallen(df_ext=df_ext_2010)
    assert "contains aggers" in str(e.value)
