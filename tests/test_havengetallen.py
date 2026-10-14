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


@pytest.mark.parametrize("correct_slotgemiddelden", [False, True])
@pytest.mark.unittest
def test_calc_havengetallen(correct_slotgemiddelden, df_ext_12_2010_2014):
    df_havengetallen, data_pd_hwlw = kw.calc_havengetallen(
        df_ext=df_ext_12_2010_2014,
        return_df_ext=True,
        correct_slotgemiddelden=correct_slotgemiddelden,
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
    assert len(data_pd_hwlw) == len(df_ext_12_2010_2014)

    # assert the havengetallen values
    hw_values_median = df_havengetallen["HW_values_median"].values
    if correct_slotgemiddelden:
        hw_values_median_expected = np.array(
            [
                1.35532189,
                1.33532189,
                1.27532189,
                1.21532189,
                1.11532189,
                1.00532189,
                0.96532189,
                0.99532189,
                1.10532189,
                1.19532189,
                1.27532189,
                1.34532189,
                1.18198856,
            ]
        )
    else:
        hw_values_median_expected = np.array(
            [
                1.31,
                1.29,
                1.23,
                1.17,
                1.07,
                0.96,
                0.92,
                0.95,
                1.06,
                1.15,
                1.23,
                1.3,
                1.13666667,
            ]
        )

    assert np.allclose(hw_values_median, hw_values_median_expected)

    # test time delays
    hw_delay_median = df_havengetallen["HW_delay_median"].values.astype(float)
    hw_delay_median_expected = np.array(
        [
            5.662e12,
            4.762e12,
            3.842e12,
            3.147e12,
            2.997e12,
            3.635e12,
            5.556e12,
            7.576e12,
            8.151e12,
            7.967e12,
            7.378e12,
            6.577e12,
            5.604e12,
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
