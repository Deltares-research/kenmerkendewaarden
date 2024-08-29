# -*- coding: utf-8 -*-

import pytest
import kenmerkendewaarden as kw
import numpy as np
import pandas as pd


@pytest.mark.unittest
def test_calc_gemiddeldgetij_outputtype(df_meas_2010):
    pred_freq = "60s"
    gemgetij_dict_raw = kw.calc_gemiddeldgetij(
        df_meas=df_meas_2010,
        df_ext=None,
        freq=pred_freq,
        nb=0,
        nf=0,
        scale_extremes=False,
        scale_period=False,
    )

    assert isinstance(gemgetij_dict_raw, dict)
    for k, v in gemgetij_dict_raw.items():
        assert isinstance(v, pd.Series)
        assert v.name == "values"
        assert isinstance(v.index, pd.TimedeltaIndex)
        assert v.index.name == "timedelta"


@pytest.mark.unittest
def test_calc_gemiddeldgetij_raw(df_meas_2010):
    pred_freq = "60s"

    gemgetij_dict_raw = kw.calc_gemiddeldgetij(
        df_meas=df_meas_2010,
        df_ext=None,
        freq=pred_freq,
        nb=0,
        nf=0,
        scale_extremes=False,
        scale_period=False,
    )

    prediction_av_raw = gemgetij_dict_raw["mean"]
    prediction_sp_raw = gemgetij_dict_raw["spring"]
    prediction_np_raw = gemgetij_dict_raw["neap"]

    assert len(prediction_av_raw) == 746
    assert np.isclose(prediction_av_raw.min(), -0.567608885905236)
    assert np.isclose(prediction_av_raw.max(), 1.1665100442336331)

    assert len(prediction_sp_raw) == 741
    assert np.isclose(prediction_sp_raw.min(), -0.5305575695904736)
    assert np.isclose(prediction_sp_raw.max(), 1.313379801706846)

    assert len(prediction_np_raw) == 755
    assert np.isclose(prediction_np_raw.min(), -0.579937620616439)
    assert np.isclose(prediction_np_raw.max(), 0.8044625899304197)


@pytest.mark.unittest
def test_calc_gemiddeldgetij_corr(df_meas_2010, df_ext_12_2010):
    pred_freq = "60s"

    gemgetij_dict_corr = kw.calc_gemiddeldgetij(
        df_meas=df_meas_2010,
        df_ext=df_ext_12_2010,
        freq=pred_freq,
        nb=2,
        nf=2,
        scale_extremes=True,
        scale_period=False,
    )

    prediction_av_corr = gemgetij_dict_corr["mean"]
    prediction_sp_corr = gemgetij_dict_corr["spring"]
    prediction_np_corr = gemgetij_dict_corr["neap"]

    assert len(prediction_av_corr) == 3726
    assert np.isclose(prediction_av_corr.min(), -0.6095833333333333)
    assert np.isclose(prediction_av_corr.max(), 1.1300000000000003)

    assert len(prediction_sp_corr) == 3701
    assert np.isclose(prediction_sp_corr.min(), -0.5700000000000001)
    assert np.isclose(prediction_sp_corr.max(), 1.3450000000000002)

    assert len(prediction_np_corr) == 3771
    assert np.isclose(prediction_np_corr.min(), -0.61)
    assert np.isclose(prediction_np_corr.max(), 0.8650000000000001)


@pytest.mark.unittest
def test_calc_gemiddeldgetij_corr_boi(df_meas_2010, df_ext_12_2010):
    pred_freq = "60s"

    gemgetij_dict_corr_boi = kw.calc_gemiddeldgetij(
        df_meas=df_meas_2010,
        df_ext=df_ext_12_2010,
        freq=pred_freq,
        nb=0,
        nf=10,
        scale_extremes=True,
        scale_period=True,
    )

    prediction_av_corr_boi = gemgetij_dict_corr_boi["mean"]
    prediction_sp_corr_boi = gemgetij_dict_corr_boi["spring"]
    prediction_np_corr_boi = gemgetij_dict_corr_boi["neap"]

    assert len(prediction_av_corr_boi) == 8196
    assert np.isclose(prediction_av_corr_boi.min(), -0.6095833333333333)
    assert np.isclose(prediction_av_corr_boi.max(), 1.1300000000000003)

    assert len(prediction_sp_corr_boi) == 8196
    assert np.isclose(prediction_sp_corr_boi.min(), -0.5700000000000001)
    assert np.isclose(prediction_sp_corr_boi.max(), 1.3450000000000002)

    assert len(prediction_np_corr_boi) == 8196
    assert np.isclose(prediction_np_corr_boi.min(), -0.61)
    assert np.isclose(prediction_np_corr_boi.max(), 0.8650000000000001)


@pytest.mark.unittest
def test_calc_gemiddeldgetij_aggers(df_meas_2010, df_ext_2010):
    pred_freq = "60s"

    with pytest.raises(ValueError) as e:
        kw.calc_gemiddeldgetij(
            df_meas=df_meas_2010,
            df_ext=df_ext_2010,
            freq=pred_freq,
            nb=0,
            nf=10,
            scale_extremes=True,
            scale_period=True,
        )
    assert "contains aggers" in str(e.value)


@pytest.mark.unittest
def test_calc_gemiddeldgetij_noext(df_meas_2010):
    pred_freq = "60s"

    with pytest.raises(ValueError) as e:
        kw.calc_gemiddeldgetij(
            df_meas=df_meas_2010,
            df_ext=None,
            freq=pred_freq,
            nb=0,
            nf=0,
            scale_extremes=True,
            scale_period=False,
        )
    assert "df_ext should be provided if scale_extremes=True" in str(e.value)


@pytest.mark.unittest
def test_calc_gemiddeldgetij_failedanalysis(df_meas_2010_2014):
    df_meas_2010_extra = df_meas_2010_2014.loc["2010":"2011-01-02"]
    pred_freq = "60s"

    with pytest.raises(ValueError) as e:
        kw.calc_gemiddeldgetij(
            df_meas=df_meas_2010_extra,
            df_ext=None,
            freq=pred_freq,
            nb=0,
            nf=0,
            scale_extremes=False,
            scale_period=False,
        )
    assert "analysis result contains nan values" in str(e.value)


@pytest.mark.unittest
def test_plot_gemiddeldgetij(df_meas_2010):
    pred_freq = "60s"

    gemgetij_dict_raw = kw.calc_gemiddeldgetij(
        df_meas=df_meas_2010,
        df_ext=None,
        freq=pred_freq,
        nb=0,
        nf=0,
        scale_extremes=False,
        scale_period=False,
    )
    kw.plot_gemiddeldgetij(
        gemgetij_dict=gemgetij_dict_raw,
        gemgetij_dict_raw=gemgetij_dict_raw,
        tick_hours=12,
    )
