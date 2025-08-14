# -*- coding: utf-8 -*-
"""
Computation of tidal indicators from waterlevel extremes for spring and neaptide
"""

import numpy as np
import pandas as pd
import logging
from kenmerkendewaarden.tidalindicators import (
    compute_actual_counts,
    compute_expected_counts,
)
from kenmerkendewaarden.utils import (
    raise_empty_df,
    raise_extremes_with_aggers,
)
from kenmerkendewaarden.havengetallen import calc_hwlw_moonculm_combi

__all__ = [
    "calc_HWLW_springneap",
]

logger = logging.getLogger(__name__)


# separate module, placing this in tidalindicators results in a circular import
def calc_HWLW_springneap(
    df_ext: pd.DataFrame, min_coverage: float = None, moonculm_offset: int = 4
):
    """
    Compute the yearly means of the extremes (high and low waters) for spring tide and
    neap tide.

    Parameters
    ----------
    df_ext : pd.DataFrame
        DataFrame with extremes (highs and lows, no aggers). The last 10 years of this
        timeseries are used to compute the havengetallen.
    min_coverage : float, optional
        The minimal required coverage (between 0 to 1) of the df_ext timeseries to
        consider the statistics to be valid. It is the factor between the actual amount
        and the expected amount of high waters in the series. Note that the expected
        amount is not an exact extimate, so min_coverage=1 will probably result in nans
        even though all extremes are present. The default is None.
    moonculm_offset : int, optional
        Offset between moonculmination and extremes. Passed on to
        `calc_HWLW_moonculm_combi`. The default is 4, which corresponds to a 2-day
        offset, which is applicable to the Dutch coast.


    Returns
    -------
    dict_hwlw_springneap : dict
        Dictionary with Dataframes with yearly means of high and low waters for spring
        and neap tide.

    """

    raise_empty_df(df_ext)
    raise_extremes_with_aggers(df_ext)

    # TODO: moonculminations cannot be computed before 1900
    # https://github.com/Deltares-research/kenmerkendewaarden/issues/184
    if df_ext.index.min().year < 1901:
        logger.warning(
            "calc_HWLW_springneap() only supports timestamps after 1900 "
            "all older data will be ignored"
        )
        df_ext = df_ext.loc["1901":]

    current_station = df_ext.attrs["station"]
    logger.info(f"computing HWLW for spring/neap tide for {current_station}")
    df_ext_culm = calc_hwlw_moonculm_combi(
        df_ext=df_ext,
        moonculm_offset=moonculm_offset,
    )

    # all HW/LW at spring/neaptide
    bool_hw = df_ext_culm["HWLWcode"] == 1
    bool_lw = df_ext_culm["HWLWcode"] == 2
    bool_spring = df_ext_culm["culm_hr"] == 0
    bool_neap = df_ext_culm["culm_hr"] == 6
    hw_spring = df_ext_culm.loc[bool_hw & bool_spring]["values"]
    lw_spring = df_ext_culm.loc[bool_lw & bool_spring]["values"]
    hw_neap = df_ext_culm.loc[bool_hw & bool_neap]["values"]
    lw_neap = df_ext_culm.loc[bool_lw & bool_neap]["values"]

    # mean HW/LW at spring/neap tide
    pi_hw_sp_y = pd.PeriodIndex(hw_spring.index, freq="Y")
    pi_lw_sp_y = pd.PeriodIndex(lw_spring.index, freq="Y")
    pi_hw_np_y = pd.PeriodIndex(hw_neap.index, freq="Y")
    pi_lw_np_y = pd.PeriodIndex(lw_neap.index, freq="Y")
    hw_spring_peryear = hw_spring.groupby(pi_hw_sp_y).mean()
    lw_spring_peryear = lw_spring.groupby(pi_lw_sp_y).mean()
    hw_neap_peryear = hw_neap.groupby(pi_hw_np_y).mean()
    lw_neap_peryear = lw_neap.groupby(pi_lw_np_y).mean()

    # replace invalids with nan (in case of too less values per month or year)
    if min_coverage is not None:
        assert 0 <= min_coverage <= 1
        # get series for coverage. Note that it is not possible to get correct expected
        # counts from spring/neap timeseries since all gaps are ignored. Therefore
        # we use the full extremes timeseries to derive the invalid years.
        ser_meas = df_ext_culm["values"]
        # count timeseries values per year/month
        wl_count_peryear = compute_actual_counts(ser_meas, freq="Y")
        # compute expected counts and multiply with min_coverage to get minimal counts
        min_count_peryear = compute_expected_counts(ser_meas, freq="Y") * min_coverage
        # get invalid years
        bool_invalid = wl_count_peryear < min_count_peryear
        years_invalid = bool_invalid.loc[bool_invalid].index

        # make sure to not have missing years in de index (happened for EEMSHVN 2021)
        years_invalid_hw_sp = years_invalid[years_invalid.isin(hw_spring_peryear.index)]
        years_invalid_lw_sp = years_invalid[years_invalid.isin(lw_spring_peryear.index)]
        years_invalid_hw_np = years_invalid[years_invalid.isin(hw_neap_peryear.index)]
        years_invalid_lw_np = years_invalid[years_invalid.isin(lw_neap_peryear.index)]
        # set all statistics that were based on too little values to nan
        hw_spring_peryear.loc[years_invalid_hw_sp] = np.nan
        lw_spring_peryear.loc[years_invalid_lw_sp] = np.nan
        hw_neap_peryear.loc[years_invalid_hw_np] = np.nan
        lw_neap_peryear.loc[years_invalid_lw_np] = np.nan

    # merge in dict
    dict_hwlw_springneap = {
        "HW_spring_mean_peryear": hw_spring_peryear,
        "LW_spring_mean_peryear": lw_spring_peryear,
        "HW_neap_mean_peryear": hw_neap_peryear,
        "LW_neap_mean_peryear": lw_neap_peryear,
    }
    return dict_hwlw_springneap
