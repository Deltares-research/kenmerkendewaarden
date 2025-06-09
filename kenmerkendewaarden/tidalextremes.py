# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 20:15:54 2025

@author: veenstra
"""

import pandas as pd
import datetime as dt
import hatyan
import logging
from kenmerkendewaarden.utils import crop_timeseries_last_nyears
from kenmerkendewaarden.tidalindicators import calc_getijcomponenten
from kenmerkendewaarden.slotgemiddelden import calc_slotgemiddelden

__all__ = [
    "calc_hat_lat_fromcomponents",
    "calc_hat_lat_frommeasurements",
]

logger = logging.getLogger(__name__)


def calc_hat_lat_fromcomponents(comp: pd.DataFrame) -> tuple:
    """
    Derive highest and lowest astronomical tide (HAT/LAT) from a component set.
    The component set is used to make a tidal prediction for an arbitrary period of 19 years with a 10 minute interval.
    The max/min values of the predictions of all years are the HAT/LAT values.
    The HAT/LAT is very dependent on the A0 of the component set. Therefore, the HAT/LAT values are
    relevant for the same year as the slotgemiddelde that is used to replace A0 in the component set.
    For instance, if the slotgemiddelde is valid for 2021.0, HAT and LAT are also relevant for that year.
    It is important to use the same tidal prediction settings as used to derive the tidal components.

    Parameters
    ----------
    comp : pd.DataFrame
        DataFrame with amplitudes and phases for a list of components.

    Returns
    -------
    tuple
        hat and lat values.

    """

    min_vallist_allyears = pd.Series(dtype=float)
    max_vallist_allyears = pd.Series(dtype=float)
    tz = comp.attrs["tzone"] # to avoid tzone warning for all years
    logger.info("generating prediction for 19 years")
    # 19 arbitrary consequtive years to capture entire nodal cycle
    for year in range(2020, 2039):  
        times_pred_all = pd.date_range(
            start=dt.datetime(year, 1, 1), end=dt.datetime(year + 1, 1, 1), freq="10min",
            tz=tz,
        )
        ts_prediction = hatyan.prediction(comp=comp, times=times_pred_all)

        min_vallist_allyears.loc[year] = ts_prediction["values"].min()
        max_vallist_allyears.loc[year] = ts_prediction["values"].max()

    logger.info("deriving hat/lat")
    hat = max_vallist_allyears.max()
    lat = min_vallist_allyears.min()
    return hat, lat


def predict_19y_peryear(comp, yearmax=2039):
    list_pred = []
    # frequency of 1min is better in theory, but 10min is faster and hat/lat values
    # differ only 2mm for HOEKVHLD
    freq = "10min"
    # 19 arbitrary consequtive years to capture entire nodal cycle
    # the chosen period does influence the results slightly
    yearmin = yearmax - 19
    for year in range(yearmin, yearmax+1):
        times_pred_all = pd.date_range(
            start=pd.Timestamp(year, 1, 1), end=pd.Timestamp(year + 1, 1, 1), freq=freq,
            tz=comp.attrs["tzone"], # to avoid tzone warning for all years
            inclusive="left", # skip last (1jan) value
        )
        ts_prediction = hatyan.prediction(comp=comp, times=times_pred_all)
        list_pred.append(ts_prediction)
    pred_all = pd.concat(list_pred, axis=0)
    return pred_all


def calc_hat_lat_frommeasurements(df_meas: pd.DataFrame) -> tuple:
    """
    

    Parameters
    ----------
    df_meas : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    tuple
        DESCRIPTION.

    """
    df_meas_19y = crop_timeseries_last_nyears(df=df_meas, nyears=19)
    df_meas_4y = crop_timeseries_last_nyears(df=df_meas, nyears=4)

    comp_19y = calc_getijcomponenten(
        df_meas_19y,
        const_list=["SA","SM"],
        analysis_perperiod=False,
        )
    comp_4y = calc_getijcomponenten(
        df_meas_4y,
        const_list=hatyan.get_const_list_hatyan("year"), 
        analysis_perperiod="Y",
        )

    comp_comb = comp_4y.copy()
    comp_comb.update(comp_19y)
    
    # overwrite A0 with slotgemiddelde
    slotgem = calc_slotgemiddelden(df_meas=df_meas)["wl_model_fit"].iloc[-1]
    comp_comb.loc["A0","A"] = slotgem

    yearmax = df_meas_19y.index.year.max()
    df_pred = predict_19y_peryear(comp_comb, yearmax=yearmax)

    lat = df_pred["values"].min()
    hat = df_pred["values"].max()
    return hat, lat
