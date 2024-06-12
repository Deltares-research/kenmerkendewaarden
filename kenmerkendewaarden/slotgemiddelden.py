# -*- coding: utf-8 -*-
"""
Computation of slotgemiddelden of waterlevels and extremes
"""

import numpy as np
import statsmodels.api as sm
import pandas as pd
from kenmerkendewaarden.data_retrieve import clip_timeseries_physical_break
from kenmerkendewaarden.tidalindicators import calc_wltidalindicators, calc_HWLWtidalindicators
import logging

__all__ = ["calc_slotgemiddelden"]

logger = logging.getLogger(__name__)


def calc_slotgemiddelden(df_meas: pd.DataFrame, df_ext: pd.DataFrame=None, 
                         only_valid: pd.Timestamp = False, clip_physical_break: bool = False, 
                         with_nodal: bool = True):
    """
    Compute slotgemiddelden from measurement timeseries and optionally also from extremes timeseries.

    Parameters
    ----------
    df_meas : pd.DataFrame
        the timeseries of measured waterlevels.
    df_ext : pd.DataFrame, optional
        the timeseries of extremes (high and low waters). The default is None.
    only_valid : pd.Timestamp, optional
        Whether to set yearly means to nans for years that do not have sufficient data coverage. The default is False.
    clip_physical_break : bool, optional
        Whether to exclude the part of the timeseries before physical breaks like estuary closures. The default is False.
    with_nodal : bool, optional
        Whether to include a nodal cycle in the linear trend model. The default is True.

    Returns
    -------
    slotgemiddelden_dict : dict
        dictionary with yearly means and model fits, optionally also for extremes.

    """

    # TODO: prevent hardcoded min_count argument for tidalindicators functions: https://github.com/Deltares-research/kenmerkendewaarden/issues/58
    if only_valid:
        min_count_ext = 1400 # 2*24*365/12.42=1410.6 (12.42 hourly extreme)
        min_count_wl = 2900 # 24*365=8760 (hourly interval), 24/3*365=2920 (3-hourly interval)
    else:
        min_count_ext = None
        min_count_wl = None
    
    # clip last value of the timeseries if this is exactly newyearsday
    if df_meas.index[-1] == pd.Timestamp(df_meas.index[-1].year,1,1, tz=df_meas.index.tz):
        df_meas = df_meas.iloc[:-1]
    
    if clip_physical_break:
        df_meas = clip_timeseries_physical_break(df_meas)
    
    
    # calculate yearly means
    dict_wltidalindicators = calc_wltidalindicators(df_meas, min_count=min_count_wl)
    wl_mean_peryear = dict_wltidalindicators['wl_mean_peryear']
    
    # fit linear models over yearly mean values
    pred_pd_wl = fit_models(wl_mean_peryear, with_nodal=with_nodal)
    
    # add to dict
    slotgemiddelden_dict = {}
    slotgemiddelden_dict["wl_mean_peryear"] = wl_mean_peryear
    slotgemiddelden_dict["wl_model_fit"] = pred_pd_wl
    
    if df_ext is not None:
        # clip last value of the timeseries if this is exactly newyearsday
        if df_ext.index[-1] == pd.Timestamp(df_ext.index[-1].year,1,1, tz=df_ext.index.tz):
            df_ext = df_ext.iloc[:-1]
        
        if clip_physical_break:
            df_ext = clip_timeseries_physical_break(df_ext)
    
        # calculate yearly means
        dict_HWLWtidalindicators = calc_HWLWtidalindicators(df_ext, min_count=min_count_ext)
        HW_mean_peryear = dict_HWLWtidalindicators['HW_mean_peryear']
        LW_mean_peryear = dict_HWLWtidalindicators['LW_mean_peryear']
    
        # fit linear models over yearly mean values
        pred_pd_HW = fit_models(HW_mean_peryear, with_nodal=with_nodal)
        pred_pd_LW = fit_models(LW_mean_peryear, with_nodal=with_nodal)
        
        # add to dict
        slotgemiddelden_dict["HW_mean_peryear"] = HW_mean_peryear
        slotgemiddelden_dict["LW_mean_peryear"] = LW_mean_peryear
        slotgemiddelden_dict["HW_model_fit"] = pred_pd_HW
        slotgemiddelden_dict["LW_model_fit"] = pred_pd_LW
    
    return slotgemiddelden_dict


def fit_models(mean_array_todate: pd.Series, with_nodal=True) -> pd.DataFrame:
    """
    Fit linear model over yearly means in mean_array_todate, including five years in the future.

    Parameters
    ----------
    mean_array_todate : pd.Series
        DESCRIPTION.

    Returns
    -------
    pred_pd : TYPE
        DESCRIPTION.

    """
    
    start = mean_array_todate.index.min()
    end = mean_array_todate.index.max() + pd.Timedelta(days=370)
    
    logger.info(f"fit linear model for {start} to {end}")
    
    # We'll just use the years. This assumes that annual waterlevels are used that are stored left-padded, the mean waterlevel for 2020 is stored as 2020-1-1. This is not logical, but common practice.
    allyears_DTI = pd.date_range(start=start, end=end, freq='YS')
    mean_array_allyears = pd.Series(mean_array_todate, index=allyears_DTI)
    
    df = pd.DataFrame({'year':mean_array_allyears.index.year, 'height':mean_array_allyears.values}) #TODO: make functions accept mean_array instead of df as argument?
    
    # below methods are copied from https://github.com/openearth/sealevel/blob/master/slr/slr/models.py #TODO: install slr package as dependency or keep separate?
    
    fit, names, X = linear_model(df, with_wind=False, with_nodal=with_nodal)
    pred_linear = fit.predict(X)
    
    linear_fit = pd.Series(pred_linear, index=allyears_DTI)
    return linear_fit


# copied from https://github.com/openearth/sealevel/blob/master/slr/slr/models.py
def linear_model(df, with_wind=True, with_ar=True, with_nodal=True, quantity='height'):
    """Define the linear model with optional wind and autoregression.
    See the latest report for a detailed description.
    """

    y = df[quantity]
    X = np.c_[df['year']-1970,
              ]
    #month = np.mod(df['year'], 1) * 12.0
    names = ['Constant', 'Trend']
    if with_nodal:
        X = np.c_[X,
                  np.cos(2*np.pi*(df['year']-1970)/18.613),
                  np.sin(2*np.pi*(df['year']-1970)/18.613)
                  ]
        names.extend(['Nodal U', 'Nodal V'])
    if with_wind:
        X = np.c_[
            X,
            df['u2'],
            df['v2']
        ]
        names.extend(['Wind $u^2$', 'Wind $v^2$'])
    X = sm.add_constant(X)
    if with_ar:
        model = sm.GLSAR(y, X, missing='drop', rho=1)
    else:
        model = sm.OLS(y, X, missing='drop')
    fit = model.fit(cov_type='HC0')
    return fit, names, X


