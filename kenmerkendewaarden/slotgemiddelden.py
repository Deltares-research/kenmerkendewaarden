# -*- coding: utf-8 -*-
"""
Computation of slotgemiddelden of waterlevels and extremes
"""

import numpy as np
import statsmodels.api as sm
import pandas as pd
import datetime as dt

__all__ = ["fit_models"]


def fit_models(mean_array_todate: pd.Series) -> pd.DataFrame:
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
    
    
    # We'll just use the years. This assumes that annual waterlevels are used that are stored left-padded, the mean waterlevel for 2020 is stored as 2020-1-1. This is not logical, but common practice.
    allyears_DTI = pd.date_range(mean_array_todate.index.min(),mean_array_todate.index.max()+dt.timedelta(days=5*360),freq='AS')
    mean_array_allyears = pd.Series(mean_array_todate,index=allyears_DTI)
    
    df = pd.DataFrame({'year':mean_array_allyears.index.year, 'height':mean_array_allyears.values}) #TODO: make functions accept mean_array instead of df as argument?
    
    # below methods are copied from https://github.com/openearth/sealevel/blob/master/slr/slr/models.py #TODO: install slr package as dependency or keep separate?
    fit, names, X = linear_model(df, with_wind=False, with_nodal=False)
    pred_linear_nonodal = fit.predict(X)
    fit, names, X = linear_model(df, with_wind=False)
    pred_linear_winodal = fit.predict(X)
    
    pred_pd = pd.DataFrame({'pred_linear_nonodal':pred_linear_nonodal,
                            'pred_linear_winodal':pred_linear_winodal},
                            index=allyears_DTI)
    return pred_pd


# copied from https://github.com/openearth/sealevel/blob/master/slr/slr/models.py
def broken_linear_model(df, with_wind=True, quantity='height', start_acceleration=1993):
    """This model fits the sea-level rise has started to rise faster in 1993."""
    y = df[quantity]
    X = np.c_[
        df['year']-1970,
        (df['year'] > start_acceleration),# * (df['year'] - start_acceleration),
        np.cos(2*np.pi*(df['year']-1970)/18.613),
        np.sin(2*np.pi*(df['year']-1970)/18.613)
    ]
    names = ['Constant', 'Trend', f'+trend ({start_acceleration})', 'Nodal U', 'Nodal V']
    if with_wind:
        X = np.c_[
            X,
            df['u2'],
            df['v2']
        ]
        names.extend(['Wind $u^2$', 'Wind $v^2$'])
    X = sm.add_constant(X)
    model_broken_linear = sm.GLSAR(y, X, rho=1, missing='drop')
    fit = model_broken_linear.iterative_fit(cov_type='HC0', missing='drop')
    return fit, names, X


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


