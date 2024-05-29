# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:14:25 2024

@author: veenstra
"""

import numpy as np
import pandas as pd
import datetime as dt
from hatyan.timeseries import calc_HWLW12345to12, calc_HWLWnumbering
from hatyan.analysis_prediction import HatyanSettings, prediction #PydanticConfig

__all__ = ["calc_wltidalindicators",
           "calc_HWLWtidalindicators",
           "calc_HWLWtidalrange",
           ]

#from pydantic import validate_arguments #TODO: enable validator (first add pydantic as dependency, plus how to validate comp df (columns A/phi, then maybe classed should be used instead)


def calc_HWLWtidalindicators(data_pd_HWLW_all, min_count=None):
    """
    computes several tidal extreme indicators from tidal extreme dataset

    Parameters
    ----------
    data_pd_HWLW_all : TYPE
        DESCRIPTION.
    min_count : int
        The minimum amount of timeseries values per year to consider the statistics to be valid.

    Returns
    -------
    dict_tidalindicators : TYPE
        DESCRIPTION.

    """
    if len(data_pd_HWLW_all['HWLWcode'].unique()) > 2: #aggers are present
        # TODO: this drops first/last value if it is a LW (and more problems), should be fixed: https://github.com/Deltares/hatyan/issues/311
        data_pd_HWLW_12 = calc_HWLW12345to12(data_pd_HWLW_all) #convert 12345 to 12 by taking minimum of 345 as 2 (laagste laagwater)
    else:
        data_pd_HWLW_12 = data_pd_HWLW_all.copy()
    
    #split to HW and LW separately, also groupby year
    data_pd_HW = data_pd_HWLW_12.loc[data_pd_HWLW_12['HWLWcode']==1]
    data_pd_LW = data_pd_HWLW_12.loc[data_pd_HWLW_12['HWLWcode']==2]
    
    #count HWLW values per year/month
    HWLW_count_peryear = data_pd_HWLW_12.groupby(pd.PeriodIndex(data_pd_HWLW_12.index, freq="y"))['values'].count()
    HWLW_count_permonth = data_pd_HWLW_12.groupby(pd.PeriodIndex(data_pd_HWLW_12.index, freq="m"))['values'].count()
    
    #yearmean HWLW from HWLW values #maybe also add *_mean_permonth
    HW_mean_peryear = data_pd_HW.groupby(pd.PeriodIndex(data_pd_HW.index, freq="y"))[['values']].mean()
    LW_mean_peryear = data_pd_LW.groupby(pd.PeriodIndex(data_pd_LW.index, freq="y"))[['values']].mean()
    
    #derive GHHW/GHWS (gemiddeld hoogwater springtij) per month
    HW_monthmax_permonth = data_pd_HW.groupby(pd.PeriodIndex(data_pd_HW.index, freq="m"))[['values']].max() #proxy for HW at spring tide
    LW_monthmin_permonth = data_pd_LW.groupby(pd.PeriodIndex(data_pd_LW.index, freq="m"))[['values']].min() #proxy for LW at spring tide
    HW_monthmin_permonth = data_pd_HW.groupby(pd.PeriodIndex(data_pd_HW.index, freq="m"))[['values']].min() #proxy for HW at neap tide
    LW_monthmax_permonth = data_pd_LW.groupby(pd.PeriodIndex(data_pd_LW.index, freq="m"))[['values']].max() #proxy for LW at neap tide
    
    #replace invalids with nan (in case of too less values per month or year)
    if min_count is not None:
        min_count_permonth = min_count/13 #not 13 but 12, to also make the threshold valid in short months
        HW_mean_peryear.loc[HWLW_count_peryear<min_count] = np.nan
        LW_mean_peryear.loc[HWLW_count_peryear<min_count] = np.nan
        HW_monthmax_permonth.loc[HWLW_count_permonth<min_count_permonth] = np.nan
        LW_monthmin_permonth.loc[HWLW_count_permonth<min_count_permonth] = np.nan
        HW_monthmin_permonth.loc[HWLW_count_permonth<min_count_permonth] = np.nan
        LW_monthmax_permonth.loc[HWLW_count_permonth<min_count_permonth] = np.nan
    
    #derive GHHW/GHWS (gemiddeld hoogwater springtij)
    HW_monthmax_mean_peryear = HW_monthmax_permonth.groupby(pd.PeriodIndex(HW_monthmax_permonth.index, freq="y"))[['values']].mean()
    LW_monthmin_mean_peryear = LW_monthmin_permonth.groupby(pd.PeriodIndex(LW_monthmin_permonth.index, freq="y"))[['values']].mean()
    HW_monthmin_mean_peryear = HW_monthmin_permonth.groupby(pd.PeriodIndex(HW_monthmin_permonth.index, freq="y"))[['values']].mean()
    LW_monthmax_mean_peryear = LW_monthmax_permonth.groupby(pd.PeriodIndex(LW_monthmax_permonth.index, freq="y"))[['values']].mean()
    
    dict_HWLWtidalindicators = {'HW_mean':data_pd_HW['values'].mean(), #GHW
                                'LW_mean':data_pd_LW['values'].mean(), #GLW
                                'HW_mean_peryear':HW_mean_peryear['values'], #GHW peryear
                                'LW_mean_peryear':LW_mean_peryear['values'], #GLW peryear
                                'HW_monthmax_permonth':HW_monthmax_permonth['values'], #GHHW/GHWS permonth
                                'LW_monthmin_permonth':LW_monthmin_permonth['values'], #GLLW/GLWS permonth
                                'HW_monthmax_mean_peryear':HW_monthmax_mean_peryear['values'], #GHHW/GHWS peryear
                                'LW_monthmin_mean_peryear':LW_monthmin_mean_peryear['values'], #GLLW/GLWS peryear
                                'HW_monthmin_mean_peryear':HW_monthmin_mean_peryear['values'], #GLHW/GHWN peryear
                                'LW_monthmax_mean_peryear':LW_monthmax_mean_peryear['values'], #GHLW/GLWN peryear
                                }
    
    for key in dict_HWLWtidalindicators.keys():
        if not hasattr(dict_HWLWtidalindicators[key],'index'):
            continue
        dict_HWLWtidalindicators[key].index = dict_HWLWtidalindicators[key].index.to_timestamp()
        
    return dict_HWLWtidalindicators


def calc_wltidalindicators(data_wl_pd, min_count=None):
    """
    computes monthly and yearly means from waterlevel timeseries

    Parameters
    ----------
    data_wl_pd : TYPE
        DESCRIPTION.
    min_count : int
        The minimum amount of timeseries values per year to consider the statistics to be valid.

    Returns
    -------
    dict_wltidalindicators : TYPE
        DESCRIPTION.

    """
    if hasattr(data_wl_pd.index[0],'tz'): #timezone present in index
        data_wl_pd.index = data_wl_pd.index.tz_localize(None)
    
    #count wl values per year/month
    wl_count_peryear = data_wl_pd.groupby(pd.PeriodIndex(data_wl_pd.index, freq="y"))['values'].count()
    wl_count_permonth = data_wl_pd.groupby(pd.PeriodIndex(data_wl_pd.index, freq="m"))['values'].count()
    
    #yearmean wl from wl values
    wl_mean_peryear = data_wl_pd.groupby(pd.PeriodIndex(data_wl_pd.index, freq="y"))[['values']].mean()
    wl_mean_permonth = data_wl_pd.groupby(pd.PeriodIndex(data_wl_pd.index, freq="m"))[['values']].mean()
    
    #replace invalids with nan (in case of too less values per month or year)
    if min_count is not None:
        min_count_permonth = min_count/12
        wl_mean_peryear.loc[wl_count_peryear<min_count] = np.nan
        wl_mean_permonth.loc[wl_count_permonth<min_count_permonth] = np.nan
        
    dict_wltidalindicators = {'wl_mean_peryear':wl_mean_peryear['values'], #yearly mean wl
                              'wl_mean_permonth':wl_mean_permonth['values'], #monthly mean wl
                              }
    
    for key in dict_wltidalindicators.keys():
        if not hasattr(dict_wltidalindicators[key],'index'):
            continue
        dict_wltidalindicators[key].index = dict_wltidalindicators[key].index.to_timestamp()
        
    return dict_wltidalindicators


def calc_HWLWtidalrange(ts_ext):
    """
    creates column 'tidalrange' in ts_ext dataframe
    """
    ts_ext = calc_HWLWnumbering(ts_ext=ts_ext)
    ts_ext['times_backup'] = ts_ext.index
    ts_ext_idxHWLWno = ts_ext.set_index('HWLWno',drop=False)
    ts_ext_idxHWLWno['tidalrange'] = ts_ext_idxHWLWno.loc[ts_ext_idxHWLWno['HWLWcode']==1,'values'] - ts_ext_idxHWLWno.loc[ts_ext_idxHWLWno['HWLWcode']==2,'values']
    ts_ext = ts_ext_idxHWLWno.set_index('times_backup')
    ts_ext.index.name = 'times'
    return ts_ext


#@validate_arguments(config=PydanticConfig)
def calc_hat_lat_fromcomponents(comp: pd.DataFrame) -> tuple:
    """
    Derive highest and lowest astronomical tide (HAT/LAT) from a component set.
    The component set is used to make a tidal prediction for an arbitrary period of 19 years with a 1 minute interval. The max/min values of the predictions of all years are the HAT/LAT values.
    The HAT/LAT is very dependent on the A0 of the component set. Therefore, the HAT/LAT values are relevant for the same year as the slotgemiddelde that is used to replace A0 in the component set. For instance, if the slotgemiddelde is valid for 2021.0, HAT and LAT are also relevant for that year.
    The HAT/LAT values are also very dependent on the hatyan_settings used, in general it is important to use the same settings as used to derive the tidal components.
    
    Parameters
    ----------
    comp : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    tuple
        DESCRIPTION.

    """
    
    xfac = comp.attrs['xfac']
    hatyan_settings = HatyanSettings(nodalfactors=True, xfac=xfac, fu_alltimes=False)
    
    min_vallist_allyears = pd.Series(dtype=float)
    max_vallist_allyears = pd.Series(dtype=float)
    for year in range(2020,2039): # 19 arbitrary consequtive years to capture entire nodal cycle
        times_pred_all = pd.date_range(start=dt.datetime(year,1,1), end=dt.datetime(year+1,1,1), freq='1min')
        ts_prediction = prediction(comp=comp, hatyan_settings=hatyan_settings, times=times_pred_all)
        
        min_vallist_allyears.loc[year] = ts_prediction['values'].min()
        max_vallist_allyears.loc[year] = ts_prediction['values'].max()
    
    hat = max_vallist_allyears.max()
    lat = min_vallist_allyears.min()
    return hat, lat
