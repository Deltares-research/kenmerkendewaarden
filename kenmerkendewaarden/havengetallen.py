# -*- coding: utf-8 -*-
"""
Computation of havengetallen
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import logging
from hatyan.astrog import astrog_culminations
from hatyan.timeseries import calc_HWLWnumbering
from kenmerkendewaarden.tidalindicators import calc_HWLWtidalrange

__all__ = ["calc_havengetallen",
           "plot_HWLW_pertimeclass",
           "plot_aardappelgrafiek",
           ]

logger = logging.getLogger(__name__)


def calc_havengetallen(df_ext:pd.DataFrame, return_df_ext=False):
    """
    havengetallen consist of the extreme (high and low) median values and the 
    extreme median time delays with respect to the moonculmination.
    Besides that it computes the tide difference for each cycle and the tidal period.
    All these indicators are derived by dividing the extremes in hour-classes 
    with respect to the moonculminination.

    Parameters
    ----------
    df_ext : pd.DataFrame
        DataFrame with extremes (highs and lows, no aggers).
    return_df : bool
        Whether to return the enriched input dataframe. Default is False.
    
    Returns
    -------
    df_havengetallen : pd.DataFrame
        DataFrame with havengetallen for all hour-classes. 
        0 corresponds to spring, 6 corresponds to neap, mean is mean.
    return_df_ext : pd.DataFrame
        An enriched copy of the input DataFrame, mainly for plotting.

    """
    # TODO: alternatively we can convert 12345 to 12 here
    assert len(df_ext["HWLWcode"].drop_duplicates()) == 2
    
    current_station = df_ext.attrs["station"]
    logger.info(f'computing havengetallen for {current_station}')
    # TODO: check culm_addtime and HWLWno+4 offsets. culm_addtime could also be 2 days or 2days +1h GMT-MET correction. 20 minutes seems odd since moonculm is about tidal wave from ocean
    # culm_addtime is a 2d and 2u20min correction, this shifts the x-axis of aardappelgrafiek
    # HW is 2 days after culmination (so 4x25min difference between length of avg moonculm and length of 2 days)
    # 1 hour (GMT to MET, alternatively we can also account for timezone differences elsewhere)
    # 20 minutes (0 to 5 meridian)
    culm_addtime = 4*dt.timedelta(hours=12,minutes=25) + dt.timedelta(hours=1) - dt.timedelta(minutes=20)
    
    # TODO: move calc_HWLW_moonculm_combi() to top since it is the same for all stations
    # TODO: we added tz_localize on 29-5-2024 (https://github.com/Deltares-research/kenmerkendewaarden/issues/30)
    # TODO: consider supporting timezones in hatyan.astrog.dT
    # this means we pass a UTC+1 timeseries as if it were a UTC timeseries
    if df_ext.index.tz is not None:
        df_ext = df_ext.tz_localize(None)
    df_ext = calc_HWLW_moonculm_combi(data_pd_HWLW_12=df_ext, culm_addtime=culm_addtime) #culm_addtime=None provides the same gemgetijkromme now delay is not used for scaling anymore
    df_havengetallen = calc_HWLW_culmhr_summary(df_ext) #TODO: maybe add tijverschil
    logger.info('computing havengetallen done')
    if return_df_ext:
        return df_havengetallen, df_ext
    else:
        return df_havengetallen


def get_moonculm_idxHWLWno(tstart,tstop):
    data_pd_moonculm = astrog_culminations(tFirst=tstart,tLast=tstop) # in UTC, which is important since data_pd_HWLW['culm_hr']=range(12) hourvalues should be in UTC since that relates to the relation dateline/sun
    data_pd_moonculm['datetime'] = data_pd_moonculm['datetime'].dt.tz_convert('UTC') #convert to UTC (is already)
    data_pd_moonculm['datetime'] = data_pd_moonculm['datetime'].dt.tz_localize(None) #remove timezone
    data_pd_moonculm = data_pd_moonculm.set_index('datetime',drop=False)
    data_pd_moonculm['values'] = data_pd_moonculm['type'] #dummy values for TA in hatyan.calc_HWLWnumbering()
    data_pd_moonculm['HWLWcode'] = 1 #all HW values since one every ~12h25m
    data_pd_moonculm = calc_HWLWnumbering(data_pd_moonculm,doHWLWcheck=False) #TODO: currently w.r.t. cadzd, is that an issue? With DELFZL the matched culmination is incorrect (since far away), but that might not be a big issue
    moonculm_idxHWLWno = data_pd_moonculm.set_index('HWLWno')
    return moonculm_idxHWLWno


def calc_HWLW_moonculm_combi(data_pd_HWLW_12,culm_addtime=None):
    moonculm_idxHWLWno = get_moonculm_idxHWLWno(tstart=data_pd_HWLW_12.index.min()-dt.timedelta(days=3),tstop=data_pd_HWLW_12.index.max())
    moonculm_idxHWLWno.index = moonculm_idxHWLWno.index + 4 #correlate HWLW to moonculmination 2 days before. TODO: check this offset in relation to culm_addtime.

    data_pd_HWLW_idxHWLWno = calc_HWLWnumbering(data_pd_HWLW_12)
    data_pd_HWLW_idxHWLWno['times'] = data_pd_HWLW_idxHWLWno.index
    data_pd_HWLW_idxHWLWno = data_pd_HWLW_idxHWLWno.set_index('HWLWno',drop=False)
    
    HW_bool = data_pd_HWLW_idxHWLWno['HWLWcode']==1
    data_pd_HWLW_idxHWLWno.loc[HW_bool,'getijperiod'] = data_pd_HWLW_idxHWLWno.loc[HW_bool,'times'].iloc[1:].values - data_pd_HWLW_idxHWLWno.loc[HW_bool,'times'].iloc[:-1] #this works properly since index is HWLW
    data_pd_HWLW_idxHWLWno.loc[HW_bool,'duurdaling'] = data_pd_HWLW_idxHWLWno.loc[~HW_bool,'times'] - data_pd_HWLW_idxHWLWno.loc[HW_bool,'times']
    data_pd_HWLW_idxHWLWno['culm_time'] = moonculm_idxHWLWno['datetime'] #couple HWLW to moonculminations two days earlier (this works since index is HWLWno)
    data_pd_HWLW_idxHWLWno['culm_hr'] = (data_pd_HWLW_idxHWLWno['culm_time'].round('h').dt.hour)%12
    data_pd_HWLW_idxHWLWno['HWLW_delay'] = data_pd_HWLW_idxHWLWno['times'] - data_pd_HWLW_idxHWLWno['culm_time']
    if culm_addtime is not None:
        data_pd_HWLW_idxHWLWno['HWLW_delay'] -= culm_addtime
    data_pd_HWLW = data_pd_HWLW_idxHWLWno.set_index('times')
    return data_pd_HWLW


def calc_HWLW_culmhr_summary(data_pd_HWLW):
    logger.info('calculate medians per hour group for LW and HW (instead of 1991 method: average of subgroups with removal of outliers)')
    data_pd_HW = data_pd_HWLW.loc[data_pd_HWLW['HWLWcode']==1]
    data_pd_LW = data_pd_HWLW.loc[data_pd_HWLW['HWLWcode']==2]
    
    HWLW_culmhr_summary = pd.DataFrame()
    HWLW_culmhr_summary['HW_values_median'] = data_pd_HW.groupby(data_pd_HW['culm_hr'])['values'].median()
    HWLW_culmhr_summary['HW_delay_median'] = data_pd_HW.groupby(data_pd_HW['culm_hr'])['HWLW_delay'].median().round('S')
    HWLW_culmhr_summary['LW_values_median'] = data_pd_LW.groupby(data_pd_LW['culm_hr'])['values'].median()
    HWLW_culmhr_summary['LW_delay_median'] = data_pd_LW.groupby(data_pd_LW['culm_hr'])['HWLW_delay'].median().round('S')
    HWLW_culmhr_summary['tijverschil'] = HWLW_culmhr_summary['HW_values_median'] - HWLW_culmhr_summary['LW_values_median']
    HWLW_culmhr_summary['getijperiod_median'] = data_pd_HW.groupby(data_pd_HW['culm_hr'])['getijperiod'].median().round('S')
    HWLW_culmhr_summary['duurdaling_median'] = data_pd_HW.groupby(data_pd_HW['culm_hr'])['duurdaling'].median().round('S')
    
    HWLW_culmhr_summary.loc['mean',:] = HWLW_culmhr_summary.mean() #add mean row to dataframe (not convenient to add immediately due to plotting with index 0-11)
    for colname in HWLW_culmhr_summary.columns: #round timedelta to make outputformat nicer
        if HWLW_culmhr_summary[colname].dtype == 'timedelta64[ns]':
            HWLW_culmhr_summary[colname] = HWLW_culmhr_summary[colname].round('S')

    return HWLW_culmhr_summary


def calc_HWLW_culmhr_summary_tidalcoeff(data_pd_HWLW_12):
    #TODO: use tidal coefficient instead?: The tidal coefficient is the size of the tide in relation to its mean. It usually varies between 20 and 120. The higher the tidal coefficient, the larger the tidal range – i.e. the difference in water height between high and low tide. This means that the sea level rises and falls back a long way. The mean value is 70. We talk of strong tides – called spring tides – from coefficient 95.  Conversely, weak tides are called neap tides. https://escales.ponant.com/en/high-low-tide/ en https://www.manche-toerisme.com/springtij
    #for HOEKVHLD, sp=0 is approx tc=1.2, np=6 is approx tc=0.8, av=mean is approx tc=1.0 (for HW, for LW it is different)
    data_pd_HWLW = data_pd_HWLW_12.copy()
    data_pd_HWLW = calc_HWLWtidalrange(data_pd_HWLW)
    data_pd_HWLW['tidalcoeff'] = data_pd_HWLW['tidalrange']/data_pd_HWLW['tidalrange'].mean()
    data_pd_HWLW['tidalcoeff_round'] = data_pd_HWLW['tidalcoeff'].round(1)
    TR_groupby_median = data_pd_HWLW.groupby('tidalcoeff_round')['tidalrange'].median()
    HW_groupby_median = data_pd_HWLW.loc[data_pd_HWLW['HWLWcode']==1].groupby('tidalcoeff_round')['values'].median()
    LW_groupby_median = data_pd_HWLW.loc[data_pd_HWLW['HWLWcode']==2].groupby('tidalcoeff_round')['values'].median()
    
    HWLW_culmhr_summary = pd.DataFrame()
    HWLW_culmhr_summary['HW_values_median'] = HW_groupby_median
    HWLW_culmhr_summary['LW_values_median'] = LW_groupby_median
    HWLW_culmhr_summary['tidalrange_median'] = TR_groupby_median
    HWLW_culmhr_summary = HWLW_culmhr_summary.loc[[0.8,1.0,1.2]] #select neap/mean/springtide
    HWLW_culmhr_summary.index = ['neap','mean','spring']
    
    return HWLW_culmhr_summary


def plot_HWLW_pertimeclass(data_pd_HWLW, HWLW_culmhr_summary):
    
    station = data_pd_HWLW.attrs["station"]
    
    HWLW_culmhr_summary = HWLW_culmhr_summary.loc[:11].copy() #remove mean column
    
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(18,8), sharex=True)
    data_pd_HW = data_pd_HWLW.loc[data_pd_HWLW['HWLWcode']==1]
    data_pd_LW = data_pd_HWLW.loc[data_pd_HWLW['HWLWcode']==2]
    ax1.set_title(f'HW values {station}')
    ax1.plot(data_pd_HW['culm_hr'],data_pd_HW['values'],'.')
    ax1.plot(HWLW_culmhr_summary['HW_values_median'],'.-')
    ax2.set_title(f'LW values {station}')
    ax2.plot(data_pd_LW['culm_hr'],data_pd_LW['values'],'.')
    ax2.plot(HWLW_culmhr_summary['LW_values_median'],'.-')
    ax3.set_title(f'HW time delays {station}')
    ax3.plot(data_pd_HW['culm_hr'],data_pd_HW['HWLW_delay'].dt.total_seconds()/3600,'.')
    ax3.plot(HWLW_culmhr_summary['HW_delay_median'].dt.total_seconds()/3600,'.-')
    ax4.set_title(f'LW time delays {station}')
    ax4.plot(data_pd_LW['culm_hr'],data_pd_LW['HWLW_delay'].dt.total_seconds()/3600,'.')
    ax4.plot(HWLW_culmhr_summary['LW_delay_median'].dt.total_seconds()/3600,'.-')
    ax4.set_xlim([0-0.5,12-0.5])
    fig.tight_layout()
    axs = np.array(((ax1,ax2),(ax3,ax4)))
    
    return fig, axs


def plot_aardappelgrafiek(HWLW_culmhr_summary):
    
    HWLW_culmhr_summary = HWLW_culmhr_summary.loc[:11].copy() #remove mean column
    
    def timeTicks(x, pos):
        d = dt.timedelta(hours=np.abs(x))
        if np.sign(x)>0:
            d_str = str(d)
        else:
            d_str = '-'+str(d)
        return d_str
    
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(7.5,4), sharex=False)
    ax1.set_title('HW')
    ax1.set_xlabel('maansverloop in uu:mm:ss' )
    ax1.set_ylabel('waterstand in m t.o.v. NAP')
    ax1.plot(HWLW_culmhr_summary['HW_delay_median'].dt.total_seconds()/3600,HWLW_culmhr_summary['HW_values_median'],'.-')
    ax1.xaxis.set_major_formatter(timeTicks)
    ax1.grid()
    ax2.set_title('LW')
    ax2.set_xlabel('maansverloop in uu:mm:ss' )
    ax2.set_ylabel('waterstand in m t.o.v. NAP')
    ax2.plot(HWLW_culmhr_summary['LW_delay_median'].dt.total_seconds()/3600,HWLW_culmhr_summary['LW_values_median'],'.-')
    ax2.xaxis.set_major_formatter(timeTicks)
    ax2.grid()
    for iH,row in HWLW_culmhr_summary.iterrows():
        ax1.text(row['HW_delay_median'].total_seconds()/3600,row['HW_values_median'], str(int(iH)))
        ax2.text(row['LW_delay_median'].total_seconds()/3600,row['LW_values_median'], str(int(iH)))
    #set equal ylims
    ax1_xlimmean = np.mean(ax1.get_xlim())
    ax2_xlimmean = np.mean(ax2.get_xlim())
    ax1_ylimmean = np.mean(ax1.get_ylim())
    ax2_ylimmean = np.mean(ax2.get_ylim())
    xlimrange = 2
    ylimrange = 1
    ax1.set_xlim([ax1_xlimmean-xlimrange/2,ax1_xlimmean+xlimrange/2])
    ax2.set_xlim([ax2_xlimmean-xlimrange/2,ax2_xlimmean+xlimrange/2])
    ax1.set_ylim([ax1_ylimmean-ylimrange/2,ax1_ylimmean+ylimrange/2])
    ax2.set_ylim([ax2_ylimmean-ylimrange/2,ax2_ylimmean+ylimrange/2])
    #plot gemtij dotted lines
    ax1.plot(ax1.get_xlim(),[HWLW_culmhr_summary['HW_values_median'].mean(),HWLW_culmhr_summary['HW_values_median'].mean()],'k--')
    ax1.plot([HWLW_culmhr_summary['HW_delay_median'].mean().total_seconds()/3600,HWLW_culmhr_summary['HW_delay_median'].mean().total_seconds()/3600],ax1.get_ylim(),'k--')
    ax2.plot(ax2.get_xlim(),[HWLW_culmhr_summary['LW_values_median'].mean(),HWLW_culmhr_summary['LW_values_median'].mean()],'k--')
    ax2.plot([HWLW_culmhr_summary['LW_delay_median'].mean().total_seconds()/3600,HWLW_culmhr_summary['LW_delay_median'].mean().total_seconds()/3600],ax2.get_ylim(),'k--')
    fig.tight_layout()
    
    axs = (ax1,ax2)
    return fig, axs
