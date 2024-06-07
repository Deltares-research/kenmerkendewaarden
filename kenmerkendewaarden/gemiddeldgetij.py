# -*- coding: utf-8 -*-
"""
Computation of gemiddelde getijkromme
"""

import os
import numpy as np
import pandas as pd
import hatyan
import logging
from kenmerkendewaarden.tidalindicators import calc_HWLWtidalrange

__all__ = ["gemiddeld_getij_av_sp_np",
           ]

logger = logging.getLogger(__name__)


def gemiddeld_getij_av_sp_np(df_meas, pred_freq="60sec", nb=0, nf=0, scale_extremes=False, scale_period=False, debug=False):
    """
    Generate an average tidal signal for average/spring/neap tide by doing a tidal 
    analysis on a timeseries of measurements. The (subsets/adjusted) resulting tidal components 
    are then used to make a raw prediction for average/spring/neap tide.
    These raw predictions can optionally be scaled in height (with havengetallen)
    and in time (to a fixed period of 12h25min). An n-number of backwards and forward repeats 
    are added before the timeseries are returned, resulting in nb+nf+1 tidal periods.
    
    df_meas: timeseries of 10 years of waterlevel measurements
    pred_freq: frequency of the prediction, a value of 60 seconds or lower is adivisable for decent results.
    nb: amount of periods to repeat backward
    nf: amount of periods to repeat forward
    scale_extremes: scale extremes with havengetallen. Should be a boolean, but is now a filepath to the havengetallen csv files
    scale_period: scale to 12h25min (for boi)
    """
    data_pd_meas_10y = df_meas
    tstop_dt = df_meas.index.max()
    
    current_station = data_pd_meas_10y.attrs["station"]
    
    # TODO: deprecate debug argument+plot (maybe use max HW instead of max tidalrange?)
    # TODO: we now call this function three times and deriving the unscaled krommes takes quite some time. Put in different function and cache it. 
    # TODO: add correctie havengetallen HW/LW av/sp/np met slotgemiddelde uit PLSS/modelfit (HW/LW av)
    
    if scale_period:
        tP_goal = pd.Timedelta(hours=12,minutes=25)
    else:
        tP_goal = None

    # TODO: derive havengetallen on the fly instead, no hardcoded files at least (and deprecate file_havget argument)
    if scale_extremes: # if not None, so e.g. file_havget
        file_havget = scale_extremes # TODO: this is a temporary solution to pass file_havget
        if not os.path.exists(file_havget):
            raise Exception(f'havengetallen file does not exist: {file_havget}')
        data_havget = pd.read_csv(file_havget, index_col=0)
        for colname in ['HW_delay_median','LW_delay_median','getijperiod_median','duurdaling_median']:
            data_havget[colname] = pd.to_timedelta(data_havget[colname])
        
        HW_sp, LW_sp = data_havget.loc['spring',['HW_values_median','LW_values_median']]
        HW_np, LW_np = data_havget.loc['neap',['HW_values_median','LW_values_median']]
        HW_av, LW_av = data_havget.loc['mean',['HW_values_median','LW_values_median']]
    else:
        HW_av = LW_av = None
        HW_sp = LW_sp = None
        HW_np = LW_np = None        
    
    #derive components via TA on measured waterlevels
    comp_frommeasurements_avg, comp_av = get_gemgetij_components(data_pd_meas_10y)
    
    times_pred_1mnth = pd.date_range(start=pd.Timestamp(tstop_dt.year,1,1,0,0)-pd.Timedelta(hours=12), end=pd.Timestamp(tstop_dt.year,2,1,0,0), freq=pred_freq) #start 12 hours in advance, to assure also corrected values on desired tstart
    comp_av.attrs['nodalfactors'] = False #nodalfactors=False to guarantee repetative signal
    comp_av.attrs['fu_alltimes'] = True # TODO: this is not true, but this setting is the default
    prediction_av = hatyan.prediction(comp_av, times=times_pred_1mnth)
    prediction_av_ext = hatyan.calc_HWLW(ts=prediction_av, calc_HWLW345=False)
    
    time_firstHW = prediction_av_ext.loc[prediction_av_ext['HWLWcode']==1].index[0] #time of first HW
    ia1 = prediction_av_ext.loc[time_firstHW:].index[0] #time of first HW
    ia2 = prediction_av_ext.loc[time_firstHW:].index[2] #time of second HW
    prediction_av_one = prediction_av.loc[ia1:ia2]
    prediction_av_ext_one = prediction_av_ext.loc[ia1:ia2]
    
    # =============================================================================
    # Hatyan predictie voor 1 jaar met gemiddelde helling maansbaan (voor afleiden spring-doodtijcyclus) >> predictie zonder nodalfactors instead
    # =============================================================================
    """
    uit: gemiddelde getijkrommen 1991.0
    Voor de ruwe krommen voor springtij en doodtij is het getij voorspeld voor een jaar met gemiddelde helling maansbaan 
    met uitsluitend zuivere combinaties van de componenten M2 en S2:
    tabel: Gebruikte componenten voor de spring- en doodtijkromme
    SM, 3MS2, mu2, M2, S2, 2SM2, 3MS4, M4, MS4, 
    4MS6, M6, 2MS6, M8, 3MS8, M10, 4MS10, M12, 5MS12
    
    In het aldus gemodelleerde getij is de vorm van iedere getijslag, gegeven de getijfase, identiek. 
    Vervolgens is aan de hand van de havengetallen een springtij- en een doodtijkromme geselecteerd.
    
    #NOTE: background on choice of components
    #below is different than provided list, these shallow ones are extra: ['S4','2SM6','M7','4MS4','2(MS)8','3M2S10','4M2S12']
    #shallow relations, derive 'zuivere harmonischen van M2 en S2' (this means averaging over eenmaaldaagse componenten, but why is that chosen?)
    #adding above extra components or oneday freqs, gives a modulation and therefore there is no repetative signal anymore. Apperantly all components in this list have an integer number of periods in one springneap cycle?
    dummy,shallowrel,dummy = hatyan.foreman.get_foreman_shallowrelations()
    bool_M2S2only = shallowrel[1].isin([1,2]) & shallowrel[3].isin(['M2','S2']) & shallowrel[5].isin(['M2','S2',np.nan]) & shallowrel.index.isin(const_list_year)
    shallowdeps_M2S2 = shallowrel.loc[bool_M2S2only,:5]
    print(shallowdeps_M2S2)
    """
    components_sn = ['A0','SM','3MS2','MU2','M2','S2','2SM2','3MS4','M4','MS4','4MS6','M6','2MS6','M8','3MS8','M10','4MS10','M12','5MS12']
    
    #make prediction with springneap components with nodalfactors=False (alternative for choosing a year with a neutral nodal factor). Using 1yr instead of 1month does not make a difference in min/max tidal range and shape, also because of nodalfactors=False. (when using more components, there is a slight difference)
    comp_frommeasurements_avg_sncomp = comp_frommeasurements_avg.loc[components_sn]
    comp_frommeasurements_avg_sncomp.attrs['nodalfactors'] = False #nodalfactors=False to make independent on chosen year
    comp_frommeasurements_avg_sncomp.attrs['fu_alltimes'] = True # TODO: this is not true, but this setting is the default
    prediction_sn = hatyan.prediction(comp_frommeasurements_avg_sncomp, times=times_pred_1mnth)
    
    prediction_sn_ext = hatyan.calc_HWLW(ts=prediction_sn, calc_HWLW345=False)
    
    #selecteer getijslag met minimale tidalrange en maximale tidalrange (werd geselecteerd adhv havengetallen in 1991.0 doc)
    prediction_sn_ext = calc_HWLWtidalrange(prediction_sn_ext)
    
    time_TRmax = prediction_sn_ext.loc[prediction_sn_ext['HWLWcode']==1,'tidalrange'].idxmax()
    is1 = prediction_sn_ext.loc[time_TRmax:].index[0]
    is2 = prediction_sn_ext.loc[time_TRmax:].index[2]
    
    time_TRmin = prediction_sn_ext.loc[prediction_sn_ext['HWLWcode']==1,'tidalrange'].idxmin()
    in1 = prediction_sn_ext.loc[time_TRmin:].index[0]
    in2 = prediction_sn_ext.loc[time_TRmin:].index[2]
    
    #select one tideperiod for springtide and one for neaptide
    prediction_sp_one = prediction_sn.loc[is1:is2]
    prediction_sp_ext_one = prediction_sn_ext.loc[is1:is2]
    prediction_np_one = prediction_sn.loc[in1:in2]
    prediction_np_ext_one = prediction_sn_ext.loc[in1:in2]
    
    # plot selection of neap/spring
    if debug:
        fig, (ax1,ax2) = hatyan.plot_timeseries(ts=prediction_sn,ts_ext=prediction_sn_ext)
        ax1.plot(prediction_sp_one['values'],'r')
        ax1.plot(prediction_np_one['values'],'r')
        ax1.legend(labels=ax1.get_legend_handles_labels()[1]+['kromme spring','kromme neap'],loc=4)
        ax1.set_ylabel('waterstand [m]')
        ax1.set_title(f'spring- en doodtijkromme {current_station}')
        # fig.savefig(os.path.join(dir_gemgetij,f'springdoodtijkromme_{current_station}_slotgem{year_slotgem}.png'))
    
    #timeseries for gele boekje (av/sp/np have different lengths, time is relative to HW of av and HW of sp/np are shifted there) #TODO: is this product still necessary?
    logger.info(f'reshape_signal GEMGETIJ: {current_station}')
    prediction_av_corr_one = reshape_signal(prediction_av_one, prediction_av_ext_one, HW_goal=HW_av, LW_goal=LW_av, tP_goal=tP_goal)
    prediction_av_corr_one.index = prediction_av_corr_one.index - prediction_av_corr_one.index[0] # make relative to first timestamp (=HW)
    if scale_period: # resampling required because of scaling
        prediction_av_corr_one = prediction_av_corr_one.resample(pred_freq).nearest()
    prediction_av = repeat_signal(prediction_av_corr_one, nb=nb, nf=nf)
    
    logger.info(f'reshape_signal SPRINGTIJ: {current_station}')
    prediction_sp_corr_one = reshape_signal(prediction_sp_one, prediction_sp_ext_one, HW_goal=HW_sp, LW_goal=LW_sp, tP_goal=tP_goal)
    prediction_sp_corr_one.index = prediction_sp_corr_one.index - prediction_sp_corr_one.index[0] # make relative to first timestamp (=HW)
    if scale_period: # resampling required because of scaling
        prediction_sp_corr_one = prediction_sp_corr_one.resample(pred_freq).nearest()
    prediction_sp = repeat_signal(prediction_sp_corr_one, nb=nb, nf=nf)
    
    logger.info(f'reshape_signal DOODTIJ: {current_station}')
    prediction_np_corr_one = reshape_signal(prediction_np_one, prediction_np_ext_one, HW_goal=HW_np, LW_goal=LW_np, tP_goal=tP_goal)
    prediction_np_corr_one.index = prediction_np_corr_one.index - prediction_np_corr_one.index[0] # make relative to first timestamp (=HW)
    if scale_period: # resampling required because of scaling
        prediction_np_corr_one = prediction_np_corr_one.resample(pred_freq).nearest()
    prediction_np = repeat_signal(prediction_np_corr_one, nb=nb, nf=nf)
    
    return prediction_av, prediction_sp, prediction_np


def get_gemgetij_components(data_pd_meas):
    # =============================================================================
    # Hatyan analyse voor 10 jaar (alle componenten voor gemiddelde getijcyclus) #TODO: maybe use original 4y period/componentfile instead? SA/SM should come from 19y analysis
    # =============================================================================
    const_list = hatyan.get_const_list_hatyan('year') #components should not be reduced, since higher harmonics are necessary
    hatyan_settings_ana = dict(nodalfactors=True, fu_alltimes=False, xfac=True, analysis_perperiod='Y', return_allperiods=True) #RWS-default settings
    comp_frommeasurements_avg, comp_frommeasurements_allyears = hatyan.analysis(data_pd_meas, const_list=const_list, **hatyan_settings_ana)
    
    # #check if all years are available
    # comp_years = comp_frommeasurements_allyears['A'].columns
    # expected_years = tstop_dt.year-tstart_dt.year
    # if len(comp_years) < expected_years:
    #     raise Exception('ERROR: analysis result contains not all years')
    
    #check if nans in analysis
    if comp_frommeasurements_avg.isnull()['A'].any():
        raise Exception('ERROR: analysis result contains nan values')
    
    # =============================================================================
    # gemiddelde getijkromme
    # =============================================================================
    """
    uit: gemiddelde getijkrommen 1991.0
    Voor meetpunten in het onbeinvloed gebied is per getijfase eerst een "ruwe kromme" berekend met de resultaten van de harmonische analyse, 
    welke daarna een weinig is bijgesteld aan de hand van de volgende slotgemiddelden:
    gemiddeld hoog- en laagwater, duur daling. Deze bijstelling bestaat uit een eenvoudige vermenigvuldiging.    
    
    Voor de ruwe krommen voor gemiddeld tij zijn uitsluitend zuivere harmonischen van M2 gebruikt: M2, M4, M6, M8, M10, M12, 
    waarbij de amplituden per component zijn vervangen door de wortel uit de kwadraatsom van de amplituden 
    van alle componenten in de betreffende band, voor zover voorkomend in de standaardset van 94 componenten. 
    Zoals te verwachten is de verhouding per component tussen deze wortel en de oorspronkelijke amplitude voor alle plaatsen gelijk.
    tabel: Verhouding tussen amplitude en oorspronkelijke amplitude
    M2 (tweemaaldaagse band) 1,06
    M4 1,28
    M6 1,65
    M8 2,18
    M10 2,86
    M12 3,46
    
    In het aldus gemodelleerde getij is de vorm van iedere getijslag identiek, met een getijduur van 12 h 25 min.
    Bij meetpunten waar zich aggers voordoen, is, afgezien van de dominantie, de vorm bepaald door de ruwe krommen; 
    dit in tegenstelling tot vroegere bepalingen. Bij spring- en doodtij is bovendien de differentiele getijduur, 
    en daarmee de duur rijzing, afgeleid uit de ruwe krommen.
    
    """
    #kwadraatsommen voor M2 tot M12
    components_av = ['M2','M4','M6','M8','M10','M12']
    comp_av = comp_frommeasurements_avg.loc[components_av]
    for comp_higherharmonics in components_av:
        iM = int(comp_higherharmonics[1:])
        bool_endswithiM = comp_frommeasurements_avg.index.str.endswith(str(iM)) & comp_frommeasurements_avg.index.str.replace(str(iM),'').str[-1].str.isalpha()
        comp_iM = comp_frommeasurements_avg.loc[bool_endswithiM]
        comp_av.loc[comp_higherharmonics,'A'] = np.sqrt((comp_iM['A']**2).sum()) #kwadraatsom
    
    comp_av.loc['A0'] = comp_frommeasurements_avg.loc['A0']
    
    logger.debug('verhouding tussen originele en kwadratensom componenten:\n'
                 f'{comp_av/comp_frommeasurements_avg.loc[components_av]}') # values are different than 1991.0 document and differs per station while the document states "Zoals te verwachten is de verhouding per component tussen deze wortel en de oorspronkelijke amplitude voor alle plaatsen gelijk"

    return comp_frommeasurements_avg, comp_av


def reshape_signal(ts, ts_ext, HW_goal, LW_goal, tP_goal=None):
    """
    scales tidal signal to provided HW/LW value and up/down going time
    tP_goal (tidal period time) is used to fix tidalperiod to 12h25m (for BOI timeseries)
    
    time_down was scaled with havengetallen before, but not anymore to avoid issues with aggers
    """
    # early escape # TODO: consider not calling function in this case
    if HW_goal is None and LW_goal is None:
        return ts
    
    # TODO: consider removing the need for ts_ext, it should be possible with min/max, although the HW of the raw timeseries are not exactly equal
    
    TR_goal = HW_goal-LW_goal
    
    #selecteer alle hoogwaters en opvolgende laagwaters
    bool_HW = ts_ext['HWLWcode']==1
    idx_HW = np.where(bool_HW)[0]
    timesHW = ts_ext.index[idx_HW]
    timesLW = ts_ext.index[idx_HW[:-1]+1] #assuming alternating 1,2,1 or 1,3,1, this is always valid in this workflow
    
    #crop from first to last HW (rest is not scaled anyway)
    ts_time_firstHW = ts_ext[bool_HW].index[0]
    ts_time_lastHW = ts_ext[bool_HW].index[-1]
    ts_corr = ts.copy().loc[ts_time_firstHW:ts_time_lastHW]

    ts_corr['times'] = ts_corr.index #this is necessary since datetimeindex with freq is not editable, and Series is editable
    for i in np.arange(0,len(timesHW)-1):
        HW1_val = ts_corr.loc[timesHW[i],'values']
        HW2_val = ts_corr.loc[timesHW[i+1],'values']
        LW_val = ts_corr.loc[timesLW[i],'values']
        TR1_val = HW1_val-LW_val
        TR2_val = HW2_val-LW_val
        tP_val = timesHW[i+1]-timesHW[i]
        if tP_goal is None:
            tP_goal = tP_val
        
        temp1 = (ts_corr.loc[timesHW[i]:timesLW[i],'values']-LW_val)/TR1_val*TR_goal+LW_goal
        temp2 = (ts_corr.loc[timesLW[i]:timesHW[i+1],'values']-LW_val)/TR2_val*TR_goal+LW_goal
        temp = pd.concat([temp1,temp2.iloc[1:]]) #.iloc[1:] since timesLW[i] is in both timeseries (values are equal)
        ts_corr['values_new'] = temp
        
        tide_HWtoHW = ts_corr.loc[timesHW[i]:timesHW[i+1]]
        ts_corr['times'] = pd.date_range(start=ts_corr.loc[timesHW[i],'times'],end=ts_corr.loc[timesHW[i],'times']+tP_goal,periods=len(tide_HWtoHW))
        
    ts_corr = ts_corr.set_index('times',drop=True)
    ts_corr['values'] = ts_corr['values_new']
    ts_corr = ts_corr.drop(['values_new'],axis=1)
    return ts_corr


def repeat_signal(ts_one_HWtoHW, nb, nf):
    """
    repeat tidal signal, necessary for sp/np, since they are computed as single tidal signal first
    """
    tidalperiod = ts_one_HWtoHW.index[-1] - ts_one_HWtoHW.index[0]
    ts_rep = pd.DataFrame()
    for iAdd in np.arange(-nb,nf+1):
        ts_add = pd.DataFrame({'values':ts_one_HWtoHW['values'].values},
                              index=ts_one_HWtoHW.index + iAdd*tidalperiod)
        ts_rep = pd.concat([ts_rep,ts_add])
    ts_rep = ts_rep.loc[~ts_rep.index.duplicated()]
    return ts_rep
