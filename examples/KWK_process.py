# -*- coding: utf-8 -*-

import os
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
plt.close('all')
import hatyan
import kenmerkendewaarden as kw # pip install git+https://github.com/Deltares-research/kenmerkendewaarden

# TODO: HW/LW numbers not always increasing (at havengetallen): ['HANSWT','BROUWHVSGT08','PETTZD','DORDT']

tstart_dt = pd.Timestamp(2011,1,1, tz="UTC+01:00")
tstop_dt = pd.Timestamp(2021,1,1, tz="UTC+01:00")
tstop_dt_naive = tstop_dt.tz_localize(None)
if ((tstop_dt.year-tstart_dt.year)==10) & (tstop_dt.month==tstop_dt.day==tstart_dt.month==tstart_dt.day==1):
    year_slotgem = tstop_dt.year
else:
    year_slotgem = 'invalid'
print(f'year_slotgem: {year_slotgem}')

# dir_base = r'p:\11208031-010-kenmerkende-waarden-k\work'
dir_base = r'p:\11210325-005-kenmerkende-waarden\work'
dir_meas = os.path.join(dir_base,'measurements_wl_18700101_20240101')
# TODO: move to full data folder (otherwise overschrijding and slotgemiddelde is completely wrong)
# dir_meas = os.path.join(dir_base,'measurements_wl_20101201_20220201')

dir_havget = os.path.join(dir_base,f'out_havengetallen_{year_slotgem}')
dir_slotgem = os.path.join(dir_base,f'out_slotgem_{year_slotgem}')
dir_gemgetij = os.path.join(dir_base,f'out_gemgetij_{year_slotgem}')
dir_overschrijding = os.path.join(dir_base,f'out_overschrijding_{year_slotgem}')
os.makedirs(dir_havget, exist_ok=True)
os.makedirs(dir_slotgem, exist_ok=True)
os.makedirs(dir_gemgetij, exist_ok=True)
os.makedirs(dir_overschrijding, exist_ok=True)

fig_alltimes_ext = [dt.datetime.strptime(x,'%Y%m%d') for x in os.path.basename(dir_meas).split('_')[2:4]]

# all stations from TK (dataTKdia)
station_list = ['A12','AWGPFM','BAALHK','BATH','BERGSDSWT','BROUWHVSGT02','BROUWHVSGT08','GATVBSLE','BRESKVHVN','CADZD',
                'D15','DELFZL','DENHDR','EEMSHVN','EURPFM','F16','F3PFM','HARVT10','HANSWT','HARLGN','HOEKVHLD','HOLWD','HUIBGT',
                'IJMDBTHVN','IJMDSMPL','J6','K13APFM','K14PFM','KATSBTN','KORNWDZBTN','KRAMMSZWT','L9PFM','LAUWOG','LICHTELGRE',
                'MARLGT','NES','NIEUWSTZL','NORTHCMRT','DENOVBTN','OOSTSDE04','OOSTSDE11','OOSTSDE14','OUDSD','OVLVHWT','Q1',
                'ROOMPBNN','ROOMPBTN','SCHAARVDND','SCHEVNGN','SCHIERMNOG','SINTANLHVSGR','STAVNSE','STELLDBTN','TERNZN','TERSLNZE','TEXNZE',
                'VLAKTVDRN','VLIELHVN','VLISSGN','WALSODN','WESTKPLE','WESTTSLG','WIERMGDN','YERSKE']
# TODO: maybe add from Dillingh 2013: DORDT, MAASMSMPL, PETTZD, ROTTDM
stat_list = ['HOEKVHLD']#,'HARVT10','VLISSGN']



# TODO: move to csv file and add as package data
#physical_break_dict for slotgemiddelden and overschrijdingsfrequenties TODO: maybe use everywhere to crop data?
physical_break_dict = {'DENOVBTN':'1933', #laatste sluitgat afsluitdijk in 1932 
                       'HARLGN':'1933', #laatste sluitgat afsluitdijk in 1932
                       'VLIELHVN':'1933', #laatste sluitgat afsluitdijk in 1932
                       } #TODO: add physical_break for STAVNSE and KATSBTN? (Oosterscheldekering)

nap_correction = False

compute_slotgem = True
compute_havengetallen = True
compute_gemgetij = True
compute_overschrijding = True


for current_station in stat_list:
    plt.close('all')
    
    print(f'loading data for {current_station}')
    # timeseries are used for slotgemiddelden, gemgetijkrommen (needs slotgem+havget)
    data_pd_meas_all = kw.read_measurements(dir_output=dir_meas, station=current_station, extremes=False, nap_correction=nap_correction)
    if data_pd_meas_all is not None:
        #crop measurement data
        data_pd_meas_10y = hatyan.crop_timeseries(data_pd_meas_all, times=slice(tstart_dt,tstop_dt-dt.timedelta(minutes=10)))#,onlyfull=False)
    
    # extremes are used for slotgemiddelden, havengetallen, overschrijding
    data_pd_HWLW_all = kw.read_measurements(dir_output=dir_meas, station=current_station, extremes=True, nap_correction=nap_correction)
    if data_pd_HWLW_all is not None:
        # TODO: make calc_HWLW12345to12() faster: https://github.com/Deltares/hatyan/issues/311
        data_pd_HWLW_all_12 = hatyan.calc_HWLW12345to12(data_pd_HWLW_all) #convert 12345 to 12 by taking minimum of 345 as 2 (laagste laagwater)
        #crop timeseries to 10y
        data_pd_HWLW_10y_12 = hatyan.crop_timeseries(data_pd_HWLW_all_12, times=slice(tstart_dt,tstop_dt),onlyfull=False)
        
        #check if amount of HWs is enough
        M2_period_timedelta = pd.Timedelta(hours=hatyan.schureman.get_schureman_freqs(['M2']).loc['M2','period [hr]'])
        numHWs_expected = (tstop_dt-tstart_dt).total_seconds()/M2_period_timedelta.total_seconds()
        numHWs = (data_pd_HWLW_10y_12['HWLWcode']==1).sum()
        if numHWs < 0.95*numHWs_expected:
            raise Exception(f'ERROR: not enough high waters present in period, {numHWs} instead of >=0.95*{int(numHWs_expected):d}')



    #### SLOTGEMIDDELDEN
    #TODO: nodal cycle is not in same phase for all stations, this is not physically correct.
    #TODO: more data is needed for proper working of fitting for some stations (2011: BAALHK, BRESKVHVN, GATVBSLE, SCHAARVDND)
    if compute_slotgem and data_pd_meas_all is not None:
        print(f'slotgemiddelden for {current_station}')
        
        #calculate yearly mean
        dict_wltidalindicators = kw.calc_wltidalindicators(data_pd_meas_all)
        wl_mean_peryear = dict_wltidalindicators['wl_mean_peryear']
        dict_wltidalindicators_valid = kw.calc_wltidalindicators(data_pd_meas_all, min_count=2900) #24*365=8760 (hourly interval), 24/3*365=2920 (3-hourly interval)
        wl_mean_peryear_valid = dict_wltidalindicators_valid['wl_mean_peryear']
        
        #derive tidal indicators like yearmean HWLW from HWLW values
        if data_pd_HWLW_all is not None:
            dict_HWLWtidalindicators = kw.calc_HWLWtidalindicators(data_pd_HWLW_all_12)
            HW_mean_peryear = dict_HWLWtidalindicators['HW_mean_peryear']
            LW_mean_peryear = dict_HWLWtidalindicators['LW_mean_peryear']
            dict_HWLWtidalindicators_valid = kw.calc_HWLWtidalindicators(data_pd_HWLW_all_12, min_count=1400) #2*24*365/12.42=1410.6 (12.42 hourly extreme)
            HW_mean_peryear_valid = dict_HWLWtidalindicators_valid['HW_mean_peryear']
            LW_mean_peryear_valid = dict_HWLWtidalindicators_valid['LW_mean_peryear']
        
        #plotting (yearly averages are plotted on 1jan)
        fig,ax1 = plt.subplots(figsize=(12,6))
        
        #get and plot validation timeseries (yearly mean wl/HW/LW)
        station_name_dict = {'HOEKVHLD':'hoek',
                             'HARVT10':'ha10'}
        if current_station in station_name_dict.keys():
            dir_meas_gemHWLWwlAB = r'p:\archivedprojects\11208031-010-kenmerkende-waarden-k\work\data_KW-RMM'
            file_yearmeanHW = os.path.join(dir_meas_gemHWLWwlAB,f'{station_name_dict[current_station]}_hw.txt')
            file_yearmeanLW = os.path.join(dir_meas_gemHWLWwlAB,f'{station_name_dict[current_station]}_lw.txt')
            file_yearmeanwl = os.path.join(dir_meas_gemHWLWwlAB,f'{station_name_dict[current_station]}_Z.txt')
            yearmeanHW = pd.read_csv(file_yearmeanHW, sep='\\s+', skiprows=1, names=['datetime','values'], parse_dates=['datetime'], na_values=-999.9, index_col='datetime')/100
            yearmeanLW = pd.read_csv(file_yearmeanLW, sep='\\s+', skiprows=1, names=['datetime','values'], parse_dates=['datetime'], na_values=-999.9, index_col='datetime')/100
            yearmeanwl = pd.read_csv(file_yearmeanwl, sep='\\s+', skiprows=1, names=['datetime','values'], parse_dates=['datetime'], na_values=-999.9, index_col='datetime')/100
            ax1.plot(yearmeanHW['values'],'+g')
            ax1.plot(yearmeanLW['values'],'+g')
            ax1.plot(yearmeanwl['values'],'+g',label='yearmean validation')
        
        #plot values
        if data_pd_HWLW_all is not None:
            ax1.plot(HW_mean_peryear,'x',color='grey')
            ax1.plot(LW_mean_peryear,'x',color='grey')
            ax1.plot(HW_mean_peryear_valid,'xr')
            ax1.plot(LW_mean_peryear_valid,'xr')
        ax1.plot(wl_mean_peryear,'x',color='grey',label='yearmean')
        ax1.plot(wl_mean_peryear_valid,'xr',label='yearmean significant')
        ax1.grid()
        ax1.set_xlim(fig_alltimes_ext) # entire period
        ax1.set_ylabel('waterstand [m]')
        ax1.set_title(f'yearly mean HW/wl/LW {current_station}')
        fig.tight_layout()
        
        if current_station in physical_break_dict.keys():
            tstart_dt_trend = physical_break_dict[current_station]
        else:
            tstart_dt_trend = None
        
        #fit linear models over yearly mean values
        wl_mean_array_todate = wl_mean_peryear_valid.loc[tstart_dt_trend:tstop_dt_naive] #remove all values after tstop_dt (is year_slotgem)
        pred_pd_wl = kw.fit_models(wl_mean_array_todate)
        ax1.plot(pred_pd_wl, ".-", label=pred_pd_wl.columns)
        ax1.set_prop_cycle(None) #reset matplotlib colors
        #2021.0 value (and future)
        ax1.plot(pred_pd_wl.loc[tstop_dt_naive:,'pred_linear_winodal'], ".k", label=f'pred_linear from {year_slotgem}')
        pred_slotgem = pred_pd_wl.loc[[tstop_dt_naive],['pred_linear_winodal']]
        pred_slotgem.to_csv(os.path.join(dir_slotgem,f'slotgem_value_{current_station}.txt'))
        ax1.legend(loc=2)
        
        if data_pd_HWLW_all is not None:
            HW_mean_array_todate = HW_mean_peryear_valid.loc[tstart_dt_trend:tstop_dt_naive] #remove all values after tstop_dt (is year_slotgem)
            pred_pd_HW = kw.fit_models(HW_mean_array_todate)
            ax1.plot(pred_pd_HW, ".-", label=pred_pd_HW.columns)
            ax1.set_prop_cycle(None) #reset matplotlib colors
            
            LW_mean_array_todate = LW_mean_peryear_valid.loc[tstart_dt_trend:tstop_dt_naive] #remove all values after tstop_dt (is year_slotgem)
            pred_pd_LW = kw.fit_models(LW_mean_array_todate)
            ax1.plot(pred_pd_LW, ".-", label=pred_pd_LW.columns)
            ax1.set_prop_cycle(None) #reset matplotlib colors

        fig.savefig(os.path.join(dir_slotgem,f'yearly_values_{current_station}'))
    
    
    
    
    ### HAVENGETALLEN 
    if compute_havengetallen and data_pd_HWLW_all is not None:
        
        print(f'havengetallen for {current_station}')
        # TODO: havengetallen are different than p:\archivedprojects\11208031-010-kenmerkende-waarden-k\work\out_havengetallen_2021\havengetallen_2021_HOEKVHLD.csv
        
        # TODO: check culm_addtime and HWLWno+4 offsets. culm_addtime could also be 2 days or 2days +1h GMT-MET correction. 20 minutes seems odd since moonculm is about tidal wave from ocean
        # culm_addtime is a 2d and 2u20min correction, this shifts the x-axis of aardappelgrafiek
        # HW is 2 days after culmination (so 4x25min difference between length of avg moonculm and length of 2 days)
        # 1 hour (GMT to MET, alternatively we can also account for timezone differences elsewhere)
        # 20 minutes (0 to 5 meridian)
        culm_addtime = 4*dt.timedelta(hours=12,minutes=25) + dt.timedelta(hours=1) - dt.timedelta(minutes=20)
        
        # TODO: move calc_HWLW_moonculm_combi() to top since it is the same for all stations
        # TODO: we added tz_localize on 29-5-2024 (https://github.com/Deltares-research/kenmerkendewaarden/issues/30)
        # this means we pass a UTC+1 timeseries as if it were a UTC timeseries
        data_pd_HWLW = kw.calc_HWLW_moonculm_combi(data_pd_HWLW_12=data_pd_HWLW_10y_12.tz_localize(None), culm_addtime=culm_addtime) #culm_addtime=None provides the same gemgetijkromme now delay is not used for scaling anymore
        HWLW_culmhr_summary = kw.calc_HWLW_culmhr_summary(data_pd_HWLW) #TODO: maybe add tijverschil
        
        print('HWLW FIGUREN PER TIJDSKLASSE, INCLUSIEF MEDIAN LINE')
        fig, axs = kw.plot_HWLW_pertimeclass(data_pd_HWLW, HWLW_culmhr_summary)
        fig.savefig(os.path.join(dir_havget,f'HWLW_pertijdsklasse_inclmedianline_{current_station}'))
        
        print('AARDAPPELGRAFIEK')
        fig, (ax1,ax2) = kw.plot_aardappelgrafiek(HWLW_culmhr_summary)
        fig.savefig(os.path.join(dir_havget, f'aardappelgrafiek_{year_slotgem}_{current_station}'))
        
        #write to csv
        HWLW_culmhr_summary_exp = HWLW_culmhr_summary.loc[[6,'mean',0]] #select neap/mean/springtide
        HWLW_culmhr_summary_exp.index = ['neap','mean','spring']
        HWLW_culmhr_summary_exp.to_csv(os.path.join(dir_havget, f'havengetallen_{year_slotgem}_{current_station}.csv'),float_format='%.3f')
    
    
    
    
    
    ##### GEMIDDELDE GETIJKROMMEN
    if compute_gemgetij and data_pd_meas_all is not None:
        
        print(f'gem getijkrommen for {current_station}')
        pred_freq = "10s" #TODO: frequency decides accuracy of tU/tD and other timings (and is writing freq of BOI timeseries)
        file_havget = os.path.join(dir_havget,f'havengetallen_{year_slotgem}_{current_station}.csv')

        # derive getijkrommes: raw, scaled to havengetallen, scaled to havengetallen and 12h25min period
        prediction_av, prediction_sp, prediction_np = kw.gemiddeld_getij_av_sp_np(
                                        df_meas=data_pd_meas_10y, pred_freq=pred_freq, nb=0, na=0, 
                                        scale_extremes=False, scale_period=False)
        prediction_av_corr, prediction_sp_corr, prediction_np_corr = kw.gemiddeld_getij_av_sp_np(
                                        df_meas=data_pd_meas_10y, pred_freq=pred_freq, nb=2, na=2, 
                                        scale_extremes=file_havget, scale_period=False)
        prediction_av_corr_boi, prediction_sp_corr_boi, prediction_np_corr_boi = kw.gemiddeld_getij_av_sp_np(
                                        df_meas=data_pd_meas_10y, pred_freq=pred_freq, nb=0, na=10, 
                                        scale_extremes=file_havget, scale_period=True)

        # write boi timeseries to csv files # TODO: maybe convert timedeltaIndex to minutes instead?
        prediction_av_corr_boi.to_csv(os.path.join(dir_gemgetij,f'gemGetijkromme_BOI_{current_station}_slotgem{year_slotgem}.csv'),float_format='%.3f',date_format='%Y-%m-%d %H:%M:%S')
        prediction_sp_corr_boi.to_csv(os.path.join(dir_gemgetij,f'springtijkromme_BOI_{current_station}_slotgem{year_slotgem}.csv'),float_format='%.3f',date_format='%Y-%m-%d %H:%M:%S')
        prediction_np_corr_boi.to_csv(os.path.join(dir_gemgetij,f'doodtijkromme_BOI_{current_station}_slotgem{year_slotgem}.csv'),float_format='%.3f',date_format='%Y-%m-%d %H:%M:%S')
        
        
        cmap = plt.get_cmap("tab10")
            
        print(f'plot getijkromme trefHW: {current_station}')
        fig_sum,ax_sum = plt.subplots(figsize=(14,7))
        ax_sum.set_title(f'getijkromme trefHW {current_station}')
        prediction_av['values'].plot(ax=ax_sum, linestyle='--', color=cmap(0), linewidth=0.7, label='gem kromme, one')
        prediction_av_corr['values'].plot(ax=ax_sum, color=cmap(0), label='gem kromme, corr')
        prediction_sp['values'].plot(ax=ax_sum, linestyle='--', color=cmap(1), linewidth=0.7, label='sp kromme, one')
        prediction_sp_corr['values'].plot(ax=ax_sum, color=cmap(1), label='sp kromme, corr')
        prediction_np['values'].plot(ax=ax_sum, linestyle='--', color=cmap(2), linewidth=0.7, label='np kromme, one')
        prediction_np_corr['values'].plot(ax=ax_sum, color=cmap(2), label='np kromme, corr')
        ax_sum.set_xticks([x*3600e9 for x in range(-15, 25, 5)]) # nanoseconds units # TODO: make multiple of 12
        ax_sum.legend(loc=4)
        ax_sum.grid()
        ax_sum.set_xlim([x*3600e9 for x in [-15.5,15.5]])
        ax_sum.set_xlabel('hours since HW (ts are shifted to this reference)')
        fig_sum.tight_layout()
        fig_sum.savefig(os.path.join(dir_gemgetij,f'gemgetij_trefHW_{current_station}'))
        
        print(f'plot BOI figure and compare to KW2020: {current_station}')
        fig_boi,ax1_boi = plt.subplots(figsize=(14,7))
        ax1_boi.set_title(f'getijkromme BOI {current_station}')
        #plot gemtij/springtij/doodtij
        prediction_av_corr_boi['values'].plot(ax=ax1_boi,color=cmap(0),label='prediction gemtij')
        prediction_sp_corr_boi['values'].plot(ax=ax1_boi,color=cmap(1),label='prediction springtij')
        prediction_np_corr_boi['values'].plot(ax=ax1_boi,color=cmap(2),label='prediction doodtij')
        ax1_boi.set_xticks([x*3600e9 for x in range(0, 6*24, 12)]) # nanoseconds units
        
        #plot validation lines if available
        dir_vali_krommen = r'p:\archivedprojects\11205258-005-kpp2020_rmm-g5\C_Work\00_KenmerkendeWaarden\07_Figuren\figures_ppSCL_2\final20201211'
        file_vali_doodtijkromme = os.path.join(dir_vali_krommen,f'doodtijkromme_{current_station}_havengetallen{year_slotgem}.csv')
        file_vali_gemtijkromme = os.path.join(dir_vali_krommen,f'gemGetijkromme_{current_station}_havengetallen{year_slotgem}.csv')
        file_vali_springtijkromme = os.path.join(dir_vali_krommen,f'springtijkromme_{current_station}_havengetallen{year_slotgem}.csv')        
        if os.path.exists(file_vali_gemtijkromme):
            data_vali_gemtij = pd.read_csv(file_vali_gemtijkromme,index_col=0,parse_dates=True)
            ax1_boi.plot(data_vali_gemtij['Water Level [m]'],'--',color=cmap(0),linewidth=0.7,label='validation KW2020 gemtij')
        if os.path.exists(file_vali_springtijkromme):
            data_vali_springtij = pd.read_csv(file_vali_springtijkromme,index_col=0,parse_dates=True)
            ax1_boi.plot(data_vali_springtij['Water Level [m]'],'--',color=cmap(1),linewidth=0.7,label='validation KW2020 springtij')
        if os.path.exists(file_vali_doodtijkromme):
            data_vali_doodtij = pd.read_csv(file_vali_doodtijkromme,index_col=0,parse_dates=True)
            ax1_boi.plot(data_vali_doodtij['Water Level [m]'],'--',color=cmap(2),linewidth=0.7, label='validation KW2020 doodtij')
        
        ax1_boi.grid()
        ax1_boi.legend(loc=4)
        ax1_boi.set_xlabel('times since first av HW (start of ts)')
        ax1_boi.set_xlim([x*3600e9 for x in [-2-4, 48-4]]) # TODO: make nicer xrange
        fig_boi.tight_layout()
        fig_boi.savefig(os.path.join(dir_gemgetij,f'gemspringdoodtijkromme_BOI_{current_station}_slotgem{year_slotgem}.png'))
    
    
    
    
    
    ###OVERSCHRIJDINGSFREQUENTIES
    #TODO: SLR trend correctie voor overschrijdingsfrequenties en evt ook voor andere KW?
    #TODO: resulting freqs seem to be shifted w.r.t. getijtafelboekje (mail PH 9-3-2022)
    #plots beoordelen: rode lijn moet ongeveer verlengde zijn van groene, als die ineens omhoog piekt komt dat door hele extreme waardes die je dan vermoedelijk ook al ziet in je groene lijn
    
    Tfreqs_interested = [5, 2, 1, 1/2, 1/5, 1/10, 1/20, 1/50, 1/100, 1/200, #overschrijdingsfreqs
                         1/500, 1/1000, 1/2000, 1/4000, 1/5000, 1/10000] #TODO: which frequencies are realistic with n years of data? probably remove this entire row >> met 40 jaar data kun je in principe tot 1/40 gaan, maar met weibull kun je extrapoleren en in theorie >> dit is voor tabel die je eruit wil hebben
    
    if compute_overschrijding and data_pd_HWLW_all is not None:
    
        print(f'overschrijdingsfrequenties for {current_station}')
        
        #clip data #TODO: do at top?
        data_pd_measext = data_pd_HWLW_all_12.loc[:tstop_dt] # only include data up to year_slotgem
        
        data_pd_HW = data_pd_measext.loc[data_pd_measext['HWLWcode']==1]
        data_pd_LW = data_pd_measext.loc[data_pd_measext['HWLWcode']!=1]
        
        #get Hydra-NL and KWK-RMM validation data (only for HOEKVHLD)
        dist_vali_exc = {}
        dist_vali_dec = {}
        if current_station =='HOEKVHLD':
            dir_vali_overschr = os.path.join(dir_base,'data_overschrijding') # TODO: this data is not reproducible yet
            stat_name = 'Hoek_van_Holland'
            print('Load Hydra-NL distribution data and other validation data')
            dist_vali_exc = {}
            dist_vali_exc['Hydra-NL'] = pd.read_csv(os.path.join(dir_vali_overschr,'Processed_HydraNL','Without_model_uncertainty',f'{stat_name}.csv'), sep=';', header=[0])
            dist_vali_exc['Hydra-NL']['values'] /= 100 # cm to m
            dist_vali_exc['Hydra-NL met modelonzekerheid'] = pd.read_csv(os.path.join(dir_vali_overschr,'Processed_HydraNL','With_model_uncertainty',f'{stat_name}_with_model_uncertainty.csv'), sep=';', header=[0])
            dist_vali_exc['Hydra-NL met modelonzekerheid']['values'] /= 100 # cm to m
            file_vali_exeed = os.path.join(dir_vali_overschr,'Tables','Exceedance_lines',f'Exceedance_lines_{stat_name}.csv')
            if os.path.exists(file_vali_exeed):
                dist_vali_exc['validation'] = pd.read_csv(file_vali_exeed,sep=';')
                dist_vali_exc['validation']['values'] /= 100
            file_vali_dec = os.path.join(dir_vali_overschr,'Tables','Deceedance_lines',f'Deceedance_lines_{stat_name}.csv')
            if os.path.exists(file_vali_dec):
                dist_vali_dec['validation'] = pd.read_csv(file_vali_dec,sep=';')
                dist_vali_dec['validation']['values'] /= 100
    
        #set station rules
        station_rule_type = 'break'
        if current_station in physical_break_dict.keys(): 
            station_break_value = physical_break_dict[current_station] #TODO: maybe better to just not select the data by doing data_pd_measext.loc[station_break_value:tstop] instead of data_pd_measext.loc[:tstop]
        else:
            station_break_value = data_pd_measext.index.min()
    
        # 1. Exceedance
        print('Exceedance')
        dist_exc = kw.compute_overschrijding(data_pd_HW, rule_type=station_rule_type, rule_value=station_break_value)
        dist_exc.update(dist_vali_exc)
        df_interp = kw.interpolate_interested_Tfreqs(dist_exc['Gecombineerd'], Tfreqs=Tfreqs_interested)
        df_interp.to_csv(os.path.join(dir_overschrijding, f'Exceedance_{current_station}.csv'), index=False, sep=';')
        
        fig, ax = kw.plot_distributions(dist_exc, name=current_station, color_map='default')
        ax.set_ylim(0,5.5)
        fig.savefig(os.path.join(dir_overschrijding, f'Exceedance_lines_{current_station}.png'))
        
        # 2. Deceedance
        print('Deceedance')
        dist_dec = kw.compute_overschrijding(data_pd_LW, rule_type=station_rule_type, rule_value=station_break_value, inverse=True)
        dist_dec.update(dist_vali_dec)
        df_interp = kw.interpolate_interested_Tfreqs(dist_dec['Gecombineerd'], Tfreqs=Tfreqs_interested)
        df_interp.to_csv(os.path.join(dir_overschrijding, f'Deceedance_{current_station}.csv'), index=False, sep=';')
        
        fig, ax = kw.plot_distributions(dist_dec, name=current_station, color_map='default')
        fig.savefig(os.path.join(dir_overschrijding, f'Deceedance_lines_{current_station}.png'))

