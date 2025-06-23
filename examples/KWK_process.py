# -*- coding: utf-8 -*-

import os
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
plt.close('all')
import hatyan
import kenmerkendewaarden as kw # pip install git+https://github.com/Deltares-research/kenmerkendewaarden

# set logging level to INFO to get log messages
import logging
logging.basicConfig() # calling basicConfig is essential to set logging level for sub-modules
logging.getLogger("kenmerkendewaarden").setLevel(level="INFO")

year_slotgem = 2021
print(f'year_slotgem: {year_slotgem}')

dir_base = r'p:\11210325-005-kenmerkende-waarden\work'
dir_meas = os.path.join(dir_base,'measurements_wl_18700101_20240101')

dir_indicators = os.path.join(dir_base,f'out_tidalindicators_{year_slotgem}')
dir_slotgem = os.path.join(dir_base,f'out_slotgemiddelden_{year_slotgem}')
dir_havget = os.path.join(dir_base,f'out_havengetallen_{year_slotgem}')
dir_gemgetij = os.path.join(dir_base,f'out_gemiddeldgetij_{year_slotgem}')
dir_overschrijding = os.path.join(dir_base,f'out_overschrijding_{year_slotgem}')
os.makedirs(dir_indicators, exist_ok=True)
os.makedirs(dir_slotgem, exist_ok=True)
os.makedirs(dir_havget, exist_ok=True)
os.makedirs(dir_gemgetij, exist_ok=True)
os.makedirs(dir_overschrijding, exist_ok=True)

fig_alltimes_ext = [dt.datetime.strptime(x,'%Y%m%d') for x in os.path.basename(dir_meas).split('_')[2:4]]

# all stations from TK (dataTKdia)
# TODO: maybe add from Dillingh 2013: DORDT, MAASMSMPL, PETTZD, ROTTDM (not downloaded)
station_list = ["A12","AWGPFM","BAALHK","BATH","BERGSDSWT","BROUWHVSGT02","BROUWHVSGT08","GATVBSLE","BRESKVHVN","CADZD",
                "D15","DELFZL","DENHDR","EEMSHVN","EURPFM","F16","F3PFM","HARVT10","HANSWT","HARLGN","HOEKVHLD","HOLWD","HUIBGT",
                "IJMDBTHVN","IJMDSMPL","J6","K13APFM","K14PFM","KATSBTN","KORNWDZBTN","KRAMMSZWT","L9PFM","LAUWOG","LICHTELGRE",
                "MARLGT","NES","NIEUWSTZL","NORTHCMRT","DENOVBTN","OOSTSDE04","OOSTSDE11","OOSTSDE14","OUDSD","OVLVHWT","Q1",
                "ROOMPBNN","ROOMPBTN","SCHAARVDND","SCHEVNGN","SCHIERMNOG","SINTANLHVSGR","STAVNSE","STELLDBTN","TERNZN","TERSLNZE","TEXNZE",
                "VLAKTVDRN","VLIELHVN","VLISSGN","WALSODN","WESTKPLE","WESTTSLG","WIERMGDN","YERSKE"]
# subset of 11 stations along the coast
station_list = ["VLISSGN","HOEKVHLD","IJMDBTHVN","HARLGN","DENHDR","DELFZL","SCHIERMNOG","VLIELHVN","STELLDBTN","SCHEVNGN","ROOMPBTN"]
# short list for testing
station_list = ["HOEKVHLD"]

stations_skip = []
# skip duplicate code stations from station_list_tk (hist/realtime)
# TODO: avoid this https://github.com/Rijkswaterstaat/wm-ws-dl/issues/12 and https://github.com/Rijkswaterstaat/wm-ws-dl/issues/20
stations_skip += ["BATH", "D15", "J6", "NES"]
# skip MSL/NAP duplicate stations from station_list_tk
# TODO: avoid this: https://github.com/Rijkswaterstaat/wm-ws-dl/issues/17
stations_skip += ["EURPFM", "LICHTELGRE", "K13APFM"]
# skip stations without extremes
stations_skip += ["A12", "AWGPFM", "F16", "F3PFM", "K14PFM", "L9PFM", "NORTHCMRT", "Q1"]
# skip stations that have no extremes before 2021-01-01
# TODO: https://github.com/Rijkswaterstaat/wm-ws-dl/issues/39
stations_skip += ["BAALHK", "GATVBSLE", "BRESKVHVN", "IJMDSMPL", "OVLVHWT", "VLAKTVDRN", "WALSODN"]
# skip STELLDBTN since it has only extremes from 1984 to 1996
# for KATSBTN/OOSTSDE04/OOSTSDE14/YERSKE no extremes between 1986 and 2020-12-31
# skip other stations with too little extremes in 2000-2020
# TODO: remove after fixing https://github.com/Rijkswaterstaat/wm-ws-dl/issues/39
stations_skip += ["STELLDBTN", "KATSBTN", "OOSTSDE04", "OOSTSDE11", "OOSTSDE14", "YERSKE"]
stations_skip += ["BROUWHVSGT02", "HOLWD", "MARLGT", "SCHAARVDND", "SINTANLHVSGR"]
# skip TEXNZE since it hass too little data in 2007 (at least lat+slotgemiddelden fail)
# TODO: remove after fixing https://github.com/Rijkswaterstaat/wm-ws-dl/issues/39
stations_skip += ["TEXNZE"]
# remove stations from station_list
for stat_remove in stations_skip:
    if stat_remove in station_list:
        print(f"removing {stat_remove} from station_list")
        station_list.remove(stat_remove)


nap_correction = False
min_coverage = 0.9
drop_duplicates = True

compute_indicators = True
compute_slotgem = True
compute_havengetallen = True
compute_gemgetij = True
compute_overschrijding = True


for current_station in station_list:
    print(f'starting process for {current_station}')
    plt.close('all')
    
    # timeseries are used for slotgemiddelden, gemgetijkrommen (needs slotgem+havget)
    df_meas_all = kw.read_measurements(dir_output=dir_meas, station=current_station, extremes=False, 
                                       nap_correction=nap_correction, drop_duplicates=drop_duplicates)
    # extremes are used for slotgemiddelden, havengetallen, overschrijding
    df_ext_12345_all = kw.read_measurements(dir_output=dir_meas, station=current_station, extremes=True,
                                      nap_correction=nap_correction, drop_duplicates=drop_duplicates)
    if df_meas_all is None or df_ext_12345_all is None:
        raise ValueError(f"missing data for {current_station}")
    
    # convert 12345 to 12 by taking minimum of 345 as 2 (laagste laagwater)
    # TODO: make calc_HWLW12345to12() faster: https://github.com/Deltares/hatyan/issues/311
    df_ext_all = hatyan.calc_HWLW12345to12(df_ext_12345_all)
    
    # crop measurement data to (excluding) year_slotgem
    df_meas_todate = df_meas_all.loc[:str(year_slotgem-1)]
    df_ext_todate = df_ext_all.loc[:str(year_slotgem-1)]
    
    # TODO: remove this filter again after fixing the source data
    # https://github.com/Deltares-research/kenmerkendewaarden/issues/196
    list_20201231 = [
        'BROUWHVSGT08', 'DELFZL', 'DENHDR', 'EEMSHVN', 'HARLGN', 'HOEKVHLD', 'HOLWD',
        'HUIBGT', 'IJMDBTHVN', 'KORNWDZBTN', 'LAUWOG', 'NIEUWSTZL', 'DENOVBTN',
        'ROOMPBTN', 'SCHEVNGN', 'TERSLNZE', 'WESTTSLG']
    if current_station in list_20201231:
        print(f"extremes after 2020-12-30 were removed for {current_station} as a "
              "temporary workaround")
        df_ext_todate = df_ext_todate.loc[:"2020-12-30"]
    
    
    #### TIDAL INDICATORS
    if compute_indicators:
        print(f'tidal indicators for {current_station}')
        # compute and plot tidal indicators
        dict_wltidalindicators = kw.calc_wltidalindicators(df_meas=df_meas_todate, min_coverage=min_coverage)
        dict_HWLWtidalindicators = kw.calc_HWLWtidalindicators(df_ext=df_ext_todate, min_coverage=min_coverage)
        # TODO: use all data after fixing https://github.com/Deltares-research/kenmerkendewaarden/issues/191
        df_ext_noduplicates = df_ext_todate.loc["1993":]
        dict_HWLW_springneap = kw.calc_HWLW_springneap(df_ext=df_ext_noduplicates, min_coverage=min_coverage)
        
        # add hat/lat
        hat, lat = kw.calc_hat_lat_frommeasurements(df_meas_todate)
        dict_HWLWtidalindicators["hat"] = hat
        dict_HWLWtidalindicators["lat"] = lat
        
        # merge dictionaries
        dict_wltidalindicators.update(dict_HWLWtidalindicators)
        dict_wltidalindicators.update(dict_HWLW_springneap)
        
        # csv for yearlymonthly indicators
        for key in ['wl_mean_peryear','wl_mean_permonth']:
            file_csv = os.path.join(dir_indicators, f'kw{year_slotgem}-{key}-{current_station}.csv')
            dict_wltidalindicators[key].to_csv(file_csv, float_format='%.3f')
        
        # plot
        fig, ax = kw.plot_tidalindicators(dict_wltidalindicators)
        fig.savefig(os.path.join(dir_indicators,f'kw{year_slotgem}-tidalindicators-{current_station}.png'))
    
    
    
    
    #### SLOTGEMIDDELDEN
    # TODO: more data is needed for proper working of fitting for some stations (2011: BAALHK, BRESKVHVN, GATVBSLE, SCHAARVDND) >> still after linear?
    if compute_slotgem:
        print(f'slotgemiddelden for {current_station}')
                
        # compute slotgemiddelden, exclude all values after tstop_dt (is year_slotgem)
        # including years with too little values and years before physical break
        slotgemiddelden_all = kw.calc_slotgemiddelden(df_meas=df_meas_todate, 
                                                      df_ext=df_ext_todate, 
                                                      min_coverage=0, clip_physical_break=True)
        # only years with enough values and after potential physical break
        slotgemiddelden_valid = kw.calc_slotgemiddelden(df_meas=df_meas_todate, 
                                                        df_ext=df_ext_todate, 
                                                        min_coverage=min_coverage, clip_physical_break=True)
        
        # plot slotgemiddelden
        fig1, ax1 = kw.plot_slotgemiddelden(slotgemiddelden_valid, slotgemiddelden_all)
        ax1.set_xlim(fig_alltimes_ext)

        # write slotgemiddelden to csv, the slotgemiddelde is the last value of the model fit
        key_list = ["wl_mean_peryear", "wl_model_fit",
                    "HW_mean_peryear", "HW_model_fit",
                    "LW_mean_peryear", "LW_model_fit",
                    "tidalrange_mean_peryear", "tidalrange_model_fit",
                    ]
        for key in key_list:
            file_csv = os.path.join(dir_slotgem, f'kw{year_slotgem}-{key}-{current_station}.csv')
            slotgemiddelden_valid[key].to_csv(file_csv, float_format='%.3f')
        
        # get and plot validation timeseries (yearly mean wl/HW/LW)
        station_name_dict = {'HOEKVHLD':'hoek',
                             'HARVT10':'ha10'}
        if current_station in station_name_dict.keys():
            dir_meas_gemHWLWwlAB = r'p:\archivedprojects\11208031-010-kenmerkende-waarden-k\work\data_KW-RMM'
            file_yearmeanHW = os.path.join(dir_meas_gemHWLWwlAB,f'{station_name_dict[current_station]}_hw.txt')
            file_yearmeanLW = os.path.join(dir_meas_gemHWLWwlAB,f'{station_name_dict[current_station]}_lw.txt')
            file_yearmeanwl = os.path.join(dir_meas_gemHWLWwlAB,f'{station_name_dict[current_station]}_Z.txt')
            csv_kwargs = dict(sep='\\s+', na_values=-999.9, skiprows=1, header=None, parse_dates=[0], index_col=0)
            yearmeanHW = pd.read_csv(file_yearmeanHW, **csv_kwargs)
            yearmeanLW = pd.read_csv(file_yearmeanLW, **csv_kwargs)
            yearmeanwl = pd.read_csv(file_yearmeanwl, **csv_kwargs)
            ax1.plot(yearmeanHW[1],'+g', zorder=0)
            ax1.plot(yearmeanLW[1],'+g', zorder=0)
            ax1.plot(yearmeanwl[1],'+g',label='yearmean validation', zorder=0)
            ax1.legend(loc=2)
        
        fig1.savefig(os.path.join(dir_slotgem,f'kw{year_slotgem}-slotgemiddelden-{current_station}'))
    
    
    
    
    ### HAVENGETALLEN 
    if compute_havengetallen:
        print(f'havengetallen for {current_station}')
        df_havengetallen, df_HWLW = kw.calc_havengetallen(df_ext=df_ext_todate, return_df_ext=True, min_coverage=min_coverage)
        
        # plot hwlw per timeclass including median
        fig, axs = kw.plot_HWLW_pertimeclass(df_ext=df_HWLW, df_havengetallen=df_havengetallen)
        fig.savefig(os.path.join(dir_havget,f'kw{year_slotgem}-HWLW_pertijdsklasse-{current_station}.png'))
        
        # plot aardappelgrafiek
        fig, (ax1,ax2) = kw.plot_aardappelgrafiek(df_havengetallen=df_havengetallen)
        fig.savefig(os.path.join(dir_havget, f'kw{year_slotgem}-aardappelgrafiek-{current_station}.png'))
        
        #write to csv
        file_csv = os.path.join(dir_havget, f'kw{year_slotgem}-havengetallen-{current_station}.csv')
        df_havengetallen.to_csv(file_csv, float_format='%.3f')
    
    
    
    
    ##### GEMIDDELDE GETIJKROMMEN
    if compute_gemgetij:
        print(f'gemiddelde getijkrommen for {current_station}')
        pred_freq = "10s" # frequency influences the accuracy of havengetallen-scaling and is writing frequency of BOI timeseries
        
        # derive getijkrommes: raw, scaled to havengetallen, scaled to havengetallen and 12h25min period
        gemgetij_raw = kw.calc_gemiddeldgetij(df_meas=df_meas_todate, df_ext=None,
                                              freq=pred_freq, nb=0, nf=0, 
                                              scale_extremes=False, scale_period=False,
                                              min_coverage=min_coverage)
        gemgetij_corr = kw.calc_gemiddeldgetij(df_meas=df_meas_todate, df_ext=df_ext_todate,
                                               freq=pred_freq, nb=1, nf=1, 
                                               scale_extremes=True, scale_period=False,
                                               min_coverage=min_coverage)
        gemgetij_corr_boi = kw.calc_gemiddeldgetij(df_meas=df_meas_todate, df_ext=df_ext_todate,
                                                   freq=pred_freq, nb=0, nf=4, 
                                                   scale_extremes=True, scale_period=True,
                                                   min_coverage=min_coverage)

        # TODO: the shape of the validation lines are different, so compare krommes to gele boekje instead
        # p:\archivedprojects\11205258-005-kpp2020_rmm-g5\C_Work\00_KenmerkendeWaarden\07_Figuren\figures_ppSCL_2\final20201211
        fig, ax = kw.plot_gemiddeldgetij(gemgetij_dict=gemgetij_corr, gemgetij_dict_raw=gemgetij_raw, tick_hours=6)
        fig.savefig(os.path.join(dir_gemgetij,f'kw{year_slotgem}-gemiddeldgetij-{current_station}.png'))
        
        # write corrected timeseries to csv files
        # TODO: better representation of negative timedeltas requested in https://github.com/pandas-dev/pandas/issues/17232#issuecomment-2205579156, maybe convert timedeltaIndex to minutes instead?
        for key in ['mean', 'spring', 'neap']:
            file_csv = os.path.join(dir_gemgetij, f'kw{year_slotgem}-gemiddeldgetij_{key}-{current_station}.csv')
            gemgetij_corr[key].to_csv(file_csv, float_format='%.3f')
        
        # plot BOI figure and compare to KW2020
        fig_boi, ax1_boi = kw.plot_gemiddeldgetij(gemgetij_dict=gemgetij_corr_boi, tick_hours=12)
        fig_boi.savefig(os.path.join(dir_gemgetij,f'kw{year_slotgem}-gemiddeldgetij_BOI-{current_station}.png'))
    
        # write BOI timeseries to csv files
        for key in ['mean', 'spring', 'neap']:
            file_boi_csv = os.path.join(dir_gemgetij, f'kw{year_slotgem}-gemiddeldgetij_BOI_{key}-{current_station}.csv')
            gemgetij_corr_boi[key].to_csv(file_boi_csv, float_format='%.3f')
    
    
    
    
    #### OVERSCHRIJDINGSFREQUENTIES
    # TODO: SLR trend correctie voor overschrijdingsfrequenties en evt ook voor andere KW?
    # TODO: resulting freqs seem to be shifted w.r.t. getijtafelboekje (mail PH 9-3-2022)
    # plots beoordelen: rode lijn moet ongeveer verlengde zijn van groene, als die ineens 
    # omhoog piekt komt dat door hele extreme waardes die je dan vermoedelijk ook al ziet in je groene lijn
    
    def initiate_dist_with_hydra_nl(station):
        """
        get Hydra-NL and KWK-RMM validation data (only available for selection of stations)
        """
        # TODO: this data is not reproducible yet: https://github.com/Deltares-research/kenmerkendewaarden/issues/107
        # TODO: HOEKVHLD Hydra values are different than old ones in validation line and p:\archivedprojects\11205258-005-kpp2020_rmm-g5\C_Work\00_KenmerkendeWaarden\Onder_overschrijdingslijnen_Boyan\Data\Processed_HydraNL

        dist_dict = {}
        dir_overschr_hydra = os.path.join(dir_base,'data_hydraNL')
        file_hydra_nl = os.path.join(dir_overschr_hydra, f'{station}.xls')
        if os.path.exists(file_hydra_nl):
            df_hydra_nl = pd.read_table(file_hydra_nl, encoding='latin-1', decimal=',', header=0)
            df_hydra_nl.index = 1/df_hydra_nl['Terugkeertijd [jaar]']
            df_hydra_nl.index.name = 'frequency'
            df_hydra_nl['values'] = df_hydra_nl['Belastingniveau [m+NAP]/Golfparameter [m]/[s]/Sterkte bekleding [-]'] * 100
            df_hydra_nl = df_hydra_nl[['values']]
            df_hydra_nl.attrs['station'] = station
            dist_dict['Hydra-NL'] = df_hydra_nl['values']
        return dist_dict

    def add_validation_dist(dist_dict, dist_type, station):
        station_names_vali_dict = {"HOEKVHLD":"Hoek_van_Holland"}
        if station not in station_names_vali_dict.keys():
            return
        dir_overschr_vali = r"p:\archivedprojects\11205258-005-kpp2020_rmm-g5\C_Work\00_KenmerkendeWaarden\Onder_overschrijdingslijnen_Boyan\Tables"
        file_validation = os.path.join(dir_overschr_vali, f'{dist_type}_lines', f'{dist_type}_lines_{station_names_vali_dict[station]}.csv')
        df_validation = pd.read_csv(file_validation, sep=';')
        df_validation = df_validation.rename({"value":"values"},axis=1)
        df_validation = df_validation.set_index("value_Tfreq", drop=True)
        df_validation.index.name = 'frequency'
        df_validation.attrs['station'] = station
        dist_dict['validation'] = df_validation['values']
    
    freqs_interested = [5, 2, 1, 1/2, 1/5, 1/10, 1/20, 1/50, 1/100, 1/200,
                         1/500, 1/1000, 1/2000, 1/4000, 1/5000, 1/10000]
    
    if compute_overschrijding:
        print(f'overschrijdingsfrequenties for {current_station}')
        
        # 1. Exceedance
        dist_exc_hydra = initiate_dist_with_hydra_nl(station=current_station)
        dist_exc = kw.calc_overschrijding(df_ext=df_ext_todate, rule_type=None, rule_value=None, 
                                          clip_physical_break=True, dist=dist_exc_hydra,
                                          interp_freqs=freqs_interested)
        add_validation_dist(dist_exc, dist_type='exceedance', station=current_station)
        dist_exc['geinterpoleerd'].to_csv(os.path.join(dir_overschrijding, f'kw{year_slotgem}-exceedance-{current_station}.csv'))
        
        fig, ax = kw.plot_overschrijding(dist_exc)
        ax.set_ylim(0,550)
        fig.savefig(os.path.join(dir_overschrijding, f'kw{year_slotgem}-exceedance-{current_station}.png'))
        
        # 2. Deceedance
        dist_dec = kw.calc_overschrijding(df_ext=df_ext_todate, rule_type=None, rule_value=None, 
                                          clip_physical_break=True, inverse=True,
                                          interp_freqs=freqs_interested)
        add_validation_dist(dist_dec, dist_type='deceedance', station=current_station)
        dist_dec['geinterpoleerd'].to_csv(os.path.join(dir_overschrijding, f'kw{year_slotgem}-deceedance-{current_station}.csv'))
        
        fig, ax = kw.plot_overschrijding(dist_dec)
        fig.savefig(os.path.join(dir_overschrijding, f'kw{year_slotgem}-deceedance-{current_station}.png'))

        # get n highest/lowest values
        df_ext_nhighest = kw.calc_highest_extemes(df_ext=df_ext_todate, num_extremes=5)
        df_ext_nhighest.to_csv(os.path.join(dir_overschrijding, f'kw{year_slotgem}-highest_extremes-{current_station}.csv'))
        df_ext_nlowest = kw.calc_highest_extemes(df_ext=df_ext_todate, num_extremes=5, ascending=True)
        df_ext_nlowest.to_csv(os.path.join(dir_overschrijding, f'kw{year_slotgem}-lowest_extremes-{current_station}.csv'))
