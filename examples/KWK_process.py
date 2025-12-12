# -*- coding: utf-8 -*-

import os
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
plt.close('all')
import hatyan
import kenmerkendewaarden as kw

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
station_list = ["a12", "ameland.westgat", "kloosterzande.baalhoek", "rilland.bath", 
                "tholen.bergsediepsluis.buiten", "brouwersdam.brouwershavensegat.2", 
                "brouwersdam.brouwershavensegat.8", "gatvanborssele", "breskens.veerhaven", 
                "cadzand.2", "d15", "delfzijl", "denhelder.marsdiep", "eemshaven.haven", 
                "europlatform", "f16", "f3", "haringvliet.10", "hansweert", "harlingen.waddenzee", 
                "hoekvanholland", "holwerd.veersteiger", "huibertgat", "ijmuiden.buitenhaven", 
                "ijgeul.1", "j6", "k13a", "k14", "kats.zandkreeksluis", "kornwerderzand.waddenzee.buitenhaven", 
                "krammersluizen.west", "l9", "lauwersoog.waddenzee", "goeree.lichteiland", "marollegat", 
                "ameland.nes", "nieuwestatenzijl.dollard", "north.cormorant", "denoever.waddenzee.voorhaven", 
                "oosterschelde.4", "oosterschelde.11", "oosterschelde.14", "texel.oudeschild", 
                "ossenisse", "q1.1", "oosterschelde.roompotsluis.binnen", "oosterschelde.roompotsluis.buiten", 
                "schaarvandenoord", "scheveningen", "schiermonnikoog.waddenzee", "sintannaland.havensteiger", 
                "stavenisse", "stellendam.buitenhaven", "terneuzen", "terschelling.noordzee", 
                "texel.noordzee", "vlaktevanderaan", "vlieland.haven", "vlissingen", "walsoorden", 
                "westkapelle", "terschelling.west", "wierumergronden", "yerseke"]

# subset of 11 stations along the coast
station_list = ["vlissingen", "hoekvanholland", "ijmuiden.buitenhaven", "harlingen.waddenzee", 
                "denhelder.marsdiep", "delfzijl", "schiermonnikoog.waddenzee", "vlieland.haven", 
                "stellendam.buitenhaven", "scheveningen", "oosterschelde.roompotsluis.buiten"]
# short list for testing
station_list = ["hoekvanholland","vlissingen"]

stations_skip = []
# skip duplicate code stations from station_list_tk (hist/realtime)
# TODO: avoid this https://github.com/Rijkswaterstaat/wm-ws-dl/issues/12 and https://github.com/Rijkswaterstaat/wm-ws-dl/issues/20
stations_skip += ["BATH", "D15", "J6", "NES"]
# skip MSL/NAP duplicate stations from station_list_tk
# TODO: avoid this: https://github.com/Rijkswaterstaat/wm-ws-dl/issues/17
stations_skip += ["EURPFM", "LICHTELGRE", "K13APFM"]
# skip stations without extremes
stations_skip += ["A12", "AWGPFM", "BAALHK", "F16", "F3PFM", "K14PFM", "L9PFM", "NORTHCMRT", "Q1"]
# skip stations that have no extremes before 2021-01-01
# TODO: remove after fixing https://github.com/Rijkswaterstaat/wm-ws-dl/issues/39
stations_skip += ["GATVBSLE", "BRESKVHVN", "IJMDSMPL", "OVLVHWT", "SINTANLHVSGR", "VLAKTVDRN", "WALSODN"]
# skip stations with too little extremes in 2000-2020
# TODO: remove after fixing https://github.com/Rijkswaterstaat/wm-ws-dl/issues/39
stations_skip += ["BROUWHVSGT02", "HOLWD", "KATSBTN", "MARLGT", "OOSTSDE04", "OOSTSDE11",
                  "OOSTSDE14", "SCHAARVDND", "STELLDBTN", "YERSKE"]
# skip TEXNZE for 2011.0 since it has too little meas/ext data in 2007
# skip BROUWHVSGT08 for 2011.0 since it has no ext data in 2010 (disappeared with DDL update of August 8)
# TODO: remove after fixing https://github.com/Rijkswaterstaat/wm-ws-dl/issues/39
if year_slotgem == 2011:
    stations_skip += ["TEXNZE", "BROUWHVSGT08"]
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
    df_meas_all = kw.read_measurements(dir_output=dir_meas, station=current_station, quantity="meas_wl", 
                                       nap_correction=nap_correction, drop_duplicates=drop_duplicates)
    # extremes are used for slotgemiddelden, havengetallen, overschrijding
    df_ext_12345_all = kw.read_measurements(dir_output=dir_meas, station=current_station, quantity="meas_ext",
                                      nap_correction=nap_correction, drop_duplicates=drop_duplicates)
    if df_meas_all is None or df_ext_12345_all is None:
        raise ValueError(f"missing data for {current_station}")
    
    # convert 12345 to 12 by taking minimum of 345 as 2 (laagste laagwater)
    df_ext_all = hatyan.calc_HWLW12345to12(df_ext_12345_all)
    
    # crop measurement data to (excluding) year_slotgem
    df_meas_todate = df_meas_all.loc[:str(year_slotgem-1)]
    df_ext_todate = df_ext_all.loc[:str(year_slotgem-1)]
    
    
    
    
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
        hat, lat = kw.calc_highest_lowest_astronomical_tide(df_meas_todate)
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

        fig, ax = kw.plot_gemiddeldgetij(gemgetij_dict=gemgetij_corr, gemgetij_dict_raw=gemgetij_raw, tick_hours=6)
        fig.savefig(os.path.join(dir_gemgetij,f'kw{year_slotgem}-gemiddeldgetij-{current_station}.png'))
        
        # write corrected timeseries to csv files
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
    # TODO: simplify input: https://github.com/Deltares-research/kenmerkendewaarden/issues/252
    # plots beoordelen: rode lijn moet ongeveer verlengde zijn van groene, als die ineens 
    # omhoog piekt komt dat door hele extreme waardes die je dan vermoedelijk ook al ziet in je groene lijn
    
    def initiate_dist_with_hydra_nl(station):
        """
        get Hydra-NL and KWK-RMM validation data (only available for selection of stations)
        """
        # TODO: this data is not reproducible yet: https://github.com/Deltares-research/kenmerkendewaarden/issues/107 (and #155)

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
                                          clip_physical_break=True,
                                          correct_trend=True, min_coverage=0.9,
                                          dist=dist_exc_hydra,
                                          interp_freqs=freqs_interested)
        add_validation_dist(dist_exc, dist_type='exceedance', station=current_station)
        dist_exc['geinterpoleerd'].to_csv(os.path.join(dir_overschrijding, f'kw{year_slotgem}-exceedance-{current_station}.csv'))
        
        fig, ax = kw.plot_overschrijding(dist_exc)
        ax.set_ylim(0,550)
        fig.savefig(os.path.join(dir_overschrijding, f'kw{year_slotgem}-exceedance-{current_station}.png'))
        
        # 2. Deceedance
        dist_dec = kw.calc_overschrijding(df_ext=df_ext_todate, rule_type=None, rule_value=None, 
                                          clip_physical_break=True,
                                          correct_trend=True, min_coverage=0.9,
                                          inverse=True,
                                          interp_freqs=freqs_interested)
        add_validation_dist(dist_dec, dist_type='deceedance', station=current_station)
        dist_dec['geinterpoleerd'].to_csv(os.path.join(dir_overschrijding, f'kw{year_slotgem}-deceedance-{current_station}.csv'))
        
        fig, ax = kw.plot_overschrijding(dist_dec)
        fig.savefig(os.path.join(dir_overschrijding, f'kw{year_slotgem}-deceedance-{current_station}.png'))

        # get n highest/lowest values
        df_ext_nhighest = kw.calc_highest_extremes(df_ext=df_ext_todate, num_extremes=5)
        df_ext_nhighest.to_csv(os.path.join(dir_overschrijding, f'kw{year_slotgem}-highest_extremes-{current_station}.csv'))
        df_ext_nlowest = kw.calc_highest_extremes(df_ext=df_ext_todate, num_extremes=5, ascending=True)
        df_ext_nlowest.to_csv(os.path.join(dir_overschrijding, f'kw{year_slotgem}-lowest_extremes-{current_station}.csv'))
