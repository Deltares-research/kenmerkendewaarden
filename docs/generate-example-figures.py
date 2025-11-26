# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 10:21:08 2025

@author: veenstra
"""

import os
import matplotlib.pyplot as plt
plt.close('all')
import hatyan
import kenmerkendewaarden as kw

# set logging level to INFO to get log messages
import logging
logging.basicConfig() # calling basicConfig is essential to set logging level for sub-modules
logging.getLogger("kenmerkendewaarden").setLevel(level="INFO")

year_slotgem = 2021

dir_figures = 'example-figures'
os.makedirs(dir_figures, exist_ok=True)

nap_correction = False
min_coverage = 0.9
drop_duplicates = True

current_station = "HOEKVHLD"


# READ MEASUREMENT DATA
dir_testdata = "../tests/testdata"
file_dia_ts = os.path.join(dir_testdata, "HOEK_KW.dia")
df_meas_all = hatyan.read_dia(file_dia_ts)
df_meas_all["values"] *= 100
df_meas_all.attrs["eenheid"] = "cm"
file_dia_ext = os.path.join(dir_testdata, "HOEKVHLD_ext.dia")
df_ext_12345_all = hatyan.read_dia(file_dia_ext, station="HOEKVHLD", block_ids="allstation")
df_ext_12345_all["values"] *= 100
df_ext_12345_all.attrs["eenheid"] = "cm"

# convert 12345 to 12 by taking minimum of 345 as 2 (laagste laagwater)
df_ext_all = hatyan.calc_HWLW12345to12(df_ext_12345_all)

# crop measurement data to (excluding) year_slotgem
df_meas_todate = df_meas_all.loc[:str(year_slotgem-1)]
df_ext_todate = df_ext_all.loc[:str(year_slotgem-1)]


#### TIDAL INDICATORS
# compute and plot tidal indicators
dict_wltidalindicators = kw.calc_wltidalindicators(df_meas=df_meas_todate, min_coverage=min_coverage)
dict_HWLWtidalindicators = kw.calc_HWLWtidalindicators(df_ext=df_ext_todate, min_coverage=min_coverage)
df_ext_noduplicates = df_ext_todate#.loc["1993":]
dict_HWLW_springneap = kw.calc_HWLW_springneap(df_ext=df_ext_noduplicates, min_coverage=min_coverage)

# add hat/lat
hat, lat = kw.calc_highest_lowest_astronomical_tide(df_meas_todate)
dict_HWLWtidalindicators["hat"] = hat
dict_HWLWtidalindicators["lat"] = lat

# merge dictionaries
dict_wltidalindicators.update(dict_HWLWtidalindicators)
dict_wltidalindicators.update(dict_HWLW_springneap)

# plot
fig, ax = kw.plot_tidalindicators(dict_wltidalindicators)
fig.savefig(os.path.join(dir_figures, f'kw{year_slotgem}-tidalindicators-{current_station}.png'))


#### SLOTGEMIDDELDEN
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
fig1.savefig(os.path.join(dir_figures, f'kw{year_slotgem}-slotgemiddelden-{current_station}'))


### HAVENGETALLEN 
df_havengetallen, df_HWLW = kw.calc_havengetallen(df_ext=df_ext_todate, return_df_ext=True, min_coverage=min_coverage)

# plot aardappelgrafiek
fig, (ax1,ax2) = kw.plot_aardappelgrafiek(df_havengetallen=df_havengetallen)
fig.savefig(os.path.join(dir_figures, f'kw{year_slotgem}-aardappelgrafiek-{current_station}.png'))


##### GEMIDDELDE GETIJKROMMEN
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

fig, ax = kw.plot_gemiddeldgetij(gemgetij_dict=gemgetij_corr, gemgetij_dict_raw=gemgetij_raw, tick_hours=6)
fig.savefig(os.path.join(dir_figures, f'kw{year_slotgem}-gemiddeldgetij-{current_station}.png'))


#### OVERSCHRIJDINGSFREQUENTIES
freqs_interested = [5, 2, 1, 1/2, 1/5, 1/10, 1/20, 1/50, 1/100, 1/200,
                     1/500, 1/1000, 1/2000, 1/4000, 1/5000, 1/10000]

# 1. Exceedance
dist_exc = kw.calc_overschrijding(df_ext=df_ext_todate, rule_type=None, rule_value=None, 
                                  clip_physical_break=True, dist=None,
                                  interp_freqs=freqs_interested)

fig, ax = kw.plot_overschrijding(dist_exc)
fig.savefig(os.path.join(dir_figures, f'kw{year_slotgem}-exceedance-{current_station}.png'))

# 2. Deceedance
dist_dec = kw.calc_overschrijding(df_ext=df_ext_todate, rule_type=None, rule_value=None, 
                                  clip_physical_break=True, inverse=True,
                                  interp_freqs=freqs_interested)

fig, ax = kw.plot_overschrijding(dist_dec)
fig.savefig(os.path.join(dir_figures, f'kw{year_slotgem}-deceedance-{current_station}.png'))
