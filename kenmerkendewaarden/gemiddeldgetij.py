# -*- coding: utf-8 -*-
"""
Computation of gemiddelde getijkromme
"""

import numpy as np
import pandas as pd
import hatyan
import logging
import matplotlib.pyplot as plt
from kenmerkendewaarden.tidalindicators import calc_HWLWtidalrange
from kenmerkendewaarden.havengetallen import calc_havengetallen
from kenmerkendewaarden.utils import TimeSeries_TimedeltaFormatter_improved
from matplotlib.ticker import MaxNLocator, MultipleLocator


__all__ = [
    "calc_gemiddeldgetij",
    "plot_gemiddeldgetij",
]

logger = logging.getLogger(__name__)


def calc_gemiddeldgetij(
    df_meas: pd.DataFrame,
    df_ext: pd.DataFrame = None,
    min_coverage: float = None,
    freq: str = "60sec",
    nb: int = 0,
    nf: int = 0,
    scale_extremes: bool = False,
    scale_period: bool = False,
    debug: bool = False,
):
    """
    Generate an average tidal signal for average/spring/neap tide by doing a tidal
    analysis on a timeseries of measurements. The (subsets/adjusted) resulting tidal components
    are then used to make a raw prediction for average/spring/neap tide.
    These raw predictions can optionally be scaled in height (with havengetallen)
    and in time (to a fixed period of 12h25min). An n-number of backwards and forward repeats
    are added before the timeseries are returned, resulting in nb+nf+1 tidal periods.

    Parameters
    ----------
    df_meas : pd.DataFrame
        Timeseries of 10 years of waterlevel measurements.
    df_ext : pd.DataFrame, optional
        Timeseries of 10 years of waterlevel extremes (1/2 only). The default is None.
    min_coverage : float, optional
        The minimal required coverage of the df_ext timeseries. Passed on to `calc_havengetallen()`. The default is None.
    freq : str, optional
        Frequency of the prediction, a value of 60 seconds or lower is adivisable for decent results. The default is "60sec".
    nb : int, optional
        Amount of periods to repeat backward. The default is 0.
    nf : int, optional
        Amount of periods to repeat forward. The default is 0.
    scale_extremes : bool, optional
        Whether to scale extremes with havengetallen. The default is False.
    scale_period : bool, optional
        Whether to scale to 12h25min (for boi). The default is False.
    debug : bool, optional
        Whether to generate a debug figure with selected spring/neap tidalperiods. The default is False.

    Returns
    -------
    gemgetij_dict : dict
        dictionary with Dataframes with gemiddeld getij for mean, spring and neap tide.

    """
    data_pd_meas_10y = df_meas
    tstop_dt = df_meas.index.max()

    current_station = data_pd_meas_10y.attrs["station"]

    # TODO: deprecate debug argument+plot (maybe use max HW instead of max tidalrange?)
    # TODO: add correctie havengetallen HW/LW av/sp/np met slotgemiddelde uit PLSS/modelfit (HW/LW av)

    if scale_period:
        tP_goal = pd.Timedelta(hours=12, minutes=25)
    else:
        tP_goal = None

    # scale extremes with havengetallen, or not
    if scale_extremes:
        if df_ext is None:
            raise ValueError("df_ext should be provided if scale_extremes=True")
        # compare station attributes
        station_attrs = [df.attrs["station"] for df in [df_meas, df_ext]]
        assert all(x == station_attrs[0] for x in station_attrs)

        df_havengetallen = calc_havengetallen(df_ext=df_ext, min_coverage=min_coverage)
        HW_sp, LW_sp = df_havengetallen.loc[
            0, ["HW_values_median", "LW_values_median"]
        ]  # spring
        HW_np, LW_np = df_havengetallen.loc[
            6, ["HW_values_median", "LW_values_median"]
        ]  # neap
        HW_av, LW_av = df_havengetallen.loc[
            "mean", ["HW_values_median", "LW_values_median"]
        ]  # mean
    else:
        HW_av = LW_av = None
        HW_sp = LW_sp = None
        HW_np = LW_np = None

    # derive components via TA on measured waterlevels
    comp_frommeasurements_avg, comp_av = get_gemgetij_components(data_pd_meas_10y)

    times_pred_1mnth = pd.date_range(
        start=pd.Timestamp(tstop_dt.year, 1, 1, 0, 0) - pd.Timedelta(hours=12),
        end=pd.Timestamp(tstop_dt.year, 2, 1, 0, 0),
        freq=freq,
    )  # start 12 hours in advance, to assure also corrected values on desired tstart
    comp_av.attrs["nodalfactors"] = (
        False  # nodalfactors=False to guarantee repetative signal
    )
    prediction_avg = hatyan.prediction(comp_av, times=times_pred_1mnth)
    prediction_avg_ext = hatyan.calc_HWLW(ts=prediction_avg, calc_HWLW345=False)

    time_firstHW = prediction_avg_ext.loc[prediction_avg_ext["HWLWcode"] == 1].index[
        0
    ]  # time of first HW
    ia1 = prediction_avg_ext.loc[time_firstHW:].index[0]  # time of first HW
    ia2 = prediction_avg_ext.loc[time_firstHW:].index[2]  # time of second HW
    prediction_avg_one = prediction_avg.loc[ia1:ia2]
    prediction_avg_ext_one = prediction_avg_ext.loc[ia1:ia2]

    # =============================================================================
    # Hatyan predictie voor 1 jaar met gemiddelde helling maansbaan (voor afleiden spring-doodtijcyclus) >> predictie zonder nodalfactors instead
    # =============================================================================
    """
    uit: gemiddelde getijkrommen 1991.0
    Voor de ruwe krommen voor springtij en doodtij is het getij voorspeld
    voor een jaar met gemiddelde helling maansbaan met
    uitsluitend zuivere combinaties van de componenten M2 en S2:
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
    """
    components_sn = [
        "A0",
        "SM",
        "3MS2",
        "MU2",
        "M2",
        "S2",
        "2SM2",
        "3MS4",
        "M4",
        "MS4",
        "4MS6",
        "M6",
        "2MS6",
        "M8",
        "3MS8",
        "M10",
        "4MS10",
        "M12",
        "5MS12",
    ]

    # make prediction with springneap components with nodalfactors=False (alternative for choosing a year with a neutral nodal factor). Using 1yr instead of 1month does not make a difference in min/max tidal range and shape, also because of nodalfactors=False. (when using more components, there is a slight difference)
    comp_frommeasurements_avg_sncomp = comp_frommeasurements_avg.loc[components_sn]
    comp_frommeasurements_avg_sncomp.attrs["nodalfactors"] = (
        False  # nodalfactors=False to make independent on chosen year
    )
    prediction_sn = hatyan.prediction(
        comp_frommeasurements_avg_sncomp, times=times_pred_1mnth
    )

    prediction_sn_ext = hatyan.calc_HWLW(ts=prediction_sn, calc_HWLW345=False)

    # selecteer getijslag met minimale tidalrange en maximale tidalrange (werd geselecteerd adhv havengetallen in 1991.0 doc)
    prediction_sn_ext = calc_HWLWtidalrange(prediction_sn_ext)

    time_TRmax = prediction_sn_ext.loc[
        prediction_sn_ext["HWLWcode"] == 1, "tidalrange"
    ].idxmax()
    is1 = prediction_sn_ext.loc[time_TRmax:].index[0]
    is2 = prediction_sn_ext.loc[time_TRmax:].index[2]

    time_TRmin = prediction_sn_ext.loc[
        prediction_sn_ext["HWLWcode"] == 1, "tidalrange"
    ].idxmin()
    in1 = prediction_sn_ext.loc[time_TRmin:].index[0]
    in2 = prediction_sn_ext.loc[time_TRmin:].index[2]

    # select one tideperiod for springtide and one for neaptide
    prediction_sp_one = prediction_sn.loc[is1:is2]
    prediction_sp_ext_one = prediction_sn_ext.loc[is1:is2]
    prediction_np_one = prediction_sn.loc[in1:in2]
    prediction_np_ext_one = prediction_sn_ext.loc[in1:in2]

    # plot selection of neap/spring
    if debug:
        fig, (ax1, ax2) = hatyan.plot_timeseries(
            ts=prediction_sn, ts_ext=prediction_sn_ext
        )
        ax1.plot(prediction_sp_one["values"], "r")
        ax1.plot(prediction_np_one["values"], "r")
        ax1.legend(
            labels=ax1.get_legend_handles_labels()[1]
            + ["kromme spring", "kromme neap"],
            loc=4,
        )
        ax1.set_ylabel("waterstand [m]")
        ax1.set_title(f"spring- en doodtijkromme {current_station}")
        # fig.savefig(os.path.join(dir_gemgetij,f'springdoodtijkromme_{current_station}_slotgem{year_slotgem}.png'))

    # timeseries for gele boekje (av/sp/np have different lengths, time is relative to HW of av and HW of sp/np are shifted there)
    logger.info(f"reshape_signal GEMGETIJ: {current_station}")
    prediction_av_corr_one = reshape_signal(
        prediction_avg_one,
        prediction_avg_ext_one,
        HW_goal=HW_av,
        LW_goal=LW_av,
        tP_goal=tP_goal,
    )
    prediction_av_corr_one.index = (
        prediction_av_corr_one.index - prediction_av_corr_one.index[0]
    )  # make relative to first timestamp (=HW)
    if scale_period:  # resampling required because of scaling
        prediction_av_corr_one = prediction_av_corr_one.resample(freq).nearest()
    prediction_av = repeat_signal(prediction_av_corr_one, nb=nb, nf=nf)

    logger.info(f"reshape_signal SPRINGTIJ: {current_station}")
    prediction_sp_corr_one = reshape_signal(
        prediction_sp_one,
        prediction_sp_ext_one,
        HW_goal=HW_sp,
        LW_goal=LW_sp,
        tP_goal=tP_goal,
    )
    prediction_sp_corr_one.index = (
        prediction_sp_corr_one.index - prediction_sp_corr_one.index[0]
    )  # make relative to first timestamp (=HW)
    if scale_period:  # resampling required because of scaling
        prediction_sp_corr_one = prediction_sp_corr_one.resample(freq).nearest()
    prediction_sp = repeat_signal(prediction_sp_corr_one, nb=nb, nf=nf)

    logger.info(f"reshape_signal DOODTIJ: {current_station}")
    prediction_np_corr_one = reshape_signal(
        prediction_np_one,
        prediction_np_ext_one,
        HW_goal=HW_np,
        LW_goal=LW_np,
        tP_goal=tP_goal,
    )
    prediction_np_corr_one.index = (
        prediction_np_corr_one.index - prediction_np_corr_one.index[0]
    )  # make relative to first timestamp (=HW)
    if scale_period:  # resampling required because of scaling
        prediction_np_corr_one = prediction_np_corr_one.resample(freq).nearest()
    prediction_np = repeat_signal(prediction_np_corr_one, nb=nb, nf=nf)

    # combine in single dictionary
    gemgetij_dict = {}
    gemgetij_dict["mean"] = prediction_av['values']
    gemgetij_dict["spring"] = prediction_sp['values']
    gemgetij_dict["neap"] = prediction_np['values']

    return gemgetij_dict


def plot_gemiddeldgetij(
    gemgetij_dict: dict, gemgetij_dict_raw: dict = None, tick_hours: int = None
):
    """
    Default plotting function for gemiddeldgetij dictionaries.

    Parameters
    ----------
    gemgetij_dict : dict
        dictionary as returned from `kw.calc_gemiddeldgetij()`.
    gemgetij_raw_dict : dict, optional
        dictionary as returned from `kw.calc_gemiddeldgetij()` e.g. with uncorrected values. The default is None.
    ticks_12h : bool, optional
        whether to use xaxis ticks of 12 hours, otherwise automatic but less nice values

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.
    ax : matplotlib.axes._axes.Axes
        Figure axis handle.

    """
    # get and compare station attributes
    station_attrs = [v.attrs["station"] for k, v in gemgetij_dict.items()]
    assert all(x == station_attrs[0] for x in station_attrs)
    station = station_attrs[0]

    logger.info(f"plot getijkromme trefHW: {station}")
    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = plt.get_cmap("tab10")

    if gemgetij_dict_raw is not None:
        prediction_av_raw = gemgetij_dict_raw["mean"]
        prediction_sp_raw = gemgetij_dict_raw["spring"]
        prediction_np_raw = gemgetij_dict_raw["neap"]
        prediction_av_raw.plot(
            ax=ax,
            linestyle="--",
            color=cmap(0),
            linewidth=0.7,
            label="gemiddeldgetij mean (raw)",
        )
        prediction_sp_raw.plot(
            ax=ax,
            linestyle="--",
            color=cmap(1),
            linewidth=0.7,
            label="gemiddeldgetij spring (raw)",
        )
        prediction_np_raw.plot(
            ax=ax,
            linestyle="--",
            color=cmap(2),
            linewidth=0.7,
            label="gemiddeldgetij neap (raw)",
        )

    prediction_av_corr = gemgetij_dict["mean"]
    prediction_sp_corr = gemgetij_dict["spring"]
    prediction_np_corr = gemgetij_dict["neap"]
    prediction_av_corr.plot(ax=ax, color=cmap(0), label="gemiddeldgetij mean")
    prediction_sp_corr.plot(
        ax=ax, color=cmap(1), label="gemiddeldgetij spring"
    )
    prediction_np_corr.plot(ax=ax, color=cmap(2), label="gemiddeldgetij neap")

    ax.set_title(f"getijkrommes for {station}")
    ax.legend(loc=4)
    ax.grid()
    ax.set_xlabel("time since high water")

    # fix timedelta ticks
    ax.xaxis.set_major_formatter(TimeSeries_TimedeltaFormatter_improved())
    # put ticks at intervals of multiples of 3 and 6, resulting in whole seconds
    ax.xaxis.set_major_locator(MaxNLocator(steps=[3, 6], integer=True))
    if tick_hours is not None:
        # put ticks at fixed 12-hour intervals
        ax.xaxis.set_major_locator(MultipleLocator(base=tick_hours * 3600e9))
    # the above avoids having to manually set tick locations based on hourly intervals (3600e9 nanoseconds)
    # ax.set_xticks([x*3600e9 for x in range(-15, 25, 5)])
    # ax.set_xlim([x*3600e9 for x in [-15.5,15.5]])

    fig.tight_layout()

    return fig, ax


def get_gemgetij_components(data_pd_meas):
    # =============================================================================
    # Hatyan analyse voor 10 jaar (alle componenten voor gemiddelde getijcyclus) #TODO: maybe use original 4y period/componentfile instead? SA/SM should come from 19y analysis
    # =============================================================================
    
    # components should not be reduced, since higher harmonics are necessary
    const_list = hatyan.get_const_list_hatyan("year")  
    hatyan_settings_ana = dict(
        nodalfactors=True,
        fu_alltimes=False,
        xfac=True,
        analysis_perperiod="Y",
        return_allperiods=True,
    )  # RWS-default settings
    comp_frommeasurements_avg, comp_frommeasurements_allyears = hatyan.analysis(
        data_pd_meas, const_list=const_list, **hatyan_settings_ana
    )

    # #check if all years are available
    # comp_years = comp_frommeasurements_allyears['A'].columns
    # expected_years = tstop_dt.year-tstart_dt.year
    # if len(comp_years) < expected_years:
    #     raise Exception('ERROR: analysis result contains not all years')

    # check if nans in analysis
    if comp_frommeasurements_avg.isnull()["A"].any():
        raise ValueError("analysis result contains nan values")

    # =============================================================================
    # gemiddelde getijkromme
    # =============================================================================
    """
    uit: gemiddelde getijkrommen 1991.0
    Voor meetpunten in het onbeinvloed gebied is per getijfase eerst een "ruwe kromme" berekend met
    de resultaten van de harmonische analyse,
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
    # kwadraatsommen voor M2 tot M12
    components_av = ["M2", "M4", "M6", "M8", "M10", "M12"]
    comp_av = comp_frommeasurements_avg.loc[components_av]
    for comp_higherharmonics in components_av:
        iM = int(comp_higherharmonics[1:])
        bool_endswithiM = (
            comp_frommeasurements_avg.index.str.endswith(str(iM))
            & comp_frommeasurements_avg.index.str.replace(str(iM), "")
            .str[-1]
            .str.isalpha()
        )
        comp_iM = comp_frommeasurements_avg.loc[bool_endswithiM]
        # kwadraatsom
        comp_av.loc[comp_higherharmonics, "A"] = np.sqrt((comp_iM["A"] ** 2).sum())

    comp_av.loc["A0"] = comp_frommeasurements_avg.loc["A0"]

    # values are different than 1991.0 document and differs per station while the
    # document states "Zoals te verwachten is de verhouding per component tussen deze
    # wortel en de oorspronkelijke amplitude voor alle plaatsen gelijk"    
    logger.debug(
        "verhouding tussen originele en kwadratensom componenten:\n"
        f"{comp_av/comp_frommeasurements_avg.loc[components_av]}"
    )

    return comp_frommeasurements_avg, comp_av


def reshape_signal(ts, ts_ext, HW_goal, LW_goal, tP_goal=None):
    """
    scales tidal signal to provided HW/LW value and up/down going time
    tP_goal (tidal period time) is used to fix tidalperiod to 12h25m (for BOI timeseries)

    time_down was scaled with havengetallen before, but not anymore to avoid issues with aggers
    """
    # early escape # TODO: should also be possible to only scale tP_goal
    if HW_goal is None and LW_goal is None:
        ts.index.name = "timedelta"
        return ts

    # TODO: consider removing the need for ts_ext, it should be possible with min/max, although the HW of the raw timeseries are not exactly equal

    TR_goal = HW_goal - LW_goal

    # selecteer alle hoogwaters en opvolgende laagwaters
    bool_HW = ts_ext["HWLWcode"] == 1
    idx_HW = np.where(bool_HW)[0]
    timesHW = ts_ext.index[idx_HW]
    # assuming alternating 1,2,1 or 1,3,1, this is always valid in this workflow
    timesLW = ts_ext.index[idx_HW[:-1] + 1]

    # crop from first to last HW (rest is not scaled anyway)
    ts_time_firstHW = ts_ext[bool_HW].index[0]
    ts_time_lastHW = ts_ext[bool_HW].index[-1]
    ts_corr = ts.copy().loc[ts_time_firstHW:ts_time_lastHW]

    # this is necessary since datetimeindex with freq is not editable, and Series is editable
    ts_corr["timedelta"] = (ts_corr.index)
    for i in np.arange(0, len(timesHW) - 1):
        HW1_val = ts_corr.loc[timesHW[i], "values"]
        HW2_val = ts_corr.loc[timesHW[i + 1], "values"]
        LW_val = ts_corr.loc[timesLW[i], "values"]
        TR1_val = HW1_val - LW_val
        TR2_val = HW2_val - LW_val
        tP_val = timesHW[i + 1] - timesHW[i]
        if tP_goal is None:
            tP_goal = tP_val

        temp1 = (
            ts_corr.loc[timesHW[i] : timesLW[i], "values"] - LW_val
        ) / TR1_val * TR_goal + LW_goal
        temp2 = (
            ts_corr.loc[timesLW[i] : timesHW[i + 1], "values"] - LW_val
        ) / TR2_val * TR_goal + LW_goal
        # .iloc[1:] since timesLW[i] is in both timeseries (values are equal)
        temp = pd.concat([temp1, temp2.iloc[1:]])
        ts_corr["values_new"] = temp

        tide_HWtoHW = ts_corr.loc[timesHW[i] : timesHW[i + 1]]
        ts_corr["timedelta"] = pd.date_range(
            start=ts_corr.loc[timesHW[i], "timedelta"],
            end=ts_corr.loc[timesHW[i], "timedelta"] + tP_goal,
            periods=len(tide_HWtoHW),
        )

    ts_corr = ts_corr.set_index("timedelta", drop=True)
    ts_corr["values"] = ts_corr["values_new"]
    ts_corr = ts_corr.drop(["values_new"], axis=1)
    return ts_corr


def repeat_signal(ts_one_HWtoHW, nb, nf):
    """
    repeat tidal signal, necessary for sp/np, since they are computed as single tidal signal first
    """
    tidalperiod = ts_one_HWtoHW.index[-1] - ts_one_HWtoHW.index[0]
    ts_rep = pd.DataFrame()
    for iAdd in np.arange(-nb, nf + 1):
        ts_add = pd.DataFrame(
            {"values": ts_one_HWtoHW["values"].values},
            index=ts_one_HWtoHW.index + iAdd * tidalperiod,
        )
        ts_rep = pd.concat([ts_rep, ts_add])
    ts_rep = ts_rep.loc[~ts_rep.index.duplicated()]
    # pass on attributes
    ts_rep.attrs = ts_one_HWtoHW.attrs
    return ts_rep
