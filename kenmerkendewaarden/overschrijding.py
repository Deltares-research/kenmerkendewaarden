# -*- coding: utf-8 -*-
"""
Computation of probabilities (overschrijdingsfrequenties) of extreme waterlevels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy import optimize, signal
from typing import Union, List
import logging
from kenmerkendewaarden.data_retrieve import clip_timeseries_physical_break
from kenmerkendewaarden.utils import raise_extremes_with_aggers

__all__ = [
    "calc_overschrijding",
    "plot_overschrijding",
]

logger = logging.getLogger(__name__)


def get_threshold_rowidx(ser):
    # TODO: base on frequency or value?
    thresholfreq = 3  # take a frequency that is at least higher than the max HYDRA frequency (which is 1)
    rowidx_tresholdfreq = np.abs(ser.index - thresholfreq).argmin()
    return rowidx_tresholdfreq


def series_copy_properties(ser, ser_reference):
    # copy attrs index name and values name from reference to series
    ser.attrs = ser_reference.attrs
    ser.index.name = ser_reference.index.name
    ser.name = ser_reference.name


def calc_overschrijding(
    df_ext: pd.DataFrame,
    dist: dict = None,
    inverse: bool = False,
    clip_physical_break: bool = False,
    rule_type: str = None,
    rule_value: (pd.Timestamp, float) = None,
    interp_freqs: list = None,
):
    """
    Compute exceedance/deceedance frequencies based on measured extreme waterlevels.

    Parameters
    ----------
    df_ext : pd.DataFrame, optional
        The timeseries of extremes (high and low waters). The default is None.
    dist : dict, optional
        A pre-filled dictionary with a Hydra-NL and/or validation distribution. The default is None.
    inverse : bool, optional
        Whether to compute deceedance instead of exceedance frequencies. The default is False.
    clip_physical_break : bool, optional
        Whether to exclude the part of the timeseries before physical breaks like estuary closures. The default is False.
    rule_type : str, optional
        break/linear/None, passed on to apply_trendanalysis(). The default is None.
    rule_value : (pd.Timestamp, float), optional
        Value corresponding to rule_type, pd.Timestamp (or anything understood by pd.Timestamp)
        in case of rule_type='break', float in case of rule_type='linear'. The default is None.
    interp_freqs : list, optional
        The frequencies to interpolate to, providing this will result in a
        "Geinterpoleerd" key in the returned dictionary. The default is None.

    Returns
    -------
    dist : dict
        A dictionary with several distributions.

    """

    raise_extremes_with_aggers(df_ext)
    # take only high or low extremes
    # TODO: this might not be useful in case of river discharge influenced stations where a filter is needed
    if inverse:
        df_extrema = df_ext.loc[df_ext["HWLWcode"] != 1]
    else:
        df_extrema = df_ext.loc[df_ext["HWLWcode"] == 1]

    if clip_physical_break:
        df_extrema = clip_timeseries_physical_break(df_extrema)
    # drop all info but the values (times-idx, HWLWcode etc)
    ser_extrema = df_extrema.copy()["values"]

    if dist is None:
        dist = {}

    logger.info(f"Calculate unfiltered distribution (inverse={inverse})")
    dist["Ongefilterd"] = distribution(ser_extrema, inverse=inverse)

    # TODO: re-enable filter for river discharge peaks
    #TODO: ext is geschikt voor getij, maar bij hoge afvoergolf wil je alleen het echte extreem. Er is dan een treshold per station nodig, is nodig om de rivierafvoerpiek te kunnen duiden.
    """# filtering is only applicable for stations with high river discharge influence, so disabled
    logger.info('Calculate filtered distribution')
    ser_peaks, threshold, _ = detect_peaks(ser_extrema)
    if metadata_station['apply_treshold']:
        temp[metadata_station['id']] = threshold
        ser_extrema_filt = filter_with_threshold(ser_extrema, ser_peaks, threshold)
    else:
        ser_extrema_filt = ser_extrema.copy()
    dist['Gefilterd'] = distribution(ser_extrema_filt.copy())
    """

    logger.info("Calculate filtered distribution with trendanalysis")
    ser_trend = apply_trendanalysis(
        ser_extrema, rule_type=rule_type, rule_value=rule_value
    )
    dist["Trendanalyse"] = distribution(ser_trend.copy(), inverse=inverse)

    logger.info("Fit Weibull to filtered distribution with trendanalysis")
    idx_maxfreq_trend = get_threshold_rowidx(dist["Trendanalyse"])
    treshold_value = dist["Trendanalyse"].iloc[idx_maxfreq_trend]
    treshold_Tfreq = dist["Trendanalyse"].index[idx_maxfreq_trend]
    dist["Weibull"] = get_weibull(
        dist["Trendanalyse"].copy(),
        threshold=treshold_value,
        Tfreqs=np.logspace(-5, np.log10(treshold_Tfreq), 5000),
        inverse=inverse,
    )

    if "Hydra-NL" in dist.keys():
        logger.info("Blend trend, weibull and Hydra-NL")
        # TODO: now based on hardcoded Hydra-NL dict key which is already part of the input dist dict, this is tricky
        ser_hydra = dist["Hydra-NL"].copy()
    else:
        logger.info("Blend trend and weibull")
        ser_hydra = None
    dist["Gecombineerd"] = blend_distributions(
        ser_trend=dist["Trendanalyse"].copy(),
        ser_weibull=dist["Weibull"].copy(),
        ser_hydra=ser_hydra,
    )

    if interp_freqs is not None:
        dist["Geinterpoleerd"] = interpolate_interested_Tfreqs(
            dist["Gecombineerd"], Tfreqs=interp_freqs
        )

    return dist


def delete_values_between_peak_trough(times_to_delete, times, values):
    mask = np.in1d(times, times_to_delete)
    return times[~mask], values[~mask]


def go_through_peak(values, _i, _i_extra, multiplier):
    while values[_i + _i_extra + multiplier] < values[_i + _i_extra]:
        _i_extra += multiplier
    return _i_extra


def filter_identified_peaks(values, times, times_to_delete):
    _values = values.copy()
    _values[~np.in1d(times, times_to_delete)] = -9999.0
    return _values


def check_peakside(values, _i, multiplier, window, threshold):
    _i_extra = 0
    check = True
    while check:
        try:
            while (values[_i + _i_extra + multiplier] <= values[_i + _i_extra]) and (
                values[_i + _i_extra + multiplier] != -9999.0
            ):
                _i_extra += multiplier
        except IndexError:
            pass
        try:
            _i1, _i2 = (_i + _i_extra + (multiplier * window)), (_i + _i_extra)
            values_sel = values[np.min([_i1, _i2]) : np.max([_i1, _i2])]
            if any(values_sel > threshold):
                new_peak = values_sel.max()
                while values[_i + _i_extra] != new_peak:
                    _i_extra += multiplier
            else:
                check = False
        except IndexError:
            pass
    return _i_extra


def detect_peaks_hkv(
    df: pd.DataFrame, window: int, inverse: bool = False, threshold: float = None
) -> pd.DataFrame:
    # TODO: still to be converted from dataframe to series
    _df = df.copy()
    if inverse:
        _df["values"] = -_df["values"]
    times, values = _df.index, _df["values"].values
    indices = np.arange(len(times))

    __df = _df.sort_values(by=["values"], axis=0, ascending=False)
    times_sorted, values_sorted = __df.index, __df["values"].values

    if threshold is None:
        threshold = df["values"].mean() + 2 * df["values"].std()
    else:
        if inverse:
            threshold = -threshold

    peaks = np.ones(len(values)) * np.nan
    t_peaks, dt_left_peaks, dt_right_peaks = peaks.copy(), peaks.copy(), peaks.copy()

    logger.info(f"Determining peaks (inverse={inverse})")
    peak_count = 0
    while len(values_sorted) != 0:
        _t, _p = times_sorted[0], values_sorted[0]
        t_peaks[peak_count], peaks[peak_count] = _t, _p

        _i = indices[_t == times][0]

        # first left peak
        _i_extra = check_peakside(
            filter_identified_peaks(values, times, times_sorted),
            _i,
            -1,
            window,
            threshold,
        )
        dt_left_peaks[peak_count] = (times[_i] - times[_i + _i_extra]) / np.timedelta64(
            1, "s"
        )
        times_sel = times[(_i + _i_extra) : (_i + 1)]
        times_sorted, values_sorted = delete_values_between_peak_trough(
            times_sel, times_sorted, values_sorted
        )

        # right peak
        _i_extra = check_peakside(
            filter_identified_peaks(values, times, times_sorted),
            _i,
            1,
            window,
            threshold,
        )
        dt_right_peaks[peak_count] = (
            times[_i + _i_extra] - times[_i]
        ) / np.timedelta64(1, "s")
        times_sel = times[(_i - 1) : (_i + _i_extra)]
        times_sorted, values_sorted = delete_values_between_peak_trough(
            times_sel, times_sorted, values_sorted
        )

        peak_count += 1

    t_peaks = t_peaks[~np.isnan(t_peaks)]
    peaks = peaks[~np.isnan(peaks)]
    dt_left_peaks = dt_left_peaks[~np.isnan(dt_left_peaks)]
    dt_right_peaks = dt_right_peaks[~np.isnan(dt_right_peaks)]

    dt_total_peaks = dt_left_peaks + dt_right_peaks

    df_peaks = pd.DataFrame(
        index=pd.to_datetime(t_peaks),
        data={
            "values": peaks,
            "dt_left": dt_left_peaks,
            "dt_right": dt_right_peaks,
            "dt_total": dt_total_peaks,
        },
    )
    df_peaks = df_peaks.loc[df_peaks["dt_total"] > 0]

    if inverse:
        df_peaks["values"] = -df_peaks["values"]

    return df_peaks


def distribution(
    ser: pd.Series,
    c: float = -0.3,
    d: float = 0.4,
    inverse: bool = False,
) -> pd.Series:
    years = get_total_years(ser)
    if inverse:
        ser = ser.sort_values(ascending=False)
    else:
        ser = ser.sort_values()
    rank = np.array(range(len(ser))) + 1
    ser.index = (1 - (rank + c) / (len(rank) + d)) * (len(rank) / years)
    ser_sorted = ser.sort_index(ascending=False)
    
    series_copy_properties(ser=ser_sorted, ser_reference=ser)
    ser_sorted.index.name = "frequency"
    return ser_sorted


def get_weibull(
    ser: pd.Series,
    threshold: float,
    Tfreqs: np.ndarray,
    col: str = None,
    inverse: bool = False,
) -> pd.Series:
    
    values = ser.values
    if inverse:
        values = -values
        threshold = -threshold
    p_val_gt_threshold = ser.index[values > threshold][0]

    def pfunc(x, p_val_gt_threshold, threshold, sigma, alpha):
        return p_val_gt_threshold * np.exp(
            -((x / sigma) ** alpha) + ((threshold / sigma) ** alpha)
        )

    def pfunc_inverse(p_X_gt_x, p_val_gt_threshold, threshold, sigma, alpha):
        return sigma * (
            ((threshold / sigma) ** alpha) - np.log(p_X_gt_x / p_val_gt_threshold)
        ) ** (1 / alpha)

    def der_pfunc(x, p_val_gt_threshold, threshold, alpha, sigma):
        return (
            -p_val_gt_threshold
            * (alpha * x ** (alpha - 1))
            * (sigma ** (-alpha))
            * np.exp(-((x / sigma) ** alpha) + ((threshold / sigma) ** alpha))
        )

    def cost_func(params, *args):
        return -np.sum(
            [
                np.log(-der_pfunc(x, args[0], args[1], params[0], params[1]))
                for x in args[2]
            ]
        )

    initial_guess = np.array([1, abs(threshold)])
    result = optimize.minimize(
        cost_func,
        x0=initial_guess,
        args=(p_val_gt_threshold, threshold, values[values > threshold]),
        method="Nelder-Mead",
        options={"maxiter": 1e4},
    )
    if result.success:
        alpha, sigma = result.x[0], result.x[1]
    else:
        raise ValueError(result.message)

    new_values = pfunc_inverse(Tfreqs, p_val_gt_threshold, threshold, sigma, alpha)
    if inverse:
        new_values = -new_values
    ser_weibull = pd.Series(new_values, index=Tfreqs).sort_index(ascending=False)

    series_copy_properties(ser=ser_weibull, ser_reference=ser)
    return ser_weibull


def filter_with_threshold(
    ser_raw: pd.Series,
    ser_filtered: pd.Series,
    threshold: float,
    inverse: bool = False,
) -> pd.Series:
    if inverse:
        return pd.concat(
            [
                ser_raw[ser_raw >= threshold],
                ser_filtered[ser_filtered < threshold],
            ],
            axis=0,
        ).sort_index()
    else:
        return pd.concat(
            [
                ser_raw[ser_raw <= threshold],
                ser_filtered[ser_filtered > threshold],
            ],
            axis=0,
        ).sort_index()


def detect_peaks(ser: pd.Series, prominence: int = 10, inverse: bool = False):
    ser = ser.copy()
    if inverse:
        ser = -1 * ser
    peak_indices = signal.find_peaks(ser.values, prominence=prominence)[0]
    ser_peaks = pd.Series(ser.iloc[peak_indices],
                          index=ser.iloc[peak_indices].index,
                          )
    threshold = determine_threshold(
        values=ser.values, peak_indices=peak_indices
    )
    return ser_peaks, threshold, peak_indices


def determine_threshold(values: np.ndarray, peak_indices: np.ndarray) -> float:
    w = signal.peak_widths(values, peak_indices)[0]
    for threshold in reversed(
        range(int(np.floor(values.min())), int(np.ceil(values.max())))
    ):
        _t = w[values[peak_indices] > threshold]
        if len(_t[_t <= 3]) > (
            0.1 * len(_t)
        ):  # min of 3 tidal periods and at least more than 10%
            break
    return threshold


def get_total_years(ser: pd.Series) -> float:
    return (ser.index[-1] - ser.index[0]).total_seconds() / (3600 * 24 * 365)


def apply_trendanalysis(
    ser: pd.Series, rule_type: str, rule_value: Union[pd.Timestamp, float]
):
    # There are 2 rule types:  - break -> Values before break are removed
    #                          - linear -> Values are increased/lowered based on value in value/year. It is assumes
    #                                      that there is no linear trend at the latest time (so it works its way back
    #                                      in the past). rule_value should be entered as going forward in time
    if rule_type == "break":
        ser_out = ser[rule_value:].copy()
    elif rule_type == "linear":
        rule_value = float(rule_value)
        ser = ser.copy()
        dx = np.array(
            [
                rule_value * x.total_seconds() / (365 * 24 * 3600)
                for x in (ser.index[-1] - ser.index)
            ]
        )
        ser = ser + dx
        ser_out = ser
    elif rule_type is None:
        ser_out = ser.copy()
    else:
        raise ValueError(
            f'Incorrect rule_type="{rule_type}" passed to function. Only "break", "linear" or None are supported'
        )
    ser_out.index.name = "frequency"
    return ser_out


def blend_distributions(
    ser_trend: pd.Series, ser_weibull: pd.Series, ser_hydra: pd.Series = None
) -> pd.DataFrame:

    # get and compare station attributes
    ser_list = [ser_trend, ser_weibull, ser_hydra]
    station_attrs = [ser.attrs["station"] for ser in ser_list if ser is not None]
    assert all(x == station_attrs[0] for x in station_attrs)

    ser_trend = ser_trend.sort_index(ascending=False)
    ser_weibull = ser_weibull.sort_index(ascending=False)

    # Trend to weibull
    idx_maxfreq_trend = get_threshold_rowidx(ser_trend)
    ser_blended1 = ser_trend.iloc[:idx_maxfreq_trend].copy()
    ser_weibull = ser_weibull.loc[ser_weibull.index < ser_blended1.index[-1]].copy()

    # Weibull to Hydra
    if ser_hydra is not None:
        ser_hydra = ser_hydra.sort_index(ascending=False)

        Tfreqs_combined = np.unique(np.concatenate((ser_weibull.index, ser_hydra.index)))
        vals_weibull = np.interp(
            Tfreqs_combined,
            np.flip(ser_weibull.index),
            np.flip(ser_weibull.values),
        )
        vals_hydra = np.interp(
            Tfreqs_combined, np.flip(ser_hydra.index), np.flip(ser_hydra.values)
        )

        Tfreq0, TfreqN = ser_hydra.index[0], 1 / 50
        Tfreqs = np.logspace(np.log10(TfreqN), np.log10(Tfreq0), int(1e5))
        vals_weibull = np.interp(
            np.log10(Tfreqs),
            np.log10(np.flip(ser_weibull.index)),
            np.flip(ser_weibull.values),
        )
        vals_hydra = np.interp(
            np.log10(Tfreqs),
            np.log10(np.flip(ser_hydra.index)),
            np.flip(ser_hydra.values),
        )
        indices = np.arange(len(Tfreqs))
        grads = np.flip(np.arange(len(indices))) / len(indices) * np.pi

        vals_blend = (
            0.5 * (np.cos(grads) + 1) * vals_weibull[indices]
            + (1 - 0.5 * (np.cos(grads) + 1)) * vals_hydra[indices]
        )

        ser_blended2 = pd.Series(vals_blend, 
                                index=Tfreqs
                                ).sort_index(ascending=False)

        ser_blended = pd.concat(
            [
                ser_blended1,
                ser_weibull.loc[
                    (ser_weibull.index > ser_blended2.index[0])
                    & (ser_weibull.index < ser_blended1.index[-1])
                ],
                ser_blended2,
                ser_hydra.loc[ser_hydra.index < ser_blended2.index[-1]],
            ],
            axis=0,
        )
    else:
        ser_blended = pd.concat(
            [ser_blended1, ser_weibull.loc[(ser_weibull.index < ser_blended1.index[-1])]],
            axis=0,
        )

    duplicated_freqs = ser_blended.index.duplicated(keep="first")
    ser_blended = ser_blended.loc[~duplicated_freqs].sort_index(ascending=False)

    series_copy_properties(ser=ser_blended, ser_reference=ser_trend)
    
    return ser_blended


def interpolate_interested_Tfreqs(
    ser: pd.Series, Tfreqs: List[float]
) -> pd.DataFrame:

    interp_vals = np.interp(Tfreqs, np.flip(ser.index), np.flip(ser.values))
    ser_interp = pd.Series(interp_vals, index=Tfreqs).sort_index(ascending=False)

    series_copy_properties(ser=ser_interp, ser_reference=ser)
    return ser_interp


def plot_overschrijding(dist: dict):
    """
    plot overschrijding/onderschrijding

    Parameters
    ----------
    dist : dict
        Dictionary as returned from `kw.calc_overschrijding()`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.
    ax : matplotlib.axes._axes.Axes
        Figure axis handle.
    """

    # get and compare station attributes
    station_attrs = [v.attrs["station"] for k, v in dist.items()]
    assert all(x == station_attrs[0] for x in station_attrs)
    station = station_attrs[0]

    color_map = {
        "Ongefilterd": "b",
        "Gefilterd": "orange",
        "Trendanalyse": "g",
        "Weibull": "r",
        "Hydra-NL": "m",
        "Hydra-NL met modelonzekerheid": "cyan",
        "Gecombineerd": "k",
        "Geinterpoleerd": "lime",
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    for k in dist.keys():
        if k in color_map.keys():
            c = color_map[k]
        else:
            c = None
        if k == "Gecombineerd":
            ax.plot(dist[k], "--", label=k, c=c)
        elif k == "Geinterpoleerd":
            ax.plot(dist[k], "o", label=k, c=c, markersize=5)
        else:
            ax.plot(dist[k], label=k, c=c)

    ax.set_title(f"Distribution for {station}")
    ax.set_xlabel("Frequency [1/yrs]")
    ax.set_xscale("log")
    ax.set_xlim([1e-5, 1e3])
    ax.invert_xaxis()
    ax.set_ylabel("Waterlevel [m]")
    ax.legend(fontsize="medium", loc="lower right")
    ax.xaxis.set_minor_locator(
        ticker.LogLocator(
            base=10.0, subs=tuple(i / 10 for i in range(1, 10)), numticks=12
        )
    )
    ax.xaxis.set_minor_formatter(ticker.NullFormatter()),
    ax.yaxis.set_minor_locator(
        ticker.MultipleLocator(0.1)
    )  # this was 10, but now meters instead of cm
    ax.yaxis.set_minor_formatter(ticker.NullFormatter()),
    ax.yaxis.set_major_formatter(
        ticker.FormatStrFormatter("%.2f")
    )  # to force 2 decimal places
    ax.grid(visible=True, which="major"), ax.grid(visible=True, which="minor", ls=":")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig, ax
