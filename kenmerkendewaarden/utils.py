# -*- coding: utf-8 -*-

import numpy as np
from matplotlib.ticker import Formatter
import logging

logger = logging.getLogger(__name__)


def raise_extremes_with_aggers(df_ext):
    # TODO: alternatively we can convert 12345 to 12 here
    hwlwcodes = df_ext["HWLWcode"].drop_duplicates()
    bool_is_12 = np.asarray([x in [1, 2] for x in set(hwlwcodes)])
    if not bool_is_12.all():
        raise ValueError(
            "df_ext should only contain extremes (HWLWcode 1/2), "
            "but it also contains aggers (HWLWcode 3/4/5). "
            "You can convert with `hatyan.calc_HWLW12345to12()`"
        )


def raise_empty_df(df):
    if df is None:
        raise TypeError("None was provided instead of a dataframe.")
    if df.empty:
        raise ValueError("Provided dataframe is empty.")


def raise_not_monotonic(df):
    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "The timeseries times (dataframe index) has to be monotonically increasing "
            "since it is assumed here that the first/last values are the min/max."
        )


def clip_timeseries_last_newyearsday(df):
    raise_not_monotonic(df)
    # clip last value of the timeseries if this is exactly newyearsday
    # so remove last timestep if equal to "yyyy-01-01 00:00:00"
    if "-01-01 00:00:00" in str(df.index[-1]):
        df = df.iloc[:-1]
    return df


def crop_timeseries_last_nyears(df, nyears):
    df = clip_timeseries_last_newyearsday(df)

    # last_year, for instance 2020
    last_year = df.index[-1].year
    # first_year, for instance 2011
    first_year = last_year - (nyears - 1)

    df_10y = df.loc[str(first_year) : str(last_year)]

    # TODO: consider enforcing nyears instead of warning if it is not the case
    # just like in `kw.calc_hat_lat_frommeasurements()`, but requires updates to tests
    actual_years = df_10y.index.year.drop_duplicates().to_numpy()
    is_exp_first = actual_years[0] == first_year
    is_exp_last = actual_years[-1] == last_year
    is_exp_amount = len(actual_years) == nyears
    if not (is_exp_first & is_exp_last & is_exp_amount):
        logger.warning(
            f"requested {nyears} years but resulted in "
            f"{len(actual_years)}: {actual_years}"
        )

    return df_10y


# TODO: fixing display of negative timedeltas was requested in
# https://github.com/pandas-dev/pandas/issues/17232#issuecomment-2205579156
class TimeSeries_TimedeltaFormatter_improved(Formatter):
    """
    Formats the ticks along an axis controlled by a :class:`TimedeltaIndex`.
    based on pandas.plotting._matplotlib.converter.TimeSeries_TimedeltaFormatter
    """

    @staticmethod
    def format_timedelta_ticks(x, pos, n_decimals: int) -> str:
        """
        Convert seconds to 'D days HH:MM:SS.F'
        """
        if x < 0:
            negative = True
            x = np.abs(x)
        else:
            negative = False
        s, ns = divmod(x, 10**9)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        decimals = int(ns * 10 ** (n_decimals - 9))
        s = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
        if n_decimals > 0:
            s += f".{decimals:0{n_decimals}d}"
        if d != 0:
            s = f"{int(d):d} days {s}"
        if negative:
            s = "-" + s
        return s

    def __call__(self, x, pos: int = 0) -> str:
        (vmin, vmax) = tuple(self.axis.get_view_interval())
        n_decimals = min(int(np.ceil(np.log10(100 * 10**9 / abs(vmax - vmin)))), 9)
        return self.format_timedelta_ticks(x, pos, n_decimals)
