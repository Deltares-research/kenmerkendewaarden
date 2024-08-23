# -*- coding: utf-8 -*-

import numpy as np
from matplotlib.ticker import Formatter


def raise_extremes_with_aggers(df_ext):
    # TODO: alternatively we can convert 12345 to 12 here
    hwlwcodes = df_ext["HWLWcode"].drop_duplicates()
    bool_is_12 = np.asarray([x in [1,2] for x in set(hwlwcodes)])
    if not bool_is_12.all():
        raise ValueError(
            "df_ext should only contain extremes (HWLWcode 1/2), "
            "but it also contains aggers (HWLWcode 3/4/5). "
            "You can convert with `hatyan.calc_HWLW12345to12()`"
        )


# TODO: fixing display of negative timedeltas was requested in https://github.com/pandas-dev/pandas/issues/17232#issuecomment-2205579156
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
