# -*- coding: utf-8 -*-

def raise_extremes_with_aggers(df_ext):
    # TODO: alternatively we can convert 12345 to 12 here
    if len(df_ext["HWLWcode"].drop_duplicates()) != 2:
        raise ValueError("df_ext should only contain extremes (HWLWcode 1/2), "
                         "but it also contains aggers (HWLWcode 3/4/5). "
                         "You can convert with `hatyan.calc_HWLW12345to12()`")
