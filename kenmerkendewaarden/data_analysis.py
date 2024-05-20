# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:23:46 2024

@author: veenstra
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all = [
    "df_amount_boxplot",
    "df_amount_pcolormesh",
    ]


def df_amount_boxplot(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df[df==0] = np.nan
    
    fig, ax = plt.subplots(figsize=(14,8))
    df.plot.box(ax=ax, rot=90, grid=True)
    ax.set_ylabel("measurements per year (0 excluded) [-]")
    fig.tight_layout()
    return fig, ax


def df_amount_pcolormesh(df, relative=False):
    df = df.copy()
    df[df==0] = np.nan
    
    if relative:
        # this is useful for ts, because the frequency was changed from hourly to 10-minute
        df_relative = df.div(df.median(axis=1), axis=0) * 100
        df_relative = df_relative.clip(upper=200)
        df = df_relative
        
    fig, ax = plt.subplots(figsize=(14,8))
    pc = ax.pcolormesh(df.columns, df.index, df.values, cmap="turbo")
    cbar = fig.colorbar(pc, ax=ax)
    ax.tick_params(axis='x', labelrotation=90)
    ax.grid()
    if relative:
        cbar.set_label("measurements per year w.r.t. year median (0 excluded) [%]")
    else:
        cbar.set_label("measurements per year (0 excluded) [-]")
    fig.tight_layout()
    return fig, ax
