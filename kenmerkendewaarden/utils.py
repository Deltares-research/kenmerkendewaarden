# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 09:39:32 2024

@author: veenstra
"""

import pandas as pd

def xarray_to_hatyan(ds):
    df = pd.DataFrame({"values":ds["Meetwaarde.Waarde_Numeriek"].to_pandas()/100,
                       "QC": ds["WaarnemingMetadata.KwaliteitswaardecodeLijst"].to_pandas(),
                       })
    if "HWLWcode" in ds.data_vars:
        df["HWLWcode"] = ds["HWLWcode"]
    
    # convert timezone back to UTC+1 # TODO: add testcase
    df.index = df.index.tz_localize("UTC").tz_convert("Etc/GMT-1")
    # remove timezone label (timestamps are still UTC+1 in fact)
    df.index = df.index.tz_localize(None)
    return df


