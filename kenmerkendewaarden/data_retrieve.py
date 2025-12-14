# -*- coding: utf-8 -*-
"""
Retrieve data from ddlpy and write to netcdf files including all metadata
"""

import os
import pandas as pd
import ddlpy
from pyproj import Transformer
import pooch
import logging
import dateutil
import hatyan
import xarray as xr

__all__ = [
    "retrieve_measurements_amount",
    "read_measurements_amount",
    "retrieve_measurements",
    "read_measurements",
]

logger = logging.getLogger(__name__)

DICT_FNAMES = {
    "meas_wl": "{station}_meas_wl.nc",
    "meas_ext": "{station}_meas_ext.nc",
    "meas_q": "{station}_meas_q.nc",
    "amount_wl": "data_amount_wl.csv",
    "amount_ext": "data_amount_ext.csv",
    "amount_q": "data_amount_q.csv",
}


def retrieve_catalog(overwrite=False, crs: int = None):
    # create cache dir %USERPROFILE%/AppData/Local/kenmerkendewaarden/kenmerkendewaarden/Cache
    dir_cache = str(pooch.os_cache("kenmerkendewaarden"))
    os.makedirs(dir_cache, exist_ok=True)

    file_catalog_pkl = os.path.join(dir_cache, "DDL_catalog.pkl")
    if os.path.exists(file_catalog_pkl) and not overwrite:
        logger.info("loading DDL locations catalog from pickle")
        locations = pd.read_pickle(file_catalog_pkl)
    else:
        logger.info("retrieving DDL locations catalog with ddlpy")
        # include Typeringen in locations catalog
        catalog_filter = [
            "ProcesTypes",
            "Eenheden",
            "Grootheden",
            "Hoedanigheden",
            "Groeperingen",
            "Parameters",
            "Compartimenten",
            "Typeringen",
        ]
        locations_full = ddlpy.locations(catalog_filter=catalog_filter)
        drop_columns = [
            x for x in locations_full.columns if x.endswith(".Omschrijving")
        ]
        # drop_columns.append("Parameter_Wat_Omschrijving") # TODO: uncomment after ddlpy 0.6.0 is released: https://github.com/Deltares/ddlpy/pull/104
        locations = locations_full.drop(columns=drop_columns)
        pd.to_pickle(locations, file_catalog_pkl)

    # TODO: manually replacing crs name with epsg, the old waterwebservices had epsg in
    # this column, would be great if new wws also has this.
    # https://github.com/Rijkswaterstaat/WaterWebservices/issues/20
    ser_crs_new = locations["Coordinatenstelsel"].replace("ETRS89", "4258").astype(int)
    locations["Coordinatenstelsel"] = ser_crs_new
    # convert coordinates to new crs
    if crs is not None:
        assert len(locations["Coordinatenstelsel"].drop_duplicates()) == 1
        epsg_in = locations["Coordinatenstelsel"].iloc[0]
        transformer = Transformer.from_crs(
            f"epsg:{epsg_in}", f"epsg:{crs}", always_xy=True
        )
        locations["Lon"], locations["Lat"] = transformer.transform(
            locations["Lon"], locations["Lat"]
        )
        locations["Coordinatenstelsel"] = str(crs)

    bool_procestype = locations["ProcesType"].isin(["meting"])
    bool_grootheid = locations["Grootheid.Code"].isin(["WATHTE"])
    bool_groepering_wl = locations["Groepering.Code"].isin([""])
    bool_groepering_ext = locations["Groepering.Code"].isin(["GETETM2", "GETETMSL2"])
    # TODO: for now we do not separately retrieve NAP and MSL for EURPFM/LICHELGRE which have both sets (https://github.com/Rijkswaterstaat/wm-ws-dl/issues/17), these stations are skipped
    # bool_hoedanigheid_nap = locations["Hoedanigheid.Code"].isin(["NAP"])
    # bool_hoedanigheid_msl = locations["Hoedanigheid.Code"].isin(["MSL"])

    # filtering locations dataframe on Typering is possible because "Typeringen" was in catalog_filter for ddlpy.locations
    bool_typering_exttypes = locations["Typering.Code"].isin(["GETETTPE"])

    # filtering locations dataframe on discharge/Q
    bool_grootheid_q = locations["Grootheid.Code"].isin(["Q"])
    bool_eenheid_q = locations["Eenheid.Code"].isin(["m3/s"])

    # select locations on grootheid/groepering/exttypes
    locs_meas_wl = locations.loc[bool_procestype & bool_grootheid & bool_groepering_wl]
    locs_meas_ext = locations.loc[
        bool_procestype & bool_grootheid & bool_groepering_ext
    ]
    locs_meas_exttype = locations.loc[
        bool_procestype & bool_typering_exttypes & bool_groepering_ext
    ]
    locs_meas_q = locations.loc[bool_procestype & bool_grootheid_q & bool_eenheid_q]
    return locs_meas_wl, locs_meas_ext, locs_meas_exttype, locs_meas_q


def raise_multiple_locations(locations):
    """
    checks the amount of rows in a ddlpy.locations dataframe.
    It allows for zero stations, since this regularly happens for extremes.
    It also allows for single stations. It raises an error in case of
    multiple stations, stricter station selection is required.
    """
    if len(locations) > 1:
        raise ValueError(
            f"multiple stations present after station subsetting:\n{locations}"
        )


def raise_incorrect_quantity(quantity):
    """
    checks whether the requested quantity is in the set of allowed quantities
    """
    allowed_quantities = [
        "meas_wl",  # measured waterlevel timeseries [cm]
        "meas_ext",  # measured waterlevel extremes [cm]
        "meas_q",  # measured discharge Q [m3/s]
    ]
    if quantity not in allowed_quantities:
        raise ValueError(
            f"quantity '{quantity}' is not allowed, choose from {allowed_quantities}"
        )


def retrieve_measurements_amount(
    dir_output: str,
    station_list: list,
    quantity: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
):
    """
    Retrieve the amount of measurements or extremes for a single station from the DDL with ddlpy.

    Parameters
    ----------
    dir_output : str
        Path where the measurement netcdf file will be stored.
    station : str
        station name, for instance "hoekvanholland".
    quantity : str
        Whether to retrieve measurements amount for waterlevel timeseries, waterlevel
        extremes or discharges.
    start_date : pd.Timestamp (or anything understood by pd.Timestamp)
        start date of the measurements to be retrieved.
    end_date : pd.Timestamp (or anything understood by pd.Timestamp)
        end date of the measurements to be retrieved.

    Returns
    -------
    None

    """
    locs_meas_wl, locs_meas_ext, _, locs_meas_q = retrieve_catalog()
    raise_incorrect_quantity(quantity)

    if quantity == "meas_wl":
        fname = DICT_FNAMES["amount_wl"]
        locs_meas = locs_meas_wl
    elif quantity == "meas_ext":
        fname = DICT_FNAMES["amount_ext"]
        locs_meas = locs_meas_ext
    elif quantity == "meas_q":
        fname = DICT_FNAMES["amount_q"]
        locs_meas = locs_meas_q
    file_csv_amount = os.path.join(dir_output, fname)

    if os.path.exists(file_csv_amount):
        raise FileExistsError(
            f"{file_csv_amount} already exists, delete file or change dir_output"
        )

    # if csv file(s) do not exist, get the measurement amount from the DDL
    amount_list = []
    for station in station_list:
        logger.info(
            f"retrieving measurement amount (quantity={quantity}) from DDL for "
            f"{station}"
        )

        bool_station = locs_meas.index.isin([station])
        loc_meas_one = locs_meas.loc[bool_station]

        raise_multiple_locations(loc_meas_one)

        def empty_df_row(station):
            empty_df = pd.DataFrame({station: []}, dtype="int64")
            empty_df.index.name = "Groeperingsperiode"
            return empty_df

        if len(loc_meas_one) == 0:
            logger.info(f"no station available (quantity={quantity})")
            # TODO: no ext station available for ["A12","AWGPFM","BAALHK","GATVBSLE","D15","F16","F3PFM","J6","K14PFM",
            #                                     "L9PFM","MAASMSMPL","NORTHCMRT","OVLVHWT","Q1","SINTANLHVSGR","WALSODN"]
            # https://github.com/Rijkswaterstaat/wm-ws-dl/issues/39
            amount_meas = empty_df_row(station)
        else:
            from ddlpy.ddlpy import NoDataError

            try:
                amount_meas = ddlpy.measurements_amount(
                    location=loc_meas_one.iloc[0],
                    start_date=start_date,
                    end_date=end_date,
                )
                amount_meas = amount_meas.rename(columns={"AantalMetingen": station})
            except NoDataError:
                logger.info(
                    f"no measurements available in this period (quantity={quantity})"
                )
                amount_meas = empty_df_row(station)

        amount_list.append(amount_meas)

    logger.info(f"write measurement amount csvs to {os.path.basename(dir_output)}")
    df_amount = pd.concat(amount_list, axis=1).sort_index()
    df_amount = df_amount.fillna(0).astype(int)

    df_amount.to_csv(file_csv_amount)


def read_measurements_amount(dir_output: str, quantity: str):
    """
    Read the measurements amount csv into a dataframe.

    Parameters
    ----------
    dir_output : str
        Path where the measurements are stored.
    quantity : str
        Whether to read measurements amount for waterlevel timeseries, waterlevel
        extremes or discharges.

    Returns
    -------
    df_amount : pd.DataFrame
        DataFrame with the amount of measurements per year.

    """
    raise_incorrect_quantity(quantity)
    if quantity == "meas_wl":
        fname = DICT_FNAMES["amount_wl"]
    elif quantity == "meas_ext":
        fname = DICT_FNAMES["amount_ext"]
    elif quantity == "meas_q":
        fname = DICT_FNAMES["amount_q"]
    file_csv_amount = os.path.join(dir_output, fname)

    if not os.path.exists(file_csv_amount):
        raise FileNotFoundError(f"{file_csv_amount} does not exist")

    logger.info("found existing data amount csv files, loading with pandas")
    df_amount = pd.read_csv(file_csv_amount)
    df_amount = df_amount.set_index("Groeperingsperiode")
    return df_amount


def retrieve_measurements(
    dir_output: str,
    station: str,
    quantity: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    always_preserve: list = None,
):
    """
    Retrieve timeseries with measurements or extremes for a single station from the DDL with ddlpy.

    Parameters
    ----------
    dir_output : str
        Path where the measurement netcdf file will be stored.
    station : str
        station name, for instance "hoekvanholland".
    quantity : str
        Whether to retrieve measurements for waterlevel timeseries, waterlevel extremes
        or discharges.
    start_date : pd.Timestamp (or anything understood by pd.Timestamp)
        start date of the measurements to be retrieved.
    end_date : pd.Timestamp (or anything understood by pd.Timestamp)
        end date of the measurements to be retrieved.
    always_preserve : list, optional
        A list of columns to preserve even if its values are constant. The default is
        None, which defaults to a predefined set.

    Returns
    -------
    None

    """

    locs_meas_wl, locs_meas_ext, locs_meas_exttype, locs_meas_q = retrieve_catalog()
    raise_incorrect_quantity(quantity)

    if always_preserve is None:
        always_preserve = [
            "WaarnemingMetadata.Statuswaarde",
            "WaarnemingMetadata.OpdrachtgevendeInstantie",
            "WaarnemingMetadata.Kwaliteitswaardecode",
            "WaardeBepalingsMethode.Code",
            "Meetwaarde.Waarde_Numeriek",
        ]

    bool_station_wl = locs_meas_wl.index.isin([station])
    bool_station_ext = locs_meas_ext.index.isin([station])
    bool_station_exttype = locs_meas_exttype.index.isin([station])
    bool_station_q = locs_meas_q.index.isin([station])
    loc_meas_wl_one = locs_meas_wl.loc[bool_station_wl]
    loc_meas_ext_one = locs_meas_ext.loc[bool_station_ext]
    loc_meas_exttype_one = locs_meas_exttype.loc[bool_station_exttype]
    loc_meas_q_one = locs_meas_q.loc[bool_station_q]

    if quantity == "meas_wl":
        fname = DICT_FNAMES["meas_wl"].format(station=station)
        loc_meas_one = loc_meas_wl_one
        freq = dateutil.rrule.MONTHLY
    elif quantity == "meas_ext":
        fname = DICT_FNAMES["meas_ext"].format(station=station)
        loc_meas_one = loc_meas_ext_one
        freq = dateutil.rrule.YEARLY
    elif quantity == "meas_q":
        fname = DICT_FNAMES["meas_q"].format(station=station)
        loc_meas_one = loc_meas_q_one
        freq = dateutil.rrule.MONTHLY

    file_nc = os.path.join(dir_output, fname)

    # retrieving waterlevel extremes or timeseries
    if os.path.exists(file_nc):
        logger.info(
            f"meas data (quantity={quantity}) for {station} already available in "
            f"{os.path.basename(dir_output)}, skipping station"
        )
        return

    raise_multiple_locations(loc_meas_one)
    if len(loc_meas_one) == 0:
        logger.info(f"no station available (quantity={quantity}), skipping station")
        return

    logger.info(
        f"retrieving measurement data (quantity={quantity}) from DDL for {station} to {os.path.basename(dir_output)}"
    )
    measurements = ddlpy.measurements(
        location=loc_meas_one.iloc[0],
        start_date=start_date,
        end_date=end_date,
        freq=freq,
    )
    if measurements.empty:
        logger.info("no data found for the requested period")
        return

    ds_meas = ddlpy.dataframe_to_xarray(
        df=measurements,
        always_preserve=always_preserve,
    )
    if quantity == "meas_ext":
        # convert extreme type to HWLWcode add extreme type and HWLcode as dataset variables
        # TODO: simplify by retrieving the extreme value and type from ddl in a single request: https://github.com/Rijkswaterstaat/wm-ws-dl/issues/19
        measurements_exttyp = ddlpy.measurements(
            location=loc_meas_exttype_one.iloc[0],
            start_date=start_date,
            end_date=end_date,
            freq=freq,
        )
        ts_meas_ext_pd = hatyan.ddlpy_to_hatyan(measurements, measurements_exttyp)
        ds_meas["extreme_type"] = xr.DataArray(
            ts_meas_ext_pd["values"].values, dims="time"
        )
        ds_meas["HWLWcode"] = xr.DataArray(
            ts_meas_ext_pd["HWLWcode"].values, dims="time"
        )

    # write to netcdf (including metadata)
    ds_meas.to_netcdf(file_nc, format="NETCDF4_CLASSIC")
    ds_meas.close()


def xarray_to_hatyan(ds):
    """
    converting the xarray dataset in the format of the
    kenmerkendewaarden netcdf files to a hatyan dataframe.
    This saves memory and prevents converting it multiple times
    in the kenmerkendewaarden code when passing it to hatyan.
    """
    values = ds["Meetwaarde.Waarde_Numeriek"].to_pandas()
    qualitycode = ds["WaarnemingMetadata.Kwaliteitswaardecode"].to_pandas()
    status = ds["WaarnemingMetadata.Statuswaarde"].to_pandas()
    df = pd.DataFrame(
        {
            "values": values,
            "qualitycode": qualitycode,
            "status": status,
        }
    )
    if "HWLWcode" in ds.data_vars:
        df["HWLWcode"] = ds["HWLWcode"]

    # convert timezone back to UTC+1
    df.index = df.index.tz_localize("UTC").tz_convert("Etc/GMT-1")

    # add attrs
    df.attrs["station"] = ds.attrs["Code"]
    df.attrs["eenheid"] = ds.attrs["Eenheid.Code"]
    return df


def drop_duplicate_times(df_meas):
    """
    First drop all duplicate time-value-combinations and then all duplicate times.
    The second step makes the first step redundant, but the distinction is still
    visible in the logging which is valuable for assessing the data.
    """
    # drop unique time-value-combinations
    df_meas_withtime = df_meas.copy()
    df_meas_withtime["time"] = df_meas.index
    dupl_timevals = df_meas_withtime.duplicated(keep="first")
    df_meas_clean1 = df_meas.loc[~dupl_timevals]
    nrows_dropped1 = len(df_meas) - len(df_meas_clean1)
    if nrows_dropped1 > 0:
        logger.warning(
            f"{nrows_dropped1} rows with duplicated time-value-combinations dropped"
        )

    # drop unique times that have unique values
    dupl_times = df_meas_clean1.index.duplicated(keep="first")
    df_meas_clean2 = df_meas_clean1.loc[~dupl_times]
    nrows_dropped2 = len(df_meas_clean1) - len(df_meas_clean2)
    if nrows_dropped2 > 0:
        logger.warning(
            f"{nrows_dropped2} rows with duplicated times dropped (unique values dropped)"
        )

    return df_meas_clean2


def read_measurements(
    dir_output: str,
    station: str,
    quantity: str,
    return_xarray: bool = False,
    nap_correction: bool = False,
    drop_duplicates: bool = False,
):
    """
    Read the measurements netcdf as a dataframe.

    Parameters
    ----------
    dir_output : str
        Path where the measurements are stored.
    station : str
        station name, for instance "hoekvanholland".
    quantity : str
        Whether to read measurements for waterlevel timeseries, waterlevel extremes
        or discharges.
    return_xarray : bool, optional
        Whether to return raw xarray.Dataset instead of a DataFrame. No support
        for nap_correction and drop_duplicates. The default is False.
    nap_correction : bool, optional
        Whether to correct for NAP2005. The default is False.
    drop_duplicates : bool, optional
        Whether to drop duplicated timesteps. The default is False.

    Returns
    -------
    df_meas : pd.DataFrame
        DataFrame with the measurements or extremes timeseries.

    """
    raise_incorrect_quantity(quantity)

    if quantity == "meas_wl":
        fname = DICT_FNAMES["meas_wl"].format(station=station)
    elif quantity == "meas_ext":
        fname = DICT_FNAMES["meas_ext"].format(station=station)
    elif quantity == "meas_q":
        fname = DICT_FNAMES["meas_q"].format(station=station)

    file_nc = os.path.join(dir_output, fname)

    if not os.path.exists(file_nc):
        # return None if file does not exist
        logger.info(f"file {fname} not found, returning None")
        return

    logger.info(f"loading {fname}")
    ds_meas = xr.open_dataset(file_nc)
    if return_xarray:
        return ds_meas

    df_meas = xarray_to_hatyan(ds_meas)

    if drop_duplicates:
        df_meas = drop_duplicate_times(df_meas)

    if nap_correction:
        # TODO: not available for all stations
        df_meas = nap2005_correction(df_meas)
    return df_meas


def clip_timeseries_physical_break(df_meas):
    # TODO: move to csv file and add as package data
    # physical_break_dict for slotgemiddelden and overschrijdingsfrequenties
    # values from chapter 6.4 from "Kenmerkende waarden kustwateren en grote rivieren" (Dillingh, 2013)
    # https://open.rijkswaterstaat.nl/open-overheid/onderzoeksrapporten/@44612/kenmerkende-waarden-kustwateren-grote
    # TODO: consider adding nearby stations like cadzand.1, cadzand.badstrand and others
    # TODO: add physical_break for kats.zandkreeksluis (Oosterscheldekering)
    # TODO: maybe use physical_break_dict everywhere to crop data?
    physical_break_dict = {
        "cadzand.2": "1966",
        "stavenisse": "1988",
        "scheveningen": "1962",
        "petten.zuid": "1977",
        "denhelder.marsdiep": "1933",
        "texel.oudeschild": "1933",
        "terschelling.west": "1933",
        "denoever.waddenzee.voorhaven": "1933",
        "harlingen.waddenzee": "1933",
        "vlieland.haven": "1941",
    }

    station = df_meas.attrs["station"]
    if station not in physical_break_dict.keys():
        logger.info(
            f"no physical_break defined for {station}, returning input timeseries"
        )
        return df_meas

    physical_break = physical_break_dict[station]
    assert isinstance(physical_break, str)
    logger.info(
        f"clipping timeseries for {station} before physical_break={physical_break}"
    )
    df_meas = df_meas.loc[physical_break:]

    return df_meas


def nap2005_correction(df_meas):
    # NAP correction for dates before 1-1-2005
    # TODO: check if ths make a difference (for havengetallen it makes a slight difference so yes. For gemgetijkromme it only makes a difference for spring/doodtij. (now only applied at gemgetij en havengetallen)). If so, make this flexible per station, where to get the data or is the RWS data already corrected for it?
    # herdefinitie van NAP (~20mm voor HvH in fig2, relevant?): https://puc.overheid.nl/PUC/Handlers/DownloadDocument.ashx?identifier=PUC_113484_31&versienummer=1
    # Dit is de rapportage waar het gebruik voor PSMSL data voor het eerst beschreven is: https://puc.overheid.nl/PUC/Handlers/DownloadDocument.ashx?identifier=PUC_137204_31&versienummer=1
    # TODO: maybe move dict to csv file and add as package data
    dict_correct_nap2005 = {
        "hoekvanholland": -0.0277,
        "haringvliet.10": -0.0210,
        "vlissingen": -0.0297,
    }

    station = df_meas.attrs["station"]
    if station not in dict_correct_nap2005.keys():
        raise KeyError(f"NAP2005 correction not defined for {station}")

    logger.info(f"applying NAP2005 correction for {station}")
    correct_value = dict_correct_nap2005[station]
    df_meas_corr = df_meas.copy(
        deep=True
    )  # make copy to avoid altering the original dataframe
    before2005bool = df_meas_corr.index < pd.Timestamp("2005-01-01 00:00:00 +01:00")
    df_meas_corr.loc[before2005bool, "values"] = (
        df_meas_corr.loc[before2005bool, "values"] + correct_value
    )

    return df_meas_corr
