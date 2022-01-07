import os

import pandas as pd


def get_skims():
    import openmatrix

    from .dataset import Dataset

    zfilename = os.path.join(os.path.dirname(__file__), "example_data", "skims.zarr")
    if os.path.exists(zfilename):
        skims = Dataset.from_zarr(zfilename, consolidated=False)
    else:
        filename = os.path.join(os.path.dirname(__file__), "example_data", "skims.omx")
        with openmatrix.open_file(filename) as f:
            skims = Dataset.from_omx_3d(
                f,
                index_names=("otaz", "dtaz", "time_period"),
                indexes=None,
                time_periods=["EA", "AM", "MD", "PM", "EV"],
                time_period_sep="__",
                max_float_precision=32,
            ).compute()
        skims.to_zarr(zfilename)
    return skims


def get_households():
    filename = os.path.join(
        os.path.dirname(__file__), "example_data", "households.csv.gz"
    )
    return pd.read_csv(filename, index_col="HHID")


def get_persons():
    filename = os.path.join(os.path.dirname(__file__), "example_data", "persons.csv.gz")
    return pd.read_csv(filename, index_col="PERID")


def get_land_use():
    filename = os.path.join(
        os.path.dirname(__file__), "example_data", "land_use.csv.gz"
    )
    return pd.read_csv(filename, index_col="TAZ")


def get_data():
    result = {
        "hhs": get_households(),
        "persons": get_persons(),
        "land_use": get_land_use(),
        "skims": get_skims(),
    }
    try:
        from addicty import Dict
    except ImportError:
        pass
    else:
        result = Dict(result)
        result.freeze()
    return result
