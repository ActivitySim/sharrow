import os

import numpy as np
import pandas as pd


def get_skims():
    import openmatrix

    from . import dataset

    zfilename = os.path.join(os.path.dirname(__file__), "example_data", "skims.zarr")
    if os.path.exists(zfilename):
        skims = dataset.from_zarr(zfilename, consolidated=False)
    else:
        filename = os.path.join(os.path.dirname(__file__), "example_data", "skims.omx")
        with openmatrix.open_file(filename) as f:
            skims = dataset.from_omx_3d(
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


def get_maz_to_taz():
    filename = os.path.join(os.path.dirname(__file__), "example_data", "maz_to_taz.csv")
    return pd.read_csv(filename, index_col="MAZ")


def get_maz_to_maz_walk():
    filename = os.path.join(
        os.path.dirname(__file__), "example_data", "maz_to_maz_walk.csv"
    )
    return pd.read_csv(filename)


def get_data():
    result = {
        "hhs": get_households(),
        "persons": get_persons(),
        "land_use": get_land_use(),
        "skims": get_skims(),
        "maz_taz": get_maz_to_taz(),
        "maz_maz_walk": get_maz_to_maz_walk(),
    }
    try:
        from addicty import Dict
    except ImportError:
        pass
    else:
        result = Dict(result)
        result.freeze()
    return result


def get_tour_mode_choice_spec(purpose="work"):
    filename = os.path.join(
        os.path.dirname(__file__), "example_data", "tour_mode_choice_spec.csv"
    )
    coeffs_filename = os.path.join(
        os.path.dirname(__file__), "example_data", "tour_mode_choice_coefs.csv"
    )
    coeffs_template_filename = os.path.join(
        os.path.dirname(__file__), "example_data", "tour_mode_choice_coef_template.csv"
    )
    spec = pd.read_csv(filename, comment="#")
    coefs = pd.read_csv(coeffs_filename, index_col="coefficient_name", comment="#")
    template = pd.read_csv(
        coeffs_template_filename, index_col="coefficient_name", comment="#"
    )
    spec_numeric = (
        spec.iloc[:, 3:]
        .applymap(lambda i: template[purpose].get(i, i))
        .applymap(lambda i: coefs.value.get(i, i))
        .astype(np.float32)
        .fillna(0)
    )
    return pd.concat([spec.iloc[:, :3], spec_numeric], axis=1)
