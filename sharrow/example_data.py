import os
from importlib.resources import as_file, files

import numpy as np
import pandas as pd


def get_skims_filename() -> str:
    """Return the path to the example skims file."""
    return os.path.join(os.path.dirname(__file__), "example_data", "skims.omx")


def get_skims_omx():
    import openmatrix

    from . import dataset

    with as_file(files("sharrow").joinpath("example_data/skims.omx")) as filename:
        skims = None
        with openmatrix.open_file(str(filename)) as f:
            skims = dataset.from_omx_3d(
                f,
                index_names=("otaz", "dtaz", "time_period"),
                indexes=None,
                time_periods=["EA", "AM", "MD", "PM", "EV"],
                time_period_sep="__",
                max_float_precision=32,
            ).compute()
    return skims


def get_skims_zarr():
    from . import dataset

    f = files("sharrow").joinpath("example_data/skims.zarr")
    with as_file(f) as zfile:
        if zfile.exists():
            skims = dataset.from_zarr(zfile, consolidated=False)
        else:
            skims = None
    return skims


def get_skims():
    from . import dataset

    f = files("sharrow").joinpath("example_data/skims.zarr")
    with as_file(f) as zfile:
        if zfile.exists():
            skims = dataset.from_zarr(zfile, consolidated=False)
        else:
            skims = get_skims_omx()
    return skims


def get_households():
    with as_file(files("sharrow").joinpath("example_data/households.csv.gz")) as f:
        return pd.read_csv(f, index_col="HHID")


def get_persons():
    with as_file(files("sharrow").joinpath("example_data/persons.csv.gz")) as f:
        return pd.read_csv(f, index_col="PERID")


def get_land_use():
    with as_file(files("sharrow").joinpath("example_data/land_use.csv.gz")) as f:
        return pd.read_csv(f, index_col="TAZ")


def get_maz_to_taz():
    with as_file(files("sharrow").joinpath("example_data/maz_to_taz.csv")) as f:
        return pd.read_csv(f, index_col="MAZ")


def get_maz_to_maz_walk():
    with as_file(files("sharrow").joinpath("example_data/maz_to_maz_walk.csv")) as f:
        return pd.read_csv(f)


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
