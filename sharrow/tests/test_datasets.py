import secrets
import tempfile
from pathlib import Path

import numpy as np
import openmatrix
import pandas as pd
import xarray as xr
from pytest import approx

import sharrow as sh


def test_dataset_construct_with_zoneids():
    tempdir = tempfile.TemporaryDirectory()
    t = Path(tempdir.name)

    with openmatrix.open_file(t.joinpath("dummy5.omx"), mode="w") as out:
        out.create_carray("/data", "Eye", obj=np.eye(5, dtype=np.float32))
        out.create_carray("/lookup", "Zone", obj=np.asarray([11, 22, 33, 44, 55]))
        shp = np.empty(2, dtype=int)
        shp[0] = 5
        shp[1] = 5
        out.root._v_attrs.SHAPE = shp

    with openmatrix.open_file(t.joinpath("dummy5.omx"), mode="r") as back:
        ds = sh.dataset.from_omx(back, indexes="Zone")

    assert sorted(ds.coords) == ["dtaz", "otaz"]
    assert ds.coords["otaz"].values == approx(np.asarray([11, 22, 33, 44, 55]))
    assert sorted(ds.variables) == ["Eye", "dtaz", "otaz"]
    assert ds["Eye"].data == approx(np.eye(5, dtype=np.float32))

    with openmatrix.open_file(t.joinpath("dummy5.omx"), mode="r") as back:
        ds0 = sh.dataset.from_omx(back, indexes="zero-based")
    assert ds0.coords["otaz"].values == approx(np.asarray([0, 1, 2, 3, 4]))

    with openmatrix.open_file(t.joinpath("dummy5.omx"), mode="r") as back:
        ds1 = sh.dataset.from_omx(back, indexes="one-based")
    assert ds1.coords["otaz"].values == approx(np.asarray([1, 2, 3, 4, 5]))


def test_dataset_categoricals():
    hhs = sh.example_data.get_households()

    def income_cat(i):
        if i < 12500:
            return "LOW"
        elif i < 45000:
            return "MID"
        else:
            return "HIGH"

    hhs["income_grp"] = hhs.income.apply(income_cat).astype(
        pd.CategoricalDtype(["LOW", "MID", "HIGH"], ordered=True)
    )
    assert hhs["income_grp"].dtype == "category"

    hd = sh.dataset.construct(hhs)
    assert hd["income_grp"].dtype == np.int8

    # affirm we can recover categorical and non-categorical data from datarrays
    pd.testing.assert_series_equal(
        hhs["income_grp"], hd.income_grp.single_dim.to_pandas()
    )
    pd.testing.assert_series_equal(hhs["income"], hd.income.single_dim.to_pandas())

    recovered_df = hd.single_dim.to_pandas()
    pd.testing.assert_frame_equal(hhs, recovered_df)


def test_load_with_ignore():
    filename = sh.example_data.get_skims_filename()
    with openmatrix.open_file(filename) as f:
        skims = sh.dataset.from_omx_3d(
            f,
            index_names=("otaz", "dtaz", "time_period"),
            indexes=None,
            time_periods=["EA", "AM", "MD", "PM", "EV"],
            time_period_sep="__",
            max_float_precision=32,
        )
    assert "DRV_COM_WLK_FAR" in skims.variables

    with openmatrix.open_file(filename) as f:
        skims1 = sh.dataset.from_omx_3d(
            f,
            index_names=("otaz", "dtaz", "time_period"),
            indexes=None,
            time_periods=["EA", "AM", "MD", "PM", "EV"],
            time_period_sep="__",
            max_float_precision=32,
            ignore=["DRV_COM_WLK_.*"],
        )
    assert "DRV_COM_WLK_FAR" not in skims1.variables

    with openmatrix.open_file(filename) as f:
        skims2 = sh.dataset.from_omx_3d(
            f,
            index_names=("otaz", "dtaz", "time_period"),
            indexes=None,
            time_periods=["EA", "AM", "MD", "PM", "EV"],
            time_period_sep="__",
            max_float_precision=32,
            ignore="DRV_COM_WLK_.*",
        )
    print(skims2)
    assert "DISTBIKE" in skims2.variables
    assert "DRV_COM_WLK_FAR" not in skims2.variables


def test_deferred_load_to_shared_memory():
    """
    Test of deferred loading of data into shared memory.

    Checks that skim data is loaded correctly into shared memory
    when using the `to_shared_memory` method with `load=False`, followed by
    a call to `reload_from_omx_3d`.
    """
    from sharrow.example_data import get_skims_filename

    skims_filename = get_skims_filename()
    with openmatrix.open_file(skims_filename) as f:
        d0 = sh.dataset.from_omx_3d(
            f,
            index_names=("otaz", "dtaz", "time_period"),
            time_periods=["EA", "AM", "MD", "PM", "EV"],
            max_float_precision=32,
        )
        token = "skims" + secrets.token_hex(5)
        d1 = d0.shm.to_shared_memory(token, mode="r", load=False)
        sh.dataset.reload_from_omx_3d(d1, [skims_filename])
        xr.testing.assert_equal(d0, d1)
        d2 = xr.Dataset.shm.from_shared_memory(token)
        xr.testing.assert_equal(d0, d2)
