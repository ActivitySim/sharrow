import secrets
import tempfile
from pathlib import Path

import numpy as np
import openmatrix
import pandas as pd
import pytest
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


def test_from_named_objects():
    from sharrow.dataset import from_named_objects

    s1 = pd.Series([1, 4, 9, 16], name="Squares")
    s2 = pd.Series([2, 3, 5, 7, 11], name="Primes")
    i1 = pd.Index([1, 4, 9, 16], name="Squares")
    a1 = xr.DataArray([1, 4, 9, 16], name="Squares")

    for obj in [s1, i1, a1]:
        ds = from_named_objects(obj, s2)
        assert "Squares" in ds.dims
        assert "Primes" in ds.dims
        assert ds.sizes == {"Squares": 4, "Primes": 5}

    with pytest.raises(ValueError):
        from_named_objects([1, 4, 9, 16], s2)


def test_dataarray_iloc():
    arr = xr.DataArray([1, 4, 9, 16, 25, 36], name="Squares", dims="s")

    assert arr.iloc[1] == 4
    xr.testing.assert_equal(arr.iloc[1:], xr.DataArray([4, 9, 16, 25, 36], dims="s"))
    xr.testing.assert_equal(arr.iloc[:2], xr.DataArray([1, 4], dims="s"))
    xr.testing.assert_equal(arr.iloc[2:4], xr.DataArray([9, 16], dims="s"))
    xr.testing.assert_equal(arr.iloc[:-2], xr.DataArray([1, 4, 9, 16], dims="s"))
    xr.testing.assert_equal(arr.iloc[-2:], xr.DataArray([25, 36], dims="s"))

    with pytest.raises(TypeError):
        arr.iloc[1] = 5  # assignment not allowed

    arr2 = xr.DataArray([2, 3, 5, 7, 11], name="Primes", dims="p")
    arr2d = arr * arr2

    with pytest.raises(TypeError):
        _tmp = arr2d.iloc[1]  # not allowed for 2D arrays

    assert arr2d.iloc[dict(s=1, p=2)] == 20

    z = arr2d.iloc[dict(s=slice(1, 2), p=slice(2, 4))]

    xr.testing.assert_equal(z, xr.DataArray([[20, 28]], dims=["s", "p"]))


def _skims_dataframe(zones=(11, 22, 33, 44), order="row-major"):
    """Create a dense skims dataframe for parquet testing."""
    n = len(zones)
    otaz = np.repeat(np.asarray(zones), n)
    dtaz = np.tile(np.asarray(zones), n)
    df = pd.DataFrame(
        {
            "otaz": otaz,
            "dtaz": dtaz,
            "DIST": (otaz * 1000 + dtaz).astype(np.float32),
            "TIME__AM": (otaz * 10 + dtaz).astype(np.float32),
            "TIME__PM": (otaz * 10 + dtaz + 0.5).astype(np.float32),
        }
    )
    if order == "column-major":
        df = df.sort_values(["dtaz", "otaz"]).reset_index(drop=True)
    return df


def _expected_skims(zones=(11, 22, 33, 44)):
    df = _skims_dataframe(zones)
    return df.set_index(["otaz", "dtaz"]).to_xarray()


def test_from_parquet_3d_row_major():
    with tempfile.TemporaryDirectory() as tempdir:
        f = Path(tempdir).joinpath("skims.parquet")
        _skims_dataframe().to_parquet(f, index=False)
        skims = sh.dataset.from_parquet_3d(f, time_periods=["AM", "PM"])
    expected = _expected_skims()
    assert skims["DIST"].dims == ("otaz", "dtaz")
    assert skims["TIME"].dims == ("otaz", "dtaz", "time_period")
    assert skims.coords["otaz"].values == approx(np.asarray([11, 22, 33, 44]))
    assert skims.coords["dtaz"].values == approx(np.asarray([11, 22, 33, 44]))
    assert list(skims.coords["time_period"].values) == ["AM", "PM"]
    assert skims["DIST"].values == approx(expected["DIST"].values)
    assert skims["TIME"].sel(time_period="AM").values == approx(
        expected["TIME__AM"].values
    )
    assert skims["TIME"].sel(time_period="PM").values == approx(
        expected["TIME__PM"].values
    )


def test_from_parquet_3d_column_major():
    with tempfile.TemporaryDirectory() as tempdir:
        f = Path(tempdir).joinpath("skims.parquet")
        _skims_dataframe(order="column-major").to_parquet(f, index=False)
        skims = sh.dataset.from_parquet_3d(f, time_periods=["AM", "PM"])
    expected = _expected_skims()
    assert skims["DIST"].values == approx(expected["DIST"].values)
    assert skims["TIME"].sel(time_period="PM").values == approx(
        expected["TIME__PM"].values
    )


def test_from_parquet_3d_sparse():
    expected = _expected_skims()
    df = _skims_dataframe()
    # drop some rows to make the data sparse, and shuffle the remainder
    df = df.drop(index=[1, 7]).sample(frac=1.0, random_state=42)
    with tempfile.TemporaryDirectory() as tempdir:
        f = Path(tempdir).joinpath("skims.parquet")
        df.to_parquet(f, index=False)
        skims = sh.dataset.from_parquet_3d(f, time_periods=["AM", "PM"])
    assert skims["DIST"].dims == ("otaz", "dtaz")
    assert skims.coords["otaz"].values == approx(np.asarray([11, 22, 33, 44]))
    dist = skims["DIST"].values
    assert np.isnan(dist[0, 1])
    assert np.isnan(dist[1, 3])
    valid = ~np.isnan(dist)
    assert dist[valid] == approx(expected["DIST"].values[valid])


def test_from_parquet_3d_unsorted_dense():
    df = _skims_dataframe()
    # a dense but improperly sorted table should raise an error
    df = df.sample(frac=1.0, random_state=42)
    with tempfile.TemporaryDirectory() as tempdir:
        f = Path(tempdir).joinpath("skims.parquet")
        df.to_parquet(f, index=False)
        with pytest.raises(ValueError):
            sh.dataset.from_parquet_3d(f, time_periods=["AM", "PM"])


def test_from_parquet_3d_multiple_files():
    expected = _expected_skims()
    df = _skims_dataframe()
    with tempfile.TemporaryDirectory() as tempdir:
        f1 = Path(tempdir).joinpath("skims1.parquet")
        f2 = Path(tempdir).joinpath("skims2.parquet")
        df[["otaz", "dtaz", "DIST", "TIME__AM"]].to_parquet(f1, index=False)
        # the second file is written in column-major order, to check that
        # each file is inspected independently
        df[["otaz", "dtaz", "TIME__PM"]].sort_values(["dtaz", "otaz"]).to_parquet(
            f2, index=False
        )
        skims = sh.dataset.from_parquet_3d([f1, f2], time_periods=["AM", "PM"])
    assert skims["DIST"].values == approx(expected["DIST"].values)
    assert skims["TIME"].sel(time_period="AM").values == approx(
        expected["TIME__AM"].values
    )
    assert skims["TIME"].sel(time_period="PM").values == approx(
        expected["TIME__PM"].values
    )


def test_from_parquet_3d_mismatched_index_order():
    expected = _expected_skims()
    df = _skims_dataframe()
    df2 = _skims_dataframe(zones=(44, 33, 22, 11))
    with tempfile.TemporaryDirectory() as tempdir:
        f1 = Path(tempdir).joinpath("skims1.parquet")
        f2 = Path(tempdir).joinpath("skims2.parquet")
        df[["otaz", "dtaz", "DIST"]].to_parquet(f1, index=False)
        df2[["otaz", "dtaz", "TIME__AM", "TIME__PM"]].to_parquet(f2, index=False)
        skims = sh.dataset.from_parquet_3d([f1, f2], time_periods=["AM", "PM"])
    assert skims.coords["otaz"].values == approx(np.asarray([11, 22, 33, 44]))
    assert skims["DIST"].values == approx(expected["DIST"].values)
    assert skims["TIME"].sel(time_period="AM").values == approx(
        expected["TIME__AM"].values
    )


def test_from_parquet_3d_ignore():
    with tempfile.TemporaryDirectory() as tempdir:
        f = Path(tempdir).joinpath("skims.parquet")
        _skims_dataframe().to_parquet(f, index=False)
        skims = sh.dataset.from_parquet_3d(
            f, time_periods=["AM", "PM"], ignore="TIME.*"
        )
    assert "TIME" not in skims.variables
    assert "DIST" in skims.variables


def _write_compressed_omx(path, matrices, complib="zlib", complevel=7, shuffle=True):
    """Write an OMX file with compressed, oddly-chunked matrix tables."""
    import tables

    filters = tables.Filters(complevel=complevel, complib=complib, shuffle=shuffle)
    n1, n2 = next(iter(matrices.values())).shape
    with openmatrix.open_file(path, mode="w") as out:
        for name, arr in matrices.items():
            out.create_carray(
                "/data",
                name,
                obj=arr,
                filters=filters,
                # chunk shape that does not evenly divide the array,
                # to exercise edge-chunk handling
                chunkshape=(7, 7),
            )
        out.create_carray("/lookup", "taz", obj=np.arange(11, 11 + n1))
        shp = np.empty(2, dtype=int)
        shp[0], shp[1] = n1, n2
        out.root._v_attrs.SHAPE = shp


def _random_matrices(n=25, seed=42):
    rng = np.random.default_rng(seed)
    return {
        "DIST": rng.random((n, n)),
        "TIME__AM": (rng.random((n, n)) * 100).astype(np.float32),
        "TIME__PM": (rng.random((n, n)) * 100).astype(np.float32),
        "COUNTS": rng.integers(0, 9999, (n, n)).astype(np.int32),
    }


def test_from_omx_compressed_zlib():
    matrices = _random_matrices()
    with tempfile.TemporaryDirectory() as tempdir:
        f = Path(tempdir).joinpath("skims.omx")
        _write_compressed_omx(f, matrices)
        with openmatrix.open_file(f, mode="r") as back:
            ds = sh.dataset.from_omx(back, indexes="taz")
            ds_renamed = sh.dataset.from_omx(
                back, indexes="taz", renames={"distance": "DIST"}
            )
            ds_limited = sh.dataset.from_omx(back, indexes="taz", renames=["COUNTS"])
    for name, arr in matrices.items():
        assert ds[name].dtype == arr.dtype
        np.testing.assert_array_equal(ds[name].values, arr)
    assert ds.coords["otaz"].values == approx(np.arange(11, 36))
    np.testing.assert_array_equal(ds_renamed["distance"].values, matrices["DIST"])
    assert sorted(ds_limited.data_vars) == ["COUNTS"]


def test_from_omx_compressed_h5py_handle():
    import h5py

    matrices = _random_matrices()
    with tempfile.TemporaryDirectory() as tempdir:
        f = Path(tempdir).joinpath("skims.omx")
        _write_compressed_omx(f, matrices)
        with h5py.File(f, mode="r") as back:
            ds = sh.dataset.from_omx(back, indexes="taz")
    for name, arr in matrices.items():
        np.testing.assert_array_equal(ds[name].values, arr)
    assert ds.coords["otaz"].values == approx(np.arange(11, 36))


def test_from_omx_3d_compressed_zlib():
    matrices = _random_matrices()
    with tempfile.TemporaryDirectory() as tempdir:
        f = Path(tempdir).joinpath("skims.omx")
        _write_compressed_omx(f, matrices)
        skims = sh.dataset.from_omx_3d(
            str(f),
            time_periods=["AM", "PM"],
            max_float_precision=64,
        ).compute()
        np.testing.assert_array_equal(skims["DIST"].values, matrices["DIST"])
        np.testing.assert_array_equal(
            skims["TIME"].sel(time_period="AM").values, matrices["TIME__AM"]
        )
        np.testing.assert_array_equal(
            skims["TIME"].sel(time_period="PM").values, matrices["TIME__PM"]
        )
        assert skims["TIME"].dtype == np.float32
        assert skims.coords["otaz"].values == approx(np.arange(11, 36))

        # also via an already-open file handle
        with openmatrix.open_file(f, mode="r") as back:
            skims2 = sh.dataset.from_omx_3d(
                back,
                time_periods=["AM", "PM"],
                max_float_precision=64,
            )
        # data remains readable after the handle is closed
        xr.testing.assert_equal(skims, skims2.compute())


def test_reload_from_omx_3d_compressed():
    matrices = _random_matrices()
    with tempfile.TemporaryDirectory() as tempdir:
        f = Path(tempdir).joinpath("skims.omx")
        _write_compressed_omx(f, matrices)
        expected = sh.dataset.from_omx_3d(
            str(f), time_periods=["AM", "PM"], max_float_precision=32
        ).compute()
        blank = xr.zeros_like(expected)
        assert not blank.equals(expected)
        sh.dataset.reload_from_omx_3d(blank, [str(f)])
        xr.testing.assert_equal(blank, expected)
        # with ignore
        blank2 = xr.zeros_like(expected)
        sh.dataset.reload_from_omx_3d(blank2, [str(f)], ignore=["COUNTS"])
        assert (blank2["COUNTS"].values == 0).all()
        np.testing.assert_array_equal(blank2["DIST"].values, expected["DIST"].values)


def test_from_omx_compressed_blosc():
    import h5py
    import hdf5plugin

    rng = np.random.default_rng(7)
    arr = rng.random((25, 25))
    with tempfile.TemporaryDirectory() as tempdir:
        f = Path(tempdir).joinpath("skims.omx")
        with h5py.File(f, mode="w") as out:
            out.create_dataset(
                "data/DIST",
                data=arr,
                chunks=(7, 7),
                **hdf5plugin.Blosc(cname="lz4", clevel=5),
            )
            out.create_dataset("lookup/taz", data=np.arange(11, 36))
            out.attrs["SHAPE"] = np.asarray([25, 25], dtype=int)
        with h5py.File(f, mode="r") as back:
            ds = sh.dataset.from_omx(back, indexes="taz")
    np.testing.assert_array_equal(ds["DIST"].values, arr)


def test_from_omx_3d_to_zarr():
    """Lazily loaded 3d skims remain readable when writing to zarr."""
    matrices = _random_matrices()
    with tempfile.TemporaryDirectory() as tempdir:
        f = Path(tempdir).joinpath("skims.omx")
        _write_compressed_omx(f, matrices)
        skims = sh.dataset.from_omx_3d(str(f), time_periods=["AM", "PM"])
        zarr_path = Path(tempdir).joinpath("skims.zarr")
        skims[["TIME"]].to_zarr(zarr_path, mode="w")
        back = xr.open_zarr(zarr_path)
        np.testing.assert_array_equal(
            back["TIME"].sel(time_period="AM").values, matrices["TIME__AM"]
        )


def test_from_omx_3d_writable_handle():
    """A file handle open for writing does not block lazy loading."""
    matrices = _random_matrices()
    with tempfile.TemporaryDirectory() as tempdir:
        f = Path(tempdir).joinpath("skims.omx")
        _write_compressed_omx(f, matrices)
        with openmatrix.open_file(f, mode="a") as back:
            skims = sh.dataset.from_omx_3d(
                back, time_periods=["AM", "PM"], max_float_precision=64
            )
            computed = skims.compute()
        np.testing.assert_array_equal(computed["DIST"].values, matrices["DIST"])
        np.testing.assert_array_equal(
            computed["TIME"].sel(time_period="PM").values, matrices["TIME__PM"]
        )
