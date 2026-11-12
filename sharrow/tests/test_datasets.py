import gc
import secrets
import tempfile
from pathlib import Path

import h5py
import numpy as np
import openmatrix
import pandas as pd
import pytest
import xarray as xr
from pytest import approx

import sharrow as sh
from sharrow import omx_reader
from sharrow.shared_memory import SharedMemDatasetAccessor


def test_dataset_construct_with_zoneids():
    tempdir = tempfile.TemporaryDirectory()
    t = Path(tempdir.name)

    with h5py.File(t.joinpath("dummy5.omx"), mode="w") as out:
        out.create_dataset("data/Eye", data=np.eye(5, dtype=np.float32))
        out.create_dataset("lookup/Zone", data=np.asarray([11, 22, 33, 44, 55]))
        out.attrs["SHAPE"] = np.asarray([5, 5], dtype=int)

    with h5py.File(t.joinpath("dummy5.omx"), mode="r") as back:
        ds = sh.dataset.from_omx(back, indexes="Zone")

    assert sorted(ds.coords) == ["dtaz", "otaz"]
    assert ds.coords["otaz"].values == approx(np.asarray([11, 22, 33, 44, 55]))
    assert sorted(ds.variables) == ["Eye", "dtaz", "otaz"]
    assert ds["Eye"].data == approx(np.eye(5, dtype=np.float32))

    with h5py.File(t.joinpath("dummy5.omx"), mode="r") as back:
        ds0 = sh.dataset.from_omx(back, indexes="zero-based")
    assert ds0.coords["otaz"].values == approx(np.asarray([0, 1, 2, 3, 4]))

    with h5py.File(t.joinpath("dummy5.omx"), mode="r") as back:
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
    with h5py.File(filename) as f:
        skims = sh.dataset.from_omx_3d(
            f,
            index_names=("otaz", "dtaz", "time_period"),
            indexes=None,
            time_periods=["EA", "AM", "MD", "PM", "EV"],
            time_period_sep="__",
            max_float_precision=32,
        )
    assert "DRV_COM_WLK_FAR" in skims.variables

    with h5py.File(filename) as f:
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

    with h5py.File(filename) as f:
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
    with h5py.File(skims_filename) as f:
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


def _write_compressed_omx(path, matrices, compression="gzip", compression_opts=7):
    """Write an OMX file with compressed, oddly-chunked matrix tables."""
    n1, n2 = next(iter(matrices.values())).shape
    with h5py.File(path, mode="w") as out:
        for name, arr in matrices.items():
            out.create_dataset(
                f"data/{name}",
                data=arr,
                compression=compression,
                compression_opts=compression_opts,
                shuffle=True,
                # chunk shape that does not evenly divide the array,
                # to exercise edge-chunk handling
                chunks=(7, 7),
            )
        out.create_dataset("lookup/taz", data=np.arange(11, 11 + n1))
        out.attrs["SHAPE"] = np.asarray([n1, n2], dtype=int)


def _random_matrices(n=25, seed=42):
    rng = np.random.default_rng(seed)
    return {
        "DIST": rng.random((n, n)),
        "TIME__AM": (rng.random((n, n)) * 100).astype(np.float32),
        "TIME__PM": (rng.random((n, n)) * 100).astype(np.float32),
        "COUNTS": rng.integers(0, 9999, (n, n)).astype(np.int32),
    }


def test_parallel_omx_reader_converts_into_noncontiguous_output(tmp_path):
    """Parallel chunk decoding casts directly into a final array view."""
    source = np.arange(625, dtype=np.float64).reshape(25, 25) / 7
    path = tmp_path / "direct-conversion.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "data",
            data=source,
            compression="gzip",
            shuffle=True,
            chunks=(7, 7),
        )

    backing = np.empty((25, 25, 2), dtype=np.float32)
    destination = backing[..., 1]
    assert not destination.flags.c_contiguous
    with h5py.File(path, "r") as handle:
        result = omx_reader.read_dataset(handle["data"], out=destination, max_workers=2)

    assert result is destination
    np.testing.assert_array_equal(destination, source.astype(np.float32))


def test_parallel_omx_reader_fallback_converts_dtype(tmp_path):
    """The native h5py fallback also converts into the requested dtype."""
    source = np.arange(36, dtype=np.float64).reshape(6, 6) / 7
    path = tmp_path / "uncompressed.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=source)

    destination = np.empty(source.shape, dtype=np.float32)
    with h5py.File(path, "r") as handle:
        omx_reader.read_dataset(handle["data"], out=destination, max_workers=2)
    np.testing.assert_array_equal(destination, source.astype(np.float32))


def test_omx_shape_is_inferred_when_attribute_is_missing(tmp_path):
    """Files written without SHAPE retain OpenMatrix-compatible behavior."""
    matrices = _random_matrices()
    path = tmp_path / "missing-shape.omx"
    _write_compressed_omx(path, matrices)
    with h5py.File(path, "r+") as handle:
        del handle.attrs["SHAPE"]

    two_dimensional = sh.dataset.from_omx(path, indexes="taz")
    three_dimensional = sh.dataset.from_omx_3d(
        path,
        indexes="taz",
        time_periods=["AM", "PM"],
        load="eager",
    )

    np.testing.assert_array_equal(two_dimensional["DIST"], matrices["DIST"])
    np.testing.assert_array_equal(
        three_dimensional["DIST"], matrices["DIST"].astype(np.float32)
    )
    np.testing.assert_array_equal(
        three_dimensional["TIME"].sel(time_period="PM"), matrices["TIME__PM"]
    )


def test_from_omx_compressed_zlib():
    matrices = _random_matrices()
    with tempfile.TemporaryDirectory() as tempdir:
        f = Path(tempdir).joinpath("skims.omx")
        _write_compressed_omx(f, matrices)
        with h5py.File(f, mode="r") as back:
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


def test_from_omx_compressed_path():
    matrices = _random_matrices()
    with tempfile.TemporaryDirectory() as tempdir:
        f = Path(tempdir).joinpath("skims.omx")
        _write_compressed_omx(f, matrices)
        ds = sh.dataset.from_omx(f, indexes="taz")
    for name, arr in matrices.items():
        np.testing.assert_array_equal(ds[name].values, arr)
    assert ds.coords["otaz"].values == approx(np.arange(11, 36))


def test_from_omx_filename_bearing_legacy_handle():
    """Legacy OMX handles are reopened by filename without being imported."""

    class LegacyOMXHandle:
        """Minimal protocol exposed by openmatrix.File and PyTables File."""

        def __init__(self, filename):
            self.filename = filename
            self.close_called = False

        def close(self):
            self.close_called = True

    matrices = _random_matrices()
    with tempfile.TemporaryDirectory() as tempdir:
        f = Path(tempdir).joinpath("skims.omx")
        _write_compressed_omx(f, matrices)
        legacy_handle = LegacyOMXHandle(f)

        expected_2d = sh.dataset.from_omx(f, indexes="taz")
        actual_2d = sh.dataset.from_omx(legacy_handle, indexes="taz")
        xr.testing.assert_equal(actual_2d, expected_2d)

        expected_3d = sh.dataset.from_omx_3d(f, time_periods=["AM", "PM"], load="eager")
        actual_3d = sh.dataset.from_omx_3d(
            [legacy_handle], time_periods=["AM", "PM"]
        ).compute()
        xr.testing.assert_equal(actual_3d, expected_3d)

        reloaded = xr.zeros_like(expected_3d)
        sh.dataset.reload_from_omx_3d(reloaded, legacy_handle)
        xr.testing.assert_equal(reloaded, expected_3d)

        # Sharrow owns only the temporary h5py handle it creates. The caller's
        # compatibility handle remains open and under caller control.
        assert not legacy_handle.close_called


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
        with h5py.File(f, mode="r") as back:
            skims2 = sh.dataset.from_omx_3d(
                back,
                time_periods=["AM", "PM"],
                max_float_precision=64,
            )
        # data remains readable after the handle is closed
        xr.testing.assert_equal(skims, skims2.compute())


def test_from_omx_3d_loading_modes():
    """Lazy batching and direct eager loading produce identical skim arrays."""
    matrices = _random_matrices()
    with tempfile.TemporaryDirectory() as tempdir:
        f = Path(tempdir).joinpath("skims.omx")
        _write_compressed_omx(f, matrices)
        lazy_variable = sh.dataset.from_omx_3d(
            f,
            time_periods=["EA", "AM", "PM"],
            task_granularity="variable",
        )
        lazy_matrix = sh.dataset.from_omx_3d(
            f,
            time_periods=["EA", "AM", "PM"],
            task_granularity="matrix",
        )
        eager = sh.dataset.from_omx_3d(f, time_periods=["EA", "AM", "PM"], load="eager")

        # One task per logical variable substantially reduces scheduler work.
        assert len(lazy_variable.__dask_graph__()) < len(lazy_matrix.__dask_graph__())
        xr.testing.assert_equal(lazy_variable.compute(), eager)
        xr.testing.assert_equal(lazy_matrix.compute(), eager)
        assert (eager["TIME"].sel(time_period="EA").values == 0).all()
        assert eager["TIME"].data[..., 0].flags.c_contiguous
        assert eager["DIST"].dtype == np.float32


def test_from_omx_3d_shared_parallel():
    """Separate OMX files can load concurrently into final shared pages."""
    matrices = _random_matrices()
    with tempfile.TemporaryDirectory() as tempdir:
        first = Path(tempdir).joinpath("first.omx")
        second = Path(tempdir).joinpath("second.omx")
        _write_compressed_omx(
            first, {name: data for name, data in matrices.items() if name != "TIME__PM"}
        )
        _write_compressed_omx(second, {"TIME__PM": matrices["TIME__PM"]})
        expected = sh.dataset.from_omx_3d(
            [first, second], time_periods=["EA", "AM", "PM"], load="eager"
        )
        token = "parallel-skims-" + secrets.token_hex(5)
        shared = sh.dataset.from_omx_3d(
            [first, second],
            time_periods=["EA", "AM", "PM"],
            load="shared",
            workers=2,
            shared_memory_key=token,
        )
        try:
            xr.testing.assert_equal(shared, expected)
            assert shared.shm.is_shared_memory
            assert shared["TIME"].data[..., 1].flags.c_contiguous
        finally:
            shared.shm.release_shared_memory()


def test_from_omx_3d_memmap_low_memory():
    """The low-memory mode loads directly into a new disk-backed array."""
    matrices = _random_matrices()
    with tempfile.TemporaryDirectory() as tempdir:
        f = Path(tempdir).joinpath("skims.omx")
        backing = Path(tempdir).joinpath("skims-memory.dat")
        _write_compressed_omx(f, matrices)
        expected = sh.dataset.from_omx_3d(
            f, time_periods=["EA", "AM", "PM"], load="eager"
        )
        mapped = sh.dataset.from_omx_3d(
            f,
            time_periods=["EA", "AM", "PM"],
            load="memmap",
            memory_path=backing,
            workers=2,
        )
        key = mapped.shm.shared_memory_key
        try:
            xr.testing.assert_equal(mapped, expected)
            assert backing.exists()
            assert Path(f"{backing}.meta.pkl").exists()
            assert isinstance(mapped.shm._shared_memory_objs_[-1], np.memmap)

            with pytest.raises(FileExistsError):
                sh.dataset.from_omx_3d(
                    f,
                    time_periods=["EA", "AM", "PM"],
                    load="memmap",
                    memory_path=backing,
                )
        finally:
            mapped.shm.release_shared_memory()
            del mapped
            gc.collect()
            SharedMemDatasetAccessor.delete_shared_memory_files(key)

        assert not backing.exists()
        assert not Path(f"{backing}.meta.pkl").exists()


def test_windows_parallel_backend_uses_threads(monkeypatch, tmp_path):
    """The Windows backend avoids spawn and caps workers to active sources."""
    matrices = _random_matrices()
    first = tmp_path / "first.omx"
    second = tmp_path / "second.omx"
    backing = tmp_path / "skims-memory.dat"
    _write_compressed_omx(
        first, {name: data for name, data in matrices.items() if name != "TIME__PM"}
    )
    _write_compressed_omx(second, {"TIME__PM": matrices["TIME__PM"]})
    expected = sh.dataset.from_omx_3d(
        [first, second], time_periods=["EA", "AM", "PM"], load="eager"
    )

    def forbid_process_pool(*args, **kwargs):
        pytest.fail("the Windows OMX backend must not create a process pool")

    monkeypatch.setattr(sh.dataset, "_omx_uses_processes", lambda: False)
    monkeypatch.setattr(
        sh.dataset.concurrent.futures, "ProcessPoolExecutor", forbid_process_pool
    )
    mapped = sh.dataset.from_omx_3d(
        [first, second],
        time_periods=["EA", "AM", "PM"],
        load="memmap",
        memory_path=backing,
        workers=100,
    )
    key = mapped.shm.shared_memory_key
    try:
        xr.testing.assert_equal(mapped, expected)
        assert sh.dataset._omx_worker_count(100, 2) == 2
    finally:
        mapped.shm.release_shared_memory()
        del mapped
        gc.collect()
        SharedMemDatasetAccessor.delete_shared_memory_files(key)


def test_memmap_metadata_failure_removes_new_backing_file(monkeypatch, tmp_path):
    """Partial memmap construction is transactional, including on Windows."""
    source = tmp_path / "skims.omx"
    backing = tmp_path / "skims-memory.dat"
    _write_compressed_omx(source, _random_matrices())

    def fail_metadata(*args, **kwargs):
        raise RuntimeError("injected metadata failure")

    monkeypatch.setattr(sh.shared_memory, "create_shared_list", fail_metadata)
    with pytest.raises(RuntimeError, match="injected metadata failure"):
        sh.dataset.from_omx_3d(
            source,
            time_periods=["EA", "AM", "PM"],
            load="memmap",
            memory_path=backing,
            workers=1,
        )

    assert not backing.exists()
    assert not Path(f"{backing}.meta.pkl").exists()


def test_from_omx_3d_memmap_failure_releases_before_delete(monkeypatch, tmp_path):
    """A failed memmap load closes its mapping before deleting backing files."""
    source = tmp_path / "skims.omx"
    backing = tmp_path / "skims-memory.dat"
    _write_compressed_omx(source, _random_matrices())

    def fail_load(*args, **kwargs):
        raise RuntimeError("injected load failure")

    cleanup_steps = []
    original_release = SharedMemDatasetAccessor.release_shared_memory
    original_delete = SharedMemDatasetAccessor.delete_shared_memory_files

    def record_release(accessor):
        cleanup_steps.append("release")
        original_release(accessor)

    def record_delete(key):
        # Model Windows' requirement that a file cannot be unlinked while its
        # memory mapping remains open.
        assert cleanup_steps == ["release"]
        cleanup_steps.append("delete")
        original_delete(key)

    monkeypatch.setattr(sh.dataset, "_load_omx_assignments", fail_load)
    monkeypatch.setattr(
        SharedMemDatasetAccessor, "release_shared_memory", record_release
    )
    monkeypatch.setattr(
        SharedMemDatasetAccessor,
        "delete_shared_memory_files",
        staticmethod(record_delete),
    )

    with pytest.raises(RuntimeError, match="injected load failure"):
        sh.dataset.from_omx_3d(
            source,
            time_periods=["EA", "AM", "PM"],
            load="memmap",
            memory_path=backing,
            workers=1,
        )

    assert cleanup_steps == ["release", "delete"]
    assert not backing.exists()
    assert not Path(f"{backing}.meta.pkl").exists()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"load": "invalid"}, "load must be"),
        ({"task_granularity": "invalid"}, "task_granularity"),
        ({"load": "shared", "workers": 0}, "positive integer"),
        ({"load": "memmap"}, "memory_path is required"),
        ({"load": "eager", "workers": 2}, "load='shared'"),
    ],
)
def test_from_omx_3d_loading_mode_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        sh.dataset.from_omx_3d(
            sh.example_data.get_skims_filename(),
            time_periods=["EA", "AM", "MD", "PM", "EV"],
            **kwargs,
        )


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
        # An h5py handle is accepted directly and remains owned by the caller.
        blank3 = xr.zeros_like(expected)
        with h5py.File(f, "r") as handle:
            sh.dataset.reload_from_omx_3d(blank3, handle)
            assert handle.id.valid
        xr.testing.assert_equal(blank3, expected)

        # The time-period dimension need not use Sharrow's default name.
        custom_dims = xr.zeros_like(expected.rename(time_period="period"))
        sh.dataset.reload_from_omx_3d(custom_dims, f)
        xr.testing.assert_equal(custom_dims, expected.rename(time_period="period"))


def test_reload_from_omx_3d_shared_parallel():
    """Parallel reload fills only selected arrays and preserves source precedence."""
    matrices = _random_matrices()
    with tempfile.TemporaryDirectory() as tempdir:
        first = Path(tempdir).joinpath("first.omx")
        second = Path(tempdir).joinpath("second.omx")
        _write_compressed_omx(
            first, {name: data for name, data in matrices.items() if name != "TIME__PM"}
        )
        _write_compressed_omx(
            second,
            {
                "DIST": matrices["DIST"] + 1,
                "TIME__PM": matrices["TIME__PM"],
            },
        )

        expected = sh.dataset.from_omx_3d(
            [first, second], time_periods=["AM", "PM"], load="eager"
        ).drop_vars("COUNTS")
        blank = xr.zeros_like(expected)
        token = "parallel-reload-" + secrets.token_hex(5)
        shared = blank.shm.to_shared_memory(
            token,
            mode="r+",
            load=False,
            array_order={
                name: "last-axis-first"
                for name, variable in blank.data_vars.items()
                if variable.ndim == 3
            },
        )
        try:
            sh.dataset.reload_from_omx_3d(shared, [first, second], workers=2)
            xr.testing.assert_equal(shared, expected)
            assert shared["TIME"].data[..., 0].flags.c_contiguous
        finally:
            shared.shm.release_shared_memory()

        with pytest.raises(ValueError, match="shared-memory-backed"):
            sh.dataset.reload_from_omx_3d(blank, [first, second], workers=2)
        with pytest.raises(ValueError, match="positive integer"):
            sh.dataset.reload_from_omx_3d(blank, [first, second], workers=0)


def test_from_omx_compressed_blosc():
    hdf5plugin = pytest.importorskip("hdf5plugin")

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
        token = "blosc-skims-" + secrets.token_hex(5)
        shared = sh.dataset.from_omx_3d(
            f,
            indexes="taz",
            time_periods=["AM"],
            load="shared",
            workers=2,
            shared_memory_key=token,
        )
    np.testing.assert_array_equal(ds["DIST"].values, arr)
    try:
        np.testing.assert_array_equal(shared["DIST"].values, arr.astype(np.float32))
    finally:
        shared.shm.release_shared_memory()


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
        with h5py.File(f, mode="a") as back:
            skims = sh.dataset.from_omx_3d(
                back, time_periods=["AM", "PM"], max_float_precision=64
            )
            computed = skims.compute()
        np.testing.assert_array_equal(computed["DIST"].values, matrices["DIST"])
        np.testing.assert_array_equal(
            computed["TIME"].sel(time_period="PM").values, matrices["TIME__PM"]
        )


@pytest.mark.parametrize("load", ["lazy", "eager", "shared", "memmap"])
@pytest.mark.parametrize("source_kind", ["path", "openmatrix", "h5py"])
def test_openmatrix_loading_modes(tmp_path, load, source_kind):
    """OpenMatrix-created files support all loaders and caller-owned handles."""
    import contextlib

    matrices = _random_matrices()
    path = tmp_path / "openmatrix.omx"
    with openmatrix.open_file(path, "w") as handle:
        for name, values in matrices.items():
            handle.create_matrix(name, obj=values, chunks=(7, 7))
        handle.create_mapping("taz", np.arange(11, 36))
    kwargs = {"memory_path": tmp_path / "skims.dat"} if load == "memmap" else {}
    opener = {"openmatrix": openmatrix.open_file, "h5py": h5py.File}
    context = (
        opener[source_kind](path, "r")
        if source_kind != "path"
        else contextlib.nullcontext(path)
    )
    with context as source:
        two_d = sh.dataset.from_omx(source, indexes="taz")
        np.testing.assert_array_equal(two_d["DIST"], matrices["DIST"])
        dataset = sh.dataset.from_omx_3d(
            source, time_periods=["AM", "PM"], load=load, **kwargs
        )
        if source_kind != "path":
            assert source.id.valid
    # Lazy arrays must still work after the caller closes the source.
    key = dataset.shm.shared_memory_key if load in {"shared", "memmap"} else None
    try:
        dataset.load()
        np.testing.assert_array_equal(dataset.coords["otaz"], np.arange(11, 36))
        np.testing.assert_array_equal(
            dataset["DIST"], matrices["DIST"].astype("float32")
        )
        for period in ["AM", "PM"]:
            np.testing.assert_array_equal(
                dataset["TIME"].sel(time_period=period), matrices[f"TIME__{period}"]
            )
        # Reload also accepts the real OpenMatrix handle, without closing it.
        dataset["DIST"].data[...] = 0
        with openmatrix.open_file(path, "r") as handle:
            sh.dataset.reload_from_omx_3d(dataset, handle)
            assert handle.id.valid
        np.testing.assert_array_equal(
            dataset["DIST"], matrices["DIST"].astype("float32")
        )
    finally:
        if key is not None:
            dataset.shm.release_shared_memory()
            del dataset
            gc.collect()
            SharedMemDatasetAccessor.delete_shared_memory_files(key)


def test_openmatrix_in_memory_handle():
    """Non-reopenable OpenMatrix files load directly and stay caller-owned."""
    with openmatrix.open_file(
        "in-memory.omx", "w", driver="core", backing_store=False
    ) as handle:
        handle.create_matrix("DIST", obj=np.eye(3))
        handle.create_mapping("taz", [11, 22, 33])
        two_d = sh.dataset.from_omx(handle, indexes="taz")
        three_d = sh.dataset.from_omx_3d(handle, time_periods=["AM"])
        assert handle.id.valid
    np.testing.assert_array_equal(two_d["DIST"], np.eye(3))
    np.testing.assert_array_equal(three_d["DIST"], np.eye(3, dtype="float32"))


def test_memmap_release_keeps_retained_views_alive(tmp_path):
    """Releasing a Dataset must not unmap storage still used by NumPy views."""
    import subprocess
    import sys

    source = tmp_path / "skims.omx"
    _write_compressed_omx(source, _random_matrices())
    code = """
import gc
import sys
import weakref
import numpy as np
import sharrow as sh

ds = sh.dataset.from_omx_3d(
    sys.argv[1], time_periods=['AM', 'PM'], load='memmap',
    memory_path=sys.argv[2], workers=1,
)
key = ds.shm.shared_memory_key
mapping = weakref.ref(ds.shm._shared_memory_objs_[0])
view = ds['DIST'].values
expected = view.copy()
ds.shm.release_shared_memory()
ds.shm.release_shared_memory()
np.testing.assert_array_equal(ds['DIST'].values, expected)
np.testing.assert_array_equal(view, expected)
del ds
gc.collect()
np.testing.assert_array_equal(view, expected)
assert mapping() is not None
del view
gc.collect()
assert mapping() is None
sh.shared_memory.delete_shared_memory_files(key)
"""
    subprocess.run(
        [sys.executable, "-c", code, str(source), str(tmp_path / "mapped.dat")],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert not (tmp_path / "mapped.dat").exists()
    assert not (tmp_path / "mapped.dat.meta.pkl").exists()


@pytest.mark.parametrize("mode", ["r", "a", "w", "memory"])
def test_reload_preserves_caller_handle(tmp_path, mode):
    """Serial reload uses the live handle even when no backing file exists."""
    path = tmp_path / "source.omx"
    values = np.arange(9, dtype=np.float64).reshape(3, 3)
    if mode in {"r", "a"}:
        with openmatrix.open_file(path, "w") as handle:
            handle.create_matrix("DIST", obj=values)
    options = {"driver": "core", "backing_store": False} if mode == "memory" else {}
    with openmatrix.open_file(
        path, "w" if mode == "memory" else mode, **options
    ) as handle:
        if mode in {"w", "memory"}:
            handle.create_matrix("DIST", obj=values)
        target = xr.Dataset({"DIST": (("o", "d"), np.zeros((3, 3), dtype="float32"))})
        sh.dataset.reload_from_omx_3d(target, handle, workers=1)
        np.testing.assert_array_equal(target.DIST, values)
        assert handle.id.valid
    if mode == "memory":
        assert not path.exists()


def test_lazy_decoder_restarts_after_fork(tmp_path):
    """A child can decode OMX after inheriting a warmed parent thread pool."""
    import os
    import subprocess
    import sys

    if not hasattr(os, "fork"):
        pytest.skip("fork is unavailable")
    source = tmp_path / "source.omx"
    _write_compressed_omx(source, _random_matrices())
    code = """
import os
import signal
import sys
import numpy as np
import sharrow as sh

expected = sh.dataset._fast_load_omx_array(sys.argv[1], 'DIST')
reader = sh.dataset._lazy_omx_reader
sh.dataset._lazy_omx_reader_lock.acquire()
pid = os.fork()
if pid == 0:
    signal.alarm(10)
    assert sh.dataset._lazy_omx_reader is None
    assert sh.dataset._lazy_omx_reader_lock.acquire(blocking=False)
    sh.dataset._lazy_omx_reader_lock.release()
    actual = sh.dataset._fast_load_omx_array(sys.argv[1], 'DIST')
    np.testing.assert_array_equal(actual, expected)
    sh.dataset._shutdown_lazy_omx_reader()
    os._exit(0)
sh.dataset._lazy_omx_reader_lock.release()
_, status = os.waitpid(pid, 0)
assert os.waitstatus_to_exitcode(status) == 0, status
assert sh.dataset._lazy_omx_reader is reader
np.testing.assert_array_equal(
    sh.dataset._fast_load_omx_array(sys.argv[1], 'DIST'), expected
)
"""
    subprocess.run(
        [sys.executable, "-c", code, str(source)],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_default_lazy_selection_reads_only_selected_period(tmp_path, monkeypatch):
    """A selective read must not decode unrequested time-period matrices."""
    path = tmp_path / "skims.omx"
    matrices = _random_matrices()
    _write_compressed_omx(path, matrices)
    reads = []
    original = sh.dataset._read_omx_dataset

    def record_read(dataset, *args, **kwargs):
        reads.append(dataset.name)
        return original(dataset, *args, **kwargs)

    monkeypatch.setattr(sh.dataset, "_read_omx_dataset", record_read)
    lazy = sh.dataset.from_omx_3d(path, time_periods=["AM", "PM"])
    actual = lazy.TIME.sel(time_period="AM").compute(scheduler="synchronous")
    np.testing.assert_array_equal(actual, matrices["TIME__AM"])
    assert reads == ["/data/TIME__AM"]
    assert lazy.TIME.chunks[-1] == (1, 1)


def test_reload_default_does_not_spawn(tmp_path, monkeypatch):
    """Legacy reload calls remain in-process even with several shared sources."""
    paths = [tmp_path / "one.omx", tmp_path / "two.omx"]
    for path in paths:
        _write_compressed_omx(path, _random_matrices())
    expected = sh.dataset.from_omx_3d(paths, time_periods=["AM", "PM"], load="eager")
    shared = xr.zeros_like(expected).shm.to_shared_memory(
        "default-reload-" + secrets.token_hex(5), load=False
    )

    def forbid_pool(*args, **kwargs):
        pytest.fail("default reload must not spawn processes")

    monkeypatch.setattr(
        sh.dataset.concurrent.futures, "ProcessPoolExecutor", forbid_pool
    )
    try:
        sh.dataset.reload_from_omx_3d(shared, paths)
        xr.testing.assert_equal(shared, expected)
    finally:
        shared.shm.release_shared_memory()


def test_parallel_file_workers_share_decoder_budget(tmp_path, monkeypatch):
    """The file-worker boundary passes a divided CPU budget to each decoder."""
    paths = [tmp_path / "one.omx", tmp_path / "two.omx"]
    for path in paths:
        _write_compressed_omx(path, _random_matrices())
    expected = sh.dataset.from_omx_3d(paths, time_periods=["AM", "PM"], load="eager")
    budgets = []
    original_pool = sh.dataset.concurrent.futures.ProcessPoolExecutor

    def recording_pool(*args, **kwargs):
        pool = original_pool(*args, **kwargs)
        submit = pool.submit

        def record_submit(fn, *args, **kwargs):
            budgets.append(args[-1])
            return submit(fn, *args, **kwargs)

        pool.submit = record_submit
        return pool

    monkeypatch.setattr(sh.dataset.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(sh.dataset, "_omx_uses_processes", lambda: True)
    monkeypatch.setattr(
        sh.dataset.concurrent.futures, "ProcessPoolExecutor", recording_pool
    )
    shared = sh.dataset.from_omx_3d(
        paths, time_periods=["AM", "PM"], load="shared", workers=2
    )
    try:
        xr.testing.assert_equal(shared, expected)
        # Only the second source has assignments when names overlap. Use the
        # actual active worker count, not the requested upper limit.
        assert budgets == []
    finally:
        shared.shm.release_shared_memory()

    with openmatrix.open_file(paths[0], "w") as handle:
        handle.create_matrix("DIST", obj=np.eye(25))
    with openmatrix.open_file(paths[1], "w") as handle:
        handle.create_matrix("TIME__AM", obj=np.eye(25))
    shared = sh.dataset.from_omx_3d(
        paths, time_periods=["AM"], load="shared", workers=2
    )
    try:
        np.testing.assert_array_equal(shared.DIST, np.eye(25))
        np.testing.assert_array_equal(shared.TIME[..., 0], np.eye(25))
        assert budgets == [4, 4]
    finally:
        shared.shm.release_shared_memory()


@pytest.mark.parametrize("load", ["lazy", "eager", "shared", "memmap"])
def test_omx_shape_error_identifies_source_and_exclusion(tmp_path, load):
    """Reject inconsistent matrices before allocation, with actionable context."""
    first, second = tmp_path / "first.omx", tmp_path / "second.omx"
    with openmatrix.open_file(first, "w") as handle:
        handle.create_matrix("DIST", obj=np.eye(3))
    with openmatrix.open_file(second, "w") as handle:
        handle.create_matrix("UNRELATED", obj=np.eye(4))
    options = {"memory_path": tmp_path / "mapped.dat"} if load == "memmap" else {}
    with pytest.raises(ValueError) as error:
        sh.dataset.from_omx_3d(
            [first, second], time_periods=["AM"], load=load, **options
        )
    message = str(error.value)
    # Diagnostics quote paths with repr, which escapes Windows backslashes.
    assert repr(str(first)) in message and repr(str(second)) in message
    assert "UNRELATED" in message
    assert "(4, 4), expected (3, 3)" in message
    assert "SHAPE metadata" in message and "ignore" in message
    assert not (tmp_path / "mapped.dat").exists()
    # Exclusion is appropriate for an unrelated matrix, not bad root metadata.
    filtered = sh.dataset.from_omx_3d(
        [first, second], time_periods=["AM"], ignore="^UNRELATED$", load="eager"
    )
    np.testing.assert_array_equal(filtered.DIST, np.eye(3))


def test_omx_stale_shape_metadata_is_rejected(tmp_path):
    """A named coordinate does not silently override inconsistent SHAPE metadata."""
    path = tmp_path / "stale.omx"
    with openmatrix.open_file(path, "w") as handle:
        handle.create_matrix("DIST", obj=np.eye(3))
        handle.create_mapping("taz", [11, 22, 33])
        handle.attrs["SHAPE"] = [4, 4]
    with pytest.raises(ValueError, match=r"expected \(4, 4\).*SHAPE metadata"):
        sh.dataset.from_omx_3d(path, time_periods=["AM"], indexes="taz")
