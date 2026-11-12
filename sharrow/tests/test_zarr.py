import json
import multiprocessing as mp
import secrets
import zipfile

import dask
import numpy as np
import pytest
import xarray as xr
from zarr.storage import ZipStore

import sharrow as sh


@pytest.fixture
def encoded_skims():
    original = xr.Dataset(
        {
            "DIST": (("otaz", "dtaz"), np.array([[0, 5.25], [-999, 7.5]])),
            "FARE": (("otaz", "dtaz"), np.array([[1, 2], [2, np.nan]])),
            "TIME": (
                ("otaz", "dtaz", "time_period"),
                np.arange(8, dtype=np.float32).reshape(2, 2, 2),
            ),
            "VALID": (("otaz", "dtaz"), np.array([[True, False], [False, True]])),
        },
        coords={"otaz": [0, 1], "dtaz": [0, 1], "time_period": ["AM", "PM"]},
        attrs={
            "ZARR_WRITE_TIME": 123.5,
            "settings": {"enabled": True, "optional": None},
        },
    )
    original.otaz.attrs["labels"] = {"zero": 0, "scale": np.float32(0.25)}
    encoded = original.digital_encoding.set(
        "DIST", scale=0.25, offset=0, missing_value=-999, bitwidth=16
    ).digital_encoding.set("FARE", by_dict=True, bitwidth=8)
    encoded.DIST.attrs["description"] = "Distance with a missing value sentinel"
    return original, encoded


def _assert_encoded_skims(original, encoded, recovered):
    xr.testing.assert_equal(recovered, encoded)
    assert recovered.attrs == encoded.attrs
    assert recovered.otaz.attrs == encoded.otaz.attrs
    assert recovered.DIST.attrs == encoded.DIST.attrs
    np.testing.assert_array_equal(
        recovered.FARE.attrs["digital_encoding"]["dictionary"],
        encoded.FARE.attrs["digital_encoding"]["dictionary"],
    )
    for name in encoded.variables:
        assert recovered[name].dtype == encoded[name].dtype
    decoded = recovered.digital_encoding.strip(["DIST", "FARE"])
    xr.testing.assert_allclose(decoded, original)


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("zipped", [False, True])
def test_encoded_skims_roundtrip(tmp_path, encoded_skims, chunked, zipped):
    original, encoded = encoded_skims
    snapshot = encoded.copy(deep=True)
    if chunked:
        encoded = encoded.chunk({"otaz": 1, "dtaz": 1, "time_period": 1})
    cache = tmp_path / ("skims.zarr.zip" if zipped else "skims.zarr")
    encoded.to_zarr_with_attr(cache)
    if zipped:
        with zipfile.ZipFile(cache) as archive:
            assert json.loads(archive.read(".zgroup"))["zarr_format"] == 2
        with ZipStore(cache, mode="r") as store:
            with sh.dataset.from_zarr_with_attr(store) as recovered:
                _assert_encoded_skims(original, snapshot, recovered.load())
    else:
        assert json.loads((cache / ".zgroup").read_text())["zarr_format"] == 2
        assert not (cache / "zarr.json").exists()
        with sh.dataset.from_zarr_with_attr(cache) as recovered:
            _assert_encoded_skims(original, snapshot, recovered.load())
    # Attribute serialization must not mutate the source dataset.
    _assert_encoded_skims(original, snapshot, encoded.compute())


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("values", [[1.0, 2.0, np.nan], [np.nan, np.nan]])
def test_dictionary_encoding_with_nan(values, chunked):
    original = xr.Dataset({"value": ("row", values)})
    if chunked:
        original = original.chunk({"row": 1})
    encoded = original.digital_encoding.set("value", by_dict=True, bitwidth=8)
    xr.testing.assert_allclose(encoded.digital_encoding.strip("value"), original)


def test_special_float_attributes(tmp_path):
    original = xr.Dataset(
        {"value": ("row", [0, 1])},
        attrs={"limits": [np.float32(-np.inf), np.float64(np.inf), np.nan]},
    )
    cache = tmp_path / "special.zarr"
    original.to_zarr_with_attr(cache)
    with sh.dataset.from_zarr_with_attr(cache) as recovered:
        np.testing.assert_array_equal(
            recovered.attrs["limits"], original.attrs["limits"]
        )


def test_existing_format2_skims():
    with sh.example_data.get_example_data_path("skims.zarr") as cache:
        assert json.loads((cache / ".zgroup").read_text())["zarr_format"] == 2
        with sh.dataset.from_zarr_with_attr(cache, consolidated=False) as recovered:
            np.testing.assert_allclose(
                recovered.DIST.values[:2, :3],
                [[0.12, 0.24, 0.44], [0.37, 0.14, 0.28]],
                atol=1e-6,
            )


def test_legacy_encoded_attributes(tmp_path):
    legacy = xr.Dataset(
        {"FARE": ("row", np.array([0, 1, 0], dtype=np.uint8))},
        attrs={"settings": " {'enabled': True, 'optional': None} "},
    )
    legacy.FARE.attrs["digital_encoding"] = " {'dictionary': [1.0, 2.0]} "
    cache = tmp_path / "legacy.zarr"
    legacy.to_zarr(cache, zarr_format=2)
    with sh.dataset.from_zarr_with_attr(cache) as recovered:
        assert recovered.attrs["settings"] == {"enabled": True, "optional": None}
        np.testing.assert_array_equal(
            recovered.digital_encoding.strip("FARE").FARE.values, [1, 2, 1]
        )


@pytest.mark.parametrize("use_path", [False, True])
@pytest.mark.parametrize("use_keyword", [False, True])
def test_zarr_zip_roundtrip(tmp_path, encoded_skims, use_path, use_keyword):
    _, encoded = encoded_skims
    plain = encoded[["TIME", "VALID"]].copy(deep=True)
    plain.attrs = {}
    for name in plain.variables:
        plain[name].attrs = {}
    cache = tmp_path / "skims.zarr.zip"
    store = cache if use_path else str(cache)
    kwargs = {"mode": "w", "compression": zipfile.ZIP_STORED, "consolidated": False}
    if use_keyword:
        plain.to_zarr_zip(store=store, **kwargs)
    else:
        plain.to_zarr_zip(store, **kwargs)
    with zipfile.ZipFile(cache) as archive:
        assert json.loads(archive.read(".zgroup"))["zarr_format"] == 2
    with ZipStore(cache, mode="r") as zipped:
        with xr.open_zarr(zipped, consolidated=False) as recovered:
            xr.testing.assert_identical(recovered.load(), plain)


def test_zarr_zip_directory_fallback(tmp_path):
    original = xr.Dataset({"value": ("row", [0, 1, 2])})
    cache = tmp_path / "plain.zarr"
    original.to_zarr_zip(cache)
    assert json.loads((cache / ".zgroup").read_text())["zarr_format"] == 2
    with xr.open_zarr(cache) as recovered:
        xr.testing.assert_identical(original, recovered.load())


def test_zarr_zip_requires_immediate_write(tmp_path):
    original = xr.Dataset({"value": ("row", [0, 1, 2])}).chunk({"row": 1})
    cache = tmp_path / "delayed.zarr.zip"
    with pytest.raises(ValueError, match="compute=True"):
        original.to_zarr_zip(cache, compute=False)
    assert not cache.exists()


@pytest.mark.parametrize("format_kwargs", [{"zarr_format": 3}, {"zarr_version": 2}])
def test_explicit_zarr_format(tmp_path, format_kwargs):
    original = xr.Dataset({"value": ("row", [0, 1, 2])}, attrs={"optional": None})
    cache = tmp_path / "explicit.zarr"
    original.to_zarr_with_attr(cache, **format_kwargs)
    version = format_kwargs.get("zarr_format", format_kwargs.get("zarr_version"))
    metadata_file = ".zgroup" if version == 2 else "zarr.json"
    assert json.loads((cache / metadata_file).read_text())["zarr_format"] == version
    with sh.dataset.from_zarr_with_attr(cache) as recovered:
        xr.testing.assert_identical(original, recovered.load())


def _copy_cache_in_worker(cache, token, destination):
    # A fork inherits Dask's cached pool without its worker threads. Use a
    # process-local scheduler for cache IO, as ActivitySim does when loading.
    with (
        dask.config.set(scheduler="synchronous"),
        sh.dataset.from_zarr_with_attr(cache) as recovered,
    ):
        recovered.load()
        shared = sh.Dataset.shm.from_shared_memory(token)
        xr.testing.assert_equal(recovered, shared)
        recovered.to_zarr_with_attr(destination)
        with sh.dataset.from_zarr_with_attr(destination) as copied:
            decoded = copied.digital_encoding.strip(["DIST", "FARE"])
            return decoded.DIST.values, decoded.FARE.values


@pytest.mark.parametrize("start_method", mp.get_all_start_methods())
def test_encoded_cache_multiprocessing(tmp_path, encoded_skims, start_method):
    original, encoded = encoded_skims
    cache = tmp_path / "skims.zarr"
    encoded.to_zarr_with_attr(cache)
    # Initialize Zarr's IO thread before forking, as in a model preload.
    with sh.dataset.from_zarr_with_attr(cache) as recovered:
        recovered.load()
        token = "zarr_test_" + secrets.token_hex(8)
        shared = recovered.shm.to_shared_memory(token)
        try:
            context = mp.get_context(start_method)
            with context.Pool(2) as pool:
                results = pool.starmap_async(
                    _copy_cache_in_worker,
                    [(cache, token, tmp_path / f"worker_{i}.zarr") for i in range(2)],
                ).get(timeout=60)
            for distance, fare in results:
                np.testing.assert_allclose(distance, original.DIST.values)
                np.testing.assert_allclose(fare, original.FARE.values)
        finally:
            shared.shm.release_shared_memory()
