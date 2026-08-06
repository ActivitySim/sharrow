from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr

from sharrow.omx import split_omx
from sharrow.translate import omx_to_zarr


@pytest.fixture
def omx_file(tmp_path: Path) -> tuple[Path, dict[str, np.ndarray]]:
    """Create a representative OMX-format HDF5 file for utility tests."""
    matrices = {
        "DIST": np.arange(9, dtype=np.float32).reshape(3, 3),
        "TIME__AM": np.arange(10, 19, dtype=np.float32).reshape(3, 3),
        "TIME__PM": np.arange(20, 29, dtype=np.float32).reshape(3, 3),
    }
    path = tmp_path / "skims.omx"
    with h5py.File(path, "w") as handle:
        handle.attrs["OMX_VERSION"] = np.bytes_(b"0.2")
        handle.attrs["SHAPE"] = np.asarray([3, 3], dtype=np.int32)
        for name, values in matrices.items():
            handle.create_dataset(
                f"data/{name}",
                data=values,
                chunks=(2, 2),
                compression="gzip",
            )
        handle.create_dataset("lookup/taz", data=np.asarray([101, 102, 103]))
    return path, matrices


def test_split_omx_with_global_lookups(omx_file, tmp_path):
    """Matrix chunks retain datasets, compression, lookups, and OMX metadata."""
    source_path, matrices = omx_file
    destination = tmp_path / "global-lookups"

    split_omx(source_path, destination, global_lookups=True, n_chunks=2)

    matrix_names = set()
    for chunk_number in range(2):
        with h5py.File(destination / f"skims-chunk{chunk_number}.omx") as handle:
            assert tuple(handle.attrs["SHAPE"]) == (3, 3)
            np.testing.assert_array_equal(handle["lookup/taz"], [101, 102, 103])
            for name, dataset in handle["data"].items():
                matrix_names.add(name)
                assert dataset.compression == "gzip"
                np.testing.assert_array_equal(dataset, matrices[name])
    assert matrix_names == set(matrices)


def test_split_omx_with_separate_lookups(omx_file, tmp_path):
    """The default split writes each lookup to a standalone OMX file."""
    source_path, matrices = omx_file
    destination = tmp_path / "separate-lookups"

    split_omx(source_path, destination)

    for name, values in matrices.items():
        with h5py.File(destination / f"{name}.omx") as handle:
            assert list(handle["data"]) == [name]
            assert list(handle["lookup"]) == []
            np.testing.assert_array_equal(handle[f"data/{name}"], values)
    with h5py.File(destination / "_taz.omx") as handle:
        assert list(handle["data"]) == []
        np.testing.assert_array_equal(handle["lookup/taz"], [101, 102, 103])


def test_omx_to_zarr(omx_file, tmp_path):
    """OMX conversion reads HDF5 matrices into the expected Zarr dimensions."""
    source_path, matrices = omx_file
    destination = tmp_path / "skims.zarr"

    omx_to_zarr(source_path, destination, time_periods=["AM", "PM"])

    with xr.open_zarr(destination) as dataset:
        np.testing.assert_array_equal(dataset["DIST"], matrices["DIST"])
        np.testing.assert_array_equal(
            dataset["TIME"].sel(time_period="AM"), matrices["TIME__AM"]
        )
        np.testing.assert_array_equal(
            dataset["TIME"].sel(time_period="PM"), matrices["TIME__PM"]
        )
        np.testing.assert_array_equal(dataset.coords["otaz"], [101, 102, 103])
