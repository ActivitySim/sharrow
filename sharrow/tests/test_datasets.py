import tempfile
from pathlib import Path

import numpy as np
import openmatrix
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
