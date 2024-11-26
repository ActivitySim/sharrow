import numpy as np
import pandas as pd

import sharrow as sh


def test_skims():
    skims = sh.example_data.get_skims()
    assert isinstance(skims, sh.Dataset)
    np.testing.assert_almost_equal(
        skims.DIST.values[:2, :3],
        np.asarray([[0.12, 0.24, 0.44], [0.37, 0.14, 0.28]]),
    )


def test_skims_zarr():
    skims = sh.example_data.get_skims_zarr()
    assert isinstance(skims, sh.Dataset)
    np.testing.assert_almost_equal(
        skims.DIST.values[:2, :3],
        np.asarray([[0.12, 0.24, 0.44], [0.37, 0.14, 0.28]]),
    )


# def test_skims_omx():
#     skims = sh.example_data.get_skims_omx()
#     assert isinstance(skims, sh.Dataset)
#     np.testing.assert_almost_equal(
#         skims.DIST.values[:2, :3],
#         np.asarray([[0.12, 0.24, 0.44], [0.37, 0.14, 0.28]]),
#     )


def test_maz_to_taz():
    maz_to_taz = sh.example_data.get_maz_to_taz()
    assert isinstance(maz_to_taz, pd.DataFrame)
    assert maz_to_taz.index.name == "MAZ"


def test_maz_to_maz_walk():
    maz_to_maz_walk = sh.example_data.get_maz_to_maz_walk()
    assert isinstance(maz_to_maz_walk, pd.DataFrame)
    assert list(maz_to_maz_walk.columns) == ["OMAZ", "DMAZ", "DISTWALK"]


def test_land_use():
    land_use = sh.example_data.get_land_use()
    assert isinstance(land_use, pd.DataFrame)
    assert land_use.index.name == "TAZ"


def test_data():
    data = sh.example_data.get_data()
    assert isinstance(data, dict)
    assert isinstance(data["hhs"], pd.DataFrame)
    assert isinstance(data["persons"], pd.DataFrame)
    assert isinstance(data["land_use"], pd.DataFrame)
    assert isinstance(data["skims"], sh.Dataset)
    assert isinstance(data["maz_taz"], pd.DataFrame)
    assert isinstance(data["maz_maz_walk"], pd.DataFrame)
