import secrets

import numpy as np
import pandas as pd
import pytest
from pytest import approx

import sharrow as sh


@pytest.fixture
def sparse_skims_setup():
    """Create skims with sparse MAZ data.

    Mirrors the setup from sparse.ipynb.
    """
    # Load example data
    skims = sh.example_data.get_skims()
    maz_taz = sh.example_data.get_maz_to_taz()
    maz_to_maz_walk = sh.example_data.get_maz_to_maz_walk()

    # Add DISTJUMP column
    maz_to_maz_walk["DISTJUMP"] = maz_to_maz_walk["DISTWALK"] + 0.07

    # Set redirection
    skims.redirection.set(
        maz_taz,
        map_to="otaz",
        name="omaz",
        map_also={"dtaz": "dmaz"},
    )

    # Add sparse blender for DISTWALK
    skims.redirection.sparse_blender(
        "DISTWALK",
        maz_to_maz_walk.OMAZ,
        maz_to_maz_walk.DMAZ,
        maz_to_maz_walk.DISTWALK,
        max_blend_distance=1.0,
        index=maz_taz.index,
    )

    # Create DISTJUMP variable as zeros

    # skims = skims.assign(DISTJUMP=xr.DataArray(
    #     np.broadcast_to(np.zeros([1]), skims["DISTWALK"].shape),
    #     dims=skims["DISTWALK"].dims,
    #     coords=skims["DISTWALK"].coords,
    # ))

    # Add sparse blender for DISTJUMP, which has no backing in the TAZ data
    skims.redirection.sparse_blender(
        "DISTJUMP",
        maz_to_maz_walk.OMAZ,
        maz_to_maz_walk.DMAZ,
        maz_to_maz_walk.DISTJUMP,
        index=maz_taz.index,
    )

    return skims, maz_taz, maz_to_maz_walk


@pytest.fixture
def trips_data():
    """Create test trips dataframe."""
    return pd.DataFrame(
        {
            "orig_maz": [100, 100, 100, 200, 200],
            "dest_maz": [100, 101, 103, 201, 202],
        }
    )


def blend(s, d, max_s):
    """Compute blended values from sparse and dense data.

    Blends sparse and dense data based on max_s threshold.
    """
    out = np.zeros_like(d)
    ratio = s / max_s
    out = d * ratio + s * (1 - ratio)
    out = np.where(s > max_s, d, out)
    out = np.where(np.isnan(s), d, out)
    return out


def test_sparse_basic_flow(sparse_skims_setup, trips_data):
    """Test basic flow with sparse blending."""
    skims, maz_taz, maz_to_maz_walk = sparse_skims_setup
    trips = trips_data

    # Create DataTree
    tree = sh.DataTree(
        base=trips,
        skims=skims,
        relationships=(
            "base.orig_maz @ skims.omaz",
            "base.dest_maz @ skims.dmaz",
        ),
    )

    # Setup flow
    flow = tree.setup_flow(
        {
            "plain_distance": "DISTWALK",
        },
        boundscheck=True,
    )

    # Test blending logic
    sparse_dat = np.array([0.01, 0.2, np.nan, 3.2, np.nan])
    dense_dat = np.array([0.12, 0.12, 0.12, 0.17, 0.17])

    expected = blend(sparse_dat, dense_dat, 1.0)
    result = flow.load().ravel()

    assert result == approx(expected)


def test_sparse_flow_with_transformations(sparse_skims_setup, trips_data):
    """Test flow with transformations on sparse data."""
    skims, maz_taz, maz_to_maz_walk = sparse_skims_setup
    trips = trips_data

    tree = sh.DataTree(
        base=trips,
        skims=skims,
        relationships=(
            "base.orig_maz @ skims.omaz",
            "base.dest_maz @ skims.dmaz",
        ),
    )

    # Setup flow with multiple transformations
    flow2 = tree.setup_flow(
        {
            "plain_distance": "DISTWALK",
            "clip_distance": "DISTWALK.clip(upper=0.15)",
            "square_distance": "DISTWALK**2",
        }
    )

    expected = np.array(
        [
            [1.1100e-02, 1.1100e-02, 1.2321e-04],
            [1.8400e-01, 1.5000e-01, 3.3856e-02],
            [1.2000e-01, 1.2000e-01, 1.4400e-02],
            [1.7000e-01, 1.5000e-01, 2.8900e-02],
            [1.7000e-01, 1.5000e-01, 2.8900e-02],
        ],
        dtype=np.float32,
    )

    result = flow2.load_dataframe().values
    assert result == approx(expected)


def test_sparse_distjump(sparse_skims_setup, trips_data):
    """Test DISTJUMP variable with infinite blend distance."""
    skims, maz_taz, maz_to_maz_walk = sparse_skims_setup
    trips = trips_data

    tree = sh.DataTree(
        base=trips,
        skims=skims,
        relationships=(
            "base.orig_maz @ skims.omaz",
            "base.dest_maz @ skims.dmaz",
        ),
    )

    flow3 = tree.setup_flow(
        {
            "jump_distance": "skims['DISTJUMP']",
        },
        boundscheck=True,
    )

    result = flow3.load()

    # DISTJUMP should have sparse values + 0.07, or 0 for missing
    sparse_dat = np.array([0.01, 0.2, np.nan, 3.2, np.nan]) + 0.07
    dense_dat = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    expected = blend(sparse_dat, dense_dat, np.inf)

    assert result.ravel() == approx(expected)


def test_sparse_at_accessor(sparse_skims_setup, trips_data):
    """Test the 'at' accessor with sparse data."""
    skims, maz_taz, maz_to_maz_walk = sparse_skims_setup
    trips = trips_data

    out = skims.at(
        omaz=trips.orig_maz,
        dmaz=trips.dest_maz,
        _names=["DIST", "DISTWALK"],
        _load=True,
    )

    # Check DIST values (dense data)
    np.testing.assert_array_almost_equal(
        out["DIST"].to_numpy(),
        np.array([0.12, 0.12, 0.12, 0.17, 0.17], dtype=np.float32),
    )

    # Check DISTWALK values (blended sparse/dense)
    np.testing.assert_array_almost_equal(
        out["DISTWALK"].to_numpy(),
        np.array([0.0111, 0.184, 0.12, 0.17, 0.17], dtype=np.float32),
    )


def test_sparse_at_accessor_with_time_period_raises(sparse_skims_setup, trips_data):
    """Test that 'at' accessor raises NotImplementedError for 3D sparse."""
    skims, maz_taz, maz_to_maz_walk = sparse_skims_setup
    trips = trips_data

    with pytest.raises(NotImplementedError):
        skims.at(
            omaz=trips.orig_maz,
            dmaz=trips.dest_maz,
            time_period=["AM", "AM", "AM", "AM", "AM"],
            _names=["DIST", "DISTWALK", "SOV_TIME"],
            _load=True,
        )


def test_sparse_iat_accessor(sparse_skims_setup):
    """Test the 'iat' accessor with sparse data."""
    skims, maz_taz, maz_to_maz_walk = sparse_skims_setup

    out = skims.iat(
        omaz=[0, 0, 0, 100, 100],
        dmaz=[0, 1, 3, 101, 102],
        _names=["DIST", "DISTWALK"],
        _load=True,
    )

    # Check DIST values
    np.testing.assert_array_almost_equal(
        out["DIST"].to_numpy(),
        np.array([0.12, 0.12, 0.12, 0.17, 0.17], dtype=np.float32),
    )

    # Check DISTWALK values
    np.testing.assert_array_almost_equal(
        out["DISTWALK"].to_numpy(),
        np.array([0.0111, 0.184, 0.12, 0.17, 0.17], dtype=np.float32),
    )


def test_sparse_shared_memory(sparse_skims_setup, trips_data):
    """Test that sparse data works correctly with shared memory."""
    skims, maz_taz, maz_to_maz_walk = sparse_skims_setup
    trips = trips_data

    # Create unique token for shared memory
    token = "skims-with-sparse" + secrets.token_hex(5)

    # Write to shared memory
    readback0 = skims.shm.to_shared_memory(token)
    assert readback0.attrs == skims.attrs

    # Read back from shared memory
    readback = sh.Dataset.shm.from_shared_memory(token)
    assert readback.attrs == skims.attrs

    # Test iat accessor on readback
    out = readback.iat(
        omaz=[0, 0, 0, 100, 100],
        dmaz=[0, 1, 3, 101, 102],
        _names=["DIST", "DISTWALK"],
        _load=True,
    )
    np.testing.assert_array_almost_equal(
        out["DIST"].to_numpy(),
        np.array([0.12, 0.12, 0.12, 0.17, 0.17], dtype=np.float32),
    )
    np.testing.assert_array_almost_equal(
        out["DISTWALK"].to_numpy(),
        np.array([0.0111, 0.184, 0.12, 0.17, 0.17], dtype=np.float32),
    )

    # Test at accessor on readback
    out = readback.at(
        omaz=trips.orig_maz,
        dmaz=trips.dest_maz,
        _names=["DIST", "DISTWALK"],
        _load=True,
    )
    np.testing.assert_array_almost_equal(
        out["DIST"].to_numpy(),
        np.array([0.12, 0.12, 0.12, 0.17, 0.17], dtype=np.float32),
    )
    np.testing.assert_array_almost_equal(
        out["DISTWALK"].to_numpy(),
        np.array([0.0111, 0.184, 0.12, 0.17, 0.17], dtype=np.float32),
    )

    # Check blenders attribute is preserved
    assert readback.redirection.blenders == {
        "DISTWALK": {"max_blend_distance": 1.0, "blend_distance_name": None},
        "DISTJUMP": {"max_blend_distance": np.inf, "blend_distance_name": None},
    }


def test_sparse_blenders_attribute(sparse_skims_setup):
    """Test that blenders attribute is set correctly."""
    skims, maz_taz, maz_to_maz_walk = sparse_skims_setup

    assert skims.redirection.blenders == {
        "DISTWALK": {"max_blend_distance": 1.0, "blend_distance_name": None},
        "DISTJUMP": {"max_blend_distance": np.inf, "blend_distance_name": None},
    }


def test_sparse_reverse(sparse_skims_setup, trips_data):
    """Test reverse operation on sparse data."""
    skims, maz_taz, maz_to_maz_walk = sparse_skims_setup
    trips = trips_data

    tree = sh.DataTree(
        base=trips,
        skims=skims,
        relationships=(
            "base.orig_maz @ skims.omaz",
            "base.dest_maz @ skims.dmaz",
        ),
    )

    flow3 = tree.setup_flow(
        {
            "plain_distance": "DISTWALK",
            "reverse_distance": 'skims.reverse("DISTWALK")',
        }
    )

    expected = np.array(
        [[0.0111, 0.0111], [0.184, 0.12], [0.12, 0.12], [0.17, 0.17], [0.17, 0.17]],
        dtype=np.float32,
    )

    result = flow3.load()
    assert result == approx(expected)


def test_sparse_reverse_iat(sparse_skims_setup):
    """Test reverse lookup using iat accessor."""
    skims, maz_taz, maz_to_maz_walk = sparse_skims_setup

    z = skims.iat(
        omaz=[0, 1, 3, 101, 102],
        dmaz=[0, 0, 0, 100, 100],
        _names=["DIST", "DISTWALK"],
        _load=True,
    )

    assert z["DISTWALK"].data == approx(np.array([0.0111, 0.12, 0.12, 0.17, 0.17]))
    assert z["DIST"].data == approx(np.array([0.12, 0.12, 0.12, 0.17, 0.17]))


def test_sparse_dense_lookup(sparse_skims_setup):
    """Test direct dense dimension lookup (bypassing sparse)."""
    skims, maz_taz, maz_to_maz_walk = sparse_skims_setup

    # Lookup using dense TAZ dimensions should bypass sparse data
    result = skims.at(
        otaz=[1, 1, 1, 16, 16],
        dtaz=[1, 1, 1, 16, 16],
        _names=["DIST", "DISTWALK"],
        _load=True,
    )

    # Should return dense data only, no sparse blending
    assert result is not None
    assert "DIST" in result
    assert "DISTWALK" in result
