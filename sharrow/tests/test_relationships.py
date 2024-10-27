import secrets
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.random import SeedSequence, default_rng
from pytest import approx, fixture, mark, raises

import sharrow
from sharrow import Dataset, DataTree, example_data
from sharrow.dataset import from_named_objects


@fixture
def households():
    households = example_data.get_households()
    prng = default_rng(SeedSequence(42))
    households["otaz_idx"] = households["TAZ"] - 1
    households["dtaz_idx"] = prng.choice(np.arange(25), 5000)
    households["timeperiod5"] = prng.choice(np.arange(5), 5000)
    households["timeperiod3"] = np.clip(households["timeperiod5"], 1, 3) - 1
    households["rownum"] = np.arange(len(households))
    return households


@fixture
def skims():
    return example_data.get_skims()


@pytest.mark.parametrize("use_array_maker", [True, False])
def test_shared_data(dataframe_regression, households, skims, use_array_maker: bool):
    tree = DataTree(
        base=households,
        skims=skims,
        relationships=(
            "base.otaz_idx->skims.otaz",
            "base.dtaz_idx->skims.dtaz",
            "base.timeperiod5->skims.time_period",
        ),
    )

    ss = tree.setup_flow(
        {
            "income": "base.income",
            "sov_time_by_income": "skims.SOV_TIME/base.income",
            "sov_cost_by_income": "skims.HOV3_TIME",
        }
    )
    result = ss.load_dataframe(tree, use_array_maker=use_array_maker)
    dataframe_regression.check(result, basename="test_shared_data")

    ss_undot = tree.setup_flow(
        {
            "income": "income",
            "sov_time_by_income": "SOV_TIME/income",
            "sov_cost_by_income": "HOV3_TIME",
        }
    )
    result = ss_undot.load_dataframe(tree, use_array_maker=use_array_maker)
    dataframe_regression.check(result, basename="test_shared_data")

    # names that are not valid Python identifiers
    s2 = tree.setup_flow(
        {
            "income > 10k": "base.income > 10_000",
            "income [up to 10k]": "base.income <= 10_000",
            "sov_time / income": "skims.SOV_TIME/base.income",
            "log1p(sov_cost_by_income)": "log1p(skims.HOV3_TIME)",
        }
    )
    result2 = s2.load_dataframe(tree, use_array_maker=use_array_maker)
    dataframe_regression.check(result2, basename="test_shared_data_2")


@pytest.mark.parametrize("use_array_maker", [True, False])
def test_subspace_fallbacks(
    dataframe_regression, households, skims, use_array_maker: bool
):
    tree = DataTree(
        base=households,
        skims=skims,
        relationships=(
            "base.otaz_idx->skims.otaz",
            "base.dtaz_idx->skims.dtaz",
            "base.timeperiod5->skims.time_period",
        ),
    )
    tree.subspace_fallbacks["df"] = ["base", "skims"]

    flow1 = tree.setup_flow(
        {
            "income": "df['income']",
            "sov_time_by_income": "df['SOV_TIME']/df['income']",
            "sov_cost_by_income": "df['HOV3_TIME']",
        }
    )
    result1 = flow1.load_dataframe(tree, use_array_maker=use_array_maker)
    dataframe_regression.check(result1, basename="test_shared_data")

    flow2 = tree.setup_flow(
        {
            "income": "income",
            "sov_time_by_income": "SOV_TIME/income",
            "sov_cost_by_income": "HOV3_TIME",
        }
    )
    result2 = flow2.load_dataframe(tree, use_array_maker=use_array_maker)
    dataframe_regression.check(result2, basename="test_shared_data")

    # names that are not valid Python identifiers
    flow3 = tree.setup_flow(
        {
            "income > 10k": "df.income > 10_000",
            "income [up to 10k]": "df.income <= 10_000",
            "sov_time / income": "df.SOV_TIME/df.income",
            "log1p(sov_cost_by_income)": "log1p(df.HOV3_TIME)",
        }
    )
    result3 = flow3.load_dataframe(tree, use_array_maker=use_array_maker)
    dataframe_regression.check(result3, basename="test_shared_data_2")


@pytest.mark.parametrize("use_array_maker", [True, False])
def test_shared_data_reversible(
    dataframe_regression, households, skims, use_array_maker: bool
):
    tree = DataTree(
        base=households,
        odt_skims=skims,
        dot_skims=skims,
        relationships=(
            "base.otaz_idx->odt_skims.otaz",
            "base.dtaz_idx->odt_skims.dtaz",
            "base.timeperiod5->odt_skims.time_period",
            "base.otaz_idx->dot_skims.dtaz",
            "base.dtaz_idx->dot_skims.otaz",
            "base.timeperiod5->dot_skims.time_period",
        ),
    )

    ss = tree.setup_flow(
        {
            "income": "base.income",
            "sov_time_by_income": "odt_skims.SOV_TIME/base.income",
            "round_trip_hov3_time": "dot_skims.HOV3_TIME + odt_skims.HOV3_TIME",
            "double_hov3_time": "odt_skims.HOV3_TIME * 2",
        }
    )
    result = ss.load_dataframe(tree, use_array_maker=use_array_maker)
    dataframe_regression.check(result, basename="test_shared_data_reversible")
    with raises(AssertionError):
        pd.testing.assert_series_equal(
            result["round_trip_hov3_time"],
            result["double_hov3_time"],
        )


@pytest.mark.parametrize("use_array_maker", [True, False])
def test_shared_data_reversible_by_label(dataframe_regression, use_array_maker: bool):
    data = example_data.get_data()
    skims = data["skims"]
    households = data["hhs"]

    # households = pd.read_csv("data/households.csv.gz")
    prng = default_rng(SeedSequence(42))
    households["otaz"] = households["TAZ"]
    households["dtaz"] = prng.choice(np.arange(1, 26), 5000)
    households["time5"] = prng.choice(["EA", "AM", "MD", "PM", "EV"], 5000)
    households["time3"] = prng.choice(["AM", "MD", "PM"], 5000)

    tree = DataTree(
        base=households,
        odt_skims=skims.rename({"otaz": "ptaz", "dtaz": "ataz"}),
        dot_skims=skims.rename({"otaz": "ataz", "dtaz": "ptaz"}),
        relationships=(
            "base.otaz @ odt_skims.ptaz",
            "base.dtaz @ odt_skims.ataz",
            "base.time5 @ odt_skims.time_period",
            "base.otaz @ dot_skims.ptaz",
            "base.dtaz @ dot_skims.ataz",
            "base.time5 @ dot_skims.time_period",
        ),
    )

    tree.digitize_relationships(inplace=True)
    ss = tree.setup_flow(
        {
            "income": "base.income",
            "sov_time_by_income": "odt_skims.SOV_TIME/base.income",
            "round_trip_hov3_time": "dot_skims.HOV3_TIME + odt_skims.HOV3_TIME",
            "double_hov3_time": "odt_skims.HOV3_TIME * 2",
        },
        extra_hash_data=(
            1,
            2,
        ),
    )
    result = ss.load_dataframe(tree, use_array_maker=use_array_maker)
    dataframe_regression.check(result, basename="test_shared_data_reversible")
    with raises(AssertionError):
        pd.testing.assert_series_equal(
            result["round_trip_hov3_time"],
            result["double_hov3_time"],
        )

    dtree = tree.digitize_relationships()
    dss = dtree.setup_flow(
        {
            "income": "base.income",
            "sov_time_by_income": "odt_skims.SOV_TIME/base.income",
            "round_trip_hov3_time": "dot_skims.HOV3_TIME + odt_skims.HOV3_TIME",
            "double_hov3_time": "odt_skims.HOV3_TIME * 2",
        },
        extra_hash_data=(
            1,
            2,
            3,
        ),
    )
    dresult = dss.load_dataframe(dtree, use_array_maker=use_array_maker)
    dataframe_regression.check(dresult, basename="test_shared_data_reversible")


@pytest.mark.parametrize("use_array_maker", [False, True])
def test_with_2d_base(dataframe_regression, use_array_maker: bool):
    pytest.importorskip("scipy", minversion="0.16")

    data = example_data.get_data()
    skims = data["skims"]
    households = data["hhs"]

    prng = default_rng(SeedSequence(42))
    households["otaz"] = households["TAZ"]
    households["otaz_idx"] = households["TAZ"] - 1
    households["dtaz"] = prng.choice(np.arange(1, 26), 5000)
    households["timeperiod5"] = prng.choice(np.arange(5), 5000)
    households["timeperiod3"] = np.clip(households["timeperiod5"], 1, 3) - 1
    households["rownum"] = np.arange(len(households))
    households["time5"] = prng.choice(["EA", "AM", "MD", "PM", "EV"], 5000)
    households["time3"] = prng.choice(["AM", "MD", "PM"], 5000)

    blank = from_named_objects(households.index, skims["dtaz"])
    assert sorted(blank.coords) == ["HHID", "dtaz"]
    assert blank.coords["HHID"].dims == ("HHID",)
    assert blank.coords["dtaz"].dims == ("dtaz",)

    tree = DataTree(
        root_node_name="base",
        base=blank,
        hh=households,
        odt_skims=skims.rename({"otaz": "ptaz", "dtaz": "ataz"}),
        dot_skims=skims.rename({"otaz": "ataz", "dtaz": "ptaz"}),
        relationships=(
            "base.HHID @ hh.HHID",
            "base.dtaz @ odt_skims.ataz",
            "base.dtaz @ dot_skims.ataz",
            "hh.otaz @ odt_skims.ptaz",
            "hh.time5 @ odt_skims.time_period",
            "hh.otaz @ dot_skims.ptaz",
            "hh.time5 @ dot_skims.time_period",
        ),
        force_digitization=True,
    )

    ss = tree.setup_flow(
        {
            "income": "hh.income",
            "sov_time_by_income": "odt_skims.SOV_TIME/hh.income",
            "round_trip_hov3_time": "dot_skims.HOV3_TIME + odt_skims.HOV3_TIME",
            "double_hov3_time": "odt_skims.HOV3_TIME * 2",
            "a_trip_hov3_time": "dot_skims.HOV3_TIME",
            "b_trip_hov3_time": "odt_skims.HOV3_TIME",
        }
    )
    result = ss.load_dataarray(tree, use_array_maker=use_array_maker)
    assert result.dims == ("HHID", "dtaz", "expressions")
    assert result.shape == (5000, 25, 6)
    result = result.to_dataset("expressions").to_dataframe()
    dataframe_regression.check(result.iloc[::83], basename="test_with_2d_base")

    dot_result = ss._load(tree, as_dataarray=True, dot=np.ones(6))
    assert dot_result.dims == (
        "HHID",
        "dtaz",
    )
    assert dot_result.shape == (
        5000,
        25,
    )

    check_vs = np.dot(result, np.ones([6])).reshape(5000, 25)
    np.testing.assert_array_almost_equal(check_vs, dot_result.to_numpy())


@pytest.mark.parametrize("use_array_maker", [True, False])
def test_mixed_dtypes(dataframe_regression, households, skims, use_array_maker: bool):
    tree = DataTree(
        base=households,
        skims=skims,
        relationships=(
            "base.otaz_idx->skims.otaz",
            "base.dtaz_idx->skims.dtaz",
            "base.timeperiod5->skims.time_period",
        ),
    )

    ss = tree.setup_flow(
        {
            "income": "base.income",
            "sov_time_by_income": "skims.SOV_TIME/base.income",
            "sov_time_by_workers": "np.where(base.workers > 0, skims.SOV_TIME / base.workers, 0)",
        }
    )
    result = ss.load_dataframe(tree, dtype=np.float32, use_array_maker=use_array_maker)
    dataframe_regression.check(result, basename="test_mixed_dtypes")

    ss_undot = tree.setup_flow(
        {
            "income": "income",
            "sov_time_by_income": "SOV_TIME/income",
            "sov_time_by_workers": "np.where(workers > 0, SOV_TIME / workers, 0)",
        }
    )
    result = ss_undot.load_dataframe(
        tree, dtype=np.float32, use_array_maker=use_array_maker
    )
    dataframe_regression.check(result, basename="test_mixed_dtypes")


@pytest.mark.parametrize("use_array_maker", [True, False])
def test_tuple_slice(dataframe_regression, households, skims, use_array_maker):
    tree = DataTree(
        base=households,
        skims=skims,
        relationships=(
            "base.otaz_idx->skims.otaz",
            "base.dtaz_idx->skims.dtaz",
        ),
    )
    ss = tree.setup_flow(
        {
            "income": "base.income",
            "sov_time_md": "skims[('SOV_TIME', 'MD')]",
        }
    )
    result = ss.load_dataframe(tree, dtype=np.float32, use_array_maker=use_array_maker)
    dataframe_regression.check(result, basename="test_tuple_slice")


@pytest.mark.parametrize("use_array_maker", [True, False])
def test_isin(dataframe_regression, households, skims, use_array_maker: bool):
    tree = DataTree(
        base=households,
        skims=skims,
        relationships=(
            "base.otaz_idx->skims.otaz",
            "base.dtaz_idx->skims.dtaz",
        ),
        extra_vars={
            "twenty_two_hundred": 2200,
            "sixteen_five": 16500,
        },
    )
    ss = tree.setup_flow(
        {
            "income": "base.income",
            "income2": "base.income.isin([361000, 197000])",
            "income3": "base.income.isin([twenty_two_hundred, sixteen_five])",
            "income4": "base.income.isin((twenty_two_hundred, 197000))",
        }
    )
    result = ss.load_dataframe(tree, dtype=np.float32, use_array_maker=use_array_maker)
    dataframe_regression.check(result, basename="test_isin")


def _get_target(q, token):
    skims_ = Dataset.shm.from_shared_memory(token)
    q.put(skims_.SOV_TIME.sum())


@mark.skipif(
    sys.version_info < (3, 8), reason="shared memory requires python3.8 or higher"
)
def test_shared_memory(skims):
    token = "skims" + secrets.token_hex(5)

    skims_2 = skims.shm.to_shared_memory(token)
    target = skims.SOV_TIME.sum()
    assert skims_2.SOV_TIME.sum() == target

    # reconstruct in same process
    skims_3 = Dataset.shm.from_shared_memory(token)
    assert skims_3.SOV_TIME.sum() == target

    # reconstruct in different process
    from multiprocessing import Process, Queue

    q = Queue()
    p = Process(target=_get_target, args=(q, token))
    p.start()
    p.join()
    assert q.get() == target


def test_relationship_init():
    r = sharrow.Relationship.from_string("Aa.bb -> Cc.dd")
    assert r.parent_data == "Aa"
    assert r.parent_name == "bb"
    assert r.child_data == "Cc"
    assert r.child_name == "dd"
    assert r.indexing == "position"

    r = sharrow.Relationship.from_string("Ee.ff @ Gg.hh")
    assert r.parent_data == "Ee"
    assert r.parent_name == "ff"
    assert r.child_data == "Gg"
    assert r.child_name == "hh"
    assert r.indexing == "label"


def test_replacement_filters(dataframe_regression, households, skims):
    tree = DataTree(
        base=households,
        skims=skims,
        relationships=(
            "base.otaz_idx->skims.otaz",
            "base.dtaz_idx->skims.dtaz",
            "base.timeperiod5->skims.time_period",
        ),
    )

    ss = tree.setup_flow(
        {
            "income": "base.income",
            "sov_time_by_income": "skims.SOV_TIME/base.income",
            "sov_cost_by_income": "skims.HOV3_TIME",
        }
    )
    result = ss._load(tree, as_dataframe=True)
    dataframe_regression.check(result, basename="test_shared_data")

    households_malformed = households.rename(columns={"income": "jncome"})
    tree_malformed = ss.tree.replace_datasets(base=households_malformed)
    with raises(KeyError, match=".*income.*"):
        ss._load(tree_malformed, as_dataframe=True)

    def rename_jncome(x):
        return x.rename({"jncome": "income"})

    ss.tree.replacement_filters["base"] = rename_jncome

    result = ss._load(
        ss.tree.replace_datasets(base=households_malformed), as_dataframe=True
    )
    dataframe_regression.check(result, basename="test_shared_data")


def test_name_in_wrong_subspace(dataframe_regression, households, skims):
    tree = DataTree(
        base=households,
        skims=skims,
        relationships=(
            "base.otaz_idx->skims.otaz",
            "base.dtaz_idx->skims.dtaz",
            "base.timeperiod5->skims.time_period",
        ),
    )

    with raises(KeyError):
        tree.setup_flow(
            {
                "income": "base.income",
                "sov_time_by_income": "base.SOV_TIME/base.income",
                "sov_cost_by_income": "base.HOV3_TIME",
            }
        )

    tree = DataTree(
        base=households,
        od_skims=skims.drop_dims("time_period"),
        odt_skims=skims,
        relationships=(
            "base.otaz_idx->od_skims.otaz",
            "base.dtaz_idx->od_skims.dtaz",
            "base.otaz_idx->odt_skims.otaz",
            "base.dtaz_idx->odt_skims.dtaz",
            "base.timeperiod5->odt_skims.time_period",
        ),
    )
    with raises(KeyError):
        tree.setup_flow(
            {
                "income": "base.income",
                "SOV_TIMEbyINC": "SOV_TIME/base.income",
                "SOV_TIMEbyINC1": "od_skims.SOV_TIME/base.income",
                "SOV_TIME": "od_skims.SOV_TIME",
                "HOV3_TIME": "od_skims.HOV3_TIME",
                "SOV_TIME_t": "odt_skims.SOV_TIME",
                "HOV3_TIME_t": "odt_skims.HOV3_TIME",
            }
        )
    with raises(KeyError):
        tree.setup_flow(
            {
                "income": "base.income",
                "SOV_TIME": "od_skims.SOV_TIME",
                "HOV3_TIME": "od_skims.HOV3_TIME",
                "SOV_TIME_t": "odt_skims.SOV_TIME",
                "HOV3_TIME_t": "odt_skims.HOV3_TIME",
            }
        )

    tree.subspace_fallbacks["od_skims"] = ["odt_skims"]

    ss_undot = tree.setup_flow(
        {
            "income": "income",
            "sov_time_by_income": "od_skims.SOV_TIME/income",
            "sov_cost_by_income": "od_skims.HOV3_TIME",
        }
    )
    result = ss_undot._load(tree, as_dataframe=True)
    dataframe_regression.check(result, basename="test_shared_data")


def test_shared_data_encoded(dataframe_regression, households, skims):
    households = sharrow.dataset.construct(households).digital_encoding.set(
        "income",
        bitwidth=32,
        scale=0.001,
    )

    tree = DataTree(
        base=households,
        skims=skims,
        relationships=(
            "base.otaz_idx->skims.otaz",
            "base.dtaz_idx->skims.dtaz",
            "base.timeperiod5->skims.time_period",
        ),
    )

    ss = tree.setup_flow(
        {
            "income": "base.income",
            "sov_time_by_income": "skims.SOV_TIME/base.income",
            "sov_cost_by_income": "skims.HOV3_TIME",
        },
        extra_hash_data=("income_encoded",),
    )
    result = ss._load(tree, as_dataframe=True)
    dataframe_regression.check(result, basename="test_shared_data")


def test_dict_encoded(dataframe_regression, skims):
    pairs = pd.DataFrame({"orig": [0, 0, 0, 1, 1, 1], "dest": [0, 1, 2, 0, 1, 2]})
    skims1 = skims.digital_encoding.set("WLK_LOC_WLK_FAR", bitwidth=8, by_dict=True)
    tree1 = DataTree(
        base=pairs,
        skims=skims1,
        relationships=(
            "base.orig -> skims.otaz",
            "base.dest -> skims.dtaz",
        ),
    )
    flow1 = tree1.setup_flow(
        {
            "d1": 'skims["WLK_LOC_WLK_FAR", "AM"]',
            "d2": 'skims["WLK_LOC_WLK_FAR", "AM"]**2',
        }
    )
    arr1 = flow1.load_dataframe()
    dataframe_regression.check(arr1)


def test_joint_dict_encoded(dataframe_regression, skims):
    pairs = pd.DataFrame({"orig": [0, 0, 0, 1, 1, 1], "dest": [0, 1, 2, 0, 1, 2]})
    skims1 = skims.digital_encoding.set(
        "WLK_LOC_WLK_FAR",
        "WLK_LOC_WLK_BOARDS",
        "WLK_LOC_WLK_IWAIT",
        "WLK_LOC_WLK_WAIT",
        joint_dict=True,
    )
    skims1 = skims1.digital_encoding.set(
        ["DISTBIKE", "DISTWALK"],
        joint_dict="jointWB",
    )
    tree1 = DataTree(
        base=pairs,
        skims=skims1,
        rskims=skims1,
        relationships=(
            "base.orig -> skims.otaz",
            "base.dest -> skims.dtaz",
            "base.orig -> rskims.dtaz",
            "base.dest -> rskims.otaz",
        ),
    )
    flow1 = tree1.setup_flow(
        {
            "f1": 'skims["WLK_LOC_WLK_FAR", "AM"]',
            "f2": 'skims["WLK_LOC_WLK_FAR", "AM"]**2',
            "w1": "skims.DISTWALK",
            "w2": 'skims.reverse("DISTWALK")',
            "w3": "rskims.DISTWALK",
            "x1": "skims.DIST",
            "x2": 'skims.reverse("DIST")',
        },
        extra_hash_data=("joint",),
    )
    arr1 = flow1.load_dataframe()
    dataframe_regression.check(arr1)


def test_isin_and_between(dataframe_regression):
    data = example_data.get_data()
    persons = data["persons"]

    tree = DataTree(
        base=persons,
        extra_vars={
            "pt1": 1,
            "pt5": 5,
            "pt34": [3, 4],
        },
    )

    ss = tree.setup_flow(
        {
            "pt": "base.ptype",
            "pt_in_15": "base.ptype.isin([pt1,pt5])",
            "pt_in_34": "base.ptype.isin(pt34)",
            "pt_tween_15": "base.ptype.between(pt1,pt5)",
            "pt_tween_25": "base.ptype.between(2,pt5)",
            "pt_tween_35": "base.ptype.between(3,5)",
        }
    )
    result = ss.load_dataframe(tree)
    pd.testing.assert_series_equal(
        result["pt"].isin([1, 5]).astype(np.float32),
        result["pt_in_15"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result["pt"].isin([3, 4]).astype(np.float32),
        result["pt_in_34"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result["pt"].between(1, 5).astype(np.float32),
        result["pt_tween_15"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result["pt"].between(2, 5).astype(np.float32),
        result["pt_tween_25"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result["pt"].between(3, 5).astype(np.float32),
        result["pt_tween_35"],
        check_names=False,
    )
    dataframe_regression.check(result)

    # test masking
    mask = (persons.index % 2) == 0

    result = ss.load_dataframe(tree, mask=mask)
    pd.testing.assert_series_equal(
        result["pt"].isin([1, 5]).astype(np.float32).where(mask, np.nan),
        result["pt_in_15"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result["pt"].isin([3, 4]).astype(np.float32).where(mask, np.nan),
        result["pt_in_34"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result["pt"].between(1, 5).astype(np.float32).where(mask, np.nan),
        result["pt_tween_15"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result["pt"].between(2, 5).astype(np.float32).where(mask, np.nan),
        result["pt_tween_25"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result["pt"].between(3, 5).astype(np.float32).where(mask, np.nan),
        result["pt_tween_35"],
        check_names=False,
    )


def test_nested_where(dataframe_regression):
    data = example_data.get_data()
    base = persons = data["persons"]

    tree = DataTree(
        base=persons,
        extra_vars={
            "pt1": 1,
            "pt5": 5,
            "pt34": [3, 4],
        },
    )

    ss = tree.setup_flow(
        {
            "pt": "base.ptype",
            "pt_shifted_1": "np.where(base.ptype<3, np.where(base.ptype<2, base.ptype*100, 0), base.ptype)",
            "pt_shifted_2": "np.where(base.ptype<3, np.where(base.ptype<2, base.ptype*100, 0), 0)",
            "pt_shifted_3": "np.where(base.ptype<3, 0, np.where(base.ptype>4, base.ptype*100, 0))",
            "pt_shifted_4": "np.where(base.ptype<3, base.ptype, np.where(base.ptype>4, base.ptype*100, 0))",
            "pt_shifted_5": "np.where(base.ptype<3, base.ptype, np.where(base.ptype>4, base.ptype*100, base.ptype))",
            "pt_shifted_6": "np.where(base.ptype<3, 0, np.where(base.ptype>4, base.ptype*100, base.ptype))",
        }
    )
    result = ss.load_dataframe(tree)
    pd.testing.assert_series_equal(
        pd.Series(
            np.where(
                base.ptype < 3,
                np.where(base.ptype < 2, base.ptype * 100, 0),
                base.ptype,
            ).astype(np.float32),
        ),
        result["pt_shifted_1"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        pd.Series(
            np.where(
                base.ptype < 3, np.where(base.ptype < 2, base.ptype * 100, 0), 0
            ).astype(np.float32),
        ),
        result["pt_shifted_2"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        pd.Series(
            np.where(
                base.ptype < 3, 0, np.where(base.ptype > 4, base.ptype * 100, 0)
            ).astype(np.float32),
        ),
        result["pt_shifted_3"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        pd.Series(
            np.where(
                base.ptype < 3,
                base.ptype,
                np.where(base.ptype > 4, base.ptype * 100, 0),
            ).astype(np.float32),
        ),
        result["pt_shifted_4"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        pd.Series(
            np.where(
                base.ptype < 3,
                base.ptype,
                np.where(base.ptype > 4, base.ptype * 100, base.ptype),
            ).astype(np.float32),
        ),
        result["pt_shifted_5"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        pd.Series(
            np.where(
                base.ptype < 3,
                0,
                np.where(base.ptype > 4, base.ptype * 100, base.ptype),
            ).astype(np.float32),
        ),
        result["pt_shifted_6"],
        check_names=False,
    )
    dataframe_regression.check(result)


def test_isna():
    data = example_data.get_data()
    data["hhs"].loc[data["hhs"].income > 200000, "income"] = np.nan
    tree = DataTree(
        base=data["hhs"],
    )
    ss = tree.setup_flow(
        {
            "missing_income": "((income < 0) | income.isna())",
            "income_is_na": "income.isna()",
        }
    )
    result = ss.load()
    assert result[0, 0] == 1
    assert result[0, 1] == 1
    assert result[:, 0].sum() == 188
    assert result[:, 1].sum() == 188

    qf = pd.DataFrame({"MixedVals": ["a", "", None, np.nan]})
    tree2 = DataTree(
        base=qf,
    )
    qf = tree2.setup_flow(
        {
            "MixedVals_is_na": "MixedVals.isna()",
        }
    )
    result = qf.load()
    assert result == approx(np.asarray([[0, 0, 1, 1]]).T)


def test_get(dataframe_regression, households, skims):
    tree = DataTree(
        base=households,
        skims=skims,
        relationships=(
            "base.otaz_idx->skims.otaz",
            "base.dtaz_idx->skims.dtaz",
            "base.timeperiod5->skims.time_period",
        ),
    )

    flow1 = tree.setup_flow(
        {
            "income": "base.get('income', 0) + base.get('missing_one', 0)",
            "sov_time_by_income": "skims.SOV_TIME/base.get('income', 0)",
            "missing_data": "base.get('missing_data', -1)",
            "missing_skim": "skims.get('missing_core', -2)",
            "sov_time_by_income_2": "skims.get('SOV_TIME')/base.income",
            "sov_cost_by_income_2": "skims.get('HOV3_TIME', 999)",
        },
    )
    result = flow1._load(tree, as_dataframe=True)
    dataframe_regression.check(result)

    tree_plus = DataTree(
        base=households.assign(missing_one=1.0),
        skims=skims,
        relationships=(
            "base.otaz_idx->skims.otaz",
            "base.dtaz_idx->skims.dtaz",
            "base.timeperiod5->skims.time_period",
        ),
    )
    flow2 = tree_plus.setup_flow(flow1.defs)
    result = flow2._load(tree_plus, as_dataframe=True)
    dataframe_regression.check(result.eval("income = income-1"))
    assert flow2.flow_hash != flow1.flow_hash

    tree.subspace_fallbacks["df"] = ["base"]
    flow3 = tree.setup_flow(
        {
            "income": "base.get('income', 0)",
            "sov_time_by_income": "skims.SOV_TIME/df.get('income', 0)",
            "missing_data": "df.get('missing_data', -1)",
            "missing_skim": "skims.get('missing_core', -2)",
            "sov_time_by_income_2": "skims.get('SOV_TIME')/df.income",
            "sov_cost_by_income_2": "skims.get('HOV3_TIME', 999)",
        },
    )
    result = flow3._load(tree, as_dataframe=True)
    dataframe_regression.check(result)

    flow4 = tree.setup_flow(
        {
            "income": "base.get('income', default=0) + df.get('missing_one', 0)",
            "sov_time_by_income": "skims.SOV_TIME/base.get('income', default=0)",
            "missing_data": "base.get('missing_data', default=-1)",
            "missing_skim": "skims.get('missing_core', default=-2)",
            "sov_time_by_income_2": "skims.get('SOV_TIME', default=0)/base.income",
            "sov_cost_by_income_2": "skims.get('HOV3_TIME', default=999)",
        },
    )
    result = flow4._load(tree, as_dataframe=True)
    assert flow4.flow_hash != flow1.flow_hash
    dataframe_regression.check(result)

    # test get when inside another function
    flow5 = tree.setup_flow(
        {
            "income": "np.power(base.get('income', default=0) + df.get('missing_one', 0), 1)",
            "sov_time_by_income": "skims.SOV_TIME/np.power(base.get('income', default=0), 1)",
            "missing_data": "np.where(np.isnan(df.get('missing_data', default=1)), 0, df.get('missing_data', default=-1))",  # noqa: E501
            "missing_skim": "(np.where(np.isnan(df.get('num_escortees', np.nan)), -2 , df.get('num_escortees', np.nan)))",  # noqa: E501
            "sov_time_by_income_2": "skims.get('SOV_TIME', default=0)/base.income",
            "sov_cost_by_income_2": "skims.get('HOV3_TIME', default=999)",
        },
    )
    result = flow5._load(tree, as_dataframe=True)
    assert "__skims__HOV3_TIME:True" in flow5._optional_get_tokens
    assert "__df__missing_data:False" in flow5._optional_get_tokens
    assert "__df__num_escortees:False" in flow5._optional_get_tokens
    dataframe_regression.check(result)


def test_get_native():
    data = example_data.get_data()
    skims = data["skims"]
    households = data["hhs"]

    prng = default_rng(SeedSequence(42))
    households["otaz"] = households["TAZ"]
    households["otaz_idx"] = households["TAZ"] - 1
    households["dtaz"] = prng.choice(np.arange(1, 26), 5000)
    households["timeperiod5"] = prng.choice(np.arange(5), 5000)
    households["timeperiod3"] = np.clip(households["timeperiod5"], 1, 3) - 1
    households["rownum"] = np.arange(len(households))
    households["time5"] = prng.choice(["EA", "AM", "MD", "PM", "EV"], 5000)
    households["time3"] = prng.choice(["AM", "MD", "PM"], 5000)

    blank = from_named_objects(households.index, skims["dtaz"])
    assert sorted(blank.coords) == ["HHID", "dtaz"]
    assert blank.coords["HHID"].dims == ("HHID",)
    assert blank.coords["dtaz"].dims == ("dtaz",)

    tree = DataTree(
        root_node_name="base",
        base=blank,
        hh=households,
        odt_skims=skims.rename({"otaz": "ptaz", "dtaz": "ataz"}),
        dot_skims=skims.rename({"otaz": "ataz", "dtaz": "ptaz"}),
        relationships=(
            "base.HHID @ hh.HHID",
            "base.dtaz @ odt_skims.ataz",
            "base.dtaz @ dot_skims.ataz",
            "hh.otaz @ odt_skims.ptaz",
            "hh.time5 @ odt_skims.time_period",
            "hh.otaz @ dot_skims.ptaz",
            "hh.time5 @ dot_skims.time_period",
        ),
        force_digitization=True,
    )

    i = tree.get("missingname", 123)
    assert i.dims == ("HHID", "dtaz")
    assert i.shape == (5000, 25)
    assert (i == 123).all()

    i = tree["income"]
    assert isinstance(i, xr.DataArray)
    assert i.dims == ("HHID", "dtaz")
    assert i.shape == (5000, 25)
    assert i.std("dtaz").max() == 0

    i = tree.get(["income", "PERSONS"], broadcast=False)
    assert isinstance(i, xr.Dataset)
    assert "HHID" in i.coords
    assert i.dims == {"HHID": 5000}

    i = tree.get("income", broadcast=False)
    assert "HHID" in i.coords
    assert i.dims == ("HHID",)
    assert i.shape == (5000,)

    i = tree.get("income", broadcast=False, coords=False)
    assert not i.coords
    assert i.dims == ("HHID",)
    assert i.shape == (5000,)

    i = tree.get("odt_skims.DIST")
    assert i.dims == ("HHID", "dtaz")
    assert i.shape == (5000, 25)
    assert (tree.get_expr("odt_skims.DIST", allow_native=False) == i.load()).all()

    with raises(KeyError):
        tree.get("xxxx.income")
    with raises(KeyError):
        tree.get("base.xxxx")
    with raises(KeyError):
        tree.get("xxxx")
    with raises(KeyError):
        tree.get("base.DIST")


def test_streaming(households, skims):
    tree = DataTree(
        base=households,
        skims=skims,
        relationships=(
            "base.otaz_idx->skims.otaz",
            "base.dtaz_idx->skims.dtaz",
            "base.timeperiod5->skims.time_period",
        ),
    )
    ss = tree.setup_flow(
        {
            "income": "base.income",
            "sov_time_by_income": "skims.SOV_TIME/base.income",
            "sov_cost_by_income": "skims.HOV3_TIME",
        }
    )
    result = ss.load(tree)
    streamer = ss.init_streamer(tree)
    for i in range(len(households)):
        assert result[i] == approx(streamer(i))


def test_streaming_2d(households, skims):
    blank = from_named_objects(households.index, skims["dtaz"])
    assert sorted(blank.coords) == ["HHID", "dtaz"]
    assert blank.coords["HHID"].dims == ("HHID",)
    assert blank.coords["dtaz"].dims == ("dtaz",)

    tree = DataTree(
        root_node_name="base",
        base=blank,
        hh=households,
        odt_skims=skims.rename({"otaz": "ptaz", "dtaz": "ataz"}),
        dot_skims=skims.rename({"otaz": "ataz", "dtaz": "ptaz"}),
        relationships=(
            "base.HHID @ hh.HHID",
            "base.dtaz @ odt_skims.ataz",
            "base.dtaz @ dot_skims.ataz",
            "hh.otaz_idx -> odt_skims.ptaz",
            "hh.timeperiod5 -> odt_skims.time_period",
            "hh.otaz_idx -> dot_skims.ptaz",
            "hh.timeperiod5 -> dot_skims.time_period",
        ),
        force_digitization=True,
    )

    ss = tree.setup_flow(
        {
            "income": "hh.income",
            "sov_time_by_income": "odt_skims.SOV_TIME/hh.income",
            "round_trip_hov3_time": "dot_skims.HOV3_TIME + odt_skims.HOV3_TIME",
            "double_hov3_time": "odt_skims.HOV3_TIME * 2",
            "a_trip_hov3_time": "dot_skims.HOV3_TIME",
            "b_trip_hov3_time": "odt_skims.HOV3_TIME",
        }
    )
    result = ss.load_dataarray(tree)
    assert result.dims == ("HHID", "dtaz", "expressions")
    assert result.shape == (5000, 25, 6)
    result = result.to_numpy()
    streamer = ss.init_streamer(tree)
    assert streamer(0).shape == (25, 6)
    for i in range(len(households)):
        assert (result[i] == streamer(i)).all()


@pytest.mark.parametrize("force_digitization", [True, False])
@pytest.mark.parametrize("engine", ["sharrow", "numexpr", "python", None])
def test_eval_1d_root(force_digitization: bool, engine: str | None):
    data = example_data.get_data()
    skim = data["skims"]
    hh = data["hhs"]

    prng = default_rng(SeedSequence(42))
    hh["otaz"] = hh["TAZ"]
    hh["otaz_idx"] = hh["TAZ"] - 1
    hh["dtaz"] = prng.choice(np.arange(1, 26), 5000)
    hh["timeperiod5"] = prng.choice(np.arange(5), 5000)
    hh["timeperiod3"] = np.clip(hh["timeperiod5"], 1, 3) - 1
    hh["rownum"] = np.arange(len(hh))
    hh["time5"] = prng.choice(["EA", "AM", "MD", "PM", "EV"], 5000)
    hh["time3"] = prng.choice(["AM", "MD", "PM"], 5000)
    hh["n_tours"] = prng.choice([0, 1, 2, 3], 5000)

    n_tours = hh["n_tours"].sum()
    tours = pd.DataFrame(
        {
            "HHID": np.repeat(hh.index, hh["n_tours"]),
            "tour_id": np.arange(1, n_tours + 1),
            "dtaz": prng.choice(np.arange(1, 26), n_tours),
            "tourtime": prng.choice(["EA", "AM", "MD", "PM", "EV"], n_tours),
        }
    ).set_index("tour_id")
    skim.load()

    tree = DataTree(
        tour=tours,
        hh=hh,
        odt_skims=skim,
        dot_skims=skim,
        relationships=(
            "tour.HHID @ hh.HHID",
            "tour.dtaz @ odt_skims.dtaz",
            "tour.dtaz @ dot_skims.otaz",
            "hh.otaz_idx -> odt_skims.otaz",
            "tour.tourtime @ odt_skims.time_period",
            "hh.otaz_idx -> dot_skims.dtaz",
            "tour.tourtime @ dot_skims.time_period",
        ),
        force_digitization=force_digitization,
    )

    sov_time = tree["SOV_TIME"]
    assert sov_time.shape == (7563,)
    assert sov_time.dims == ("tour_id",)
    assert sov_time[:3].values == approx([4.94, 5.02, 3.61])
    assert sov_time[-3:].values == approx([7.17, 5.53, 2.65])

    sov_time_by_income = tree["SOV_TIME"] / tree["income"]
    assert sov_time_by_income.shape == (7563,)
    assert sov_time_by_income.dims == ("tour_id",)
    assert sov_time_by_income[:3].values == approx(
        [8.34177652e-05, 8.47686589e-05, 1.64090904e-03]
    )
    assert sov_time_by_income[-3:].values == approx(
        [5.47328250e-04, 5.36893224e-05, 2.57281563e-05]
    )

    t = tree.eval("SOV_TIME", engine=engine)
    assert "expressions" in t.coords, f"missing expression coords with engine={engine}"
    assert t.coords["expressions"] == "SOV_TIME"
    xr.testing.assert_allclose(t.drop_vars("expressions"), sov_time)

    t = tree.eval("SOV_TIME / income", engine=engine)
    assert "expressions" in t.coords, f"missing expression coords with engine={engine}"
    assert t.coords["expressions"] == "SOV_TIME / income"
    xr.testing.assert_allclose(t.drop_vars("expressions"), sov_time_by_income)

    dot_time = tree["dot_skims.SOV_TIME"]
    assert dot_time.shape == (7563,)
    assert dot_time.dims == ("tour_id",)
    assert dot_time[:3].values == approx([5.3, 4.69, 3.58])
    assert dot_time[-3:].values == approx([7.36, 5.67, 3.08])

    dot_time_by_income = tree["dot_skims.SOV_TIME"] / tree["hh.income"]
    assert dot_time_by_income.shape == (7563,)
    assert dot_time_by_income.dims == ("tour_id",)
    assert dot_time_by_income[:3].values == approx(
        [8.94967948e-05, 7.91962185e-05, 1.62727269e-03]
    )
    assert dot_time_by_income[-3:].values == approx(
        [5.61832071e-04, 5.50485444e-05, 2.99029119e-05]
    )

    t = tree.eval("dot_skims.SOV_TIME", engine=engine)
    assert "expressions" in t.coords, f"missing expression coords with engine={engine}"
    assert t.coords["expressions"] == "dot_skims.SOV_TIME"
    xr.testing.assert_allclose(t.drop_vars("expressions"), dot_time)

    t = tree.eval("dot_skims.SOV_TIME / income", engine=engine)
    assert "expressions" in t.coords, f"missing expression coords with engine={engine}"
    assert t.coords["expressions"] == "dot_skims.SOV_TIME / income"
    xr.testing.assert_allclose(t.drop_vars("expressions"), dot_time_by_income)

    tm = tree.eval_many(
        {"SOV_TIME": "SOV_TIME", "SOV_TIME_by_income": "SOV_TIME/income"},
        engine=engine,
        result_type="dataarray",
    )
    assert tm.dims == ("tour_id", "expressions")
    assert tm.coords["expressions"].values.tolist() == [
        "SOV_TIME",
        "SOV_TIME_by_income",
    ]
    xr.testing.assert_allclose(
        tm.drop_vars(["expressions", "source"]),
        xr.concat(
            [
                sov_time.expand_dims("expressions", -1),
                sov_time_by_income.expand_dims("expressions", -1),
            ],
            "expressions",
        ),
    )

    tm = tree.eval_many(
        {"SOV_TIME": "SOV_TIME", "SOV_TIME_by_income": "SOV_TIME/income"},
        engine=engine,
        result_type="dataset",
    )
    assert tm.sizes == {"tour_id": 7563}
    xr.testing.assert_allclose(
        tm,
        xr.merge(
            [
                sov_time.rename("SOV_TIME"),
                sov_time_by_income.rename("SOV_TIME_by_income"),
            ]
        ),
    )


@pytest.mark.parametrize("force_digitization", [True, False])
@pytest.mark.parametrize("engine", ["sharrow", "numexpr", "python", None])
def test_eval_2d_root(
    force_digitization: bool, dataframe_regression, engine: str | None
):
    pytest.importorskip("scipy", minversion="0.16")

    data = example_data.get_data()
    skims = data["skims"]
    households = data["hhs"]

    prng = default_rng(SeedSequence(42))
    households["otaz"] = households["TAZ"]
    households["otaz_idx"] = households["TAZ"] - 1
    households["dtaz"] = prng.choice(np.arange(1, 26), 5000)
    households["timeperiod5"] = prng.choice(np.arange(5), 5000)
    households["timeperiod3"] = np.clip(households["timeperiod5"], 1, 3) - 1
    households["rownum"] = np.arange(len(households))
    households["time5"] = prng.choice(["EA", "AM", "MD", "PM", "EV"], 5000)
    households["time3"] = prng.choice(["AM", "MD", "PM"], 5000)

    blank = from_named_objects(households.index, skims["dtaz"])
    assert sorted(blank.coords) == ["HHID", "dtaz"]
    assert blank.coords["HHID"].dims == ("HHID",)
    assert blank.coords["dtaz"].dims == ("dtaz",)

    skims.load()

    tree = DataTree(
        root_node_name="base",
        base=blank,
        hh=households,
        odt_skims=skims.rename({"otaz": "ptaz", "dtaz": "ataz"}),
        dot_skims=skims.rename({"otaz": "ataz", "dtaz": "ptaz"}),
        relationships=(
            "base.HHID @ hh.HHID",
            "base.dtaz @ odt_skims.ataz",
            "base.dtaz @ dot_skims.ataz",
            "hh.otaz @ odt_skims.ptaz",
            "hh.time5 @ odt_skims.time_period",
            "hh.otaz @ dot_skims.ptaz",
            "hh.time5 @ dot_skims.time_period",
        ),
        force_digitization=force_digitization,
    )

    exprs = {
        "income": "income",
        "sov_time_by_income": "SOV_TIME/income",
        "a_trip_hov_time": "log(exp(HOV2_TIME / 0.5) + exp(HOV3_TIME / 0.5)) * 0.5",
        "round_trip_hov3_time": "dot_skims.HOV3_TIME + odt_skims.HOV3_TIME",
    }

    arrays = {
        k: tree.eval(v, engine=engine, name=k)
        .drop_vars("expressions")
        .assign_attrs(expression=v)
        for (k, v) in exprs.items()
    }
    # result = xr.concat(list(arrays.values()), "expressions")
    result = xr.Dataset(arrays).to_dataframe()
    dataframe_regression.check(result.iloc[::83], basename="test_eval_2d_root")

    result2 = tree.eval_many(exprs, engine=engine, result_type="dataset").to_dataframe()
    dataframe_regression.check(result2.iloc[::83], basename="test_eval_2d_root")
