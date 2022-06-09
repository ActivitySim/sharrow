import secrets
import sys

import numpy as np
import pandas as pd
from numpy.random import SeedSequence, default_rng
from pytest import mark, raises

import sharrow
from sharrow import Dataset, DataTree, example_data
from sharrow.dataset import from_named_objects


def test_shared_data(dataframe_regression):

    data = example_data.get_data()
    skims = data["skims"]
    households = data["hhs"]

    prng = default_rng(SeedSequence(42))
    households["otaz_idx"] = households["TAZ"] - 1
    households["dtaz_idx"] = prng.choice(np.arange(25), 5000)
    households["timeperiod5"] = prng.choice(np.arange(5), 5000)
    households["timeperiod3"] = np.clip(households["timeperiod5"], 1, 3) - 1
    households["rownum"] = np.arange(len(households))

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
    dataframe_regression.check(result)

    ss_undot = tree.setup_flow(
        {
            "income": "income",
            "sov_time_by_income": "SOV_TIME/income",
            "sov_cost_by_income": "HOV3_TIME",
        }
    )
    result = ss_undot._load(tree, as_dataframe=True)
    dataframe_regression.check(result)

    # names that are not valid Python identifiers
    s2 = tree.setup_flow(
        {
            "income > 10k": "base.income > 10_000",
            "income [up to 10k]": "base.income <= 10_000",
            "sov_time / income": "skims.SOV_TIME/base.income",
            "log1p(sov_cost_by_income)": "log1p(skims.HOV3_TIME)",
        }
    )
    result2 = s2._load(tree, as_dataframe=True)
    dataframe_regression.check(result2, basename="test_shared_data_2")


def test_shared_data_reversible(dataframe_regression):

    data = example_data.get_data()
    skims = data["skims"]
    households = data["hhs"]

    prng = default_rng(SeedSequence(42))
    households["otaz_idx"] = households["TAZ"] - 1
    households["dtaz_idx"] = prng.choice(np.arange(25), 5000)
    households["timeperiod5"] = prng.choice(np.arange(5), 5000)
    households["timeperiod3"] = np.clip(households["timeperiod5"], 1, 3) - 1
    households["rownum"] = np.arange(len(households))

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
    result = ss._load(tree, as_dataframe=True)
    dataframe_regression.check(result)
    with raises(AssertionError):
        pd.testing.assert_series_equal(
            result["round_trip_hov3_time"],
            result["double_hov3_time"],
        )


def test_shared_data_reversible_by_label(dataframe_regression):
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
    result = ss._load(tree, as_dataframe=True)
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
    dresult = dss._load(dtree, as_dataframe=True)
    dataframe_regression.check(dresult, basename="test_shared_data_reversible")


def test_with_2d_base(dataframe_regression):
    data = example_data.get_data()
    skims = data["skims"]
    households = data["hhs"]

    prng = default_rng(SeedSequence(42))
    households["otaz"] = households["TAZ"]
    households["otaz_idx"] = households["TAZ"] - 1
    households["dtaz"] = prng.choice(np.arange(1, 26), 5000)
    # households['dtaz_idx'] = prng.choice(np.arange(25), 5000)
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
    result = ss._load(tree, as_dataarray=True)
    assert result.dims == ("HHID", "dtaz", "expressions")
    assert result.shape == (5000, 25, 6)
    result = result.to_dataset("expressions").to_dataframe()
    dataframe_regression.check(result.iloc[::83])

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


def test_mixed_dtypes(dataframe_regression):
    data = example_data.get_data()
    skims = data["skims"]
    households = data["hhs"]

    prng = default_rng(SeedSequence(42))
    households["otaz_idx"] = households["TAZ"] - 1
    households["dtaz_idx"] = prng.choice(np.arange(25), 5000)
    households["timeperiod5"] = prng.choice(np.arange(5), 5000)
    households["timeperiod3"] = np.clip(households["timeperiod5"], 1, 3) - 1
    households["rownum"] = np.arange(len(households))

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
    result = ss._load(tree, as_dataframe=True, dtype=np.float32)
    dataframe_regression.check(result)

    ss_undot = tree.setup_flow(
        {
            "income": "income",
            "sov_time_by_income": "SOV_TIME/income",
            "sov_time_by_workers": "np.where(workers > 0, SOV_TIME / workers, 0)",
        }
    )
    result = ss_undot._load(tree, as_dataframe=True, dtype=np.float32)
    dataframe_regression.check(result)


def test_tuple_slice(dataframe_regression):
    data = example_data.get_data()
    skims = data["skims"]
    households = data["hhs"]

    prng = default_rng(SeedSequence(42))
    households["otaz_idx"] = households["TAZ"] - 1
    households["dtaz_idx"] = prng.choice(np.arange(25), 5000)
    households["timeperiod5"] = prng.choice(np.arange(5), 5000)
    households["timeperiod3"] = np.clip(households["timeperiod5"], 1, 3) - 1
    households["rownum"] = np.arange(len(households))
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
    result = ss._load(tree, as_dataframe=True, dtype=np.float32)
    dataframe_regression.check(result)


def test_isin(dataframe_regression):
    data = example_data.get_data()
    skims = data["skims"]
    households = data["hhs"]

    prng = default_rng(SeedSequence(42))
    households["otaz_idx"] = households["TAZ"] - 1
    households["dtaz_idx"] = prng.choice(np.arange(25), 5000)
    households["timeperiod5"] = prng.choice(np.arange(5), 5000)
    households["timeperiod3"] = np.clip(households["timeperiod5"], 1, 3) - 1
    households["rownum"] = np.arange(len(households))
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
    result = ss._load(tree, as_dataframe=True, dtype=np.float32)
    dataframe_regression.check(result)


def _get_target(q, token):
    skims_ = Dataset.shm.from_shared_memory(token)
    q.put(skims_.SOV_TIME.sum())


@mark.skipif(
    sys.version_info < (3, 8), reason="shared memory requires python3.8 or higher"
)
def test_shared_memory():

    skims = example_data.get_skims()
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


def test_replacement_filters(dataframe_regression):
    data = example_data.get_data()
    skims = data["skims"]
    households = data["hhs"]

    prng = default_rng(SeedSequence(42))
    households["otaz_idx"] = households["TAZ"] - 1
    households["dtaz_idx"] = prng.choice(np.arange(25), 5000)
    households["timeperiod5"] = prng.choice(np.arange(5), 5000)
    households["timeperiod3"] = np.clip(households["timeperiod5"], 1, 3) - 1
    households["rownum"] = np.arange(len(households))

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


def test_name_in_wrong_subspace(dataframe_regression):
    data = example_data.get_data()
    skims = data["skims"]
    households = data["hhs"]

    prng = default_rng(SeedSequence(42))
    households["otaz_idx"] = households["TAZ"] - 1
    households["dtaz_idx"] = prng.choice(np.arange(25), 5000)
    households["timeperiod5"] = prng.choice(np.arange(5), 5000)
    households["timeperiod3"] = np.clip(households["timeperiod5"], 1, 3) - 1
    households["rownum"] = np.arange(len(households))

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


def test_shared_data_encoded(dataframe_regression):

    data = example_data.get_data()
    skims = data["skims"]
    households = data["hhs"]

    prng = default_rng(SeedSequence(42))
    households["otaz_idx"] = households["TAZ"] - 1
    households["dtaz_idx"] = prng.choice(np.arange(25), 5000)
    households["timeperiod5"] = prng.choice(np.arange(5), 5000)
    households["timeperiod3"] = np.clip(households["timeperiod5"], 1, 3) - 1
    households["rownum"] = np.arange(len(households))
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


def test_dict_encoded(dataframe_regression):
    data = example_data.get_data()
    skims = data["skims"]
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


def test_joint_dict_encoded(dataframe_regression):
    data = example_data.get_data()
    skims = data["skims"]
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
