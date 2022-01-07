import os
import pathlib

import numpy as np
import pandas as pd

# import pyarrow as pa
# import pyarrow.feather as pf
# import larch
from numpy.random import SeedSequence, default_rng
from pytest import approx, fixture, raises

from sharrow import Dataset, DataTree, example_data


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
    result = ss.load(tree, as_dataframe=True)
    dataframe_regression.check(result)

    ss_undot = tree.setup_flow(
        {
            "income": "income",
            "sov_time_by_income": "SOV_TIME/income",
            "sov_cost_by_income": "HOV3_TIME",
        }
    )
    result = ss_undot.load(tree, as_dataframe=True)
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
    result2 = s2.load(tree, as_dataframe=True)
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
    result = ss.load(tree, as_dataframe=True)
    dataframe_regression.check(result)
    with raises(AssertionError):
        pd.testing.assert_series_equal(
            result["round_trip_hov3_time"], result["double_hov3_time"],
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
        extra_hash_data=(1, 2,),
    )
    result = ss.load(tree, as_dataframe=True)
    dataframe_regression.check(result, basename="test_shared_data_reversible")
    with raises(AssertionError):
        pd.testing.assert_series_equal(
            result["round_trip_hov3_time"], result["double_hov3_time"],
        )

    dtree = tree.digitize_relationships()
    dss = dtree.setup_flow(
        {
            "income": "base.income",
            "sov_time_by_income": "odt_skims.SOV_TIME/base.income",
            "round_trip_hov3_time": "dot_skims.HOV3_TIME + odt_skims.HOV3_TIME",
            "double_hov3_time": "odt_skims.HOV3_TIME * 2",
        },
        extra_hash_data=(1, 2, 3,),
    )
    dresult = dss.load(dtree, as_dataframe=True)
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

    blank = Dataset.from_named_objects(households.index, skims["dtaz"])
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
            "income": "base.income",
            "sov_time_by_income": "odt_skims.SOV_TIME/base.income",
            "round_trip_hov3_time": "dot_skims.HOV3_TIME + odt_skims.HOV3_TIME",
            "double_hov3_time": "odt_skims.HOV3_TIME * 2",
            "a_trip_hov3_time": "dot_skims.HOV3_TIME",
            "b_trip_hov3_time": "odt_skims.HOV3_TIME",
        }
    )
    result = ss.load(tree, as_dataarray=True)
    assert result.dims == ("HHID", "dtaz", "expressions")
    assert result.shape == (5000, 25, 6)
    result = result.to_dataset("expressions").to_dataframe()
    dataframe_regression.check(result)

    dot_result = ss.load(tree, as_dataarray=True, dot=np.ones(6))
    assert dot_result.dims == ("HHID", "dtaz",)
    assert dot_result.shape == (5000, 25,)

    check_vs = np.dot(result, np.ones([6])).reshape(5000, 25)
    np.testing.assert_array_almost_equal(check_vs, dot_result.to_numpy())


def _get_target(q):
    skims_ = Dataset.from_shared_memory("skims")
    q.put(skims_.SOV_TIME.sum())


def test_shared_memory():

    skims = example_data.get_skims()

    skims_2 = skims.to_shared_memory("skims")
    target = skims.SOV_TIME.sum()
    assert skims_2.SOV_TIME.sum() == target

    # reconstruct in same process
    skims_3 = Dataset.from_shared_memory("skims")
    assert skims_3.SOV_TIME.sum() == target

    # reconstruct in different process
    from multiprocessing import Process, Queue

    q = Queue()
    p = Process(target=_get_target, args=(q,))
    p.start()
    p.join()
    assert q.get() == target
