from __future__ import annotations

from enum import IntEnum

import numpy as np
import pandas as pd
import xarray as xr

import sharrow


def test_simple_cat(tours_dataset: xr.Dataset):
    tree = sharrow.DataTree(tours=tours_dataset)

    assert all(tours_dataset.TourMode.cat.categories == ["Bus", "Car", "Walk"])

    expr = "tours.TourMode == 'Bus'"
    f = tree.setup_flow({expr: expr})
    a = f.load_dataarray(dtype=np.int8)
    a = a.isel(expressions=0)
    assert all(a == np.asarray([0, 1, 0, 0, 0]))

    tour_mode_bus = tree.get_expr(expr)
    assert all(tour_mode_bus == np.asarray([0, 1, 0, 0, 0]))


def test_2_level_tree_cat(
    tours_dataset: xr.Dataset,
    person_dataset: xr.Dataset,
):
    tree = sharrow.DataTree(tours=tours_dataset)
    tree.add_dataset("persons", person_dataset, "tours.person_id @ persons.person_id")

    assert all(tours_dataset.TourMode.cat.categories == ["Bus", "Car", "Walk"])

    expr = "tours.TourMode == 'Bus'"
    f = tree.setup_flow({expr: expr})
    a = f.load_dataarray(dtype=np.int8)
    a = a.isel(expressions=0)
    assert all(a == np.asarray([0, 1, 0, 0, 0]))

    tour_mode_bus = tree.get_expr(expr)
    assert all(tour_mode_bus == np.asarray([0, 1, 0, 0, 0]))

    work_mode_bus = tree.get_expr("WorkMode == 'Walk'")
    assert all(work_mode_bus == np.asarray([0, 0, 0, 0, 1]))

    work_mode_bus1 = tree.get_expr("persons.WorkMode == 'Walk'")
    assert all(work_mode_bus1 == np.asarray([0, 0, 0, 0, 1]))


def test_3_level_tree_cat(
    tours_dataset: xr.Dataset,
    person_dataset: xr.Dataset,
    household_dataset: xr.Dataset,
):
    tree = sharrow.DataTree(tours=tours_dataset)
    tree.add_dataset("persons", person_dataset, "tours.person_id @ persons.person_id")
    tree.add_dataset(
        "households", person_dataset, "persons.household_id @ households.household_id"
    )

    assert all(tours_dataset.TourMode.cat.categories == ["Bus", "Car", "Walk"])

    expr = "tours.TourMode == 'Bus'"
    f = tree.setup_flow({expr: expr})
    a = f.load_dataarray(dtype=np.int8)
    a = a.isel(expressions=0)
    assert all(a == np.asarray([0, 1, 0, 0, 0]))

    tour_mode_bus = tree.get_expr(expr)
    assert all(tour_mode_bus == np.asarray([0, 1, 0, 0, 0]))

    work_mode_bus = tree.get_expr("WorkMode == 'Walk'")
    assert all(work_mode_bus == np.asarray([0, 0, 0, 0, 1]))

    work_mode_bus1 = tree.get_expr("persons.WorkMode == 'Walk'")
    assert all(work_mode_bus1 == np.asarray([0, 0, 0, 0, 1]))


def test_rootless_tree_cat(
    tours_dataset: xr.Dataset,
    person_dataset: xr.Dataset,
    household_dataset: xr.Dataset,
):
    tree = sharrow.DataTree(tours=tours_dataset, root_node_name=False)
    tree.add_dataset("persons", person_dataset, "tours.person_id @ persons.person_id")
    tree.add_dataset(
        "households", person_dataset, "persons.household_id @ households.household_id"
    )

    assert all(tours_dataset.TourMode.cat.categories == ["Bus", "Car", "Walk"])

    expr = "tours.TourMode == 'Bus'"
    f = tree.setup_flow({expr: expr}, with_root_node_name="tours")
    a = f.load_dataarray(dtype=np.int8)
    a = a.isel(expressions=0)
    assert all(a == np.asarray([0, 1, 0, 0, 0]))


def test_int_enum_categorical():
    class TourMode(IntEnum):
        Car = 1
        Bus = 2
        Walk = 3

    df = pd.DataFrame(
        {
            "TourMode": ["Car", "Bus", "Car", "Car", "Walk"],
            "person_id": [441, 445, 552, 556, 934],
        },
        index=pd.Index([4411, 4451, 5521, 5561, 9341], name="tour_id"),
    )
    df["TourMode2"] = df["TourMode"].as_int_enum(TourMode)
    assert df["TourMode2"].dtype == "category"
    assert all(df["TourMode2"].cat.categories == ["_0", "Car", "Bus", "Walk"])
    assert all(df["TourMode2"].cat.codes == [1, 2, 1, 1, 3])
