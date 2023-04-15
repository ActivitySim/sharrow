from __future__ import annotations

import numpy as np
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
