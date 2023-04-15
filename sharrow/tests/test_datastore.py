from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import xarray as xr

from sharrow.datastore import DataStore, ReadOnlyError


def test_datasstore_checkpointing(tmp_path: Path, person_dataset):
    tm = DataStore(directory=tmp_path)
    tm["persons"] = person_dataset
    tm.make_checkpoint("init_persons")

    person_dataset["DoubleAge"] = person_dataset["Age"] * 2
    tm.update("persons", person_dataset["DoubleAge"])
    tm.make_checkpoint("annot_persons")

    tm2 = DataStore(directory=tmp_path)
    tm2.restore_checkpoint("annot_persons")
    xr.testing.assert_equal(tm2.get_dataset("persons"), person_dataset)

    tm2.restore_checkpoint("init_persons")
    assert "DoubleAge" not in tm2.get_dataset("persons")

    tm_ro = DataStore(directory=tmp_path, mode="r")
    with pytest.raises(ReadOnlyError):
        tm_ro.make_checkpoint("will-fail")


def test_datasstore_checkpointing_parquet(tmp_path: Path, person_dataset):
    tm = DataStore(directory=tmp_path, storage_format="parquet")
    tm["persons"] = person_dataset
    tm.make_checkpoint("init_persons")

    person_dataset["DoubleAge"] = person_dataset["Age"] * 2
    tm.update("persons", person_dataset["DoubleAge"])
    tm.make_checkpoint("annot_persons")

    tm2 = DataStore(directory=tmp_path)
    tm2.restore_checkpoint("annot_persons")
    xr.testing.assert_equal(tm2.get_dataset("persons"), person_dataset)

    tm2.restore_checkpoint("init_persons")
    assert "DoubleAge" not in tm2.get_dataset("persons")

    tm_ro = DataStore(directory=tmp_path, mode="r")
    with pytest.raises(ReadOnlyError):
        tm_ro.make_checkpoint("will-fail")


def test_datasstore_relationships(
    tmp_path: Path, person_dataset, household_dataset, tours_dataset
):
    pth = tmp_path.joinpath("relations")

    if pth.exists():
        shutil.rmtree(pth)

    pth.mkdir(parents=True, exist_ok=True)
    tm = DataStore(directory=pth)

    tm["persons"] = person_dataset
    tm.make_checkpoint("init_persons")

    tm["households"] = household_dataset
    tm.add_relationship("persons.household_id @ households.household_id")
    tm.make_checkpoint("init_households")

    tm["tours"] = tours_dataset
    tm.add_relationship("tours.person_id @ persons.person_id")
    tm.make_checkpoint("init_tours")

    tm.digitize_relationships()
    assert tm.relationships_are_digitized

    tm.make_checkpoint("digitized")

    tm2 = DataStore(directory=pth, mode="r")
    tm2.read_metadata("*")
    tm2.restore_checkpoint("init_households")

    assert sorted(tm2.get_dataset("persons")) == [
        "Age",
        "Income",
        "Name",
        "WorkMode",
        "household_id",
    ]

    assert sorted(tm2.get_dataset("households")) == [
        "n_cars",
    ]

    tm2.restore_checkpoint("digitized")
    assert sorted(tm2.get_dataset("persons")) == [
        "Age",
        "Income",
        "Name",
        "WorkMode",
        "digitizedOffsethousehold_id_households_household_id",
        "household_id",
    ]

    double_age = tm2.get_dataset("persons")["Age"] * 2
    with pytest.raises(ReadOnlyError):
        tm2.update("persons", double_age.rename("doubleAge"))

    with pytest.raises(ReadOnlyError):
        tm2.make_checkpoint("age-x2")

    tm.update("persons", double_age.rename("doubleAge"))
    assert sorted(tm.get_dataset("persons")) == [
        "Age",
        "Income",
        "Name",
        "WorkMode",
        "digitizedOffsethousehold_id_households_household_id",
        "doubleAge",
        "household_id",
    ]

    tm.make_checkpoint("age-x2")
    tm2.read_metadata()
    tm2.restore_checkpoint("age-x2")

    person_restored = tm2.get_dataframe("persons")
    print(person_restored.WorkMode.dtype)
    assert person_restored.WorkMode.dtype == "category"
