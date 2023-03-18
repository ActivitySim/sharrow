import pytest
import xarray as xr
import pandas as pd
from sharrow.dataset import construct


@pytest.fixture
def person_dataset() -> xr.Dataset:
    """
    Sample persons dataset with dummy data.
    """
    df = pd.DataFrame(
        {
            "Income": [45, 88, 56, 15, 71],
            "Name": ["Andre", "Bruce", "Carol", "David", "Eugene"],
            "Age": [14, 25, 55, 8, 21],
            "WorkMode": ["Car", "Bus", "Car", "Car", "Walk"],
            "household_id": [11, 11, 22, 22, 33],
        },
        index=pd.Index([441, 445, 552, 556, 934], name="person_id"),
    )
    df["WorkMode"] = df["WorkMode"].astype("category")
    return construct(df)


@pytest.fixture
def household_dataset() -> xr.Dataset:
    """
    Sample household dataset with dummy data.
    """
    df = pd.DataFrame(
        {
            "n_cars": [1, 2, 1],
        },
        index=pd.Index([11, 22, 33], name="household_id"),
    )
    return construct(df)


@pytest.fixture
def tours_dataset() -> xr.Dataset:
    """
    Sample tours dataset with dummy data.
    """
    df = pd.DataFrame(
        {
            "TourMode": ["Car", "Bus", "Car", "Car", "Walk"],
            "person_id": [441, 445, 552, 556, 934],
        },
        index=pd.Index([4411, 4451, 5521, 5561, 9341], name="tour_id"),
    )
    df["TourMode"] = df["TourMode"].astype("category")
    return construct(df)
