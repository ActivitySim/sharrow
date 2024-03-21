import pandas as pd
import pytest
import xarray as xr

from sharrow.dataset import construct


@pytest.fixture
def person_dataset() -> xr.Dataset:
    """Sample persons dataset with dummy data."""
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
    """Sample household dataset with dummy data."""
    df = pd.DataFrame(
        {
            "n_cars": [1, 2, 1],
        },
        index=pd.Index([11, 22, 33], name="household_id"),
    )
    return construct(df)


@pytest.fixture
def tours_dataset() -> xr.Dataset:
    """Sample tours dataset with dummy data."""
    df = pd.DataFrame(
        {
            "TourMode": ["Car", "Bus", "Car", "Car", "Walk"],
            "person_id": [441, 445, 552, 556, 934],
            "origin": [101, 102, 103, 101, 103],
            "destination": [102, 103, 103, 101, 102],
            "time_period": ["AM", "MD", "AM", "AM", "MD"],
            "origin_idx": [0, 1, 2, 0, 2],
            "destination_idx": [1, 2, 2, 0, 1],
            "time_period_alt": ["AM", "MD", "AM", "AM", "MD"],
        },
        index=pd.Index([4411, 4451, 5521, 5561, 9341], name="tour_id"),
    )
    df["TourMode"] = df["TourMode"].astype("category")
    df["time_period"] = df["time_period"].astype(
        pd.CategoricalDtype(categories=["AM", "MD"], ordered=False)
    )
    # time_period_alt is intentionally constructed backwards to test that bad ordering is handled
    df["time_period_alt"] = df["time_period_alt"].astype(
        pd.CategoricalDtype(categories=["MD", "AM"], ordered=False)
    )
    return construct(df)


@pytest.fixture
def skims_dataset() -> xr.Dataset:
    """Sample skims dataset with dummy data."""
    return xr.Dataset(
        {
            "cartime": xr.DataArray(
                [
                    [[11.1, 22.2, 33.3], [44.4, 55.5, 66.6], [77.7, 88.8, 99.9]],
                    [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]],
                ],
                dims=["timeperiod", "otaz", "dtaz"],
                coords={
                    "timeperiod": ["AM", "MD"],
                    "otaz": [101, 102, 103],
                    "dtaz": [101, 102, 103],
                },
            )
        }
    )
