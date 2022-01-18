import numpy as np
import pandas as pd
import xarray as xr


def _iat(source, *, _names=None, _load=False, _index_name=None, **idxs):
    loaders = {}
    if _index_name is None:
        _index_name = "index"
    for k, v in idxs.items():
        loaders[k] = xr.DataArray(v, dims=[_index_name])
    if _names:
        ds = source[_names]
    else:
        ds = source
    if _load:
        ds = ds._load()
    return ds.isel(**loaders)


def _at(source, *, _names=None, _load=False, _index_name=None, **idxs):
    loaders = {}
    if _index_name is None:
        _index_name = "index"
    for k, v in idxs.items():
        loaders[k] = xr.DataArray(v, dims=[_index_name])
    if _names:
        ds = source[_names]
    else:
        ds = source
    if _load:
        ds = ds._load()
    return ds.sel(**loaders)


def gather(source, indexes):
    """
    Extract values by label on the coordinates indicated by columns of a DataFrame.

    Parameters
    ----------
    source : xarray.DataArray or xarray.Dataset
        The source of the values to extract.
    indexes : Mapping[str, array-like]
        The keys of `indexes` (if given as a dataframe, the column names)
        should match the named dimensions of `source`.  The resulting extracted
        data will have a shape one row per row of `df`, and columns matching
        the data variables in `source`, and each value is looked up by the labels.

    Returns
    -------
    pd.DataFrame
    """
    result = _at(source, **indexes).reset_coords(drop=True)
    return result


def igather(source, positions):
    """
    Extract values by position on the coordinates indicated by columns of a DataFrame.

    Parameters
    ----------
    source : xarray.DataArray or xarray.Dataset
    positions : pd.DataFrame or Mapping[str, array-like]
        The columns (or keys) of `df` should match the named dimensions of
        this Dataset.  The resulting extracted DataFrame will have one row
        per row of `df`, columns matching the data variables in this dataset,
        and each value is looked up by the positions.

    Returns
    -------
    pd.DataFrame
    """
    result = _iat(source, **positions).reset_coords(drop=True)
    return result


class DatasetWrapper:
    def __init__(self, dataset, orig_key, dest_key, time_key=None):
        """

        Parameters
        ----------
        skim_dict: SkimDict

        orig_key: str
            name of column in dataframe to use as implicit orig for lookups
        dest_key: str
            name of column in dataframe to use as implicit dest for lookups
        """
        self.dataset = dataset
        self.orig_key = orig_key
        self.dest_key = dest_key
        self.time_key = time_key
        self.df = None

    def set_df(self, df):
        """
        Set the dataframe

        Parameters
        ----------
        df : DataFrame
            The dataframe which contains the origin and destination ids

        Returns
        -------
        self (to facilitate chaining)
        """
        assert (
            self.orig_key in df
        ), f"orig_key '{self.orig_key}' not in df columns: {list(df.columns)}"
        assert (
            self.dest_key in df
        ), f"dest_key '{self.dest_key}' not in df columns: {list(df.columns)}"
        if self.time_key:
            assert (
                self.time_key in df
            ), f"time_key '{self.time_key}' not in df columns: {list(df.columns)}"
        self.df = df

        # TODO allow non-1 offsets
        positions = {
            "otaz": df[self.orig_key] - 1,
            "dtaz": df[self.dest_key] - 1,
        }
        if self.time_key:
            time_map = {j: i for i, j in enumerate(self.dataset.indexes["time_period"])}
            positions["time_period"] = pd.Series(
                np.vectorize(time_map.get)(df[self.time_key]),
                index=df.index,
            )
        self.positions = pd.DataFrame(positions)

        return self

    def lookup(self, key, reverse=False):
        """
        Generally not called by the user - use __getitem__ instead

        Parameters
        ----------
        key : hashable
             The key (identifier) for this skim object

        od : bool (optional)
            od=True means lookup standard origin-destination skim value
            od=False means lookup destination-origin skim value

        Returns
        -------
        impedances: pd.Series
            A Series of impedances which are elements of the Skim object and
            with the same index as df
        """

        assert self.df is not None, "Call set_df first"
        if reverse:
            x = self.positions.rename(columns={"otaz": "dtaz", "dtaz": "otaz"})
        else:
            x = self.positions

        # Return a series, consistent with ActivitySim SkimWrapper
        return igather(self.dataset[key], x).to_series()

    def __getitem__(self, key):
        """
        Get the lookup for an available skim object (df and orig/dest and column names implicit)

        Parameters
        ----------
        key : hashable
             The key (identifier) for the skim object

        Returns
        -------
        impedances: pd.Series with the same index as df
            A Series of impedances values from the single Skim with specified key, indexed byt orig/dest pair
        """
        return self.lookup(key)
