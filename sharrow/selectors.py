import pandas as pd
import xarray as xr

from .accessors import register_dataset_method


class _Df_Accessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self, *args, **kwargs) -> xr.Dataset:
        raise NotImplementedError()

    def df(self, df):
        """
        Extract values by label on the coordinates indicated by columns of a DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame or Mapping[str, array-like]
            The columns (or keys) of `df` should match the named dimensions of
            this Dataset.  The resulting extracted DataFrame will have one row
            per row of `df`, columns matching the data variables in this dataset,
            and each value is looked from the source Dataset.

        Returns
        -------
        pandas.DataFrame
        """
        result = self(**df).reset_coords(drop=True).to_dataframe()
        if isinstance(df, pd.DataFrame):
            result.index = df.index
        return result

    def _filter(
        self,
        *,
        _name=None,
        _names=None,
        _load=False,
        _index_name=None,
        _func=None,
        **idxs,
    ):
        loaders = {}
        if _index_name is None:
            _index_name = "index"
        for k, v in idxs.items():
            loaders[k] = xr.DataArray(v, dims=[_index_name])
        if _name is not None:
            assert isinstance(_name, str)
            _names = [_name]
        if _names:
            _baggage = list(self._obj.digital_encoding.baggage(_names))
            _all_names = list(_names) + _baggage
            ds = self._obj[_all_names]
        else:
            _baggage = []
            ds = self._obj
        if _load:
            ds = ds.load()
        if _names:
            result = (
                getattr(ds, _func)(**loaders)
                .digital_encoding.strip(_names)
                .drop_vars(_baggage)
            )
            if _name is not None:
                result = result[_name]
            return result
        return getattr(ds, _func)(**loaders)


@xr.register_dataset_accessor("iat")
class _Iat_Accessor(_Df_Accessor):
    """
    Multi-dimensional fancy indexing by position.

    Provide the dataset dimensions to index up as keywords, each with
    a value giving an array (one dimensional) of positions to extract.

    All other arguments are keyword-only arguments beginning with an
    underscore.

    Parameters
    ----------
    _name : str, optional
        Only process this variable of this Dataset, and return a DataArray.
    _names : Collection[str], optional
        Only include these variables of this Dataset.
    _load : bool, default False
        Call `load` on the result, which will trigger a compute
        operation if the data underlying this Dataset is in dask,
        otherwise this does nothing.
    _index_name : str, default "index"
        The name to use for the resulting dataset's dimension.
    **idxs : Mapping[str, Any]
        Positions to extract.

    Returns
    -------
    Dataset or DataArray
    """

    def __call__(
        self, *, _name=None, _names=None, _load=False, _index_name=None, **idxs
    ):
        return self._filter(
            _name=_name,
            _names=_names,
            _load=_load,
            _index_name=_index_name,
            _func="isel",
            **idxs,
        )


@xr.register_dataset_accessor("at")
class _At_Accessor(_Df_Accessor):
    """
    Multi-dimensional fancy indexing by label.

    Provide the dataset dimensions to index up as keywords, each with
    a value giving an array (one dimensional) of labels to extract.

    All other arguments are keyword-only arguments beginning with an
    underscore.

    Parameters
    ----------
    _name : str, optional
        Only process this variable of this Dataset, and return a DataArray.
    _names : Collection[str], optional
        Only include these variables of this Dataset.
    _load : bool, default False
        Call `load` on the result, which will trigger a compute
        operation if the data underlying this Dataset is in dask,
        otherwise this does nothing.
    _index_name : str, default "index"
        The name to use for the resulting dataset's dimension.
    **idxs : Mapping[str, Any]
        Labels to extract.

    Returns
    -------
    Dataset or DataArray
    """

    def __call__(
        self, *, _name=None, _names=None, _load=False, _index_name=None, **idxs
    ):
        return self._filter(
            _name=_name,
            _names=_names,
            _load=_load,
            _index_name=_index_name,
            _func="sel",
            **idxs,
        )


@register_dataset_method
def keep_dims(self, keep_dims, *, errors="raise"):
    """
    Keep only certain dimensions and associated variables from this dataset.

    Parameters
    ----------
    keep_dims : hashable or iterable of hashable
        Dimension or dimensions to keep.
    errors : {"raise", "ignore"}, optional
        If 'raise' (default), raises a ValueError error if any of the
        dimensions passed are not in the dataset. If 'ignore', any given
        dimensions that are in the dataset are dropped and no error is raised.

    Returns
    -------
    obj : Dataset
        The dataset without the given dimensions (or any variables
        containing those dimensions)
    """
    if isinstance(keep_dims, str):
        keep_dims = {keep_dims}
    else:
        keep_dims = set(keep_dims)
    all_dims = set(self.dims)
    if errors == "raise":
        missing_dims = keep_dims - all_dims
        if missing_dims:
            raise ValueError(
                f"{self.__class__} does not contain the dimensions: {missing_dims}"
            )
    return self.drop_dims([i for i in all_dims if i not in keep_dims])
