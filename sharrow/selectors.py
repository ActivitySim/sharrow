import numpy as np
import pandas as pd
import xarray as xr

from .accessors import register_dataset_method


class _Df_Accessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self, *args, **kwargs) -> xr.Dataset:
        raise NotImplementedError()

    def df(self, df, *, append=False, mapping=None):
        """
        Extract values based on the coordinates indicated by columns of a DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame or Mapping[str, array-like]
            The columns (or keys) of `df` should match the named dimensions of
            this Dataset.  The resulting extracted DataFrame will have one row
            per row of `df`, columns matching the data variables in this dataset,
            and each value is looked from the source Dataset.
        append : bool or str, default False
            Assign the results of this extraction to variables in a copy of the
            dataframe `df`.  Set to a string to make that a prefix for the variable
            names.
        mapping : dict, optional
            Apply this rename mapping to the column names before extracting data.

        Returns
        -------
        pandas.DataFrame
        """
        if mapping is not None:
            df_ = df.rename(columns=mapping)
        else:
            df_ = df
        keys = {i: df_[i] for i in self._obj.dims}
        result = self(**keys).reset_coords(drop=True).to_dataframe()
        if isinstance(df, pd.DataFrame):
            result.index = df.index
        if append:
            if isinstance(append, str):
                result = result.add_prefix(append)
            result = df.assign(**result)
        return result

    def ds(self, ds, *, append=False, mapping=None, compute=False):
        """
        Extract values based on the coordinates indicated by variables of a Dataset.

        Parameters
        ----------
        ds : xr.Dataset or Mapping[str, array-like]
            The variables (or keys) of `ds` should match the named dimensions of
            this Dataset, and should all be the same shape.  The resulting
            extracted Dataset will have that common shape, variables matching the
            data variables in this dataset, and each value is looked from the
            source Dataset.
        append : bool or str, default False
            Assign the results of this extraction to variables in a copy of the
            dataset `ds`.  Set to a string to make that a prefix for the variable
            names.
        mapping : dict, optional
            Apply this rename mapping to the variable names before extracting data.
        compute : bool, default False
            Trigger a `compute` on dask arrays in the result.

        Returns
        -------
        xarray.Dataset
        """
        if mapping is not None:
            ds_ = ds.rename(mapping)
        else:
            ds_ = ds
        keys = {i: ds_[i].data for i in self._obj.dims}
        key_dims = None
        for i in self._obj.dims:
            if key_dims is None:
                key_dims = ds_[i].dims
            else:
                assert key_dims == ds_[i].dims
        if len(key_dims) > 1:
            raise NotImplementedError(
                "dataset with more than one dimension not implemented"
            )
        else:
            result = self(**keys, _index_name=key_dims[0]).reset_coords(drop=True)
        if compute:
            result = result.compute()
        if append:
            if isinstance(append, str):
                renames = {k: f"{append}{k}" for k in result.variables}
                result = result.rename(renames)
            result = ds.assign(result)
        return result

    def _filter(
        self,
        *,
        _name=None,
        _names=None,
        _load=False,
        _index_name=None,
        _func=None,
        _raw_idxs=None,
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
            for n in _names:
                if self._obj.redirection.is_blended(n):
                    dims = self._obj[n].dims
                    result[n] = xr.DataArray(
                        self._obj.redirection.get_blended(
                            n,
                            result[n].to_numpy(),
                            _raw_idxs[dims[0]],
                            _raw_idxs[dims[1]],
                        ),
                        dims=result[n].dims,
                    )
            if _name is not None:
                result = result[_name]
            return result
        else:
            result = getattr(ds, _func)(**loaders)
            names = list(result.keys())
            for n in names:
                if self._obj.redirection.is_blended(n):
                    dims = self._obj[n].dims
                    result[n] = xr.DataArray(
                        self._obj.redirection.get_blended(
                            n,
                            result[n].to_numpy(),
                            _raw_idxs[dims[0]],
                            _raw_idxs[dims[1]],
                        ),
                        dims=result[n].dims,
                    )
            return result


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
        modified_idxs = {}
        raw_idxs = {}
        for k, v in idxs.items():
            target = self._obj.redirection.target(k)
            if target is None:
                raw_idxs[k] = modified_idxs[k] = v
            else:
                v_ = np.asarray(v)
                modified_idxs[target] = self._obj[f"_digitized_{target}_of_{k}"][
                    v_
                ].to_numpy()
                raw_idxs[target] = v_  # self._obj[k][v_].to_numpy()
        return self._filter(
            _name=_name,
            _names=_names,
            _load=_load,
            _index_name=_index_name,
            _func="isel",
            _raw_idxs=raw_idxs,
            **modified_idxs,
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
        modified_idxs = {}
        raw_idxs = {}
        any_redirection = False
        for k, v in idxs.items():
            target = self._obj.redirection.target(k)
            if target is None:
                if any_redirection:
                    raise NotImplementedError(
                        "redirection with `at` must be applied to all "
                        "dimensions or none"
                    )
                raw_idxs[k] = np.asarray(v)
                modified_idxs[k] = v
            else:
                upstream = xr.DataArray(np.asarray(v), dims=["index"])
                downstream = self._obj[k]
                mapper = {i: j for (j, i) in enumerate(downstream.to_numpy())}
                offsets = xr.apply_ufunc(np.vectorize(mapper.get), upstream)
                raw_idxs[target] = np.asarray(offsets)
                modified_idxs[target] = self._obj[f"_digitized_{target}_of_{k}"][
                    offsets
                ].to_numpy()
                any_redirection = True
        return self._filter(
            _name=_name,
            _names=_names,
            _load=_load,
            _index_name=_index_name,
            _func="isel" if any_redirection else "sel",
            _raw_idxs=raw_idxs,
            **modified_idxs,
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
