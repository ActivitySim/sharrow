import io
import os
import textwrap
import sys
import importlib
import inspect
import re
import hashlib
import base64

import dask
import numpy as np
import xarray as xr
import numba as nb
import pandas as pd
import pyarrow as pa
import dask.array as da
import warnings
import logging
import sharrow
import time
from collections.abc import Sequence

from typing import (
    Any,
    Dict,
    Hashable,
    Mapping,
    Sequence,
)


from .aster import expression_for_numba, extract_all_name_tokens, extract_names_2
from .maths import piece, hard_sigmoid, transpose_leading, clip
from .table import Table
from .filewrite import rewrite
from .shared_memory import *
from . import __version__

logger = logging.getLogger("sharrow")

well_known_names = {
    'nb', 'np', 'pd', 'xr', 'pa',
    'log', 'exp', 'log1p', 'expm1', 'max', 'min',
    'piece', 'hard_sigmoid', 'transpose_leading', 'clip',
}


def one_based(n):
    return pd.RangeIndex(1, n+1)


def zero_based(n):
    return pd.RangeIndex(0, n)


def clean(s):
    """
    Convert any string into a similar python identifier.

    If any modification of the string is made, or if the string
    is longer than 120 characters, it is truncated and a hash of the
    original string is added to the end, to ensure every
    string maps to a unique cleaned name.

    Parameters
    ----------
    s : str

    Returns
    -------
    cleaned : str
    """
    if not isinstance(s, str):
        s = f"{type(s)}-{s}"
    cleaned = re.sub('\W|^(?=\d)','_', s)
    if cleaned != s or len(cleaned) > 120:
        # digest size 15 creates a 24 character base32 string
        h = base64.b32encode(
            hashlib.blake2b(s.encode(), digest_size=15).digest()
        ).decode()
        cleaned = f"{cleaned[:90]}_{h}"
    return cleaned


def coerce_to_range_index(idx):
    if isinstance(idx, pd.RangeIndex):
        return idx
    if isinstance(idx, (pd.Int64Index, pd.Float64Index, pd.UInt64Index)):
        if idx.is_monotonic_increasing and idx[-1] - idx[0] == idx.size - 1:
            return pd.RangeIndex(idx[0], idx[0]+idx.size)
    return idx


def is_dict_like(value: Any) -> bool:
    return hasattr(value, "keys") and hasattr(value, "__getitem__")



class _LocIndexer:
    __slots__ = ("dataset",)

    def __init__(self, dataset: "Dataset"):
        self.dataset = dataset

    def __getitem__(self, key: Mapping[Hashable, Any]) -> "Dataset":
        if not is_dict_like(key):
            if len(self.dataset.dims) == 1:
                dim_name = self.dataset.dims.__iter__().__next__()
                key = {dim_name: key}
            else:
                raise TypeError(
                    "can only lookup dictionaries from Dataset.loc, "
                    "unless there is only one dimension"
                )
        return self.dataset.sel(key)


class _iLocIndexer:
    __slots__ = ("dataset",)

    def __init__(self, dataset: "Dataset"):
        self.dataset = dataset

    def __getitem__(self, key: Mapping[Hashable, Any]) -> "Dataset":
        if not is_dict_like(key):
            if len(self.dataset.dims) == 1:
                dim_name = self.dataset.dims.__iter__().__next__()
                key = {dim_name: key}
            else:
                raise TypeError(
                    "can only lookup dictionaries from Dataset.iloc, "
                    "unless there is only one dimension"
                )
        return self.dataset.isel(key)


class Dataset(xr.Dataset):


    __slots__ = (
        '_shared_memory_key_',
        '_shared_memory_objs_',
        '_shared_memory_owned_',
        '__global_shared_memory_pool'
    )

    def __init__(self, *args, **kwargs):
        super(Dataset, self).__init__(*args, **kwargs)
        if len(args) == 1 and len(kwargs)==0 and isinstance(args[0], xr.Dataset):
            self.attrs = args[0].attrs

    @classmethod
    def construct(cls, source):
        if isinstance(source, pd.DataFrame):
            source = cls.from_dataframe(source)
            #source = cls.from_dataframe_fast(source) # older xarray was slow
        elif isinstance(source, (Table, pa.Table)):
            source = cls.from_table(source)
        elif isinstance(source, (pa.Table)):
            source = cls.from_table(source)
        elif isinstance(source, cls):
            pass # don't do the superclass things
        elif isinstance(source, xr.Dataset):
            source = cls(source)
        elif isinstance(source, Sequence) and all(isinstance(i, str) for i in source):
            source = cls.from_table(pa.table({i:[] for i in source}))
        else:
            raise TypeError(f"source cannot be type {type(source)}")
        return source

    def update(self, other):
        super().update(other)
        if isinstance(other, Dataset):
            match_names = self.match_names
            match_names.update(other.match_names)
            self.match_names = match_names
        return self  # deprecated return for consistency until xarray 0.19

    @classmethod
    def from_table(
        cls, tbl, index_name="index", index=None,
    ):
        """
        Convert a pyarrow.Table into an xarray.Dataset

        Parameters
        ----------
        tbl : Table
            Table from which to use data and indices.
        index_name : str, default 'index'
            This name will be given to the default dimension index, if
            none is given.  Ignored if `index` is given explicitly.
        index : Index-like, optional
            Use this index instead of a default RangeIndex.

        Returns
        -------
        New Dataset.
        """
        if len(set(tbl.column_names)) != len(tbl.column_names):
            raise ValueError("cannot convert Table with non-unique columns")
        if index is None:
            index = pd.RangeIndex(len(tbl), name=index_name)
        else:
            if len(index) != len(tbl):
                raise ValueError(
                    f"length of index ({len(index)}) does not match length of table ({len(tbl)})"
                )
        if isinstance(index, pd.MultiIndex) and not index.is_unique:
            raise ValueError(
                "cannot attach a non-unique MultiIndex and convert into xarray"
            )
        arrays = [
            (tbl.column_names[n], np.asarray(tbl.column(n)))
            for n in range(len(tbl.column_names))
        ]
        result = cls()
        if isinstance(index, pd.MultiIndex):
            dims = tuple(
                name if name is not None else "level_%i" % n
                for n, name in enumerate(index.names)
            )
            for dim, lev in zip(dims, index.levels):
                result[dim] = (dim, lev)
        else:
            index_name = index.name if index.name is not None else "index"
            dims = (index_name,)
            result[index_name] = (dims, index)

        result._set_numpy_data_from_dataframe(index, arrays, dims)
        return result

    @classmethod
    def from_omx(
        cls, omx, index_names=("otaz", "dtaz"), indexes="one-based", renames=None,
    ):
        # handle both larch.OMX and openmatrix.open_file versions
        if "larch" in type(omx).__module__:
            omx_data = omx.data
            omx_shape = omx.shape
        else:
            omx_data = omx.root["data"]
            omx_shape = omx.shape()

        arrays = {}
        if renames is None:
            for k in omx_data._v_children:
                arrays[k] = omx_data[k][:]
        elif isinstance(renames, dict):
            for new_k, old_k in renames.items():
                arrays[new_k] = omx_data[old_k][:]
        else:
            for k in renames:
                arrays[k] = omx_data[k][:]
        d = {
            "dims": index_names,
            "data_vars": {k: {"dims": index_names, "data": arrays[k]} for k in arrays},
        }
        if indexes == "one-based":
            indexes = {
                index_names[0]: one_based(omx_shape[0]),
                index_names[1]: one_based(omx_shape[1]),
            }
        elif indexes == "zero-based":
            indexes = {
                index_names[0]: zero_based(omx_shape[0]),
                index_names[1]: zero_based(omx_shape[1]),
            }
        if indexes is not None:
            d["coords"] = {
                index_name: {"dims": index_name, "data": index}
                for index_name, index in indexes.items()
            }
        return cls.from_dict(d)

    @classmethod
    def from_omx_3d(
            cls,
            omx,
            index_names=("otaz", "dtaz", "time_period"),
            indexes=None,
            *,
            time_periods=None,
            time_period_sep="__",
    ):
        # handle both larch.OMX and openmatrix.open_file versions
        if "larch" in type(omx).__module__:
            omx_data = omx.data
            omx_shape = omx.shape
            omx_lookup = omx.lookup
        else:
            omx_data = omx.root["data"]
            omx_shape = omx.shape()
            omx_lookup = omx.root["lookup"]

        data_names = list(omx_data._v_children.keys())
        n1, n2 = omx_shape
        if indexes is None:
            # default reads mapping if only one lookup is included, otherwise one-based
            if len(omx_lookup._v_children) == 1:
                ranger = None
            else:
                ranger = one_based
        elif indexes == "one-based":
            ranger = one_based
        elif indexes == "zero-based":
            ranger = zero_based
        elif indexes in set(omx_lookup._v_children):
            ranger = None
        else:
            raise NotImplementedError("only one-based, zero-based, and named indexes are implemented")
        if ranger is not None:
            r1 = ranger(n1)
            r2 = ranger(n2)
        else:
            r1 = r2 = pd.Index(omx_lookup[indexes])

        if time_periods is None:
            raise ValueError("must give time periods explicitly")

        ds = cls()
        for k in data_names:
            if time_period_sep in k:
                base_k, time_k = k.split(time_period_sep, 1)
                if base_k not in ds:
                    ds[base_k] = xr.DataArray(
                        np.float32(0),
                        dims=index_names,
                        coords={
                            index_names[0]: r1,
                            index_names[1]: r2,
                            index_names[2]: time_periods,
                        },
                    )
                ds[base_k].loc[:, :, time_k] = omx_data[k][:]
            else:
                ds[k] = xr.DataArray(
                    omx_data[k][:],
                    dims=index_names[:2],
                    coords={
                        index_names[0]: r1,
                        index_names[1]: r2,
                    },
                )
        return ds

    @classmethod
    def from_amx(
        cls, amx, index_names=("otaz", "dtaz"), indexes="one-based", renames=None,
    ):
        arrays = {}
        if renames is None:
            for k in amx.list_matrices():
                arrays[k] = amx[k][:]
        elif isinstance(renames, dict):
            for new_k, old_k in renames.items():
                arrays[new_k] = amx[old_k]
        else:
            for k in renames:
                arrays[k] = amx[k]
        d = {
            "dims": index_names,
            "data_vars": {k: {"dims": index_names, "data": arrays[k]} for k in arrays},
        }
        if indexes == "one-based":
            indexes = {index_names[i]: "1" for i in range(len(index_names))}
        elif indexes == "zero-based":
            indexes = {index_names[i]: "0" for i in range(len(index_names))}
        if isinstance(indexes, (list, tuple)):
            indexes = dict(zip(index_names, indexes))
        if isinstance(indexes, dict):
            for n, i in enumerate(index_names):
                if indexes.get(i) == "1":
                    indexes[i] = one_based(amx.shape[n])
                elif indexes.get(i) == "0":
                    indexes[i] = zero_based(amx.shape[n])
        if indexes is not None:
            d["coords"] = {
                index_name: {"dims": index_name, "data": index}
                for index_name, index in indexes.items()
            }
        return cls.from_dict(d)

    @classmethod
    def from_zarr(cls, store, *args, **kwargs):
        return cls(xr.open_zarr(store, *args, **kwargs))

    def to_zarr(self, *args, **kwargs):
        """
        Write dataset contents to a zarr group.

        Parameters
        ----------
        store : MutableMapping, str or Path, optional
            Store or path to directory in file system.  If given with a
            ".zarr.zip" extension, and keyword arguments limited to 'mode' and
            'compression', then a ZipStore will be created, populated, and then
            immediately closed.
        chunk_store : MutableMapping, str or Path, optional
            Store or path to directory in file system only for Zarr array chunks.
            Requires zarr-python v2.4.0 or later.
        mode : {"w", "w-", "a", None}, optional
            Persistence mode: "w" means create (overwrite if exists);
            "w-" means create (fail if exists);
            "a" means override existing variables (create if does not exist).
            If ``append_dim`` is set, ``mode`` can be omitted as it is
            internally set to ``"a"``. Otherwise, ``mode`` will default to
            `w-` if not set.
        synchronizer : object, optional
            Zarr array synchronizer.
        group : str, optional
            Group path. (a.k.a. `path` in zarr terminology.)
        encoding : dict, optional
            Nested dictionary with variable names as keys and dictionaries of
            variable specific encodings as values, e.g.,
            ``{"my_variable": {"dtype": "int16", "scale_factor": 0.1,}, ...}``
        compute : bool, optional
            If True write array data immediately, otherwise return a
            ``dask.delayed.Delayed`` object that can be computed to write
            array data later. Metadata is always updated eagerly.
        consolidated : bool, optional
            If True, apply zarr's `consolidate_metadata` function to the store
            after writing metadata.
        append_dim : hashable, optional
            If set, the dimension along which the data will be appended. All
            other dimensions on overriden variables must remain the same size.
        region : dict, optional
            Optional mapping from dimension names to integer slices along
            dataset dimensions to indicate the region of existing zarr array(s)
            in which to write this dataset's data. For example,
            ``{'x': slice(0, 1000), 'y': slice(10000, 11000)}`` would indicate
            that values should be written to the region ``0:1000`` along ``x``
            and ``10000:11000`` along ``y``.

            Two restrictions apply to the use of ``region``:

            - If ``region`` is set, _all_ variables in a dataset must have at
              least one dimension in common with the region. Other variables
              should be written in a separate call to ``to_zarr()``.
            - Dimensions cannot be included in both ``region`` and
              ``append_dim`` at the same time. To create empty arrays to fill
              in with ``region``, use a separate call to ``to_zarr()`` with
              ``compute=False``. See "Appending to existing Zarr stores" in
              the reference documentation for full details.
        compression : int, optional
            Only used for ".zarr.zip" files.  By default zarr uses blosc
            compression for chunks, so adding another layer of compression here
            is typically redundant.

        References
        ----------
        https://zarr.readthedocs.io/

        Notes
        -----
        Zarr chunking behavior:
            If chunks are found in the encoding argument or attribute
            corresponding to any DataArray, those chunks are used.
            If a DataArray is a dask array, it is written with those chunks.
            If not other chunks are found, Zarr uses its own heuristics to
            choose automatic chunk sizes.
        """
        if (
            len(args) == 1
            and isinstance(args[0], str)
            and args[0].endswith(".zarr.zip")
        ):
            if {"compression", "mode"}.issuperset(kwargs.keys()):
                import zarr

                with zarr.ZipStore(args[0], **kwargs) as store:
                    self.to_zarr(store)
                return
        return super().to_zarr(*args, **kwargs)

    def iat(self, *, _names=None, _load=False, _index_name=None, **idxs):
        loaders = {}
        if _index_name is None:
            _index_name = "index"
        for k, v in idxs.items():
            loaders[k] = xr.DataArray(v, dims=[_index_name])
        if _names:
            ds = self[_names]
        else:
            ds = self
        if _load:
            ds = ds.load()
        return ds.isel(**loaders)

    def at(self, *, _names=None, _load=False, _index_name=None, **idxs):
        loaders = {}
        if _index_name is None:
            _index_name = "index"
        for k, v in idxs.items():
            loaders[k] = xr.DataArray(v, dims=[_index_name])
        if _names:
            ds = self[_names]
        else:
            ds = self
        if _load:
            ds = ds.load()
        return ds.sel(**loaders)

    def at_df(self, df):
        """
        Extract values by label on the coordinates indicated by columns of a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame or Mapping[str, array-like]
            The columns (or keys) of `df` should match the named dimensions of
            this Dataset.  The resulting extracted DataFrame will have one row
            per row of `df`, columns matching the data variables in this dataset,
            and each value is looked up by the labels.

        Returns
        -------
        pd.DataFrame
        """
        result = self.at(**df).reset_coords(drop=True).to_dataframe()
        if isinstance(df, pd.DataFrame):
            result.index = df.index
        return result

    def iat_df(self, df):
        """
        Extract values by position on the coordinates indicated by columns of a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame or Mapping[str, array-like]
            The columns (or keys) of `df` should match the named dimensions of
            this Dataset.  The resulting extracted DataFrame will have one row
            per row of `df`, columns matching the data variables in this dataset,
            and each value is looked up by the positions.

        Returns
        -------
        pd.DataFrame
        """
        result = self.iat(**df).reset_coords(drop=True).to_dataframe()
        if isinstance(df, pd.DataFrame):
            result.index = df.index
        return result

    def select_and_rename(self, name_dict=None, **names):
        """
        Select and rename variables from this Dataset

        Parameters
        ----------
        name_dict, **names: dict
            The keys or keyword arguments give the current names of the
            variables that will be selected out of this Dataset.  The values
            give the new names of the same variables in the resulting Dataset.

        Returns
        -------
        Dataset
        """
        if name_dict is None:
            name_dict = names
        else:
            name_dict.update(names)
        return self[list(name_dict.keys())].rename(name_dict)

    # def squash_index(self, indexes_dict=None, *, set_match_names=True, **indexes):
    #     if indexes_dict is None:
    #         indexes_dict = indexes
    #     else:
    #         indexes_dict.update(indexes)
    #     ds = self.reset_index(list(indexes_dict.keys()), drop=True)
    #     ds = ds.rename(**indexes_dict)
    #     if set_match_names:
    #         ds = ds.set_match_names({v: v for v in indexes_dict.values()})
    #     return ds

    def _repr_html_(self):
        html = super()._repr_html_()
        html = html.replace("xarray.Dataset", "sharrow.Dataset")
        return html

    def __repr__(self):
        r = super().__repr__()
        r = r.replace("xarray.Dataset", "sharrow.Dataset")
        return r

    @property
    def match_names(self):
        """
        Mapping[str,str]

        The keys of this mapping give the named dimensions of this Dataset,
        and the values of this mapping give either a named index or column
        in the shared data. If the value is a plain string, the target must
        be an exact match with no processing of the matching data.

        If a match_name target begins with an '@', the match is a dynamic
        match, where the particular index-position values are created based
        on data in the main or other source[s]. This allows for match columns
        that do not exist yet, including columns where the key column exists
        but is a label-based or offset-based match that needs to be processed
        into index-position values.

        """
        result = {}
        for k in self.attrs.keys():
            if k.startswith("match_names_"):
                result[k[12:]] = self.attrs.get(k)
        for k in self.indexes.keys():
            if k not in result:
                result[k] = None
        return result

    @match_names.setter
    def match_names(self, names):
        if names is None:
            existing_match_name_keys = list(self.match_names.keys())
            for k in existing_match_name_keys:
                del self.attrs[k]
            return
        if isinstance(names, str):
            dims = list(self.dims.keys())
            assert len(dims) == 1
            names = {dims[0]: names}
        for k in names.keys():
            if k not in self.dims:
                raise ValueError(f"'{k}' not in dims")
        for k, v in names.items():
            if v is not None:
                self.attrs[f"match_names_{k}"] = v
            elif f"match_names_{k}" in self.attrs:
                del self.attrs[f"match_names_{k}"]

    def set_match_names(self, names):
        """
        Create a copy of this dataset with the given match_names for flowing.

        Parameters
        ----------
        names : str or Mapping[str,str]

        Returns
        -------
        sharrow.dataset.Dataset
        """
        result = self.copy()
        result.match_names = names
        return result

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
            labels that are in the dataset are dropped and no error is raised.

        Returns
        -------
        obj : Dataset
            The dataset without the given dimensions (or any variables
            containing those dimensions)
        errors : {"raise", "ignore"}, optional
            If 'raise' (default), raises a ValueError error if
            any of the dimensions passed are not
            in the dataset. If 'ignore', any given dimensions that are in the
            dataset are dropped and no error is raised.
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
                    "Dataset does not contain the dimensions: %s" % missing_dims
                )
        return self.drop_dims([i for i in all_dims if i not in keep_dims])


    ### Pro

    @property
    def loc(self):
        """
        Attribute for location based indexing. Only supports __getitem__,
        and only when the key is a dict of the form {dim: labels}, or when
        there is only one dimension.
        """
        return _LocIndexer(self)

    @property
    def iloc(self):
        """
        Attribute for position based indexing. Only supports __getitem__,
        and only when there is only one dimension.
        """
        return _iLocIndexer(self)


    @classmethod
    def from_dataframe_fast(cls, dataframe: pd.DataFrame, sparse: bool = False) -> "Dataset":
        """Convert a pandas.DataFrame into an xarray.Dataset

        Each column will be converted into an independent variable in the
        Dataset. If the dataframe's index is a MultiIndex, it will be expanded
        into a tensor product of one-dimensional indices (filling in missing
        values with NaN). This method will produce a Dataset very similar to
        that on which the 'to_dataframe' method was called, except with
        possibly redundant dimensions (since all dataset variables will have
        the same dimensionality)

        Parameters
        ----------
        dataframe : DataFrame
            DataFrame from which to copy data and indices.
        sparse : bool, default: False
            If true, create a sparse arrays instead of dense numpy arrays. This
            can potentially save a large amount of memory if the DataFrame has
            a MultiIndex. Requires the sparse package (sparse.pydata.org).

        Returns
        -------
        New Dataset.

        See Also
        --------
        xarray.DataArray.from_series
        pandas.DataFrame.to_xarray
        """

        # this is much faster than the default xarray version when not
        # using a MultiIndex.

        if isinstance(dataframe.index, pd.MultiIndex) or sparse:
            return super().from_dataframe(dataframe, sparse)

        if not dataframe.columns.is_unique:
            raise ValueError("cannot convert DataFrame with non-unique columns")

        if isinstance(dataframe.index, pd.CategoricalIndex):
            idx = dataframe.index.remove_unused_categories()
        else:
            idx = dataframe.index

        index_name = idx.name if idx.name is not None else "index"
        dims = (index_name,)

        # Cast to a NumPy array first, in case the Series is a pandas Extension
        # array (which doesn't have a valid NumPy dtype)
        # TODO: allow users to control how this casting happens, e.g., by
        # forwarding arguments to pandas.Series.to_numpy?
        arrays = {
            k: xr.DataArray(np.asarray(v), dims=dims)
            for k, v in dataframe.items()
        }

        return cls(arrays).assign_coords({index_name: dataframe.index})

    def match_names_on(self, key):
        dims = self[key].dims
        match_names = self.match_names
        result = []
        for dim in dims:
            next_dim = match_names.get(dim, None)
            result.append(_dyno(dim, next_dim))
        return tuple(result)

    def squash_index(self, indexes_dict=None, *, set_match_names=True, **indexes):
        if indexes_dict is None:
            indexes_dict = indexes
        else:
            indexes_dict.update(indexes)
        ds = super().squash_index(
            indexes_dict,
            set_match_names=set_match_names,
        )
        if set_match_names:
            ds = ds.set_match_names({v: v for v in indexes_dict.values()})
        return ds

    def release_shared_memory(self):
        """
        Release shared memory allocated to this Dataset.
        """
        release_shared_memory(self._shared_memory_key_)

    @classmethod
    def delete_shared_memory_files(cls, key):
        delete_shared_memory_files(key)

    def to_shared_memory(self, key=None, mode="r+"):
        """
        Load this Dataset into shared memory.

        The returned Dataset object references the shared memory and is the
        "owner" of this data.  When this object is destroyed, the data backing
        it may also be freed, which can result in a segfault or other unfortunate
        condition if that memory is still accessed from elsewhere.

        Parameters
        ----------
        key : str
            An identifying key for this shared memory.  Use the same key
            in `from_shared_memory` to recreate this Dataset elsewhere.
        mode : {‘r+’, ‘r’, ‘w+’, ‘c’}, optional
            This methid returns a copy of the Dataset in shared memory.
            If memmapped, that copy can be opened in various modes.
            See numpy.memmap() for details.

        Returns
        -------
        Dataset
        """
        logger.info(f"sharrow.Dataset.to_shared_memory({key})")
        if key is None:
            import random
            key = random.randbytes(4).hex()
        self._shared_memory_key_ = key
        self._shared_memory_owned_ = False
        self._shared_memory_objs_ = []
        import pickle
        from multiprocessing.shared_memory import SharedMemory, ShareableList
        wrappers = []
        sizes = []
        names = []
        position = 0

        def emit(k, a, is_coord):
            nonlocal names, wrappers, sizes, position
            wrappers.append({
                'dims': a.dims,
                'name': a.name,
                'attrs': a.attrs,
                'dtype': a.dtype,
                'shape': a.shape,
                'coord': is_coord,
                'nbytes': a.nbytes,
                'position': position,
            })
            sizes.append(a.nbytes)
            names.append(k)
            position += a.nbytes

        for k, a in self.coords.items():
            emit(k, a, True)
        for k in self.variables:
            if k in names: continue
            a = self[k]
            emit(k, a, False)

        mem = create_shared_memory_array(key, size=position)
        if key.startswith("memmap:"):
            buffer = memoryview(mem)
        else:
            buffer = mem.buf

        # @dask.delayed
        # def read_chunk(key_, size_, pos_, arr):
        #     mem_ = open_shared_memory_array(key_, mode='r+')
        #     if key_.startswith("memmap:"):
        #         buffer_ = memoryview(mem_)
        #     else:
        #         buffer_ = mem_.buf
        #     mem_arr_ = np.ndarray(shape=arr.shape, dtype=arr.dtype, buffer=buffer_[pos_:pos_ + size_])
        #     da.store(arr, mem_arr_, lock=False, compute=True)

        tasks = []
        for w in wrappers:
            _size = w['nbytes']
            _name = w['name']
            _pos = w['position']
            a = self[_name]
            mem_arr = np.ndarray(shape=a.shape, dtype=a.dtype, buffer=buffer[_pos:_pos+_size])
            if isinstance(a, xr.DataArray) and isinstance(a.data, da.Array):
                tasks.append(da.store(a.data, mem_arr, lock=False, compute=False))
                #tasks.append(read_chunk(key, _size, _pos, a.data))
            else:
                mem_arr[:] = a[:]
        if tasks:
            dask.compute(tasks, scheduler='threads')

        if key.startswith("memmap:"):
            mem.flush()

        create_shared_list([pickle.dumps(i) for i in wrappers], key)
        return type(self).from_shared_memory(key, own_data=True, mode=mode)

    @property
    def shared_memory_key(self):
        try:
            return self._shared_memory_key_
        except AttributeError:
            raise ValueError("this dataset is not in shared memory")

    @classmethod
    def from_shared_memory(cls, key, own_data=False, mode="r+"):
        """
        Connect to an existing Dataset in shared memory.

        Parameters
        ----------
        key : str
            An identifying key for this shared memory.  Use the same key
            in `from_shared_memory` to recreate this Dataset elsewhere.
        own_data : bool, default False
            The returned Dataset object references the shared memory but is
            not the "owner" of this data unless this flag is set.

        Returns
        -------
        Dataset
        """
        import pickle
        from xarray import DataArray
        _shared_memory_objs_ = []

        shr_list = read_shared_list(key)
        try:
            _shared_memory_objs_.append(shr_list.shm)
        except AttributeError:
            # for memmap, list is loaded from pickle, not shared ram
            pass
        mem = open_shared_memory_array(key, mode=mode)
        _shared_memory_objs_.append(mem)
        if key.startswith("memmap:"):
            buffer = memoryview(mem)
        else:
            buffer = mem.buf

        content = {}

        for w in shr_list:
            t = pickle.loads(w)
            shape = t.pop('shape')
            dtype = t.pop('dtype')
            name = t.pop('name')
            coord = t.pop('coord', False)
            position = t.pop('position')
            nbytes = t.pop('nbytes')
            mem_arr = np.ndarray(shape, dtype=dtype, buffer=buffer[position:position+nbytes])
            content[name] = DataArray(mem_arr, **t)

        self = cls(content)
        self._shared_memory_key_ = key
        self._shared_memory_owned_ = own_data
        self._shared_memory_objs_ = _shared_memory_objs_

        return self

    @property
    def shared_memory_size(self):
        try:
            return sum(i.size for i in self._shared_memory_objs_)
        except AttributeError:
            raise ValueError("this dataset is not in shared memory")

    @property
    def is_shared_memory(self):
        try:
            return sum(i.size for i in self._shared_memory_objs_) > 0
        except AttributeError:
            return False

    @classmethod
    def preload_shared_memory_size(cls, key):
        """
        Compute the size in bytes of a shared Dataset without actually loading it.

        Parameters
        ----------
        key : str
            The identifying key for this shared memory.

        Returns
        -------
        int
        """
        import pickle
        from multiprocessing.shared_memory import SharedMemory, ShareableList
        _shared_memory_key_ = key
        memsize = 0
        try:
            n = get_shared_list_nbytes(key)
        except FileNotFoundError:
            pass
        else:
            memsize += n
        try:
            mem = open_shared_memory_array(key, mode="r")
        except FileNotFoundError:
            pass
        else:
            memsize += mem.size
        return memsize

    @classmethod
    def from_omx_3d(
            cls,
            omx,
            index_names=("otaz", "dtaz", "time_period"),
            indexes=None,
            *,
            time_periods=None,
            time_period_sep="__",
            max_float_precision=32,
    ):
        if not isinstance(omx, (list, tuple)):
            omx = [omx]

        # handle both larch.OMX and openmatrix.open_file versions
        if "larch" in type(omx[0]).__module__:
            omx_shape = omx[0].shape
            omx_lookup = omx[0].lookup
        else:
            omx_shape = omx[0].shape()
            omx_lookup = omx[0].root["lookup"]
        omx_data = []
        omx_data_map = {}
        for n, i in enumerate(omx):
            if "larch" in type(i).__module__:
                omx_data.append(i.data)
                for k in i.data._v_children:
                    omx_data_map[k] = n
            else:
                omx_data.append(i.root["data"])
                for k in i.root["data"]._v_children:
                    omx_data_map[k] = n

        import dask.array

        data_names = list(omx_data_map.keys())
        n1, n2 = omx_shape
        if indexes is None:
            # default reads mapping if only one lookup is included, otherwise one-based
            if len(omx_lookup._v_children) == 1:
                ranger = None
                indexes = list(omx_lookup._v_children)[0]
            else:
                ranger = one_based
        elif indexes == "one-based":
            ranger = one_based
        elif indexes == "zero-based":
            ranger = zero_based
        elif indexes in set(omx_lookup._v_children):
            ranger = None
        else:
            raise NotImplementedError("only one-based, zero-based, and named indexes are implemented")
        if ranger is not None:
            r1 = ranger(n1)
            r2 = ranger(n2)
        else:
            r1 = r2 = pd.Index(omx_lookup[indexes])

        if time_periods is None:
            raise ValueError("must give time periods explicitly")

        time_periods_map = {t: n for n, t in enumerate(time_periods)}

        pending_3d = {}
        content = {}

        for k in data_names:
            if time_period_sep in k:
                base_k, time_k = k.split(time_period_sep, 1)
                if base_k not in pending_3d:
                    pending_3d[base_k] = [None] * len(time_periods)
                pending_3d[base_k][time_periods_map[time_k]] = dask.array.from_array(omx_data[omx_data_map[k]][k])
            else:
                content[k] = xr.DataArray(
                    dask.array.from_array(omx_data[omx_data_map[k]][k]),
                    dims=index_names[:2],
                    coords={
                        index_names[0]: r1,
                        index_names[1]: r2,
                    },
                )
        for base_k, darrs in pending_3d.items():
            # find a prototype array
            prototype = None
            for i in darrs:
                prototype = i
                if prototype is not None:
                    break
            if prototype is None:
                raise ValueError("no prototype")
            darrs_ = [(i if i is not None else dask.array.zeros_like(prototype)) for i in darrs]
            content[base_k] = xr.DataArray(
                dask.array.stack(darrs_, axis=-1),
                dims=index_names,
                coords={
                    index_names[0]: r1,
                    index_names[1]: r2,
                    index_names[2]: time_periods,
                },
            )
        for i in content:
            if np.issubdtype(content[i].dtype, np.floating):
                if content[i].dtype.itemsize > max_float_precision/8:
                    content[i] = content[i].astype(f"float{max_float_precision}")
        return cls(content)

    def max_float_precision(self, p=32):
        """
        Set the maximum precision for floating point values.

        This modifies the Dataset in-place.

        Parameters
        ----------
        p : {64, 32, 16}
            The max precision to set.

        Returns
        -------
        self
        """
        for i in self:
            if np.issubdtype(self[i].dtype, np.floating):
                if self[i].dtype.itemsize > p/8:
                    self[i] = self[i].astype(f"float{p}")
        return self

    @property
    def digital_encodings(self):
        """
        dict: All digital_encoding attributes from Dataset variables.
        """
        result = {}
        for k in self.variables:
            try:
                k_attrs = self._variables[k].attrs
            except:
                k_attrs = self[k].attrs
            if 'digital_encoding' in k_attrs:
                result[k] = k_attrs['digital_encoding']
        return result

    def set_digital_encoding(self, name, *args, **kwargs):
        logger.info(f"set_digital_encoding({name})")
        from .digital_encoding import array_encode
        result = self.copy()
        result[name] = array_encode(self[name], *args, **kwargs)
        return result

    def interchange_dims(self, dim1, dim2):
        p21 = "PLACEHOLD21"
        p12 = "PLACEHOLD12"
        s1 = {dim1: p12, dim2: p21}
        s2 = {p12: dim2, p21: dim1}
        rv = {}
        vr = {}
        if dim1 in self.variables:
            rv[dim1] = p12
            vr[p12] = dim2
        if dim2 in self.variables:
            rv[dim2] = p21
            vr[p21] = dim1
        return self.rename_dims(s1).rename_vars(rv).rename_dims(s2).rename_vars(vr)

    def rename_dims_and_coords(self, dims_dict=None, **dims_kwargs):
        from xarray.core.utils import either_dict_or_kwargs
        dims_dict = either_dict_or_kwargs(dims_dict, dims_kwargs, "rename_dims_and_coords")
        out = self.rename_dims(dims_dict)
        coords_dict = {}
        for k in out.coords:
            if k in dims_dict:
                coords_dict[k] = dims_dict[k]
        return out.rename_vars(coords_dict)

    def rename_or_ignore(self, dims_dict=None, **dims_kwargs):
        from xarray.core.utils import either_dict_or_kwargs
        dims_dict = either_dict_or_kwargs(dims_dict, dims_kwargs, "rename_dims_and_coords")
        dims_dict = {k: v for (k, v) in dims_dict.items() if (k in self.dims or k in self._variables)}
        return self.rename(dims_dict)

    def explode(self):
        dims = self.dims
        out = self.rename_dims({f"{k}": f"{k}_" for k in dims})
        out = out.reset_coords()
        out = out.broadcast_like(out)
        return out

    @classmethod
    def from_named_objects(cls, *args):
        objs = {}
        for n, a in enumerate(args):
            try:
                name = a.name
            except AttributeError:
                raise ValueError(f"argument {n} has no name")
            if name is None:
                raise ValueError(f"the name for argument {n} is None")
            objs[name] = a
        return cls(objs)

    def ensure_integer(self, names, bitwidth=32, inplace=False):
        """
        Convert dataset variables to integers, if they are not already integers.
        Parameters
        ----------
        names : Iterable[str]
            Variable names in this dataset to convert.
        bitwidth : int, default 32
            Bit width of integers that are created when a conversion is made.
            Note that variables that are already integer are not modified,
            even if their bit width differs from this.
        inplace : bool, default False
            Whether to make the conversion in-place on this Dataset, or
            return a copy.

        Returns
        -------
        Dataset
        """
        if inplace:
            result = self
        else:
            result = self.copy()
        for name in names:
            if name not in result:
                continue
            if not np.issubdtype(result[name].dtype, np.integer):
                result[name] = result[name].astype(f"int{bitwidth}")
        if not inplace:
            return result


def filter_name_tokens(expr, matchable_names=None):
    name_tokens = extract_all_name_tokens(expr)
    name_tokens -= {'_args', '_inputs', '_outputs', 'np'}
    name_tokens -= well_known_names
    if matchable_names:
        name_tokens &= matchable_names
    return name_tokens


def _dyno(k, v):
    if isinstance(v, str) and v[0]=="@":
        return f"__dynamic_{k}{v}"
    elif v is None:
        return f"__dynamic_{k}"
    else:
        return v


def _flip_flop_def(v):
    if "# sharrow:" in v:
        return v.split("# sharrow:", 1)[1].strip()
    else:
        return v

