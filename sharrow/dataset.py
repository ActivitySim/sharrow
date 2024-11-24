from __future__ import annotations

import ast
import base64
import hashlib
import logging
import re
import time
from collections.abc import Hashable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr
from xarray import DataArray, Dataset

from .accessors import register_dataset_method
from .aster import extract_all_name_tokens
from .categorical import _Categorical  # noqa
from .shared_memory import si_units
from .table import Table

if TYPE_CHECKING:
    import openmatrix


logger = logging.getLogger("sharrow")

well_known_names = {
    "nb",
    "np",
    "pd",
    "xr",
    "pa",
    "log",
    "exp",
    "log1p",
    "expm1",
    "max",
    "min",
    "piece",
    "hard_sigmoid",
    "transpose_leading",
    "clip",
}


def one_based(n):
    return pd.RangeIndex(1, n + 1)


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
    cleaned = re.sub(r"\W|^(?=\d)", "_", s)
    if cleaned != s or len(cleaned) > 120:
        # digest size 15 creates a 24 character base32 string
        h = base64.b32encode(
            hashlib.blake2b(s.encode(), digest_size=15).digest()
        ).decode()
        cleaned = f"{cleaned[:90]}_{h}"
    return cleaned


def construct(source):
    """
    Create Datasets from various similar objects.

    Parameters
    ----------
    source : pandas.DataFrame, pyarrow.Table, xarray.Dataset, or Sequence[str]
        The source from which to create a Dataset.  DataFrame and Table objects
        are converted to Datasets that have one dimension (the rows) and
        separate variables for each of the columns.  A list of strings
        creates a dataset with those named empty variables.

    Returns
    -------
    Dataset
    """
    if isinstance(source, pd.DataFrame):
        source = dataset_from_dataframe_fast(source)  # xarray default can be slow
    elif isinstance(source, (Table, pa.Table)):
        source = from_table(source)
    elif isinstance(source, xr.Dataset):
        pass  # don't do the superclass things
    elif isinstance(source, Sequence) and all(isinstance(i, str) for i in source):
        source = from_table(pa.table({i: [] for i in source}))
    else:
        source = xr.Dataset(source)
    return source


def dataset_from_dataframe_fast(
    dataframe: pd.DataFrame,
    sparse: bool = False,
    preserve_cat: bool = True,
) -> Dataset:
    """Convert a pandas.DataFrame into an xarray.Dataset.

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
    preserve_cat : bool, default True
        If true, preserve encoding of categorical columns.  Xarray lacks an
        official implementation of a categorical datatype, so sharrow's
        dictionary-based digital encoding is applied instead. Note that in
        native xarray usage, the resulting variable will look like integer
        values instead of the category values.  The `dataset.cat` accessor
        can be used to interact with the categorical data.

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
        return Dataset.from_dataframe(dataframe, sparse)

    if not dataframe.columns.is_unique:
        # if the dataframe has non-unique column names, but all the duplicate
        # names contain the same data, we can recover safely by dropping the
        # duplicates, otherwise throw an error.
        cannot_fix = False
        dupe_columns = dataframe.columns.duplicated()
        dupe_column_names = dataframe.columns[dupe_columns]
        for j in dupe_column_names:
            subframe = dataframe[j]
            ref_col = subframe.iloc[:, 0]
            for k in range(1, len(subframe.columns)):
                if not ref_col.equals(subframe.iloc[:, k]):
                    cannot_fix = True
                    break
                if cannot_fix:
                    break
        dupe_column_names = [f"- {i}" for i in dupe_column_names]
        logger.error(
            "DataFrame has non-unique columns\n" + "\n".join(dupe_column_names)
        )
        if cannot_fix:
            raise ValueError("cannot convert DataFrame with non-unique columns")
        else:
            dataframe = dataframe.loc[:, ~dupe_columns]

    if isinstance(dataframe.index, pd.CategoricalIndex):
        idx = dataframe.index.remove_unused_categories()
    else:
        idx = dataframe.index

    index_name = idx.name if idx.name is not None else "index"
    # Cast to a NumPy array first, in case the Series is a pandas Extension
    # array (which doesn't have a valid NumPy dtype)
    arrays = {}
    for name in dataframe.columns:
        if name != index_name:
            if dataframe[name].dtype == "category" and preserve_cat:
                cat = dataframe[name].cat
                categories = np.asarray(cat.categories)
                if categories.dtype.kind == "O":
                    categories = categories.astype(str)
                arrays[name] = (
                    [index_name],
                    np.asarray(cat.codes),
                    {
                        "digital_encoding": {
                            "dictionary": categories,
                            "ordered": cat.ordered,
                        }
                    },
                )
            else:
                arrays[name] = ([index_name], np.asarray(dataframe[name].values))
    return Dataset(arrays, coords={index_name: (index_name, dataframe.index.values)})


def from_table(
    tbl,
    index_name="index",
    index=None,
):
    """
    Convert a pyarrow.Table into an xarray.Dataset.

    Parameters
    ----------
    tbl : Table
        Table from which to use data and indices.
    index_name : str, default 'index'
        This name will be given to the default dimension index, if
        none is given.  Ignored if `index` is given explicitly and
        it already has a name.
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
    arrays = []
    metadata = {}
    for n in range(len(tbl.column_names)):
        c = tbl.column(n)
        if isinstance(c.type, pa.DictionaryType):
            cc = c.combine_chunks()
            arrays.append((tbl.column_names[n], np.asarray(cc.indices)))
            metadata[tbl.column_names[n]] = {
                "digital_encoding": {
                    "dictionary": cc.dictionary,
                    "ordered": cc.type.ordered,
                }
            }
        else:
            arrays.append((tbl.column_names[n], np.asarray(c)))
    result = xr.Dataset()
    if isinstance(index, pd.MultiIndex):
        dims = tuple(
            name if name is not None else f"level_{n}"
            for n, name in enumerate(index.names)
        )
        for dim, lev in zip(dims, index.levels):
            result[dim] = (dim, lev)
    else:
        try:
            if index.name is not None:
                index_name = index.name
        except AttributeError:
            pass
        dims = (index_name,)
        result[index_name] = (dims, index)

    result._set_numpy_data_from_dataframe(index, arrays, dims)
    for k, v in metadata.items():
        result[k].attrs.update(v)
    return result


def from_omx(
    omx: openmatrix.File,
    index_names=("otaz", "dtaz"),
    indexes="one-based",
    renames=None,
):
    """
    Create a Dataset from an OMX file.

    Parameters
    ----------
    omx : openmatrix.File or larch.OMX
        An OMX-format file, opened for reading.
    index_names : tuple, default ("otaz", "dtaz")
        Should be a tuple of length 3, giving the names of the three
        dimensions.  The first two names are the native dimensions from
        the open matrix file, the last is the name of the implicit
        dimension that is created by parsing array names.
    indexes : str or tuple[str], optional
        The name of a 'lookup' in the OMX file, which will be used to
        populate the coordinates for the two native dimensions.  Or,
        specify "one-based" or "zero-based" to assume sequential and
        consecutive numbering starting with 1 or 0 respectively. For
        non-square OMX data, this must be given as a tuple, relating
        indexes as above for each dimension of `index_names`.
    renames : Mapping or Collection, optional
        Limit the import only to these data elements.  If given as a
        mapping, the keys will be the names of variables in the resulting
        dataset, and the values give the names of data matrix tables in the
        OMX file.  If given as a list or other non-mapping collection,
        elements are not renamed but only elements in the collection are
        included.

    Returns
    -------
    Dataset
    """
    # handle both larch.OMX and openmatrix.open_file versions
    if "lar" in type(omx).__module__:
        omx_data = omx.data
        omx_lookup = omx.lookup
        omx_shape = omx.shape
    else:
        omx_data = omx.root["data"]
        omx_lookup = omx.root["lookup"]
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
    elif isinstance(indexes, str):
        if indexes in omx_lookup:
            if omx_shape[0] != omx_shape[1]:
                raise ValueError("singleton arbitrary coordinates on non-square arrays")
            ixs = np.asarray(omx_lookup[indexes])
            indexes = {
                index_names[0]: ixs,
                index_names[1]: ixs,
            }
        else:
            raise KeyError(f"{indexes} not found in OMX lookups")
    elif isinstance(indexes, tuple):
        indexes_ = {}
        for n, (name, i) in enumerate(zip(index_names, indexes)):
            if i == "one-based":
                indexes_[name] = one_based(omx_shape[n])
            elif i == "zero-based":
                indexes_[name] = zero_based(omx_shape[n])
            elif isinstance(i, str):
                if i in omx_lookup:
                    indexes_[name] = np.asarray(omx_lookup[i])
                else:
                    raise KeyError(f"{i} not found in OMX lookups")
        indexes = indexes_
    if indexes is not None:
        d["coords"] = {
            index_name: {"dims": index_name, "data": index}
            for index_name, index in indexes.items()
        }
    return xr.Dataset.from_dict(d)


def _should_ignore(ignore, x):
    if ignore is not None:
        for i in ignore:
            if re.match(i, x):
                return True
    return False


def from_omx_3d(
    omx: openmatrix.File | str | Iterable[openmatrix.File | str],
    index_names=("otaz", "dtaz", "time_period"),
    indexes=None,
    *,
    time_periods=None,
    time_period_sep="__",
    max_float_precision=32,
    ignore=None,
):
    """
    Create a Dataset from an OMX file with an implicit third dimension.

    Parameters
    ----------
    omx : openmatrix.File or larch.OMX
        An OMX-format file, opened for reading.
    index_names : tuple, default ("otaz", "dtaz", "time_period")
        Should be a tuple of length 3, giving the names of the three
        dimensions.  The first two names are the native dimensions from
        the open matrix file, the last is the name of the implicit
        dimension that is created by parsing array names.
    indexes : str, optional
        The name of a 'lookup' in the OMX file, which will be used to
        populate the coordinates for the two native dimensions.  Or,
        specify "one-based" or "zero-based" to assume sequential and
        consecutive numbering starting with 1 or 0 respectively.
    time_periods : list-like, required keyword argument
        A list of index values from which the third dimension is constructed
        for all variables with a third dimension.
    time_period_sep : str, default "__" (double underscore)
        The presence of this separator within the name of any table in the
        OMX file indicates that table is to be considered a page in a
        three dimensional variable.  The portion of the name preceding the
        first instance of this separator is the name of the resulting
        variable, and the portion of the name after the first instance of
        this separator is the label of the position for this page, which
        should appear in `time_periods`.
    max_float_precision : int, default 32
        When loading, reduce all floats in the OMX file to this level of
        precision, generally to save memory if they were stored as double
        precision but that level of detail is unneeded in the present
        application.
    ignore : list-like, optional
        A list of regular expressions that will be used to filter out
        variables from the dataset.  If any of the regular expressions
        match the name of a variable, that variable will not be included
        in the loaded dataset. This is useful for excluding variables that
        are not needed in the current application.

    Returns
    -------
    Dataset
    """
    if not isinstance(omx, (list, tuple)):
        omx = [omx]

    use_file_handles = []
    opened_file_handles = []
    for filename in omx:
        if isinstance(filename, str):
            import openmatrix

            h = openmatrix.open_file(filename)
            opened_file_handles.append(h)
            use_file_handles.append(h)
        else:
            use_file_handles.append(filename)
    omx = use_file_handles

    try:
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
        if ignore is not None:
            if isinstance(ignore, str):
                ignore = [ignore]
            data_names = [i for i in data_names if not _should_ignore(ignore, i)]
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
            raise NotImplementedError(
                "only one-based, zero-based, and named indexes are implemented"
            )
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
                pending_3d[base_k][time_periods_map[time_k]] = dask.array.from_array(
                    omx_data[omx_data_map[k]][k]
                )
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
            darrs_ = [
                (i if i is not None else dask.array.zeros_like(prototype))
                for i in darrs
            ]
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
                if content[i].dtype.itemsize > max_float_precision / 8:
                    content[i] = content[i].astype(f"float{max_float_precision}")
        return xr.Dataset(content)
    finally:
        for h in opened_file_handles:
            h.close()


def reload_from_omx_3d(
    dataset: xr.Dataset,
    omx: Iterable[str],
    *,
    time_period_sep="__",
    ignore=None,
) -> None:
    """
    Reload the content of a dataset from OMX files.

    This loads the data from the OMX files into the dataset, replacing
    the existing data in the dataset.  The dataset must have been created
    by `from_omx_3d` or a similar function. Note that `from_omx_3d` will
    create a dataset backed by `dask.array` objects; this function allows for
    loading the data without going through dask, which may have poor performance
    on some platforms.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to reload into.
    omx : Iterable[str]
        The list of OMX file names to load from.
    time_period_sep : str, default "__"
        The separator used to identify time periods in the dataset.
    ignore : list-like, optional
        A list of regular expressions that will be used to filter out
        variables from the dataset.  If any of the regular expressions
        match the name of a variable, that variable will not be included
        in the load process. This is useful for excluding variables that
        are not found in the target dataset.
    """
    if isinstance(ignore, str):
        ignore = [ignore]

    use_file_handles = []
    opened_file_handles = []
    for filename in omx:
        if isinstance(filename, (str, Path)):
            import openmatrix

            h = openmatrix.open_file(filename)
            opened_file_handles.append(h)
            use_file_handles.append(h)
        else:
            use_file_handles.append(filename)
    omx = use_file_handles

    bytes_loaded = 0

    try:
        t0 = time.time()
        for filename, f in zip(omx, use_file_handles):
            if isinstance(filename, str):
                logger.info(f"loading into dataset from {filename}")
            for data_name in f.root.data._v_children:
                if _should_ignore(ignore, data_name):
                    logger.info(f"ignoring {data_name}")
                    continue
                t1 = time.time()
                filters = f.root.data[data_name].filters
                filter_note = f"{filters.complib}/{filters.complevel}"

                if time_period_sep in data_name:
                    data_name_x, data_name_t = data_name.split(time_period_sep, 1)
                    if data_name_x not in dataset:
                        logger.info(
                            f"skipping {data_name} because {data_name_x} not in dataset"
                        )
                        continue
                    if len(dataset[data_name_x].dims) != 3:
                        raise ValueError(
                            f"dataset variable {data_name_x} has "
                            f"{len(dataset[data_name_x].dims)} dimensions, expected 3"
                        )
                    raw = dataset[data_name_x].sel(time_period=data_name_t).data
                    raw[:, :] = f.root.data[data_name][:, :]
                else:
                    if len(dataset[data_name].dims) != 2:
                        raise ValueError(
                            f"dataset variable {data_name} has "
                            f"{len(dataset[data_name].dims)} dimensions, expected 2"
                        )
                    raw = dataset[data_name].data
                    raw[:, :] = f.root.data[data_name][:, :]
                bytes_loaded += raw.nbytes
                logger.info(
                    f"loaded {data_name} ({filter_note}) to dataset "
                    f"in {time.time() - t1:.2f}s, {si_units(bytes_loaded)}"
                )
        logger.info(f"loading to dataset complete in {time.time() - t0:.2f}s")
    finally:
        for h in opened_file_handles:
            h.close()


def from_amx(
    amx,
    index_names=("otaz", "dtaz"),
    indexes="one-based",
    renames=None,
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
    return xr.Dataset.from_dict(d)


def from_zarr(store, *args, **kwargs):
    """
    Load and decode a dataset from a Zarr store.

    The `store` object should be a valid store for a Zarr group. `store`
    variables must contain dimension metadata encoded in the
    `_ARRAY_DIMENSIONS` attribute.

    Parameters
    ----------
    store : MutableMapping or str
        A MutableMapping where a Zarr Group has been stored or a path to a
        directory in file system where a Zarr DirectoryStore has been stored.
    synchronizer : object, optional
        Array synchronizer provided to zarr
    group : str, optional
        Group path. (a.k.a. `path` in zarr terminology.)
    chunks : int or dict or tuple or {None, 'auto'}, optional
        Chunk sizes along each dimension, e.g., ``5`` or
        ``{'x': 5, 'y': 5}``. If `chunks='auto'`, dask chunks are created
        based on the variable's zarr chunks. If `chunks=None`, zarr array
        data will lazily convert to numpy arrays upon access. This accepts
        all the chunk specifications as Dask does.
    overwrite_encoded_chunks : bool, optional
        Whether to drop the zarr chunks encoded for each variable when a
        dataset is loaded with specified chunk sizes (default: False)
    decode_cf : bool, optional
        Whether to decode these variables, assuming they were saved according
        to CF conventions.
    mask_and_scale : bool, optional
        If True, replace array values equal to `_FillValue` with NA and scale
        values according to the formula `original_values * scale_factor +
        add_offset`, where `_FillValue`, `scale_factor` and `add_offset` are
        taken from variable attributes (if they exist).  If the `_FillValue` or
        `missing_value` attribute contains multiple values a warning will be
        issued and all array values matching one of the multiple values will
        be replaced by NA.
    decode_times : bool, optional
        If True, decode times encoded in the standard NetCDF datetime format
        into datetime objects. Otherwise, leave them encoded as numbers.
    concat_characters : bool, optional
        If True, concatenate along the last dimension of character arrays to
        form string arrays. Dimensions will only be concatenated over (and
        removed) if they have no corresponding variable and if they are only
        used as the last dimension of character arrays.
    decode_coords : bool, optional
        If True, decode the 'coordinates' attribute to identify coordinates in
        the resulting dataset.
    drop_variables : str or iterable, optional
        A variable or list of variables to exclude from being parsed from the
        dataset. This may be useful to drop variables with problems or
        inconsistent values.
    consolidated : bool, optional
        Whether to open the store using zarr's consolidated metadata
        capability. Only works for stores that have already been consolidated.
        By default (`consolidate=None`), attempts to read consolidated metadata,
        falling back to read non-consolidated metadata if that fails.
    chunk_store : MutableMapping, optional
        A separate Zarr store only for chunk data.
    storage_options : dict, optional
        Any additional parameters for the storage backend (ignored for local
        paths).
    decode_timedelta : bool, optional
        If True, decode variables and coordinates with time units in
        {'days', 'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds'}
        into timedelta objects. If False, leave them encoded as numbers.
        If None (default), assume the same value of decode_time.
    use_cftime : bool, optional
        Only relevant if encoded dates come from a standard calendar
        (e.g. "gregorian", "proleptic_gregorian", "standard", or not
        specified).  If None (default), attempt to decode times to
        ``np.datetime64[ns]`` objects; if this is not possible, decode times to
        ``cftime.datetime`` objects. If True, always decode times to
        ``cftime.datetime`` objects, regardless of whether or not they can be
        represented using ``np.datetime64[ns]`` objects.  If False, always
        decode times to ``np.datetime64[ns]`` objects; if this is not possible
        raise an error.

    Returns
    -------
    dataset : Dataset
        The newly created dataset.

    References
    ----------
    http://zarr.readthedocs.io/
    """
    return xr.open_zarr(store, *args, **kwargs)


def from_zarr_with_attr(*args, **kwargs):
    obj = from_zarr(*args, **kwargs)
    for k in obj:
        attrs = {}
        for aname, avalue in obj[k].attrs.items():
            attrs[aname] = _from_evalable_string(avalue)
        obj[k] = obj[k].assign_attrs(attrs)
    attrs = {}
    for aname, avalue in obj.attrs.items():
        attrs[aname] = _from_evalable_string(avalue)
    obj = obj.assign_attrs(attrs)
    return obj


def coerce_to_range_index(idx):
    if isinstance(idx, pd.RangeIndex):
        return idx
    if isinstance(idx, (pd.Int64Index, pd.Float64Index, pd.UInt64Index)):
        if idx.is_monotonic_increasing and idx[-1] - idx[0] == idx.size - 1:
            return pd.RangeIndex(idx[0], idx[0] + idx.size)
    return idx


def is_dict_like(value: Any) -> bool:
    return hasattr(value, "keys") and hasattr(value, "__getitem__")


@xr.register_dataset_accessor("single_dim")
class _SingleDim:
    """Convenience accessor for single-dimension datasets."""

    __slots__ = ("dataset", "dim_name")

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        if len(self.dataset.dims) != 1:
            raise ValueError("single_dim implies a single dimension dataset")
        self.dim_name = self.dataset.dims.__iter__().__next__()

    @property
    def coords(self):
        return self.dataset.coords[self.dim_name]

    @property
    def index(self):
        return self.dataset.indexes[self.dim_name]

    @property
    def size(self):
        return self.dataset.dims[self.dim_name]

    def _to_pydict(self):
        columns = [k for k in self.dataset.variables if k != self.dim_name]
        data = []
        for k in columns:
            a = self.dataset._variables[k]
            if (
                "digital_encoding" in a.attrs
                and "dictionary" in a.attrs["digital_encoding"]
            ):
                de = a.attrs["digital_encoding"]
                data.append(
                    pd.Categorical.from_codes(
                        a.values,
                        de["dictionary"],
                        de.get("ordered"),
                    )
                )
            else:
                data.append(a.values)
        return dict(zip(columns, data))

    def to_pyarrow(self) -> pa.Table:
        columns = [k for k in self.dataset.variables if k != self.dim_name]
        data = []
        for k in columns:
            a = self.dataset._variables[k]
            if (
                "digital_encoding" in a.attrs
                and "dictionary" in a.attrs["digital_encoding"]
            ):
                de = a.attrs["digital_encoding"]
                data.append(
                    pa.DictionaryArray.from_arrays(
                        a.values,
                        de["dictionary"],
                        ordered=de.get("ordered", False),
                    )
                )
            else:
                data.append(pa.array(a.values))
        content = dict(zip(columns, data))
        content[self.dim_name] = self.index
        return pa.Table.from_pydict(content)

    def to_parquet(self, filename):
        import pyarrow.parquet as pq

        t = self.to_pyarrow()
        pq.write_table(t, filename)

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert this dataset into a pandas DataFrame.

        The resulting DataFrame is always a copy of the data in the dataset.

        Returns
        -------
        pandas.DataFrame
        """
        return pd.DataFrame(self._to_pydict(), index=self.index, copy=True)

    def eval(
        self,
        expr: str,
        parser: str = "pandas",
        engine: str | None = None,
        local_dict=None,
        global_dict=None,
    ):
        """
        Evaluate a Python expression as a string using various backends.

        Parameters
        ----------
        expr : str
            The expression to evaluate. This string cannot contain any Python
            `statements
            <https://docs.python.org/3/reference/simple_stmts.html#simple-statements>`__,
            only Python `expressions
            <https://docs.python.org/3/reference/simple_stmts.html#expression-statements>`__.
        parser : {'pandas', 'python'}, default 'pandas'
            The parser to use to construct the syntax tree from the expression. The
            default of ``'pandas'`` parses code slightly different than standard
            Python. Alternatively, you can parse an expression using the
            ``'python'`` parser to retain strict Python semantics.  See the
            :ref:`enhancing performance <enhancingperf.eval>` documentation for
            more details.
        engine : {'python', 'numexpr'}, default 'numexpr'
            The engine used to evaluate the expression. Supported engines are
            - None : tries to use ``numexpr``, falls back to ``python``
            - ``'numexpr'`` : This default engine evaluates pandas objects using
              numexpr for large speed ups in complex expressions with large frames.
            - ``'python'`` : Performs operations as if you had ``eval``'d in top
              level python. This engine is generally not that useful.
        local_dict : dict or None, optional
            A dictionary of local variables, taken from locals() by default.
        global_dict : dict or None, optional
            A dictionary of global variables, taken from globals() by default.

        Returns
        -------
        DataArray or numeric scalar
        """
        result = pd.eval(
            expr,
            parser=parser,
            engine=engine,
            local_dict=local_dict,
            global_dict=global_dict,
            resolvers=[self.dataset],
        )
        if result.size == self.size:
            return DataArray(np.asarray(result), coords=self.dataset.coords)
        else:
            return result


@xr.register_dataarray_accessor("single_dim")
class _SingleDimArray:
    """Convenience accessor for single-dimension datasets."""

    __slots__ = ("dataarray", "dim_name")

    def __init__(self, dataarray: DataArray):
        self.dataarray = dataarray
        if len(self.dataarray.dims) != 1:
            raise ValueError("single_dim implies a single dimension dataset")
        self.dim_name = self.dataarray.dims[0]

    @property
    def coords(self):
        return self.dataarray.coords[self.dim_name]

    @property
    def index(self):
        return self.dataarray.indexes[self.dim_name]

    def rename(self, name: str) -> DataArray:
        """Rename the single dimension."""
        if self.dim_name == name:
            return self.dataarray
        return self.dataarray.rename({self.dim_name: name})

    def to_pandas(self) -> pd.Series:
        """
        Convert this array into a pandas Series.

        If this array is categorical (i.e. with a simple dictionary-based
        digital encoding) then the result will be a Series with categorical dtype.

        The DataArray's `name` attribute is preserved in the result.
        """
        if self.dataarray.cat.is_categorical():
            return pd.Series(
                pd.Categorical.from_codes(
                    self.dataarray,
                    self.dataarray.cat.categories,
                    self.dataarray.cat.ordered,
                ),
                index=self.index,
                name=self.dataarray.name,
            )
        else:
            result = self.dataarray.to_pandas()
            if self.dataarray.name:
                result = result.rename(self.dataarray.name)
            return result

    def to_pyarrow(self):
        if self.dataarray.cat.is_categorical():
            return pa.DictionaryArray.from_arrays(
                self.dataarray.data, self.dataarray.cat.categories
            )
        else:
            return pa.array(self.dataarray.data)


@xr.register_dataset_accessor("iloc")
class _iLocIndexer:
    """
    Purely integer-location based indexing for selection by position on 1-d Datasets.

    In many ways, a dataset with a single dimensions is like a pandas DataFrame,
    with the one dimension giving the rows, and the variables as columns. This
    analogy eventually breaks down (DataFrame columns are ordered, Dataset
    variables are not) but the similarities are enough that it’s sometimes
    convenient to have iloc functionality enabled. This only works for indexing
    on the rows, but if there’s only the one dimension the complexity of isel
    is not needed.
    """

    __slots__ = ("dataset",)

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __getitem__(self, key: Mapping[Hashable, Any]) -> Dataset:
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


@xr.register_dataarray_accessor("iloc")
class _iLocArrayIndexer:
    """
    Purely integer-location based indexing for selection by position on 1-d DataArrays.

    In many ways, a dataset with a single dimensions is like a pandas DataFrame,
    with the one dimension giving the rows, and the variables as columns. This
    analogy eventually breaks down (DataFrame columns are ordered, Dataset
    variables are not) but the similarities are enough that it’s sometimes
    convenient to have iloc functionality enabled. This only works for indexing
    on the rows, but if there’s only the one dimension the complexity of isel
    is not needed.
    """

    __slots__ = ("dataarray",)

    def __init__(self, dataarray: DataArray):
        self.dataarray = dataarray

    def __getitem__(self, key: Mapping[Hashable, Any]) -> DataArray:
        if not is_dict_like(key):
            if len(self.dataarray.dims) == 1:
                dim_name = self.dataarray.dims.__iter__().__next__()
                key = {dim_name: key}
            else:
                raise TypeError(
                    "can only lookup dictionaries from DataArray.iloc, "
                    "unless there is only one dimension"
                )
        return self.dataarray.isel(key)


xr.Dataset.rename_dims_and_coords = xr.Dataset.rename


@register_dataset_method
def rename_or_ignore(self, dims_dict=None, **dims_kwargs):
    from xarray.core.utils import either_dict_or_kwargs

    dims_dict = either_dict_or_kwargs(dims_dict, dims_kwargs, "rename_dims_and_coords")
    dims_dict = {
        k: v for (k, v) in dims_dict.items() if (k in self.dims or k in self._variables)
    }
    return self.rename(dims_dict)


@register_dataset_method
def to_zarr_zip(self, *args, **kwargs):
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
    if len(args) == 1 and isinstance(args[0], str) and args[0].endswith(".zarr.zip"):
        if {"compression", "mode"}.issuperset(kwargs.keys()):
            import zarr

            with zarr.ZipStore(args[0], **kwargs) as store:
                self.to_zarr(store)
            return
    return super().to_zarr(*args, **kwargs)


def _to_ast_literal(x):
    if isinstance(x, dict):
        return (
            "{"
            + ", ".join(
                f"{_to_ast_literal(k)}: {_to_ast_literal(v)}" for k, v in x.items()
            )
            + "}"
        )
    elif isinstance(x, list):
        return "[" + ", ".join(_to_ast_literal(i) for i in x) + "]"
    elif isinstance(x, tuple):
        return "(" + ", ".join(_to_ast_literal(i) for i in x) + ")"
    elif isinstance(x, pd.Index):
        return _to_ast_literal(x.to_list())
    elif isinstance(x, np.ndarray):
        return _to_ast_literal(list(x))
    elif isinstance(x, np.str_):
        return repr(str(x))
    else:
        return repr(x)


def _to_evalable_string(x):
    if x is None:
        return " < None > "
    elif x is True:
        return " < True > "
    elif x is False:
        return " < False > "
    else:
        return f" {_to_ast_literal(x)} "


def _from_evalable_string(x):
    if isinstance(x, str):
        # if x.startswith(" {") and x.endswith("} "):
        #     return ast.literal_eval(x[1:-1])
        if x == " < None > ":
            return None
        if x == " < True > ":
            return True
        if x == " < False > ":
            return False
        if x.startswith(" ") and x.endswith(" "):
            try:
                return ast.literal_eval(x.strip(" "))
            except Exception:
                print(x)
                raise
    else:
        return x


@register_dataset_method
def to_zarr_with_attr(self, *args, **kwargs):
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
    obj = self.copy()
    for k in self:
        attrs = {}
        for aname, avalue in self[k].attrs.items():
            attrs[aname] = _to_evalable_string(avalue)
        obj[k] = self[k].assign_attrs(attrs)
    if hasattr(self, "coords"):
        for k in self.coords:
            attrs = {}
            for aname, avalue in self.coords[k].attrs.items():
                attrs[aname] = _to_evalable_string(avalue)
            obj.coords[k] = self.coords[k].assign_attrs(attrs)
    attrs = {}
    for aname, avalue in self.attrs.items():
        attrs[aname] = _to_evalable_string(avalue)
    obj = obj.assign_attrs(attrs)
    return obj.to_zarr(*args, **kwargs)


@register_dataset_method
def to_table(self):
    """
    Convert dataset contents to a pyarrow Table.

    This dataset must not contain more than one dimension.
    """
    assert isinstance(self, Dataset)
    if len(self.dims) != 1:
        raise ValueError("Only 1-dim datasets can be converted to tables")

    import pyarrow as pa

    from .relationships import sparse_array_type

    def to_numpy(var):
        """Coerces wrapped data to numpy and returns a numpy.ndarray."""
        data = var.data
        if hasattr(data, "chunks"):
            data = data.compute()
        if isinstance(data, sparse_array_type):
            data = data.todense()
        return np.asarray(data)

    pydict = {}
    for i in self.variables:
        dictionary = self[i].attrs.get("DICTIONARY", None)
        if dictionary is not None:
            pydict[i] = pa.DictionaryArray.from_arrays(
                to_numpy(self[i]),
                dictionary,
            )
        else:
            pydict[i] = pa.array(to_numpy(self[i]))
    return pa.Table.from_pydict(pydict)


@register_dataset_method
def select_and_rename(self, name_dict=None, **names):
    """
    Select and rename variables from this Dataset.

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


@register_dataset_method
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
            if self[i].dtype.itemsize > p / 8:
                self[i] = self[i].astype(f"float{p}")
    return self


@register_dataset_method
def interchange_dims(self, dim1, dim2):
    """
    Rename a pair of dimensions by swapping their names.

    Parameters
    ----------
    dim1, dim2 : str
        The names of the two dimensions to swap.

    Returns
    -------
    Dataset
    """
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


def from_named_objects(*args):
    """
    Create a Dataset by populating it with named objects.

    A mapping of names to values is first created, and then that mapping is
    used in the standard constructor to initialize a Dataset.

    Parameters
    ----------
    *args : Any
        A collection of objects, each exposing a `name` attribute.

    Returns
    -------
    Dataset
    """
    objs = {}
    for n, a in enumerate(args):
        try:
            name = a.name
        except AttributeError:
            raise ValueError(f"argument {n} has no name") from None
        if name is None:
            raise ValueError(f"the name for argument {n} is None")
        objs[name] = np.asarray(a)
    return xr.Dataset(objs)


@register_dataset_method
def ensure_integer(dataset, names, bitwidth=32, inplace=False):
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
        result = dataset
    else:
        result = dataset.copy()
    for name in names:
        if name not in result:
            continue
        if not np.issubdtype(result[name].dtype, np.integer):
            result[name] = result[name].astype(f"int{bitwidth}")
    if not inplace:
        return result


def filter_name_tokens(expr, matchable_names=None):
    name_tokens = extract_all_name_tokens(expr)
    name_tokens -= {"_args", "_inputs", "_outputs", "np"}
    name_tokens -= well_known_names
    if matchable_names:
        name_tokens &= matchable_names
    return name_tokens


def _dyno(k, v):
    if isinstance(v, str) and v[0] == "@":
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
