from __future__ import annotations

import ast
import base64
import concurrent.futures
import contextlib
import hashlib
import logging
import multiprocessing
import os
import re
import secrets
import time
from collections.abc import Hashable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr
from xarray import DataArray, Dataset

from . import omx_reader
from .accessors import register_dataset_method
from .aster import extract_all_name_tokens
from .categorical import _Categorical  # noqa
from .shared_memory import si_units
from .table import Table

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


def _group_names(grp) -> list[str]:
    """List the child names of an HDF5 group."""
    return list(grp.keys())


def omx_file_name(omx) -> str | None:
    """Resolve the on-disk filename of an OMX HDF5 file, if possible.

    Filename discovery is deliberately structural so callers can continue to
    pass handles created by optional OMX libraries.  In particular, both
    ``openmatrix.File`` and PyTables ``File`` expose a ``filename`` attribute;
    Sharrow can reopen that path with h5py without importing either package.
    """
    filename = omx_reader.h5_filename(omx)
    if filename is None:
        filename = getattr(omx, "filename", None)
        try:
            filename = os.fspath(filename)
        except TypeError:
            return None
        if not os.path.isfile(filename):
            return None
    return filename


def from_omx(
    omx: h5py.File | str | os.PathLike,
    index_names=("otaz", "dtaz"),
    indexes="one-based",
    renames=None,
):
    """
    Create a Dataset from an OMX file.

    Parameters
    ----------
    omx : h5py.File, path-like, or filename-bearing OMX handle
        An OMX-format HDF5 file, its path, or a compatible open handle such as
        an ``openmatrix.File``. Filename-bearing handles are reopened through
        h5py and remain owned by the caller.
    index_names : tuple, default ("otaz", "dtaz")
        The names of the two matrix dimensions.
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
    if isinstance(omx, (str, os.PathLike)):
        with h5py.File(omx, "r") as handle:
            return from_omx(
                handle,
                index_names=index_names,
                indexes=indexes,
                renames=renames,
            )
    if not isinstance(omx, h5py.File):
        filename = omx_file_name(omx)
        if filename is None:
            raise TypeError(
                "omx must be an h5py.File, path-like, or filename-bearing OMX handle"
            )
        with h5py.File(filename, "r") as handle:
            return from_omx(
                handle,
                index_names=index_names,
                indexes=indexes,
                renames=renames,
            )

    omx_data = omx["data"]
    omx_lookup = omx["lookup"]
    omx_shape = tuple(int(i) for i in omx.attrs["SHAPE"])

    if renames is None:
        data_names = _group_names(omx_data)
        rename_pairs = [(k, k) for k in data_names]
    elif isinstance(renames, dict):
        rename_pairs = list(renames.items())
    else:
        rename_pairs = [(k, k) for k in renames]

    arrays = {}
    filename = omx_file_name(omx)
    if _is_reopenable(filename):
        # fast path: parallel chunk decoding via h5py
        with h5py.File(filename, "r") as f5:
            f5_data = f5["data"]
            with concurrent.futures.ThreadPoolExecutor() as pool:
                for new_k, old_k in rename_pairs:
                    arrays[new_k] = omx_reader.read_dataset(
                        f5_data[old_k], executor=pool
                    )
    else:
        for new_k, old_k in rename_pairs:
            arrays[new_k] = omx_data[old_k][:]
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


def _omx_target_dtype(dtype, max_float_precision):
    """Return the in-memory dtype after applying the precision limit."""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.floating):
        max_dtype = np.dtype(f"float{max_float_precision}")
        if dtype.itemsize > max_dtype.itemsize:
            return max_dtype
    return dtype


def _empty_omx_3d(shape, dtype):
    """Allocate a 3-D array whose individual last-axis pages are contiguous.

    OMX stores each time-period page as an independent C-contiguous 2-D HDF5
    dataset.  Making the logical last axis physically outermost lets h5py read
    each source page directly into the result without a matrix-sized temporary.
    """
    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape)) * dtype.itemsize
    buffer = np.empty(nbytes, dtype=np.uint8)
    strides = (
        shape[1] * dtype.itemsize,
        dtype.itemsize,
        shape[0] * shape[1] * dtype.itemsize,
    )
    return np.ndarray(shape, dtype=dtype, buffer=buffer, strides=strides)


def _read_omx_dataset(dset, out=None, dtype=None):
    """Read one HDF5 dataset with native h5py decompression and conversion."""
    if out is not None:
        if out.flags.c_contiguous:
            # HDF5 converts directly to the destination dtype when needed.
            dset.read_direct(out)
        else:
            # h5py requires a C-contiguous destination.  This fallback is used
            # for ordinary C-order 3-D arrays whose last-axis pages are strided.
            out[...] = dset.astype(out.dtype)[()]
        return out
    if dtype is None or np.dtype(dtype) == dset.dtype:
        return dset[()]
    return dset.astype(np.dtype(dtype))[()]


def _fast_load_omx_array(filename, name, dtype=None):
    """Load one matrix table through h5py's native HDF5 filter pipeline."""
    with h5py.File(filename, "r") as f:
        return _read_omx_dataset(f["data"][name], dtype=dtype)


def _load_omx_variable(page_sources, shape, dtype):
    """Load all pages of one logical OMX variable in a single Dask task."""
    result = _empty_omx_3d(shape, dtype)
    open_files = {}
    try:
        for period, source in enumerate(page_sources):
            if source is None:
                result[..., period].fill(0)
                continue
            filename, data_name, eager_array = source
            if eager_array is not None:
                result[..., period] = eager_array
                continue
            if filename not in open_files:
                open_files[filename] = h5py.File(filename, "r")
            _read_omx_dataset(
                open_files[filename]["data"][data_name], result[..., period]
            )
    finally:
        for handle in open_files.values():
            handle.close()
    return result


def _load_omx_assignments(dataset, source, assignments):
    """Load all selected matrices from one source into a prepared Dataset."""
    if isinstance(source, h5py.File):
        file_context = contextlib.nullcontext(source)
    else:
        file_context = h5py.File(source, "r")
    bytes_loaded = 0
    with file_context as handle:
        data_group = handle["data"]
        for data_name, variable_name, period in assignments:
            target = dataset[variable_name].data
            if period is not None:
                target = target[..., period]
            _read_omx_dataset(data_group[data_name], target)
            bytes_loaded += target.nbytes
    return bytes_loaded


def _load_omx_shared_worker(shared_memory_key, source, assignments):
    """Process worker that fills disjoint pages of a shared OMX Dataset."""
    target = xr.Dataset.shm.from_shared_memory(shared_memory_key, mode="r+")
    bytes_loaded = _load_omx_assignments(target, source, assignments)
    if shared_memory_key.startswith("memmap:"):
        for memory_object in target.shm._shared_memory_objs_:
            flush = getattr(memory_object, "flush", None)
            if flush is not None:
                flush()
    return bytes_loaded


def _is_reopenable(filename) -> bool:
    """Check whether a file can be independently opened for reading with h5py.

    Reopening can fail if the file name is unknown (e.g. an in-memory file),
    or if the file is already open elsewhere in a mode that locks it.
    """
    if filename is None:
        return False
    try:
        with h5py.File(filename, "r"):
            pass
    except Exception:  # noqa: BLE001
        return False
    return True


def from_omx_3d(
    omx: h5py.File | str | os.PathLike | Iterable[h5py.File | str | os.PathLike],
    index_names=("otaz", "dtaz", "time_period"),
    indexes=None,
    *,
    time_periods=None,
    time_period_sep="__",
    max_float_precision=32,
    ignore=None,
    load="lazy",
    task_granularity="variable",
    workers=None,
    memory_path=None,
    shared_memory_key=None,
):
    """
    Create a Dataset from an OMX file with an implicit third dimension.

    Parameters
    ----------
    omx : h5py.File, path-like, filename-bearing OMX handle, or iterable
        One or more OMX-format HDF5 files, paths, or compatible open handles
        such as ``openmatrix.File``. Filename-bearing handles are reopened
        through h5py and remain owned by the caller.
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
    load : {"lazy", "eager", "shared", "memmap"}, default "lazy"
        Loading mode. ``"lazy"`` returns Dask arrays. ``"eager"`` loads into
        ordinary NumPy arrays in the calling process. ``"shared"`` uses
        process-parallel reads into shared memory and is the fastest mode for
        multiple large OMX files. ``"memmap"`` uses the same parallel loader
        with disk-backed arrays, substantially reducing resident memory at the
        cost of additional storage I/O.
    task_granularity : {"variable", "matrix"}, default "variable"
        Dask task granularity for lazy loading. Grouping all time-period pages
        of a variable minimizes graph and file-open overhead. Matrix granularity
        can use less memory when only selected periods are subsequently loaded.
    workers : int, optional
        Number of file-level worker processes for ``"shared"`` or ``"memmap"``.
        The default uses up to one worker per source file. Eager loading uses
        one worker in the calling process.
    memory_path : path-like, optional
        New backing file to create when ``load="memmap"``. The associated
        metadata is stored alongside it with a ``.meta.pkl`` suffix. Delete
        both when finished with
        ``result.shm.delete_shared_memory_files(result.shm.shared_memory_key)``.
    shared_memory_key : str, optional
        Key used to identify an explicitly shared dataset. A unique key is
        generated by default. Call ``result.shm.release_shared_memory()`` when
        a shared result is no longer needed.

    Returns
    -------
    Dataset
        Lazy Dask-backed, ordinary in-memory, shared-memory-backed, or
        memory-mapped according to ``load``.
    """
    if load is True:
        load = "eager"
    elif load is False:
        load = "lazy"
    if load not in {"lazy", "eager", "shared", "memmap"}:
        raise ValueError("load must be 'lazy', 'eager', 'shared', or 'memmap'")
    if task_granularity not in {"variable", "matrix"}:
        raise ValueError("task_granularity must be 'variable' or 'matrix'")
    if workers is not None and (not isinstance(workers, int) or workers < 1):
        raise ValueError("workers must be a positive integer")
    if load == "eager" and workers not in {None, 1}:
        raise ValueError(
            "load='eager' uses one process; use load='shared' for parallel reads"
        )
    if load == "memmap" and memory_path is None:
        raise ValueError("memory_path is required when load='memmap'")
    if load != "memmap" and memory_path is not None:
        raise ValueError("memory_path is only used when load='memmap'")

    if isinstance(omx, (h5py.File, str, os.PathLike)) or omx_file_name(omx):
        omx_sources = [omx]
    else:
        omx_sources = list(omx)
    if not omx_sources:
        raise ValueError("at least one OMX file is required")

    use_file_handles = []
    opened_file_handles = []
    try:
        for source in omx_sources:
            if isinstance(source, (str, os.PathLike)):
                h = h5py.File(source, "r")
                opened_file_handles.append(h)
                use_file_handles.append(h)
            elif isinstance(source, h5py.File):
                use_file_handles.append(source)
            else:
                # Preserve compatibility with openmatrix/PyTables and similar
                # optional wrappers without importing those dependencies. They
                # expose the backing HDF5 path through ``filename``.
                filename = omx_file_name(source)
                if filename is None:
                    raise TypeError(
                        "omx entries must be h5py.File, path-like, or "
                        "filename-bearing OMX handles"
                    )
                h = h5py.File(filename, "r")
                opened_file_handles.append(h)
                use_file_handles.append(h)
    except Exception:
        for handle in opened_file_handles:
            handle.close()
        raise
    omx_handles = use_file_handles

    try:
        omx_shape = tuple(int(i) for i in omx_handles[0].attrs["SHAPE"])
        omx_lookup = omx_handles[0]["lookup"]
        omx_data = [handle["data"] for handle in omx_handles]
        omx_data_map = {}
        matrix_metadata = {}
        for source_number, data_group in enumerate(omx_data):
            for data_name in _group_names(data_group):
                node = data_group[data_name]
                omx_data_map[data_name] = source_number
                matrix_metadata[data_name] = (tuple(node.shape), np.dtype(node.dtype))

        omx_filenames = [omx_file_name(i) for i in omx_handles]
        omx_reopenable = [_is_reopenable(i) for i in omx_filenames]

        data_names = list(omx_data_map.keys())
        if ignore is not None:
            if isinstance(ignore, str):
                ignore = [ignore]
            data_names = [i for i in data_names if not _should_ignore(ignore, i)]
        n1, n2 = omx_shape
        if indexes is None:
            # default reads mapping if only one lookup is included, otherwise one-based
            lookup_names = _group_names(omx_lookup)
            if len(lookup_names) == 1:
                ranger = None
                indexes = lookup_names[0]
            else:
                ranger = one_based
        elif indexes == "one-based":
            ranger = one_based
        elif indexes == "zero-based":
            ranger = zero_based
        elif indexes in set(_group_names(omx_lookup)):
            ranger = None
        else:
            raise NotImplementedError(
                "only one-based, zero-based, and named indexes are implemented"
            )
        if ranger is not None:
            r1 = ranger(n1)
            r2 = ranger(n2)
        else:
            r1 = r2 = pd.Index(np.asarray(omx_lookup[indexes]))

        if time_periods is None:
            raise ValueError("must give time periods explicitly")

        time_periods_map = {t: n for n, t in enumerate(time_periods)}

        pending_3d = {}
        variable_specs = {}
        for data_name in data_names:
            source_number = omx_data_map[data_name]
            matrix_shape, matrix_dtype = matrix_metadata[data_name]
            if matrix_shape != omx_shape:
                raise ValueError(
                    f"matrix {data_name!r} has shape {matrix_shape}, expected {omx_shape}"
                )
            entry = (source_number, data_name, matrix_dtype)
            if time_period_sep in data_name:
                base_name, period_name = data_name.split(time_period_sep, 1)
                if period_name not in time_periods_map:
                    raise KeyError(
                        f"time period {period_name!r} from {data_name!r} is not in "
                        "time_periods"
                    )
                pending_3d.setdefault(base_name, [None] * len(time_periods))[
                    time_periods_map[period_name]
                ] = entry
            else:
                variable_specs[data_name] = {
                    "pages": [entry],
                    "shape": omx_shape,
                    "dtype": _omx_target_dtype(matrix_dtype, max_float_precision),
                }
        for base_name, pages in pending_3d.items():
            source_dtypes = [page[2] for page in pages if page is not None]
            variable_specs[base_name] = {
                "pages": pages,
                "shape": omx_shape + (len(time_periods),),
                "dtype": _omx_target_dtype(
                    np.result_type(*source_dtypes), max_float_precision
                ),
            }

        coords = {index_names[0]: r1, index_names[1]: r2}
        if pending_3d:
            coords[index_names[2]] = time_periods

        # Each assignment belongs to exactly one source file. Duplicate OMX
        # names retain the established last-file-wins behavior.
        assignments = [[] for _ in omx_handles]
        for variable_name, spec in variable_specs.items():
            is_3d = len(spec["shape"]) == 3
            for period, page in enumerate(spec["pages"]):
                if page is None:
                    continue
                source_number, data_name, _ = page
                assignments[source_number].append(
                    (data_name, variable_name, period if is_3d else None)
                )

        if load == "lazy":
            import dask
            import dask.array

            eager_arrays = {}

            def source_descriptor(page):
                if page is None:
                    return None
                source_number, data_name, _ = page
                if omx_reopenable[source_number]:
                    return (str(omx_filenames[source_number]), data_name, None)
                if data_name not in eager_arrays:
                    eager_arrays[data_name] = np.asarray(
                        omx_data[source_number][data_name][()]
                    )
                return (None, data_name, eager_arrays[data_name])

            content = {}
            for variable_name, spec in variable_specs.items():
                dtype = spec["dtype"]
                if len(spec["shape"]) == 2:
                    page = spec["pages"][0]
                    descriptor = source_descriptor(page)
                    filename, data_name, eager_array = descriptor
                    if eager_array is not None:
                        array = dask.array.from_array(eager_array).astype(dtype)
                    else:
                        array = dask.array.from_delayed(
                            dask.delayed(_fast_load_omx_array)(
                                filename, data_name, dtype
                            ),
                            shape=spec["shape"],
                            dtype=dtype,
                        )
                elif task_granularity == "variable":
                    page_sources = [source_descriptor(i) for i in spec["pages"]]
                    array = dask.array.from_delayed(
                        dask.delayed(_load_omx_variable)(
                            page_sources, spec["shape"], dtype
                        ),
                        shape=spec["shape"],
                        dtype=dtype,
                    )
                else:
                    page_arrays = []
                    for page in spec["pages"]:
                        descriptor = source_descriptor(page)
                        if descriptor is None:
                            page_arrays.append(
                                dask.array.zeros(
                                    omx_shape, chunks=omx_shape, dtype=dtype
                                )
                            )
                            continue
                        filename, data_name, eager_array = descriptor
                        if eager_array is not None:
                            page_array = dask.array.from_array(eager_array).astype(
                                dtype
                            )
                        else:
                            page_array = dask.array.from_delayed(
                                dask.delayed(_fast_load_omx_array)(
                                    filename, data_name, dtype
                                ),
                                shape=omx_shape,
                                dtype=dtype,
                            )
                        page_arrays.append(page_array)
                    array = dask.array.stack(page_arrays, axis=-1)
                dims = index_names if array.ndim == 3 else index_names[:2]
                content[variable_name] = (dims, array)
            return xr.Dataset(content, coords=coords)

        if load == "eager":
            content = {}
            for variable_name, spec in variable_specs.items():
                if len(spec["shape"]) == 3:
                    array = _empty_omx_3d(spec["shape"], spec["dtype"])
                    for period, page in enumerate(spec["pages"]):
                        if page is None:
                            array[..., period].fill(0)
                else:
                    array = np.empty(spec["shape"], dtype=spec["dtype"])
                dims = index_names if array.ndim == 3 else index_names[:2]
                content[variable_name] = (dims, array)
            result = xr.Dataset(content, coords=coords)
            for source, source_assignments in zip(omx_handles, assignments):
                if source_assignments:
                    _load_omx_assignments(result, source, source_assignments)
            return result

        # Shared and memory-mapped modes use a lightweight template to reserve
        # one contiguous backing buffer, then independent processes fill each
        # source file directly into disjoint final-array pages.
        if not all(omx_reopenable[n] for n, batch in enumerate(assignments) if batch):
            raise ValueError(
                f"load={load!r} requires path-backed OMX files that can be reopened"
            )
        import dask.array

        template_content = {}
        array_order = {}
        for variable_name, spec in variable_specs.items():
            dims = index_names if len(spec["shape"]) == 3 else index_names[:2]
            template_content[variable_name] = (
                dims,
                dask.array.empty(
                    spec["shape"], chunks=spec["shape"], dtype=spec["dtype"]
                ),
            )
            if len(spec["shape"]) == 3:
                array_order[variable_name] = "last-axis-first"
        template = xr.Dataset(template_content, coords=coords)

        if load == "memmap":
            memory_path = Path(memory_path).expanduser().resolve()
            metadata_path = Path(f"{memory_path}.meta.pkl")
            if memory_path.exists() or metadata_path.exists():
                raise FileExistsError(
                    f"memory_path and metadata path must not already exist: {memory_path}"
                )
            memory_path.parent.mkdir(parents=True, exist_ok=True)
            key = f"memmap:{memory_path}"
        else:
            if shared_memory_key is not None and shared_memory_key.startswith(
                "memmap:"
            ):
                raise ValueError("shared_memory_key must not start with 'memmap:'")
            key = shared_memory_key or f"omx-{secrets.token_hex(8)}"

        result = template.shm.to_shared_memory(
            key, mode="r+", load=False, array_order=array_order
        )
        if hasattr(result.shm, "tasks"):
            del result.shm.tasks
        if hasattr(result.shm, "task_names"):
            del result.shm.task_names

        active_batches = [
            (str(omx_filenames[n]), batch)
            for n, batch in enumerate(assignments)
            if batch
        ]
        worker_count = workers or max(1, min(len(active_batches), os.cpu_count() or 1))
        started = time.time()
        try:
            if worker_count == 1:
                bytes_loaded = sum(
                    _load_omx_assignments(result, source, batch)
                    for source, batch in active_batches
                )
            else:
                # Spawn avoids inheriting any HDF5 state held by the caller.
                mp_context = multiprocessing.get_context("spawn")
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=worker_count, mp_context=mp_context
                ) as pool:
                    futures = [
                        pool.submit(_load_omx_shared_worker, key, source, batch)
                        for source, batch in active_batches
                    ]
                    bytes_loaded = sum(future.result() for future in futures)
            if load == "memmap":
                for memory_object in result.shm._shared_memory_objs_:
                    flush = getattr(memory_object, "flush", None)
                    if flush is not None:
                        flush()
            logger.info(
                "loaded %s from %d OMX files with %d worker(s) in %.2fs",
                si_units(bytes_loaded),
                len(active_batches),
                worker_count,
                time.time() - started,
            )
            return result
        except Exception:
            if load == "shared":
                result.shm.release_shared_memory()
            else:
                result.shm.delete_shared_memory_files(key)
            raise
    finally:
        for h in opened_file_handles:
            h.close()


def reload_from_omx_3d(
    dataset: xr.Dataset,
    omx: h5py.File | str | os.PathLike | Iterable[h5py.File | str | os.PathLike],
    *,
    time_period_sep="__",
    ignore=None,
) -> None:
    """
    Reload the content of a dataset from OMX files.

    This loads the data from the OMX files into the dataset, replacing
    the existing data in the dataset.  The dataset must have been created
    by `from_omx_3d` or a similar function. By default, `from_omx_3d` creates
    a dataset backed by `dask.array` objects; this function allows for loading
    the data without going through Dask.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to reload into.
    omx : h5py.File, path-like, filename-bearing OMX handle, or iterable
        One or more OMX-format HDF5 files, paths, or compatible open handles
        such as ``openmatrix.File``.
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
    if isinstance(omx, (h5py.File, str, os.PathLike)) or omx_file_name(omx):
        omx = [omx]

    bytes_loaded = 0
    t0 = time.time()

    def _load_one(dset, data_name, filter_note):
        nonlocal bytes_loaded
        t1 = time.time()
        if time_period_sep in data_name:
            data_name_x, data_name_t = data_name.split(time_period_sep, 1)
            if data_name_x not in dataset:
                logger.info(
                    f"skipping {data_name} because {data_name_x} not in dataset"
                )
                return
            if len(dataset[data_name_x].dims) != 3:
                raise ValueError(
                    f"dataset variable {data_name_x} has "
                    f"{len(dataset[data_name_x].dims)} dimensions, expected 3"
                )
            period_dimension = dataset[data_name_x].dims[-1]
            raw = dataset[data_name_x].sel({period_dimension: data_name_t}).data
        else:
            if data_name not in dataset:
                logger.info(f"skipping {data_name} because it is not in dataset")
                return
            if len(dataset[data_name].dims) != 2:
                raise ValueError(
                    f"dataset variable {data_name} has "
                    f"{len(dataset[data_name].dims)} dimensions, expected 2"
                )
            raw = dataset[data_name].data
        _read_omx_dataset(dset, raw)
        bytes_loaded += raw.nbytes
        logger.debug(
            f"loaded {data_name} ({filter_note}) to dataset "
            f"in {time.time() - t1:.2f}s, {si_units(bytes_loaded)}"
        )

    for source in omx:
        if isinstance(source, (str, os.PathLike)):
            logger.info(f"loading into dataset from {source}")
            file_context = h5py.File(source, "r")
        elif isinstance(source, h5py.File):
            logger.info(f"loading into dataset from {source.filename}")
            file_context = contextlib.nullcontext(source)
        else:
            filename = omx_file_name(source)
            if filename is None:
                raise TypeError(
                    "omx entries must be h5py.File, path-like, or "
                    "filename-bearing OMX handles"
                )
            logger.info(f"loading into dataset from {filename}")
            file_context = h5py.File(filename, "r")
        with file_context as handle:
            data_group = handle["data"]
            for data_name, dset in data_group.items():
                if _should_ignore(ignore, data_name):
                    logger.info(f"ignoring {data_name}")
                    continue
                filter_note = f"{dset.compression}/{dset.compression_opts}"
                _load_one(dset, data_name, filter_note)
    logger.info(f"loading to dataset complete in {time.time() - t0:.2f}s")


def _parquet_layout(labels_0, labels_1):
    """
    Determine the layout of a two dimensional index given as two columns.

    Parameters
    ----------
    labels_0, labels_1 : array-like
        The values of the two index columns, which together identify the
        position of each row of a table in a two dimensional array.

    Returns
    -------
    layout : {"row-major", "column-major", "unsorted-dense", "sparse"}
    index_0, index_1 : array-like or None
        The unique labels for each dimension, in the order they appear in
        the resulting array.  These are None if the layout is "sparse", as
        the arrangement of a sparse table is resolved elsewhere.
    """
    n_rows = len(labels_0)
    unique_0 = pd.unique(labels_0)
    unique_1 = pd.unique(labels_1)
    n_0 = len(unique_0)
    n_1 = len(unique_1)
    if n_0 * n_1 != n_rows:
        return "sparse", None, None

    # row-major: the first index changes slowly, the second changes quickly
    row_major_0 = labels_0[::n_1]
    row_major_1 = labels_1[:n_1]
    if len(row_major_0) == n_0 and np.array_equal(
        labels_0, np.repeat(row_major_0, n_1)
    ):
        if np.array_equal(labels_1, np.tile(row_major_1, n_0)):
            return "row-major", row_major_0, row_major_1

    # column-major: the second index changes slowly, the first changes quickly
    column_major_0 = labels_0[:n_0]
    column_major_1 = labels_1[::n_0]
    if len(column_major_1) == n_1 and np.array_equal(
        labels_1, np.repeat(column_major_1, n_0)
    ):
        if np.array_equal(labels_0, np.tile(column_major_0, n_1)):
            return "column-major", column_major_0, column_major_1

    # The number of rows matches a dense array, but the rows are not in
    # either dense ordering.  This is only actually dense if every possible
    # pair of labels appears exactly once.
    codes_0 = pd.Index(unique_0).get_indexer(labels_0)
    codes_1 = pd.Index(unique_1).get_indexer(labels_1)
    flat = codes_0.astype(np.int64) * n_1 + codes_1
    if len(np.unique(flat)) == n_rows:
        return "unsorted-dense", unique_0, unique_1
    return "sparse", None, None


def _parquet_column_to_numpy(table, name):
    column = table.column(name)
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    if isinstance(column, pa.ChunkedArray):
        # older versions of pyarrow return a ChunkedArray from combine_chunks
        return column.to_numpy()
    return column.to_numpy(zero_copy_only=False)


def _parquet_data_names(schema_names, index_names, ignore):
    data_names = [i for i in schema_names if i not in index_names]
    if ignore is not None:
        if isinstance(ignore, str):
            ignore = [ignore]
        data_names = [i for i in data_names if not _should_ignore(ignore, i)]
    return data_names


def _read_one_parquet_3d(filename, index_names, ignore):
    """
    Read the matrix tables in one parquet file into two dimensional arrays.

    Parameters
    ----------
    filename : path-like
        The parquet file to read.
    index_names : tuple[str, str]
        The names of the columns in the parquet file that give the position
        of each row in the two dimensional arrays.
    ignore : list-like or None
        Regular expressions for matrix table names to skip.

    Returns
    -------
    arrays : dict[str, array-like]
        Two dimensional arrays, one for each matrix table in the file.
    index_0, index_1 : array-like
        The labels for each dimension of the arrays.
    """
    import pyarrow.parquet as pq

    with pq.ParquetFile(filename) as pf:
        schema_names = pf.schema_arrow.names
        for i in index_names:
            if i not in schema_names:
                raise KeyError(f"index column {i!r} not found in {filename}")
        data_names = _parquet_data_names(schema_names, index_names, ignore)

        index_table = pf.read(columns=list(index_names))
        labels_0 = _parquet_column_to_numpy(index_table, index_names[0])
        labels_1 = _parquet_column_to_numpy(index_table, index_names[1])
        layout, index_0, index_1 = _parquet_layout(labels_0, labels_1)
        logger.info(f"parquet file {filename} has a {layout} layout")
        del index_table, labels_0, labels_1

        if layout == "unsorted-dense":
            raise ValueError(
                f"the data in {filename} is dense but is not sorted into "
                f"row-major or column-major order"
            )

        if layout == "sparse":
            # fall back to the generic xarray loader for sparse data
            df = pf.read(columns=list(index_names) + data_names).to_pandas()
            ds = df.set_index(list(index_names)).to_xarray()
            arrays = {k: ds[k].to_numpy() for k in data_names}
            return arrays, ds[index_names[0]].to_numpy(), ds[index_names[1]].to_numpy()

        n_0 = len(index_0)
        n_1 = len(index_1)
        arrays = {}
        for k in data_names:
            content = _parquet_column_to_numpy(pf.read(columns=[k]), k)
            if layout == "row-major":
                arrays[k] = content.reshape(n_0, n_1)
            else:
                arrays[k] = content.reshape(n_1, n_0).transpose()
        return arrays, index_0, index_1


def from_parquet_3d(
    parquet,
    index_names=("otaz", "dtaz", "time_period"),
    *,
    time_periods=None,
    time_period_sep="__",
    max_float_precision=32,
    ignore=None,
):
    """
    Create a Dataset from parquet file(s) with an implicit third dimension.

    The parquet file(s) should contain two index columns, which give the
    position of each row in the two "native" dimensions, plus any number of
    other columns, each of which gives the values of one matrix table.  The
    matrix tables are named in the same manner as they would be in an OMX
    file, including using a separator (typically a double underscore) to
    identify time periods, which are assembled into a third dimension.

    Parameters
    ----------
    parquet : path-like or Iterable[path-like]
        The parquet file(s) to read.  When multiple files are given, the
        matrix tables from all files are combined into a single dataset.
        Each file is checked independently for its data layout, so the
        index columns need not be in the same order in every file.
    index_names : tuple, default ("otaz", "dtaz", "time_period")
        Should be a tuple of length 3, giving the names of the three
        dimensions.  The first two names are the names of the index columns
        in the parquet file(s), the last is the name of the implicit
        dimension that is created by parsing matrix table names.
    time_periods : list-like, optional
        A list of index values from which the third dimension is constructed
        for all variables with a third dimension.  Required if any matrix
        table name contains `time_period_sep`.
    time_period_sep : str, default "__" (double underscore)
        The presence of this separator within the name of any matrix table
        indicates that table is to be considered a page in a three
        dimensional variable.  The portion of the name preceding the first
        instance of this separator is the name of the resulting variable,
        and the portion of the name after the first instance of this
        separator is the label of the position for this page, which should
        appear in `time_periods`.
    max_float_precision : int, default 32
        When loading, reduce all floats to this level of precision,
        generally to save memory if they were stored as double precision but
        that level of detail is unneeded in the present application.
    ignore : str or list-like, optional
        A list of regular expressions that will be used to filter out
        variables from the dataset.  If any of the regular expressions
        match the name of a variable, that variable will not be included
        in the loaded dataset.

    Returns
    -------
    Dataset
    """
    if isinstance(parquet, (str, Path)) or not isinstance(parquet, Iterable):
        parquet = [parquet]

    if len(index_names) != 3:
        raise ValueError("index_names must have length 3")

    time_periods_map = None
    if time_periods is not None:
        time_periods = list(time_periods)
        time_periods_map = {t: n for n, t in enumerate(time_periods)}

    index_0 = None
    index_1 = None
    content = {}
    pending_3d = {}

    for filename in parquet:
        arrays, file_index_0, file_index_1 = _read_one_parquet_3d(
            filename, tuple(index_names[:2]), ignore
        )
        if index_0 is None:
            index_0 = file_index_0
            index_1 = file_index_1
        elif not (
            np.array_equal(index_0, file_index_0)
            and np.array_equal(index_1, file_index_1)
        ):
            # the labels in this file are not in the same order as the
            # labels in the first file, so rearrange this file's data
            take_0 = pd.Index(file_index_0).get_indexer(index_0)
            take_1 = pd.Index(file_index_1).get_indexer(index_1)
            if (take_0 < 0).any() or (take_1 < 0).any():
                raise ValueError(
                    f"the index labels in {filename} do not match those in "
                    f"the other parquet file(s)"
                )
            arrays = {k: v[take_0][:, take_1] for k, v in arrays.items()}

        for k, v in arrays.items():
            if time_period_sep in k:
                base_k, time_k = k.split(time_period_sep, 1)
                if time_periods_map is None:
                    raise ValueError("must give time periods explicitly")
                if time_k not in time_periods_map:
                    raise KeyError(f"time period {time_k!r} not in time_periods")
                if base_k not in pending_3d:
                    pending_3d[base_k] = [None] * len(time_periods)
                pending_3d[base_k][time_periods_map[time_k]] = v
            else:
                content[k] = xr.DataArray(
                    v,
                    dims=index_names[:2],
                    coords={
                        index_names[0]: index_0,
                        index_names[1]: index_1,
                    },
                )

    for base_k, arrs in pending_3d.items():
        prototype = None
        for i in arrs:
            if i is not None:
                prototype = i
                break
        if prototype is None:
            raise ValueError("no prototype")
        arrs_ = [(i if i is not None else np.zeros_like(prototype)) for i in arrs]
        content[base_k] = xr.DataArray(
            np.stack(arrs_, axis=-1),
            dims=index_names,
            coords={
                index_names[0]: index_0,
                index_names[1]: index_1,
                index_names[2]: time_periods,
            },
        )

    for i in content:
        if np.issubdtype(content[i].dtype, np.floating):
            if content[i].dtype.itemsize > max_float_precision / 8:
                content[i] = content[i].astype(f"float{max_float_precision}")
    return xr.Dataset(content)


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
