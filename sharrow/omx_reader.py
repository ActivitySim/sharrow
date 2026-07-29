"""Fast, low-memory readers for HDF5-backed OMX matrix files.

The functions in this module bypass the HDF5 library's single-threaded
decompression path.  Raw (still compressed) chunks are read from the file one
at a time and handed to a thread pool, where the HDF5 filter pipeline is undone
by extension code that releases the GIL.  Decompressed chunks are written
directly into the destination array, so peak memory usage is the size of the
result plus a small, bounded number of in-flight chunks.

If a dataset uses a filter this module does not know how to invert, reading
transparently falls back to h5py, which will use the registered HDF5 filter
plugins instead.

This module is adapted from the ``omx_fast_reader`` module of the `wring
<https://github.com/driftlesslabs/wring>`_ project.
"""

from __future__ import annotations

import concurrent.futures
import os
import zlib
from collections.abc import Sequence

import blosc2
import h5py
import hdf5plugin  # noqa: F401  (registers blosc/blosc2/zstd/etc HDF5 filters)
import numpy as np

__all__ = [
    "read_dataset",
    "h5_filename",
]

H5Z_FILTER_DEFLATE = 1
H5Z_FILTER_SHUFFLE = 2
H5Z_FILTER_FLETCHER32 = 3
H5Z_FILTER_BLOSC = 32001
H5Z_FILTER_BLOSC2 = 32026

#: Filters this module can invert without calling into the HDF5 filter pipeline.
SUPPORTED_FILTERS = frozenset(
    {
        H5Z_FILTER_DEFLATE,
        H5Z_FILTER_SHUFFLE,
        H5Z_FILTER_FLETCHER32,
        H5Z_FILTER_BLOSC,
        H5Z_FILTER_BLOSC2,
    }
)


def h5_filename(omx) -> str | None:
    """Resolve the on-disk filename of an OMX-like object.

    Parameters
    ----------
    omx : str, os.PathLike, h5py.File, tables.File, or larch.OMX
        An OMX file reference: a path, an h5py file, a pytables file (which
        includes ``openmatrix.File``), or a larch OMX object.

    Returns
    -------
    str or None
        The path of the underlying file, or None if it cannot be determined.
    """
    if isinstance(omx, (str, os.PathLike)):
        return os.fspath(omx)
    # h5py.File, tables.File (and openmatrix.File), and larch.OMX all expose
    # the underlying file path as a `filename` attribute.
    filename = getattr(omx, "filename", None)
    if isinstance(filename, (str, os.PathLike)):
        filename = os.fspath(filename)
        if os.path.exists(filename):
            return filename
    return None


def _unshuffle(data: bytes, element_size: int) -> bytes:
    """Invert the HDF5 shuffle filter.

    The shuffle filter de-interleaves the bytes of each element, so that all
    first bytes are stored together, then all second bytes, and so on.

    Parameters
    ----------
    data : bytes
        The shuffled byte stream.
    element_size : int
        Size in bytes of one array element.

    Returns
    -------
    bytes
        The original (un-shuffled) byte stream.
    """
    if element_size <= 1:
        return data
    n_elements = len(data) // element_size
    n_shuffled = n_elements * element_size
    shuffled = np.frombuffer(data, dtype=np.uint8, count=n_shuffled)
    unshuffled = shuffled.reshape(element_size, n_elements).T.tobytes()
    # HDF5 leaves any trailing bytes that do not fill a whole element untouched
    return unshuffled + data[n_shuffled:]


def _read_filter_pipeline(dset: h5py.Dataset) -> list:
    """Read the ordered HDF5 filter pipeline of a dataset.

    Parameters
    ----------
    dset : h5py.Dataset
        The dataset to inspect.

    Returns
    -------
    list[tuple[int, tuple[int, ...]]]
        Filters in the order they were applied when the data was written,
        each as a ``(filter_id, cd_values)`` pair.
    """
    plist = dset.id.get_create_plist()
    pipeline = []
    for i in range(plist.get_nfilters()):
        filter_id, _flags, cd_values, _name = plist.get_filter(i)
        pipeline.append((int(filter_id), tuple(int(c) for c in cd_values)))
    return pipeline


def _pipeline_is_supported(pipeline: Sequence) -> bool:
    """Report whether every filter in `pipeline` can be inverted by this module."""
    return all(filter_id in SUPPORTED_FILTERS for filter_id, _ in pipeline)


def _decode_blosc2_cframe(data: bytes):
    """Decode a chunk written by the Blosc2 HDF5 filter.

    Depending on how the data was written, the chunk may be an n-dimensional
    ``b2nd`` container, a plain blosc2 super-chunk, or a bare blosc2 chunk.
    All three are self-describing, so each form is tried in turn.

    Parameters
    ----------
    data : bytes
        The raw chunk as stored in the HDF5 file.

    Returns
    -------
    bytes or numpy.ndarray
        The decompressed payload.
    """
    try:
        return blosc2.ndarray_from_cframe(data)[:]
    except (RuntimeError, ValueError):
        pass
    try:
        return blosc2.schunk_from_cframe(data)[:]
    except (RuntimeError, ValueError):
        pass
    return blosc2.decompress(data)


def _decompress_chunk(
    raw_bytes: bytes,
    filter_mask: int,
    pipeline: Sequence,
    dtype: np.dtype,
    chunk_shape: tuple,
) -> np.ndarray:
    """Undo a dataset's filter pipeline for one raw chunk.

    Parameters
    ----------
    raw_bytes : bytes
        The raw, still-filtered bytes of the chunk as stored in the file.
    filter_mask : int
        Bit mask returned by ``read_direct_chunk``; a set bit means the filter
        at that position in the pipeline was skipped for this chunk.
    pipeline : Sequence
        The dataset filter pipeline, in write order.
    dtype : numpy.dtype
        Element type of the dataset.
    chunk_shape : tuple[int, ...]
        Shape of a full chunk.

    Returns
    -------
    numpy.ndarray
        The decoded chunk, with shape `chunk_shape` and dtype `dtype`.

    Raises
    ------
    NotImplementedError
        If the pipeline contains a filter this module cannot invert.
    """
    data = raw_bytes

    # Filters are applied in order when writing, so undo them in reverse order.
    for position, (filter_id, cd_values) in reversed(list(enumerate(pipeline))):
        if filter_mask & (1 << position):
            # This filter was skipped for this particular chunk
            continue
        if isinstance(data, np.ndarray):
            # Only the innermost (last) filter may emit an array instead of bytes
            data = data.tobytes()
        if filter_id == H5Z_FILTER_DEFLATE:
            data = zlib.decompress(data)
        elif filter_id == H5Z_FILTER_SHUFFLE:
            element_size = (cd_values[0] if cd_values else 0) or dtype.itemsize
            data = _unshuffle(data, element_size)
        elif filter_id == H5Z_FILTER_FLETCHER32:
            # Trailing 4-byte checksum, not verified here
            data = data[:-4]
        elif filter_id == H5Z_FILTER_BLOSC:
            data = blosc2.decompress(data)
        elif filter_id == H5Z_FILTER_BLOSC2:
            data = _decode_blosc2_cframe(data)
        else:
            raise NotImplementedError(f"unsupported HDF5 filter id {filter_id}")

    if isinstance(data, np.ndarray):
        if data.dtype != dtype:
            data = data.view(dtype)
        return data.reshape(chunk_shape)
    return np.frombuffer(data, dtype=dtype).reshape(chunk_shape)


def _write_chunk(
    out: np.ndarray,
    offset: tuple,
    raw_bytes: bytes,
    filter_mask: int,
    pipeline: Sequence,
    chunk_shape: tuple,
) -> None:
    """Decode one raw chunk and write it into its place in `out`.

    Chunks map onto disjoint regions of `out`, so this is safe to call
    concurrently from several threads.

    Parameters
    ----------
    out : numpy.ndarray
        Destination array, shaped like the full dataset.
    offset : tuple[int, ...]
        Index in the dataset of the chunk's first element.
    raw_bytes : bytes
        Raw, still-filtered chunk bytes.
    filter_mask : int
        Per-chunk filter mask from ``read_direct_chunk``.
    pipeline : Sequence
        The dataset filter pipeline, in write order.
    chunk_shape : tuple[int, ...]
        Shape of a full chunk.
    """
    chunk = _decompress_chunk(raw_bytes, filter_mask, pipeline, out.dtype, chunk_shape)

    # Edge chunks are stored padded out to the full chunk shape, so the part
    # that actually lands in the dataset may be smaller than the chunk itself.
    target = tuple(
        slice(start, min(start + size, extent))
        for start, size, extent in zip(offset, chunk_shape, out.shape)
    )
    source = tuple(slice(0, s.stop - s.start) for s in target)
    out[target] = chunk[source]


def _fallback_read(dset: h5py.Dataset, out: np.ndarray) -> None:
    """Read `dset` into `out` using h5py's ordinary (serial) read path."""
    if out.flags.c_contiguous:
        dset.read_direct(out)
    else:
        out[...] = dset[()]


def _load_dataset_into(
    dset: h5py.Dataset,
    out: np.ndarray,
    executor: concurrent.futures.ThreadPoolExecutor,
    max_pending: int,
) -> None:
    """Fill `out` with the contents of `dset`, decoding chunks in parallel.

    Falls back to a plain h5py read when the dataset is not chunked, is not
    fully written, or uses a filter that cannot be inverted here.

    Parameters
    ----------
    dset : h5py.Dataset
        Source dataset.
    out : numpy.ndarray
        Destination array; must have the same shape and dtype as `dset`.
    executor : concurrent.futures.ThreadPoolExecutor
        Pool used to decode chunks.
    max_pending : int
        Maximum number of raw chunks held in memory awaiting decoding.
    """
    chunk_shape = dset.chunks
    if chunk_shape is None or dset.size == 0:
        _fallback_read(dset, out)
        return

    pipeline = _read_filter_pipeline(dset)
    if not _pipeline_is_supported(pipeline):
        _fallback_read(dset, out)
        return

    dset_id = dset.id
    try:
        num_chunks = dset_id.get_num_chunks()
    except (OSError, AttributeError, ValueError):
        _fallback_read(dset, out)
        return

    expected_chunks = 1
    for extent, size in zip(dset.shape, chunk_shape):
        expected_chunks *= -(-extent // size)
    if num_chunks != expected_chunks:
        # Sparsely allocated dataset; let HDF5 supply the fill values.
        _fallback_read(dset, out)
        return

    pending = set()
    for i in range(num_chunks):
        offset = dset_id.get_chunk_info(i).chunk_offset
        filter_mask, raw_bytes = dset_id.read_direct_chunk(offset)

        if len(pending) >= max_pending:
            done, pending = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                future.result()

        pending.add(
            executor.submit(
                _write_chunk,
                out,
                offset,
                raw_bytes,
                filter_mask,
                pipeline,
                chunk_shape,
            )
        )

    for future in concurrent.futures.as_completed(pending):
        future.result()


def _max_pending(max_workers: int | None) -> int:
    """Compute how many raw chunks may queue up, given a worker count."""
    workers = max_workers or (os.cpu_count() or 1)
    return max(2 * workers, 4)


def read_dataset(
    dset: h5py.Dataset,
    *,
    out: np.ndarray | None = None,
    executor: concurrent.futures.ThreadPoolExecutor | None = None,
    max_workers: int | None = None,
) -> np.ndarray:
    """Read an already-open HDF5 dataset, decoding its chunks in parallel.

    Parameters
    ----------
    dset : h5py.Dataset
        The dataset to read.
    out : numpy.ndarray, optional
        Destination array, which must have the same shape and dtype as
        `dset`.  It may be a non-contiguous view (e.g. a slice of a larger
        array).  If not given, a new array is allocated.
    executor : concurrent.futures.ThreadPoolExecutor, optional
        Pool used to decode chunks.  If not given, a temporary pool is created
        and shut down before returning.
    max_workers : int, optional
        Size of the temporary thread pool.  Ignored when `executor` is given.

    Returns
    -------
    numpy.ndarray
        The dataset contents (same object as `out` when given).
    """
    if out is None:
        out = np.empty(dset.shape, dtype=dset.dtype)
    else:
        if tuple(out.shape) != tuple(dset.shape):
            raise ValueError(f"out has shape {out.shape}, expected {dset.shape}")
        if out.dtype != dset.dtype:
            raise ValueError(f"out has dtype {out.dtype}, expected {dset.dtype}")
    if executor is not None:
        workers = getattr(executor, "_max_workers", None)
        _load_dataset_into(dset, out, executor, _max_pending(workers))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            _load_dataset_into(dset, out, pool, _max_pending(max_workers))
    return out
