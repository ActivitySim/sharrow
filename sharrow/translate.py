import logging

from .dataset import from_omx_3d

logger = logging.getLogger("sharrow.translate")


def omx_to_zarr(
    omx_filenames,
    zarr_directory,
    index_names=("otaz", "dtaz", "time_period"),
    indexes=None,
    *,
    time_periods=None,
    time_period_sep="__",
    max_float_precision=64,
):
    """Convert OMX matrices to format-2 Zarr without implicit float64 loss.

    ``max_float_precision`` defaults to 64 for conversion; pass 32 explicitly
    to reduce floating-point precision. Stored chunks contain at most 256 rows,
    256 columns, and one time period, so slice reads do not fetch whole variables.
    Input decoding still operates on one source matrix at a time.
    """
    logger.info(f"reading metadata from {omx_filenames}")
    dataset = from_omx_3d(
        omx_filenames,
        index_names=index_names,
        indexes=indexes,
        time_periods=time_periods,
        time_period_sep=time_period_sep,
        max_float_precision=max_float_precision,
        task_granularity="matrix",
    )
    # Set storage chunks independently of the loader's Dask task granularity.
    # Period-only chunking would still leave huge matrices as oversized chunks.
    dataset = dataset.chunk(
        {
            dim: 1 if dim == index_names[2] else max(1, min(256, size))
            for dim, size in dataset.sizes.items()
        }
    )

    logger.info(f"writing to {zarr_directory}")
    for name in dataset.data_vars:
        logger.info(f" - {name}")
    dataset.to_zarr(zarr_directory, mode="a", zarr_format=2)
