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
):
    """Convert one or more OMX-format HDF5 files into a Zarr store."""
    logger.info(f"reading metadata from {omx_filenames}")
    dataset = from_omx_3d(
        omx_filenames,
        index_names=index_names,
        indexes=indexes,
        time_periods=time_periods,
        time_period_sep=time_period_sep,
    )

    logger.info(f"writing to {zarr_directory}")
    for name in dataset.data_vars:
        logger.info(f" - {name}")
    dataset.to_zarr(zarr_directory, mode="a")
