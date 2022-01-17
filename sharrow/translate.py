import logging

import numpy as np
import pandas as pd
import xarray as xr
from larch import OMX

from sharrow.dataset import Dataset

from .dataset import one_based, zero_based

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

    bucket = {}

    r1 = r2 = None

    for omx_filename in omx_filenames:
        logger.info(f"reading metadata from {omx_filename}")

        omx = OMX(omx_filename)
        omx_data = omx.data
        omx_shape = omx.shape
        omx_lookup = omx.lookup

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

        bucket.update({i: omx.data[i] for i in omx.data._v_children})

    data_names = list(bucket.keys())

    logger.info(f"writing to {zarr_directory}")
    for k in data_names:
        logger.info(f" - {k}")
        ds = Dataset()
        if time_period_sep in k:
            base_k, time_k = k.split(time_period_sep, 1)
            ds[base_k] = xr.DataArray(
                np.float32(0),
                dims=index_names,
                coords={
                    index_names[0]: r1,
                    index_names[1]: r2,
                    index_names[2]: time_periods,
                },
            )
            ds[base_k].loc[:, :, time_k] = bucket[k][:]
        else:
            ds[k] = xr.DataArray(
                bucket[k][:],
                dims=index_names[:2],
                coords={
                    index_names[0]: r1,
                    index_names[1]: r2,
                },
            )
        ds.to_zarr(zarr_directory, mode="a")
