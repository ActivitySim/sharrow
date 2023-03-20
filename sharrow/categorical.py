from __future__ import annotations

import numpy as np
import xarray as xr


class ArrayIsNotCategoricalError(TypeError):
    """The array is not an encoded categorical array."""


@xr.register_dataarray_accessor("cat")
class _Categorical:
    """
    Accessor for pseudo-categorical arrays.
    """

    __slots__ = ("dataarray",)

    def __init__(self, dataarray: xr.DataArray):
        self.dataarray = dataarray

    @property
    def categories(self):
        try:
            return self.dataarray.attrs["digital_encoding"]["dictionary"]
        except KeyError:
            raise ArrayIsNotCategoricalError() from None

    @property
    def ordered(self):
        return self.dataarray.attrs["digital_encoding"].get("ordered", True)

    def category_array(self) -> np.ndarray:
        arr = np.asarray(self.categories)
        if arr.dtype.kind == "O":
            arr = arr.astype(str)
        return arr

    def is_categorical(self) -> bool:
        return "dictionary" in self.dataarray.attrs.get("digital_encoding", {})
