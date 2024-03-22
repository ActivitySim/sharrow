from __future__ import annotations

from enum import IntEnum
from functools import reduce

import numpy as np
import pandas as pd
import xarray as xr


class ArrayIsNotCategoricalError(TypeError):
    """The array is not an encoded categorical array."""


@xr.register_dataarray_accessor("cat")
class _Categorical:
    """Accessor for pseudo-categorical arrays."""

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

    def is_same_categories(self, other: xr.DataArray) -> bool:
        """Check if the categories of two categorical arrays are the same.

        Parameters
        ----------
        other : xr.DataArray or categorical accessor
            The other array to compare.
        """
        if not self.is_categorical():
            return False
        if isinstance(other, xr.DataArray):
            if not other.cat.is_categorical():
                return False
            if not np.array_equal(self.categories, other.cat.categories):
                return False
            if self.ordered != other.cat.ordered:
                return False
            return True
        elif isinstance(other, _Categorical):
            return self.is_same_categories(other.dataarray)
        else:
            raise TypeError(f"cannot compare categorical array with {type(other)}")


def _interpret_enum(e: type[IntEnum], value: int | str) -> IntEnum:
    """
    Convert a string or integer into an Enum value.

    The

    Parameters
    ----------
    e : Type[IntEnum]
        The enum to use in interpretation.
    value: int or str
        The value to convert.  Integer and simple string values are converted
        to their corresponding value.  Multiple string values can also be given
        joined by the pipe operator, in the style of flags (e.g. "Red|Octagon").
    """
    if isinstance(value, int):
        return e(value)
    return reduce(lambda x, y: x | y, [getattr(e, v) for v in value.split("|")])


def get_enum_name(e: type[IntEnum], value: int) -> str:
    """
    Get the name of an enum by value, or a placeholder name if not found.

    This allows for dummy placeholder names is the enum is does not contain
    all consecutive values between 0 and the maximum value, inclusive.

    Parameters
    ----------
    e : Type[IntEnum]
        The enum to use in interpretation.
    value : int
        The value for which to find a name.  If not found in `e`, this
        function will generate a new name as a string by prefixing `value`
        with an underscore.

    Returns
    -------
    str
    """
    result = e._value2member_map_.get(value, f"_{value}")
    try:
        return result.name
    except AttributeError:
        return result


def int_enum_to_categorical_dtype(e: type[IntEnum]) -> pd.CategoricalDtype:
    """
    Convert an integer-valued enum to a pandas CategoricalDtype.

    Parameters
    ----------
    e : Type[IntEnum]

    Returns
    -------
    pd.CategoricalDtype
    """
    max_enum_value = int(max(e))
    categories = [get_enum_name(e, i) for i in range(max_enum_value + 1)]
    return pd.CategoricalDtype(categories=categories)


def as_int_enum(
    s: pd.Series,
    e: type[IntEnum],
    dtype: type[np.integer] | None = None,
    categorical: bool = True,
) -> pd.Series:
    """
    Encode a pandas Series as categorical, consistent with an IntEnum.

    Parameters
    ----------
    s : pd.Series
    e : Type[IntEnum]
    dtype : Type[np.integer], optional
        Specific dtype to use for the code point encoding.  It is typically not
        necessary to give this explicitly as the function will automatically
        select the best (most efficient) bitwidth.
    categorical : bool, default True
        If set to false, the returned series will simply be integer encoded with
        no formal Categorical dtype.

    Returns
    -------
    pd.Series
    """
    min_enum_value = int(min(e))
    max_enum_value = int(max(e))
    assert min_enum_value >= 0
    if dtype is None:
        if max_enum_value < 256 and min_enum_value >= 0:
            dtype = np.uint8
        elif max_enum_value < 128 and min_enum_value >= -128:
            dtype = np.int8
        elif max_enum_value < 65536 and min_enum_value >= 0:
            dtype = np.uint16
        elif max_enum_value < 32768 and min_enum_value >= -32768:
            dtype = np.int16
        elif max_enum_value < 2_147_483_648 and min_enum_value >= -2_147_483_648:
            dtype = np.int32
        else:
            dtype = np.int64
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    result = s.apply(lambda x: _interpret_enum(e, x)).astype(dtype)
    if categorical:
        categories = [get_enum_name(e, i) for i in range(max_enum_value + 1)]
        result = pd.Categorical.from_codes(codes=result, categories=categories)
    return result


@pd.api.extensions.register_series_accessor("as_int_enum")
class _AsIntEnum:
    """
    Encode a pandas Series as categorical, consistent with an IntEnum.

    Parameters
    ----------
    s : pd.Series
    e : Type[IntEnum]
    dtype : Type[np.integer], optional
        Specific dtype to use for the code point encoding.  It is typically not
        necessary to give this explicitly as the function will automatically
        select the best (most efficient) bitwidth.
    categorical : bool, default True
        If set to false, the returned series will simply be integer encoded with
        no formal Categorical dtype.

    Returns
    -------
    pd.Series
    """

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __call__(
        self: pd.Series,
        e: type[IntEnum],
        dtype: type[np.integer] | None = None,
        categorical: bool = True,
    ):
        return as_int_enum(self._obj, e, dtype, categorical)
