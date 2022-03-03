import dask.array as da
import numpy as np
import xarray as xr


def array_encode(
    x,
    missing_value=None,
    bitwidth=16,
    min_value=None,
    max_value=None,
    scale=None,
    offset=None,
    by_dict=False,
):
    """
    Encode a float DataArray into integers.

    Parameters
    ----------
    x : DataArray
        The original array to be encoded.
    missing_value : Numeric, optional
        If the current array has "missing" values encoded with something
        other that NaN, give that value here.
    bitwidth : {16, 8}
        Number of bits to use in the encoded integers.
    min_value, max_value : Numeric, optional
        Explicitly give the min and max values represented in the array.
        If not given, they are inferred from `x`. It is useful to give
        these values if x does not necessarily include all the values that
        might need to be inserted into `x` later.
    scale : Numeric, optional
        Explicitly give the scaling factor.  This is inferred from the
        min and max values if not provided.
    offset : Numeric, optional
        Explicitly give the offset factor.  This is inferred from the
        min value if not provided.
    by_dict : bool or {8, 16, 32}, optional
        Encode by dictionary, using this bitwidth, or set to `True`
        and give the `bitwidth` argument.  If given, all other
        arguments setting encoding parameters are ignored.

    Returns
    -------
    encoded_values : DataArray
    """
    x = array_decode(x)
    if by_dict:
        if by_dict is True:
            by_dict = bitwidth
        return digitize_by_dictionary(x, bitwidth=by_dict)
    if missing_value is not None:
        legit_values = x.where(x != missing_value)
    else:
        legit_values = x.where(~x.isnull())
    if min_value is None:
        min_value = float(legit_values.min())
    if offset is None:
        offset = min_value
    if max_value is None:
        max_value = float(legit_values.max())
    width = max_value - offset
    n = 1 << (bitwidth - 1)
    if scale is None:
        scale = width / n
    encoded_values = ((x - offset) / scale).astype(f"int{bitwidth}")
    encoded_values = encoded_values.where(x != missing_value, -1)
    encoded_values.attrs["digital_encoding"] = {
        "scale": scale,
        "offset": offset,
        "missing_value": missing_value,
    }
    return encoded_values


def array_decode(x, digital_encoding=None):
    if digital_encoding is None:
        if "digital_encoding" not in x.attrs:
            return x
        digital_encoding = x.attrs["digital_encoding"]
    dictionary = digital_encoding.get("dictionary", None)
    if dictionary is not None:
        result = x.copy()
        result.data = dictionary[x.to_numpy()]
        result.attrs.pop("digital_encoding", None)
        return result
    scale = digital_encoding.get("scale", 1)
    offset = digital_encoding.get("offset", 0)
    missing_value = digital_encoding.get("missing_value", None)
    result = x * scale + offset
    if missing_value is not None:
        result = result.where(x >= 0, missing_value)
    result.attrs.pop("digital_encoding", None)
    return result


def smash_bins(values, bincounts, final_width=255, preserve_minmax=True):
    big = 1 << 31
    z = n = bincounts.shape[0]
    if preserve_minmax:
        bincounts[0] = big
        bincounts[-1] = big

    while n > final_width:
        dissolve = np.argmin(bincounts)
        lower_neighbor = dissolve - 1
        while lower_neighbor >= 0 and bincounts[lower_neighbor] == big:
            lower_neighbor -= 1
        upper_neighbor = dissolve + 1
        while upper_neighbor < z and bincounts[upper_neighbor] == big:
            upper_neighbor += 1
        if lower_neighbor >= 0:
            lower_distance = values[dissolve] - values[lower_neighbor]
        else:
            lower_distance = np.inf
        if upper_neighbor < z:
            upper_distance = values[upper_neighbor] - values[dissolve]
        else:
            upper_distance = np.inf
        if lower_distance < upper_distance:
            bincounts[lower_neighbor] += bincounts[dissolve]
        else:
            bincounts[upper_neighbor] += bincounts[dissolve]
        bincounts[dissolve] = big
        n -= 1

    if preserve_minmax:
        bincounts[0] = 1
        bincounts[-1] = 1

    return values[bincounts < big]


def find_bins(values, final_width=255):
    u, b = np.unique(values, return_counts=True)
    return smash_bins(u, b, final_width=final_width)


def digitize_by_dictionary(arr, bitwidth=8):
    result = arr.copy()
    bins = find_bins(arr, final_width=1 << bitwidth)
    bin_edges = (bins[1:] - bins[:-1]) / 2 + bins[:-1]
    try:
        arr_data = arr.data
    except AttributeError:
        pass
    else:
        if isinstance(arr_data, da.Array):
            result.data = da.digitize(arr_data, bin_edges).astype(f"uint{bitwidth}")
            result.attrs["digital_encoding"] = {
                "dictionary": bins,
            }
            return result
    # fall back to numpy digitize
    result.data = np.digitize(arr, bin_edges).astype(f"uint{bitwidth}")
    result.attrs["digital_encoding"] = {
        "dictionary": bins,
    }
    return result


@xr.register_dataset_accessor("digital_encoding")
class _DigitalEncodings:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def info(self):
        """
        All digital_encoding attributes from Dataset variables.

        Returns
        -------
        dict
        """
        result = {}
        for k in self._obj.variables:
            k_attrs = self._obj._variables[k].attrs
            if "digital_encoding" in k_attrs:
                result[k] = k_attrs["digital_encoding"]
        return result

    def set(self, name, *args, **kwargs):
        """
        Digitally encode one or more variables in this dataset.

        All variables are encoded using the same given parameters.
        To encode various variables differently, make multiple calls
        to this function.

        Parameters
        ----------
        name : str or Collection[str]
            The name(s) of the variable to be encoded.
        missing_value : Numeric, optional
            If the current array has "missing" values encoded with something
            other that NaN, give that value here.
        bitwidth : {16, 8}
            Number of bits to use in the encoded integers.
        min_value, max_value : Numeric, optional
            Explicitly give the min and max values represented in the array.
            If not given, they are inferred from `x`. It is useful to give
            these values if x does not necessarily include all the values that
            might need to be inserted into `x` later.
        scale : Numeric, optional
            Explicitly give the scaling factor.  This is inferred from the
            min and max values if not provided.
        offset : Numeric, optional
            Explicitly give the offset factor.  This is inferred from the
            min value if not provided.
        by_dict : {8, 16, 32}, optional
            Encode by dictionary, using this bitwidth.  If given, all
            arguments other than this and `x` are ignored.

        Returns
        -------
        Dataset
            A copy of the dataset, with the named variable digitally encoded.
        """
        updates = {}
        if isinstance(name, str):
            updates[name] = array_encode(self._obj[name], *args, **kwargs)
        else:
            for n in name:
                updates[n] = array_encode(self._obj[n], *args, **kwargs)
        return self._obj.assign(updates)

    def strip(self, name):
        """
        Digitally decode one or more variables in this dataset.

        Parameters
        ----------
        name : str or Collection[str]
            The name(s) of the variable to be decoded.
        """
        updates = {}
        if isinstance(name, str):
            updates[name] = array_decode(self._obj[name])
        else:
            for n in name:
                updates[n] = array_decode(self._obj[n])
        return self._obj.assign(updates)
