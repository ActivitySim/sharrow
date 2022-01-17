import numba as nb
import numpy as np


@nb.njit(cache=True)
def piece(x, low_bound, high_bound=None):
    """
    Clip the values in an array.

    This function differs from the usual `numpy.clip`
    in that the result is shifted by `low_bound` if it is
    given, so that the valid result range is a contiguous
    block of non-negative values starting from 0.

    Parameters
    ----------
    x : array-like
        Array containing elements to clip
    low_bound, high_bound : scalar, array-like, or None
        Minimum and maximum values. If None, clipping is not
        performed on the corresponding edge. Both may be
        None, which results in a noop. Both are broadcast
        against `x`.

    Returns
    -------
    clipped_array : array-like
    """
    if low_bound is None:
        if high_bound is None:
            return x
        else:
            if x < high_bound:
                return x
            else:
                return high_bound
    else:
        if high_bound is None:
            if x > low_bound:
                return x - low_bound
            else:
                return 0.0
        else:
            if x > low_bound:
                if x < high_bound:
                    return x - low_bound
                else:
                    return high_bound - low_bound
            else:
                return 0.0


@nb.njit(cache=True)
def hard_sigmoid(x, zero_bound, one_bound):
    """
    Apply a piecewise linear sigmoid function.

    Parameters
    ----------
    x : array-like
        Array containing elements to clip
    zero_bound, one_bound : scalar, array-like, or None
        Inflection points of the piecewise linear sigmoid
        function.

    Returns
    -------
    clipped_array : array-like
    """
    if zero_bound < one_bound:
        if x <= zero_bound:
            return 0.0
        if x >= one_bound:
            return 1.0
        return (x - zero_bound) / (one_bound - zero_bound)
    else:
        if x >= zero_bound:
            return 0.0
        if x <= one_bound:
            return 1.0
        return (zero_bound - x) / (zero_bound - one_bound)


@nb.njit(cache=True)
def transpose_leading(j):
    if j.ndim == 2:
        return j.transpose()
    elif j.ndim == 3:
        return j.transpose(1, 0, 2)
    elif j.ndim == 4:
        return j.transpose(1, 0, 2, 3)
    elif j.ndim == 5:
        return j.transpose(1, 0, 2, 3, 4)
    else:
        raise ValueError("too many dims for transpose")


# @nb.njit(
#     [
#         (nb.float32, nb.optional(nb.float32), nb.optional(nb.float32)),
#         (nb.float64, nb.optional(nb.float64), nb.optional(nb.float64)),
#     ],
#     cache=True
# )
@nb.njit(cache=True)
def clip(a, lower=None, upper=None):
    # Applies the pandas keyword names for the numpy clip function.
    # TODO: use the numba implementation of np.clip https://github.com/numba/numba/pull/6808

    lower_is_none = lower is None
    upper_is_none = upper is None

    if lower_is_none:
        if upper_is_none:
            return a
        else:
            return np.minimum(a, upper)
    elif upper_is_none:
        return np.maximum(a, lower)
    else:
        return np.maximum(np.minimum(a, upper), lower)


@nb.njit(cache=True)
def digital_decode(encoded_value, scale, offset, missing_value):
    if encoded_value < 0:
        return missing_value
    else:
        return (encoded_value * scale) + offset
