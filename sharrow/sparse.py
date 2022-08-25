import numba as nb
import numpy as np
import pandas as pd
import scipy.sparse
import sparse
import xarray as xr


@nb.njit
def _get_idx(indices, indptr, data, i, j):
    pool_lb = indptr[i]
    pool_ub = indptr[i + 1] - 1
    if pool_ub < pool_lb:
        # This indicates there are no values at all for row i
        return np.nan
    idx_lo = indices[pool_lb]
    idx_hi = indices[pool_ub]
    # check top and bottom
    if j == idx_lo:
        # The lower bound on possible j values is the j value, return it
        return data[pool_lb]
    elif j == idx_hi:
        # The upper bound on possible j values is the j value, return it
        return data[pool_ub]
    # check if out of original range
    elif j < idx_lo or j > idx_hi:
        # the j value is outside the possible range, there is no value to return
        return np.nan

    # The j value is somewhere inside the bounds, so conduct an efficient search to
    # see if we can find it.
    span = pool_ub - pool_lb - 1
    # assume the j values are uniformly distributed between bounds, guess at the
    # approximate location of the target j
    peek = (j - idx_lo) / (idx_hi - idx_lo)
    while span > 3:
        candidate = int(peek * span) + pool_lb
        if candidate <= pool_lb:
            candidate = pool_lb + 1
        if candidate >= pool_ub:
            candidate = pool_ub - 1
        # printd(f"{peek=}  {span=}  {pool_lb=}  {int(peek * span)=}  {candidate=}  ???")
        if j == indices[candidate]:
            # printd(f"{pool_lb=}  {pool_ub=}  {indices[pool_lb]=}  {indices[pool_ub]=}  {span=}  {i=}  {j=} *")
            return data[candidate]
        elif j < indices[candidate]:
            pool_ub = candidate
            idx_hi = indices[pool_ub]
            span = pool_ub - pool_lb - 1
            # printd(f"{pool_lb=}  {pool_ub=}  {indices[pool_lb]=}  {indices[pool_ub]=}  {span=}  {i=}  {j=} <")
        elif j > indices[candidate]:
            pool_lb = candidate
            idx_lo = indices[pool_lb]
            span = pool_ub - pool_lb - 1
            # printd(f"{pool_lb=}  {pool_ub=}  {indices[pool_lb]=}  {indices[pool_ub]=}  {span=}  {i=}  {j=} >")
        peek = (j - idx_lo) / (idx_hi - idx_lo)
    while pool_lb < pool_ub:
        pool_lb += 1
        if j == indices[pool_lb]:
            # printd(f"{pool_lb=}  {pool_ub=}  {indices[pool_lb]=}  {indices[pool_ub]=}  {span=}  {i=}  {j=} *")
            return data[pool_lb]
    # printd("no match")
    return np.nan


class SparseArray2D:
    def __init__(self, i, j, data, shape=None):
        if isinstance(data, scipy.sparse.csr_matrix):
            self._sparse_data = data
        else:
            self._sparse_data = scipy.sparse.coo_matrix(
                (data, (i, j)), shape=shape
            ).tocsr()
        self._sparse_data.sort_indices()

    def __getitem__(self, item):
        if isinstance(item, tuple) and len(item) == 2:
            if isinstance(item[0], int) and isinstance(item[1], int):
                return _get_idx(
                    self._sparse_data.indices,
                    self._sparse_data.indptr,
                    self._sparse_data.data,
                    item[0],
                    item[1],
                )
        return type(self)(None, None, self._sparse_data[item])


@xr.register_dataset_accessor("redirection")
class RedirectionAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def set(self, m2t, map_to, map_also=None, name=None):
        """
        Parameters
        ----------
        m2t : pandas.Series
            Mapping maz's to tazs
        """

        if name is None:
            name = f"redirect_{map_to}"

        mapper = {i: j for (j, i) in enumerate(self._obj[map_to].to_numpy())}
        if isinstance(m2t, pd.DataFrame) and m2t.shape[1] == 1:
            m2t = m2t.iloc[:, 0]
        if isinstance(m2t, pd.Series):
            m2t = xr.DataArray(m2t, dims=name)
        offsets = xr.apply_ufunc(np.vectorize(mapper.get), m2t)
        self._obj[f"_digitized_{map_to}_of_{name}"] = offsets
        self._obj.attrs[f"dim_redirection_{name}"] = map_to

        if map_also is not None:
            for i, j in map_also.items():
                self._obj[f"_digitized_{i}_of_{j}"] = offsets.rename({name: j})
                self._obj.attrs[f"dim_redirection_{j}"] = i

    def sparse_blender(
        self,
        name,
        i,
        j,
        data,
        shape=None,
        max_blend_distance=None,
        blend_distance_name=None,
        index=None,
        i_dim="omaz",
        j_dim="dmaz",
    ):
        i = np.asarray(i)
        j = np.asarray(j)
        if index is not None and shape is None:
            shape = (len(index), len(index))
        if index is not None:
            mapper = {i: j for (j, i) in enumerate(index)}

            @np.vectorize
            def apply_mapper(x):
                return mapper.get(x, -1)

            i_ = apply_mapper(i)
            j_ = apply_mapper(j)
        else:
            i_ = i
            j_ = j

        sparse_data = sparse.GCXS(
            sparse.COO((i_, j_), data, shape=shape), compressed_axes=(0,)
        )
        self._obj[f"_s_{name}"] = xr.DataArray(
            sparse_data,
            dims=(i_dim, j_dim),
        )
        if not max_blend_distance:
            max_blend_distance = np.inf
        self._obj.attrs[f"blend_{name}_max"] = max_blend_distance
        self._obj.attrs[f"blend_{name}_dist"] = blend_distance_name

    def is_blended(self, name):
        return f"_s_{name}" in self._obj
        # return (
        #     (f"_{name}_indices" in self._obj)
        #     and (f"_{name}_indptr" in self._obj)
        #     and (f"_{name}_data" in self._obj)
        # )

    def get_blended(self, name, backstop_values, i, j):
        return get_blended_2_arr(
            backstop_values,
            np.asarray(self._obj[f"_s_{name}"].data.indices),
            np.asarray(self._obj[f"_s_{name}"].data.indptr),
            np.asarray(self._obj[f"_s_{name}"].data.data),
            i,
            j,
            blend_limit=self._obj.attrs[f"blend_{name}_max"],
        )

    def target(self, name):
        return self._obj.attrs.get(f"dim_redirection_{name}", None)

    @property
    def blenders(self):
        b = {}
        for k, v in self._obj.attrs.items():
            if k.startswith("blend_") and k.endswith("_max"):
                name = k[6:-4]
                d = b.get(name, {})
                d["max_blend_distance"] = v
                b[name] = d
            if k.startswith("blend_") and k.endswith("_dist"):
                name = k[6:-5]
                d = b.get(name, {})
                d["blend_distance_name"] = v
                b[name] = d
        return b


@nb.njit
def isnan_fast_safe(x):
    if int(x) == -9223372036854775808:
        return True
    else:
        return False


@nb.njit
def get_blended_2(backstop_value, indices, indptr, data, i, j, blend_limit=np.inf):
    dtype = type(backstop_value)
    micro_v = dtype(_get_idx(indices, indptr, data, i, j))
    if isnan_fast_safe(micro_v) or micro_v > blend_limit:
        return backstop_value
    if blend_limit == np.inf:
        return micro_v
    macro_ratio = dtype(micro_v / blend_limit)
    return macro_ratio * backstop_value + (1 - macro_ratio) * micro_v


@nb.njit
def get_blended_2_arr(
    backstop_values_, indices, indptr, data, i_, j_, blend_limit=np.inf
):
    out = np.zeros_like(backstop_values_)
    for z in range(backstop_values_.size):
        out[z] = get_blended_2(
            backstop_values_[z],
            indices,
            indptr,
            data,
            i_[z],
            j_[z],
            blend_limit=blend_limit,
        )
    return out
