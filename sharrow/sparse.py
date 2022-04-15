import numba as nb
import numpy as np
import pandas as pd
import scipy.sparse
import xarray as xr


@nb.njit
def _get_idx(indices, indptr, data, i, j):
    pool_lb = indptr[i]
    pool_ub = indptr[i + 1] - 1
    idx_lo = indices[pool_lb]
    idx_hi = indices[pool_ub]
    # check top and bottom
    if j == idx_lo:
        # printd(f"{pool_lb=}  {pool_ub=}  {indices[pool_lb]=}  {indices[pool_ub]=}  !!!lo  {i=}  {j=}")
        return data[pool_lb]
    elif j == idx_hi:
        # printd(f"{pool_lb=}  {pool_ub=}  {indices[pool_lb]=}  {indices[pool_ub]=}  !!!hi  {i=}  {j=}")
        return data[pool_ub]
    # check if out of original range
    elif j < idx_lo or j > idx_hi:
        # printd(f"{pool_lb=}  {pool_ub=}  {indices[pool_lb]=}  {indices[pool_ub]=}  :(  {i=}  {j=}")
        return np.nan

    span = pool_ub - pool_lb - 1
    # printd(f"{pool_lb=}  {pool_ub=}  {indices[pool_lb]=}  {indices[pool_ub]=}  {span=}  {i=}  {j=}")
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
        self.blenders = {}

    def set(self, m2t, map_to, map_also=None, name=None):
        """
        Parameters
        ----------
        m2t : pandas.Series
            Mapping maz's to tazs
        """

        if name is None:
            name = f"redirect_{map_to}"
        dim_redirection = self._obj.attrs.get("dim_redirection", {})

        mapper = {i: j for (j, i) in enumerate(self._obj[map_to].to_numpy())}
        if isinstance(m2t, pd.DataFrame) and m2t.shape[1] == 1:
            m2t = m2t.iloc[:, 0]
        if isinstance(m2t, pd.Series):
            m2t = xr.DataArray(m2t, dims=name)
        offsets = xr.apply_ufunc(np.vectorize(mapper.get), m2t)
        self._obj[f"_digitized_{name}"] = offsets
        dim_redirection[map_to] = name
        self._obj.attrs[f"dim_redirection_{map_to}"] = name

        if map_also is not None:
            for i, j in map_also.items():
                self._obj[f"_digitized_{j}"] = offsets.rename({name: j})
                dim_redirection[i] = j
                self._obj.attrs[f"dim_redirection_{i}"] = j

    def sparse_blender(
        self,
        name,
        i,
        j,
        data,
        shape=None,
        max_blend_distance=None,
        blend_distance_name=None,
    ):
        sparse_data = scipy.sparse.coo_matrix((data, (i, j)), shape=shape).tocsr()
        sparse_data.sort_indices()
        self._obj[f"_{name}_indices"] = xr.DataArray(
            sparse_data.indices, dims=f"{name}_indices"
        )
        self._obj[f"_{name}_indptr"] = xr.DataArray(
            sparse_data.indptr, dims=f"{name}_indptr"
        )
        self._obj[f"_{name}_data"] = xr.DataArray(
            sparse_data.data, dims=f"{name}_indices"
        )
        if not max_blend_distance:
            max_blend_distance = np.inf
        self.blenders[name] = dict(
            max_blend_distance=max_blend_distance,
            blend_distance_name=blend_distance_name,
        )

    def is_blended(self, name):
        return (
            (f"_{name}_indices" in self._obj)
            and (f"_{name}_indptr" in self._obj)
            and (f"_{name}_data" in self._obj)
        )

    def target(self, name):
        return self._obj.attrs.get(f"dim_redirection_{name}", None)


@nb.njit
def get_blended_2(backstop_value, indices, indptr, data, i, j, blend_limit=np.inf):
    dtype = type(backstop_value)
    micro_v = dtype(_get_idx(indices, indptr, data, i, j))
    if np.isnan(micro_v) or micro_v > blend_limit:
        return backstop_value
    if blend_limit == np.inf:
        return micro_v
    macro_ratio = dtype(micro_v / blend_limit)
    return macro_ratio * backstop_value + (1 - macro_ratio) * micro_v
