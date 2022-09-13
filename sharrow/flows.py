import ast
import base64
import hashlib
import importlib
import inspect
import io
import logging
import os
import re
import sys
import textwrap
import time
import warnings

import numba as nb
import numpy as np
import pandas as pd
import xarray as xr

from ._infer_version import __version__
from .aster import expression_for_numba, extract_all_name_tokens, extract_names_2
from .filewrite import blacken, rewrite
from .relationships import DataTree
from .table import Table

logger = logging.getLogger("sharrow")

well_known_names = {
    "nb",
    "np",
    "pd",
    "xr",
    "pa",
    "log",
    "exp",
    "log1p",
    "expm1",
    "max",
    "min",
    "piece",
    "hard_sigmoid",
    "transpose_leading",
    "clip",
}


def one_based(n):
    return pd.RangeIndex(1, n + 1)


def zero_based(n):
    return pd.RangeIndex(0, n)


def clean(s):
    """
    Convert any string into a similar python identifier.

    If any modification of the string is made, or if the string
    is longer than 120 characters, it is truncated and a hash of the
    original string is added to the end, to ensure every
    string maps to a unique cleaned name.

    Parameters
    ----------
    s : str

    Returns
    -------
    cleaned : str
    """
    if not isinstance(s, str):
        s = f"{type(s)}-{s}"
    cleaned = re.sub(r"\W|^(?=\d)", "_", s)
    if cleaned != s or len(cleaned) > 120:
        # digest size 15 creates a 24 character base32 string
        h = base64.b32encode(
            hashlib.blake2b(s.encode(), digest_size=15).digest()
        ).decode()
        cleaned = f"{cleaned[:90]}_{h}"
    return cleaned


def presorted(sortable, presort=None, exclude=None):
    """
    Sort a collection, with certain items appearing first.

    Parameters
    ----------
    sortable : Collection
        Elements to sort.
    presort : Iterable, optional
        Pre-sorted elements, which are yielded first, in this order,
        if they appear in `sortable`.

    Yields
    ------
    Any
        The elements of sortable.
    """
    queue = set(sortable)
    if presort is not None:
        for j in presort:
            if j in queue:
                if exclude is None or j not in exclude:
                    yield j
                queue.remove(j)
    for i in sorted(queue):
        if exclude is None or i not in exclude:
            yield i


def _flip_flop_def(v):
    if isinstance(v, str) and "# sharrow:" in v:
        return v.split("# sharrow:", 1)[1].strip()
    else:
        return v


well_known_names |= {
    "_args",
    "_inputs",
    "_outputs",
}
ARG_NAMES = {f"_arg{n:02}" for n in range(100)}
well_known_names |= ARG_NAMES


def filter_name_tokens(expr, matchable_names=None):
    name_tokens = extract_all_name_tokens(expr)
    arg_tokens = name_tokens & ARG_NAMES
    name_tokens -= well_known_names
    if matchable_names:
        name_tokens &= matchable_names
    return name_tokens, arg_tokens


def coerce_to_range_index(idx):
    if isinstance(idx, pd.RangeIndex):
        return idx
    if isinstance(idx, (pd.Int64Index, pd.Float64Index, pd.UInt64Index)):
        if idx.is_monotonic_increasing and idx[-1] - idx[0] == idx.size - 1:
            return pd.RangeIndex(idx[0], idx[0] + idx.size)
    return idx


FUNCTION_TEMPLATE = """

# {init_expr}
@nb.jit(cache=True, error_model='{error_model}', boundscheck={boundscheck}, nopython={nopython}, fastmath={fastmath})
def {fname}(
    {argtokens}
    _outputs,
    {nametokens}
):
    return {expr}

"""


IRUNNER_1D_TEMPLATE = """
@nb.jit(cache=True, parallel=True, error_model='{error_model}', boundscheck={boundscheck}, nopython={nopython}, fastmath={fastmath})
def irunner(
    argshape,
    {joined_namespace_names}
    dtype=np.{dtype},
):
    result = np.empty((argshape[0], {len_self_raw_functions}), dtype=dtype)
    if argshape[0] > 1000:
        for j0 in nb.prange(argshape[0]):
            linemaker(result[j0], j0, {joined_namespace_names})
    else:
        for j0 in range(argshape[0]):
            linemaker(result[j0], j0, {joined_namespace_names})
    return result
"""

IRUNNER_2D_TEMPLATE = """
@nb.jit(cache=True, parallel=True, error_model='{error_model}', boundscheck={boundscheck}, nopython={nopython}, fastmath={fastmath})
def irunner(
    argshape,
    {joined_namespace_names}
    dtype=np.{dtype},
):
    result = np.empty((argshape[0], argshape[1], {len_self_raw_functions}), dtype=dtype)
    if argshape[0] * argshape[1] > 1000:
        for j0 in nb.prange(argshape[0]):
          for j1 in range(argshape[1]):
            linemaker(result[j0, j1], j0, j1, {joined_namespace_names})
    else:
        for j0 in range(argshape[0]):
          for j1 in range(argshape[1]):
            linemaker(result[j0, j1], j0, j1, {joined_namespace_names})
    return result
"""

IDOTTER_1D_TEMPLATE = """
@nb.jit(cache=True, parallel=True, error_model='{error_model}', boundscheck={boundscheck}, nopython={nopython}, fastmath={fastmath})
def idotter(
    argshape,
    {joined_namespace_names}
    dtype=np.{dtype},
    dotarray=None,
):
    if dotarray is None:
        raise ValueError("dotarray cannot be None")
    assert dotarray.ndim == 2
    result = np.empty((argshape[0], dotarray.shape[1]), dtype=dtype)
    if argshape[0] > 1000:
        for j0 in nb.prange(argshape[0]):
            intermediate = np.zeros({len_self_raw_functions}, dtype=dtype)
            {meta_code_stack_dot}
            np.dot(intermediate, dotarray, out=result[j0,:])
    else:
        intermediate = np.zeros({len_self_raw_functions}, dtype=dtype)
        for j0 in range(argshape[0]):
            {meta_code_stack_dot}
            np.dot(intermediate, dotarray, out=result[j0,:])
    return result
"""

IDOTTER_2D_TEMPLATE = """
@nb.jit(cache=True, parallel=True, error_model='{error_model}', boundscheck={boundscheck}, nopython={nopython}, fastmath={fastmath})
def idotter(
    argshape,
    {joined_namespace_names}
    dtype=np.{dtype},
    dotarray=None,
):
    if dotarray is None:
        raise ValueError("dotarray cannot be None")
    assert dotarray.ndim == 2
    result = np.empty((argshape[0], argshape[1], dotarray.shape[1]), dtype=dtype)
    if argshape[0] > 1000:
        for j0 in nb.prange(argshape[0]):
          for j1 in range(argshape[1]):
            intermediate = np.zeros({len_self_raw_functions}, dtype=dtype)
            {meta_code_stack_dot}
            np.dot(intermediate, dotarray, out=result[j0,j1,:])
    else:
        intermediate = np.zeros({len_self_raw_functions}, dtype=dtype)
        for j0 in range(argshape[0]):
          for j1 in range(argshape[1]):
            {meta_code_stack_dot}
            np.dot(intermediate, dotarray, out=result[j0,j1,:])
    return result
"""

ILINER_1D_TEMPLATE = """
@nb.jit(cache=True, error_model='{error_model}', boundscheck={boundscheck}, nopython={nopython}, fastmath={fastmath})
def linemaker(
    intermediate, j0,
    {joined_namespace_names}
):
            {meta_code_stack_dot}

"""

ILINER_2D_TEMPLATE = """
@nb.jit(cache=True, error_model='{error_model}', boundscheck={boundscheck}, nopython={nopython}, fastmath={fastmath})
def linemaker(
    intermediate, j0, j1,
    {joined_namespace_names}
):
            {meta_code_stack_dot}

"""


MNL_GENERIC_TEMPLATE = """
@nb.jit(cache=True, error_model='{error_model}', boundscheck={boundscheck}, nopython={nopython}, fastmath={fastmath})
def _sample_choices_maker(
        prob_array,
        random_array,
        out_choices,
        out_choice_probs,
):
    '''
    Random sample of alternatives.

    Parameters
    ----------
    prob_array : array of float, shape (n_alts)
    random_array : array of float, shape (n_samples)
    out_choices : array of int, shape (n_samples) output
    out_choice_probs : array of float, shape (n_samples) output

    '''
    sample_size = random_array.size
    n_alts = prob_array.size

    random_points = np.sort(random_array)
    a = 0
    s = 0
    unique_s = 0
    z = 0.0
    for a in range(n_alts):
        z += prob_array[a]
        while s < sample_size and z > random_points[s]:
            out_choices[s] = a
            out_choice_probs[s] = prob_array[a]
            s += 1
        if s >= sample_size:
            break
    if s < sample_size:
        # rare condition, only if a random point is greater than 1 (a bug)
        # or if the sum of probabilities is less than 1 and a random point
        # is greater than that sum, which due to the limits of numerical
        # precision can technically happen
        a = n_alts-1
        while prob_array[a] < 1e-30 and a > 0:
            # slip back to the last choice with non-trivial prob
            a -= 1
        while s < sample_size:
            out_choices[s] = a
            out_choice_probs[s] = prob_array[a]
            s += 1



@nb.jit(cache=True, error_model='{error_model}', boundscheck={boundscheck}, nopython={nopython}, fastmath={fastmath})
def _sample_choices_maker_counted(
        prob_array,
        random_array,
        out_choices,
        out_choice_probs,
        out_pick_count,
):
    '''
    Random sample of alternatives.

    Parameters
    ----------
    prob_array : array of float, shape (n_alts)
    random_array : array of float, shape (n_samples)
    out_choices : array of int, shape (n_samples) output
    out_choice_probs : array of float, shape (n_samples) output
    out_pick_count : array of int, shape (n_samples) output

    '''
    sample_size = random_array.size
    n_alts = prob_array.size

    random_points = np.sort(random_array)
    a = 0
    s = 0
    unique_s = -1
    z = 0.0
    out_pick_count[:] = 0
    for a in range(n_alts):
        z += prob_array[a]
        if s < sample_size and z > random_points[s]:
            unique_s += 1
        while s < sample_size and z > random_points[s]:
            out_choices[unique_s] = a
            out_choice_probs[unique_s] = prob_array[a]
            out_pick_count[unique_s] += 1
            s += 1
        if s >= sample_size:
            break
    if s < sample_size:
        # rare condition, only if a random point is greater than 1 (a bug)
        # or if the sum of probabilities is less than 1 and a random point
        # is greater than that sum, which due to the limits of numerical
        # precision can technically happen
        a = n_alts-1
        while prob_array[a] < 1e-30 and a > 0:
            # slip back to the last choice with non-trivial prob
            a -= 1
        if out_choices[unique_s] != a:
            unique_s += 1
        while s < sample_size:
            out_choices[unique_s] = a
            out_choice_probs[unique_s] = prob_array[a]
            out_pick_count[unique_s] += 1
            s += 1

"""

MNL_1D_TEMPLATE = (
    MNL_GENERIC_TEMPLATE
    + """

@nb.jit(cache=True, parallel=True, error_model='{error_model}', boundscheck={boundscheck}, nopython={nopython}, fastmath={fastmath})
def mnl_transform(
    argshape,
    {joined_namespace_names}
    dtype=np.{dtype},
    dotarray=None,
    random_draws=None,
    pick_counted=False,
    logsums=False,
):
    if dotarray is None:
        raise ValueError("dotarray cannot be None")
    assert dotarray.ndim == 2
    result = np.full((argshape[0], random_draws.shape[1]), -1, dtype=np.int32)
    result_p = np.zeros((argshape[0], random_draws.shape[1]), dtype=dtype)
    if pick_counted:
        pick_count = np.zeros((argshape[0], random_draws.shape[1]), dtype=np.int32)
    else:
        pick_count = np.zeros((argshape[0], 0), dtype=np.int32)
    if logsums:
        _logsums = np.zeros((argshape[0], ), dtype=dtype)
    else:
        _logsums = np.zeros((0, ), dtype=dtype)
    if argshape[0] > 1000:
        for j0 in nb.prange(argshape[0]):
            intermediate = np.zeros({len_self_raw_functions}, dtype=dtype)
            {meta_code_stack_dot}
            partial = np.exp(np.dot(intermediate, dotarray))
            local_sum = np.sum(partial)
            partial /= local_sum
            if logsums:
                _logsums[j0] = np.log(local_sum)
            if pick_counted:
                _sample_choices_maker_counted(partial, random_draws[j0], result[j0], result_p[j0], pick_count[j0])
            else:
                _sample_choices_maker(partial, random_draws[j0], result[j0], result_p[j0])
    else:
        intermediate = np.zeros({len_self_raw_functions}, dtype=dtype)
        for j0 in range(argshape[0]):
            {meta_code_stack_dot}
            partial = np.exp(np.dot(intermediate, dotarray))
            local_sum = np.sum(partial)
            partial /= local_sum
            if logsums:
                _logsums[j0] = np.log(local_sum)
            if pick_counted:
                _sample_choices_maker_counted(partial, random_draws[j0], result[j0], result_p[j0], pick_count[j0])
            else:
                _sample_choices_maker(partial, random_draws[j0], result[j0], result_p[j0])
    return result, result_p, pick_count, _logsums

"""
)


MNL_2D_TEMPLATE = (
    MNL_GENERIC_TEMPLATE
    + """

@nb.jit(cache=True, parallel=True, error_model='{error_model}', boundscheck={boundscheck}, nopython={nopython}, fastmath={fastmath})
def mnl_transform(
    argshape,
    {joined_namespace_names}
    dtype=np.{dtype},
    dotarray=None,
    random_draws=None,
    pick_counted=False,
    logsums=False,
):
    if dotarray is None:
        raise ValueError("dotarray cannot be None")
    assert dotarray.ndim == 2
    assert dotarray.shape[1] == 1
    if random_draws is None:
        raise ValueError("random_draws cannot be None")
    assert random_draws.ndim == 2
    assert random_draws.shape[0] == argshape[0]

    result = np.full((argshape[0], random_draws.shape[1]), -1, dtype=np.int32)
    result_p = np.zeros((argshape[0], random_draws.shape[1]), dtype=dtype)
    if pick_counted:
        pick_count = np.zeros((argshape[0], random_draws.shape[1]), dtype=np.int32)
    else:
        pick_count = np.zeros((argshape[0], 0), dtype=np.int32)
    if logsums:
        _logsums = np.zeros((argshape[0], ), dtype=dtype)
    else:
        _logsums = np.zeros((0, ), dtype=dtype)
    if argshape[0] > 1000:
        for j0 in nb.prange(argshape[0]):
          partial = np.zeros(argshape[1], dtype=dtype)
          for j1 in range(argshape[1]):
            intermediate = np.zeros({len_self_raw_functions}, dtype=dtype)
            {meta_code_stack_dot}
            partial[j1] = np.exp(np.dot(intermediate, dotarray))[0]
          local_sum = np.sum(partial)
          if logsums:
            _logsums[j0] = np.log(local_sum)
          partial /= local_sum
          if pick_counted:
            _sample_choices_maker_counted(partial, random_draws[j0], result[j0], result_p[j0], pick_count[j0])
          else:
            _sample_choices_maker(partial, random_draws[j0], result[j0], result_p[j0])
    else:
        intermediate = np.zeros({len_self_raw_functions}, dtype=dtype)
        partial = np.zeros(argshape[1], dtype=dtype)
        for j0 in range(argshape[0]):
          for j1 in range(argshape[1]):
            {meta_code_stack_dot}
            partial[j1] = np.exp(np.dot(intermediate, dotarray))[0]
          local_sum = np.sum(partial)
          if logsums:
            _logsums[j0] = np.log(local_sum)
          partial /= local_sum
          if pick_counted:
            _sample_choices_maker_counted(partial, random_draws[j0], result[j0], result_p[j0], pick_count[j0])
          else:
            _sample_choices_maker(partial, random_draws[j0], result[j0], result_p[j0])
    return result, result_p, pick_count, _logsums
"""
)


class Flow:
    """
    A prepared data flow.

    Parameters
    ----------
    tree : DataTree
        The tree from whence the output will be constructed.
    defs : Mapping[str,str]
        Gives the names and definitions for the variables to create in the
        generated output.
    error_model : {'numpy', 'python'}, default 'numpy'
        The error_model option controls the divide-by-zero behavior. Setting
        it to ‘python’ causes divide-by-zero to raise exception like
        CPython. Setting it to ‘numpy’ causes divide-by-zero to set the
        result to +/-inf or nan.
    cache_dir : Path-like, optional
        A location to write out generated python and numba code. If not
        provided, a unique temporary directory is created.
    name : str, optional
        The name of this Flow used for writing out cached files. If not
        provided, a unique name is generated. If `cache_dir` is given,
        be sure to avoid name conflicts with other flow's in the same
        directory.
    dtype : str, default "float32"
        The name of the numpy dtype that will be used for the output.
    boundscheck : bool, default False
        If True, boundscheck enables bounds checking for array indices, and
        out of bounds accesses will raise IndexError. The default is to not
        do bounds checking, which is faster but can produce garbage results
        or segfaults if there are problems, so try turning this on for
        debugging if you are getting unexplained errors or crashes.
    nopython : bool, default True
        Compile using numba's `nopython` mode.  Provided for debugging only,
        as there's little point in turning this off for production code, as
        all the speed benefits of sharrow will be lost.
    fastmath : bool, default True
        If true, fastmath enables the use of "fast" floating point transforms,
        which can improve performance but can result in tiny distortions in
        results.  See numba docs for details.
    parallel : bool, default True
        Enable or disable parallel computation for certain functions.
    readme : str, optional
        A string to inject as a comment at the top of the flow Python file.
    flow_library : Mapping[str,Flow], optional
        An in-memory cache of precompiled Flow objects.  Using this can result
        in performance improvements when repeatedly using the same definitions.
    extra_hash_data : Tuple[Hashable], optional
        Additional data used for generating the flow hash.  Useful to prevent
        conflicts when using a flow_library with multiple similar flows.
    write_hash_audit : bool, default True
        Writes a hash audit log into a comment in the flow Python file, for
        debugging purposes.
    hashing_level : int, default 1
        Level of detail to write into flow hashes.  Increase detail to avoid
        hash conflicts for similar flows.

    """

    def __new__(
        cls,
        tree,
        defs,
        error_model="numpy",
        cache_dir=None,
        name=None,
        dtype="float32",
        boundscheck=False,
        nopython=True,
        fastmath=True,
        parallel=True,
        readme=None,
        flow_library=None,
        extra_hash_data=(),
        write_hash_audit=True,
        hashing_level=1,
        dim_order=None,
        dim_exclude=None,
    ):
        assert isinstance(tree, DataTree)
        tree.digitize_relationships(inplace=True)

        self = super().__new__(cls)
        # clean defs with hidden values
        defs = {k: _flip_flop_def(v) for k, v in defs.items()}

        # start init up to flow_hash
        self.__initialize_1(
            tree,
            defs,
            cache_dir=cache_dir,
            extra_hash_data=extra_hash_data,
            hashing_level=hashing_level,
            dim_order=dim_order,
            dim_exclude=dim_exclude,
            error_model=error_model,
            boundscheck=boundscheck,
            nopython=nopython,
            fastmath=fastmath,
        )
        # return from library if available
        if flow_library is not None and self.flow_hash in flow_library:
            logger.info(f"flow exists in library: {self.flow_hash}")
            result = flow_library[self.flow_hash]
            result.tree = tree
            return result
        # otherwise finish normal init
        self.__initialize_2(
            defs,
            error_model=error_model,
            name=name,
            dtype=dtype,
            boundscheck=boundscheck,
            nopython=nopython,
            fastmath=fastmath,
            readme=readme,
            parallel=parallel,
            extra_hash_data=extra_hash_data,
            write_hash_audit=write_hash_audit,
        )
        if flow_library is not None:
            flow_library[self.flow_hash] = self
        return self

    def __initialize_1(
        self,
        tree,
        defs,
        cache_dir=None,
        extra_hash_data=(),
        error_model="numpy",
        boundscheck=False,
        nopython=True,
        fastmath=True,
        hashing_level=1,
        dim_order=None,
        dim_exclude=None,
    ):
        """
        Initialize up to the flow_hash

        See main docstring for arguments.
        """
        if cache_dir is None:
            import tempfile

            self.temp_cache_dir = tempfile.TemporaryDirectory()
            self.cache_dir = self.temp_cache_dir.name
        else:
            self.cache_dir = cache_dir

        self.tree = tree
        self._raw_functions = {}
        self._secondary_flows = {}
        self.dim_order = dim_order
        self.dim_exclude = dim_exclude

        all_raw_names = set()
        all_name_tokens = set()
        for k, expr in defs.items():
            plain_names, attribute_pairs, subscript_pairs = extract_names_2(expr)
            all_raw_names |= plain_names
            if self.tree.root_node_name:
                all_raw_names |= attribute_pairs.get(self.tree.root_node_name, set())
                all_raw_names |= subscript_pairs.get(self.tree.root_node_name, set())

        dimensions_ordered = presorted(self.tree.dims, self.dim_order, self.dim_exclude)
        index_slots = {i: n for n, i in enumerate(dimensions_ordered)}
        self.arg_name_positions = index_slots
        self.arg_names = dimensions_ordered
        self.output_name_positions = {}

        self._used_extra_vars = {}
        if self.tree.extra_vars:
            for k, v in self.tree.extra_vars.items():
                if k in all_raw_names:
                    self._used_extra_vars[k] = v

        self._used_extra_funcs = set()
        if self.tree.extra_funcs:
            for f in self.tree.extra_funcs:
                if f.__name__ in all_raw_names:
                    self._used_extra_funcs.add(f.__name__)

        self._used_aux_vars = []
        for aux_var in self.tree.aux_vars:
            if aux_var in all_raw_names:
                self._used_aux_vars.append(aux_var)

        self._hashing_level = hashing_level
        if self._hashing_level > 1:
            func_code, all_name_tokens = self.init_sub_funcs(
                defs,
                error_model=error_model,
                boundscheck=boundscheck,
                nopython=nopython,
                fastmath=fastmath,
            )
            self._func_code = func_code
            self._namespace_names = sorted(all_name_tokens)
        else:
            self._func_code = None
            self._namespace_names = None

        self.encoding_dictionaries = {}

        # compute the complete hash including defs, used_extra_vars, and namespace_names
        # digest size 20 creates a base32 encoded 32 character flow_hash string
        flow_hash = hashlib.blake2b(digest_size=20)
        flow_hash_audit = []

        def _flow_hash_push(x):
            nonlocal flow_hash, flow_hash_audit
            y = str(x)
            flow_hash.update(y.encode("utf8"))
            flow_hash_audit.append(y.replace("\n", "\n#    "))

        _flow_hash_push("---DataTree Flow---")
        for k, v in defs.items():
            _flow_hash_push(k)
            _flow_hash_push(v)
        for k in sorted(self._used_extra_vars):
            v = self._used_extra_vars[k]
            _flow_hash_push(k)
            _flow_hash_push(v)
        for k in sorted(self._used_aux_vars):
            _flow_hash_push(f"aux_var:{k}")
        for k in sorted(self._used_extra_funcs):
            _flow_hash_push(f"func:{k}")
        _flow_hash_push("---DataTree---")
        for k in self.arg_names:
            _flow_hash_push(f"arg:{k}")
        for k in self.tree._hash_features():
            if self._hashing_level > 0 or True:  # or not k.startswith("relationship:"):
                _flow_hash_push(k)
        if self.dim_order:
            _flow_hash_push("---dim-order---")
            for k in self.dim_order:
                _flow_hash_push(k)
        for sname, sdata in self.tree.subspaces_iter():
            digital_encoding_hashes = set()
            for iname, idata in sdata.digital_encoding.info().items():
                digital_encoding_hashes.add(f"digital_encoding:{sname}:{iname}:{idata}")
            # ensure these are hashed in a stable ordering
            for ihash in sorted(digital_encoding_hashes):
                _flow_hash_push(ihash)
        if self._hashing_level > 1:
            for k in sorted(self._namespace_names):
                if k.startswith("__base__"):
                    continue
                _flow_hash_push(k)
                parts = k.split("__")
                if len(parts) > 2:
                    try:
                        digital_encoding = self.tree.subspaces[parts[1]][
                            "__".join(parts[2:])
                        ].attrs["digital_encoding"]
                    except (AttributeError, KeyError):
                        pass
                    else:
                        if digital_encoding:
                            for de_k in sorted(digital_encoding.keys()):
                                de_v = digital_encoding[de_k]
                                if de_k == "dictionary":
                                    self.encoding_dictionaries[k] = de_v
                                _flow_hash_push((k, "digital_encoding", de_k, de_v))
        for k in extra_hash_data:
            _flow_hash_push(k)

        _flow_hash_push(f"boundscheck={boundscheck}")
        _flow_hash_push(f"error_model={error_model}")
        _flow_hash_push(f"fastmath={fastmath}")

        self.flow_hash = base64.b32encode(flow_hash.digest()).decode()
        self.flow_hash_audit = "]\n# [".join(flow_hash_audit)

    def _index_slots(self):
        return {
            i: n
            for n, i in enumerate(
                presorted(self.tree.dims, self.dim_order, self.dim_exclude)
            )
        }

    def init_sub_funcs(
        self,
        defs,
        error_model="numpy",
        boundscheck=False,
        nopython=True,
        fastmath=True,
    ):
        func_code = ""
        all_name_tokens = set()
        index_slots = {
            i: n
            for n, i in enumerate(
                presorted(self.tree.dims, self.dim_order, self.dim_exclude)
            )
        }
        self.arg_name_positions = index_slots
        candidate_names = self.tree.namespace_names()
        candidate_names |= set(f"__aux_var__{i}" for i in self.tree.aux_vars.keys())

        meta_data = {}

        if self.tree.relationships_are_digitized:
            for spacename, spacearrays in self.tree.subspaces.items():
                dim_slots = {}
                spacekeys = list(spacearrays.keys()) + list(spacearrays.coords.keys())
                for k1 in spacekeys:
                    try:
                        spacearrays_vars = spacearrays._variables
                    except AttributeError:
                        spacearrays_vars = spacearrays
                    try:
                        toks, blends = self.tree._arg_tokenizer(
                            spacename,
                            spacearray=spacearrays_vars[k1],
                            spacearrayname=k1,
                            exclude_dims=self.dim_exclude,
                        )
                    except ValueError:
                        pass
                    else:
                        dim_slots[k1] = toks
                try:
                    digital_encodings = spacearrays.digital_encoding.info()
                except AttributeError:
                    digital_encodings = {}
                blenders = spacearrays.redirection.blenders
                meta_data[spacename] = (dim_slots, digital_encodings, blenders)
        else:
            for spacename, spacearrays in self.tree.subspaces.items():
                dim_slots = {}
                spacekeys = list(spacearrays.keys()) + list(spacearrays.coords.keys())
                for k1 in spacekeys:
                    try:
                        _dims = spacearrays._variables[k1].dims
                    except AttributeError:
                        _dims = spacearrays[k1].dims
                    dim_slots[k1] = [index_slots[z] for z in _dims]
                try:
                    digital_encodings = spacearrays.digital_encoding.info()
                except AttributeError:
                    digital_encodings = {}
                blenders = spacearrays.redirection.blenders
                meta_data[spacename] = (dim_slots, digital_encodings, blenders)

        # write individual function files for each expression
        for n, (k, expr) in enumerate(defs.items()):
            expr = str(expr).lstrip()
            prior_expr = init_expr = expr
            other_way = True
            while other_way:
                other_way = False
                # if other_way is triggered, there may be residual other terms
                # that were not addressed, so this loop should be applied again.
                for spacename, spacearrays in self.tree.subspaces.items():
                    dim_slots, digital_encodings, blenders = meta_data[spacename]
                    try:
                        expr = expression_for_numba(
                            expr,
                            spacename,
                            dim_slots,
                            dim_slots,
                            digital_encodings=digital_encodings,
                            extra_vars=self.tree.extra_vars,
                            blenders=blenders,
                        )
                    except KeyError as key_err:
                        if ".." in key_err.args[0]:
                            topkey, attrkey = key_err.args[0].split("..")
                        else:
                            raise
                        # check if we can resolve this name on any other subspace
                        for other_spacename in self.tree.subspace_fallbacks.get(
                            topkey, []
                        ):
                            dim_slots, digital_encodings, blenders = meta_data[
                                other_spacename
                            ]
                            try:
                                expr = expression_for_numba(
                                    expr,
                                    spacename,
                                    dim_slots,
                                    dim_slots,
                                    digital_encodings=digital_encodings,
                                    prefer_name=other_spacename,
                                    extra_vars=self.tree.extra_vars,
                                    blenders=blenders,
                                )
                            except KeyError as err:  # noqa: F841
                                pass
                            else:
                                other_way = True
                                # at least one variable was found in a fallback
                                break
                        if not other_way:
                            raise
                if prior_expr == expr:
                    # nothing was changed, break out of loop
                    break
                else:
                    # something was changed, run the loop again to confirm
                    # nothing else needs to change
                    prior_expr = expr

            # now find instances where an identifier is previously created in this flow.
            expr = expression_for_numba(
                expr,
                "",
                (),
                self.output_name_positions,
                "_outputs",
                extra_vars=self.tree.extra_vars,
            )

            aux_tokens = {
                k: ast.parse(f"__aux_var__{k}", mode="eval").body
                for k in self.tree.aux_vars.keys()
            }

            # now handle aux vars
            expr = expression_for_numba(
                expr,
                "",
                (),
                spacevars=aux_tokens,
                prefer_name="aux_var",
                extra_vars=self.tree.extra_vars,
            )

            if (k == init_expr) and (init_expr == expr) and k.isidentifier():
                logger.error(f"unable to rewrite '{k}' to itself")
                raise ValueError(f"unable to rewrite '{k}' to itself")
            logger.debug(f"[{k}] rewrite {init_expr} -> {expr}")
            if not candidate_names:
                raise ValueError("there are no candidate namespace names loaded")
            f_name_tokens, f_arg_tokens = filter_name_tokens(expr, candidate_names)
            all_name_tokens |= f_name_tokens
            argtokens = sorted(f_arg_tokens)
            argtokens_ = ", ".join(argtokens)
            if argtokens_:
                argtokens_ += ", "
            func_code += FUNCTION_TEMPLATE.format(
                expr=expr,
                fname=clean(k),
                argtokens=argtokens_,
                nametokens=", ".join(sorted(f_name_tokens)),
                error_model=error_model,
                extra_imports="\n".join(
                    [
                        "from .extra_funcs import *" if self.tree.extra_funcs else "",
                        "from .extra_vars import *" if self._used_extra_vars else "",
                    ]
                ),
                boundscheck=boundscheck,
                nopython=nopython,
                fastmath=fastmath,
                init_expr=init_expr if k == init_expr else f"{k}: {init_expr}",
            )
            self._raw_functions[k] = (init_expr, expr, f_name_tokens, argtokens)
            self.output_name_positions[k] = n

        return func_code, all_name_tokens

    def __initialize_2(
        self,
        defs,
        error_model="numpy",
        name=None,
        dtype="float32",
        boundscheck=False,
        nopython=True,
        fastmath=True,
        readme=None,
        parallel=True,
        extra_hash_data=(),
        write_hash_audit=True,
    ):
        """

        Parameters
        ----------
        tree : DataTree
        defs : Dict[str,str]
            Gives the names and definitions for the columns to create in our
            generated table.
        error_model : {'numpy', 'python'}, default 'numpy'
            The error_model option controls the divide-by-zero behavior. Setting
            it to ‘python’ causes divide-by-zero to raise exception like
            CPython. Setting it to ‘numpy’ causes divide-by-zero to set the
            result to +/-inf or nan.
        cache_dir : Path-like, optional
            A location to write out generated python and numba code. If not
            provided, a unique temporary directory is created.
        name : str, optional
            The name of this Flow used for writing out cached files. If not
            provided, a unique name is generated. If `cache_dir` is given,
            be sure to avoid name conflicts with other flow's in the same
            directory.
        """

        if self._hashing_level <= 1:
            func_code, all_name_tokens = self.init_sub_funcs(
                defs,
                error_model=error_model,
                boundscheck=boundscheck,
                nopython=nopython,
                fastmath=fastmath,
            )
            self._func_code = func_code
            self._namespace_names = sorted(all_name_tokens)
            for k in sorted(self._namespace_names):
                if k.startswith("__base__"):
                    continue
                parts = k.split("__")
                if len(parts) > 2:
                    try:
                        digital_encoding = self.tree.subspaces[parts[1]][
                            "__".join(parts[2:])
                        ].attrs["digital_encoding"]
                    except (AttributeError, KeyError):
                        pass
                    else:
                        if digital_encoding:
                            for de_k in sorted(digital_encoding.keys()):
                                de_v = digital_encoding[de_k]
                                if de_k == "dictionary":
                                    self.encoding_dictionaries[k] = de_v

        # assign flow name based on hash unless otherwise given
        if name is None:
            name = f"flow_{self.flow_hash}"
        self.name = name

        # create the package directory for the flow if it does not exist
        os.makedirs(self.cache_dir, exist_ok=True)

        # if an existing __init__ file matches the hash, just use it
        init_file = os.path.join(self.cache_dir, self.name, "__init__.py")
        if os.path.isfile(init_file):
            with open(init_file, "rt") as f:
                content = f.read()
            s = re.search("""flow_hash = ['"](.*)['"]""", content)
        else:
            s = None
        if s and s.group(1) == self.flow_hash:
            logger.info(f"using existing flow code {self.flow_hash}")
            writing = False
        else:
            logger.info(f"writing fresh flow code {self.flow_hash}")
            writing = True

        if writing:
            dependencies = {
                "import numpy as np",
                "import numba as nb",
                "import pandas as pd",
                "import pyarrow as pa",
                "import xarray as xr",
                "import sharrow as sh",
                "import inspect",
                "import warnings",
                "from contextlib import suppress",
                "from numpy import log, exp, log1p, expm1",
                "from sharrow.maths import piece, hard_sigmoid, transpose_leading, clip, digital_decode",
                "from sharrow.sparse import get_blended_2",
            }

            func_code = self._func_code

            # write extra_funcs file, if there are any extra_funcs
            if self.tree.extra_funcs:
                try:
                    import cloudpickle as pickle
                except ModuleNotFoundError:
                    import pickle
                func_code += "\n\n# extra_funcs\n"
                for x_func in self.tree.extra_funcs:
                    if x_func.__name__ in self._used_extra_funcs:
                        if x_func.__module__ == "__main__":
                            dependencies.add("import pickle")
                            func_code += f"\n\n{x_func.__name__} = pickle.loads({repr(pickle.dumps(x_func))})\n"
                        else:
                            func_code += f"\n\nfrom {x_func.__module__} import {x_func.__name__}\n"

            # write extra_vars file, if there are any used extra_vars
            if self._used_extra_vars:
                try:
                    import cloudpickle as pickle
                except ModuleNotFoundError:
                    import pickle
                buffer = io.StringIO()
                # any_pickle = False
                for x_name, x_var in self._used_extra_vars.items():
                    if isinstance(x_var, (float, int, str)):
                        buffer.write(f"{x_name} = {x_var!r}\n")
                    else:
                        buffer.write(
                            f"{x_name} = pickle.loads({repr(pickle.dumps(x_var))})\n"
                        )
                        dependencies.add("import pickle")
                with io.StringIO() as x_code:
                    x_code.write("\n")
                    x_code.write(buffer.getvalue())
                    func_code += "\n\n# extra_vars\n"
                    func_code += x_code.getvalue()

            # write encoding dictionaries, if there are any used
            if len(self.encoding_dictionaries):
                dependencies.add("import pickle")
                try:
                    import cloudpickle as pickle
                except ModuleNotFoundError:
                    import pickle
                buffer = io.StringIO()
                for x_name, x_dict in self.encoding_dictionaries.items():
                    buffer.write(
                        f"__encoding_dict{x_name} = pickle.loads({repr(pickle.dumps(x_dict))})\n"
                    )
                with io.StringIO() as x_code:
                    x_code.write("\n")
                    x_code.write(buffer.getvalue())
                    func_code += "\n\n# encoding dictionaries\n"
                    func_code += x_code.getvalue()

            # write the master module for this flow
            os.makedirs(os.path.join(self.cache_dir, self.name), exist_ok=True)
            with rewrite(
                os.path.join(self.cache_dir, self.name, "__init__.py"), "wt"
            ) as f_code:

                f_code.write(
                    textwrap.dedent(
                        f"""
                # this module generated automatically using sharrow version {__version__}
                # generation time: {time.strftime('%d %B %Y %I:%M:%S %p')}
                """
                    )[1:]
                )

                if readme:
                    f_code.write(
                        textwrap.indent(
                            textwrap.dedent(readme),
                            "# ",
                            lambda line: True,
                        )
                    )
                    f_code.write("\n\n")

                dependencies_ = set()
                for depend in sorted(dependencies):
                    if depend.startswith("import ") and "." not in depend:
                        f_code.write(f"{depend}\n")
                        dependencies_.add(depend)
                dependencies -= dependencies_
                for depend in sorted(dependencies):
                    if depend.startswith("import "):
                        f_code.write(f"{depend}\n")
                        dependencies_.add(depend)
                dependencies -= dependencies_
                for depend in sorted(dependencies):
                    if depend.startswith("from ") and "from ." not in depend:
                        f_code.write(f"{depend}\n")
                        dependencies_.add(depend)
                dependencies -= dependencies_
                for depend in sorted(dependencies):
                    f_code.write(f"{depend}\n")

                f_code.write("\n\n# namespace names\n")
                for k in sorted(self._namespace_names):
                    f_code.write(f"# - {k}\n")

                if extra_hash_data:
                    f_code.write("\n\n# extra_hash_data\n")
                    for k in extra_hash_data:
                        f_code.write(f"# - {str(k)}\n")

                f_code.write("\n\n# function code\n")
                f_code.write(f"\n\n{blacken(func_code)}")

                f_code.write("\n\n# machinery code\n\n")

                if self.tree.relationships_are_digitized:

                    root_dims = list(
                        presorted(
                            self.tree.root_dataset.dims,
                            self.dim_order,
                            self.dim_exclude,
                        )
                    )
                    n_root_dims = len(root_dims)

                    if n_root_dims == 1:
                        js = "j0"
                    elif n_root_dims == 2:
                        js = "j0, j1"
                    else:
                        raise NotImplementedError(
                            f"n_root_dims only supported up to 2, not {n_root_dims}"
                        )

                    meta_code = []
                    meta_code_dot = []
                    for n, k in enumerate(self._raw_functions):
                        f_name_tokens = self._raw_functions[k][2]
                        f_arg_tokens = self._raw_functions[k][3]
                        f_name_tokens = ", ".join(sorted(f_name_tokens))
                        f_args_j = ", ".join([f"j{argn[-1]}" for argn in f_arg_tokens])
                        if f_args_j:
                            f_args_j += ", "
                        meta_code.append(
                            f"result[{js}, {n}] = ({clean(k)}({f_args_j}result[{js}], {f_name_tokens})).item()"
                        )
                        meta_code_dot.append(
                            f"intermediate[{n}] = ({clean(k)}({f_args_j}intermediate, {f_name_tokens})).item()"
                        )
                    meta_code_stack = textwrap.indent(
                        "\n".join(meta_code), " " * 12
                    ).lstrip()
                    meta_code_stack_dot = textwrap.indent(
                        "\n".join(meta_code_dot), " " * 12
                    ).lstrip()
                    len_self_raw_functions = len(self._raw_functions)
                    joined_namespace_names = "\n    ".join(
                        f"{nn}," for nn in self._namespace_names
                    )
                    linefeed = "\n                           "
                    if not meta_code_stack_dot:
                        meta_code_stack_dot = "pass"
                    if n_root_dims == 1:
                        meta_template = IRUNNER_1D_TEMPLATE.format(**locals()).format(
                            **locals()
                        )
                        meta_template_dot = IDOTTER_1D_TEMPLATE.format(
                            **locals()
                        ).format(**locals())
                        line_template = ILINER_1D_TEMPLATE.format(**locals()).format(
                            **locals()
                        )
                        mnl_template = MNL_1D_TEMPLATE.format(**locals()).format(
                            **locals()
                        )
                    elif n_root_dims == 2:
                        meta_template = IRUNNER_2D_TEMPLATE.format(**locals()).format(
                            **locals()
                        )
                        meta_template_dot = IDOTTER_2D_TEMPLATE.format(
                            **locals()
                        ).format(**locals())
                        line_template = ILINER_2D_TEMPLATE.format(**locals()).format(
                            **locals()
                        )
                        mnl_template = MNL_2D_TEMPLATE.format(**locals()).format(
                            **locals()
                        )
                    else:
                        raise ValueError(f"invalid n_root_dims {n_root_dims}")

                else:

                    raise RuntimeError("digitization is now required")

                f_code.write(blacken(textwrap.dedent(line_template)))
                f_code.write("\n\n")
                f_code.write(blacken(textwrap.dedent(mnl_template)))
                f_code.write("\n\n")
                f_code.write(blacken(textwrap.dedent(meta_template)))
                f_code.write("\n\n")
                f_code.write(blacken(textwrap.dedent(meta_template_dot)))
                f_code.write("\n\n")
                f_code.write(blacken(self._spill(self._namespace_names)))
                if write_hash_audit:
                    f_code.write("\n\n# hash audit\n# [")
                    f_code.write(self.flow_hash_audit)
                    f_code.write("]\n")
                f_code.write("\n\n")
                f_code.write(
                    "# Greetings, tinkerer!  The `flow_hash` included here is a safety \n"
                    "# measure to prevent unknowing users creating a mess by modifying \n"
                    "# the code in this module so that it no longer matches the expected \n"
                    "# variable definitions. If you want to modify this code, you should \n"
                    "# delete this hash to allow the code to run without any checks, but \n"
                    "# you do so at your own risk. \n"
                )
                f_code.write(f"flow_hash = {self.flow_hash!r}\n")

        if str(self.cache_dir) not in sys.path:
            logger.debug(f"inserting {self.cache_dir} into sys.path")
            sys.path.insert(0, str(self.cache_dir))
        importlib.invalidate_caches()
        logger.debug(f"importing {self.name}")
        module = importlib.import_module(self.name)
        sys.path = sys.path[1:]
        self._runner = getattr(module, "runner", None)
        self._dotter = getattr(module, "dotter", None)
        self._irunner = getattr(module, "irunner", None)
        self._imnl = getattr(module, "mnl_transform", None)
        self._idotter = getattr(module, "idotter", None)
        if not writing:
            self.function_names = module.function_names
            self.output_name_positions = module.output_name_positions

    def load_raw(self, rg, args, runner=None, dtype=None, dot=None):
        assert isinstance(rg, DataTree)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=nb.NumbaExperimentalFeatureWarning
            )
            assembled_args = [args.get(k) for k in self.arg_name_positions.keys()]
            for aa in assembled_args:
                if aa.dtype.kind != "i":
                    warnings.warn("position arguments are not all integers")
            try:
                if runner is None:
                    if dot is None:
                        runner_ = self._runner
                    else:
                        runner_ = self._dotter
                else:
                    runner_ = runner
                named_args = inspect.getfullargspec(runner_.py_func).args
                arguments = []
                for arg in named_args:
                    if arg in {"dtype", "dotarray", "inputarray", "argarray"}:
                        continue
                    if arg.startswith("_arg"):
                        continue
                    arg_value = rg.get_named_array(arg)
                    # aux_vars get passed through as is, not forced to be arrays
                    if arg.startswith("__aux_var"):
                        arguments.append(arg_value)
                    else:
                        arguments.append(np.asarray(arg_value))
                kwargs = {}
                if dtype is not None:
                    kwargs["dtype"] = dtype
                # else:
                #     kwargs['dtype'] = np.float64
                if dot is not None:
                    kwargs["dotarray"] = dot
                # logger.debug(f"load_raw calling runner with {assembled_args.shape=}, {assembled_inputs.shape=}")
                return runner_(*assembled_args, *arguments, **kwargs)
            except nb.TypingError as err:
                _raw_functions = getattr(self, "_raw_functions", {})
                logger.error(f"nb.TypingError in {len(_raw_functions)} functions")
                for k, v in _raw_functions.items():
                    logger.error(f"{k} = {v[0]} = {v[1]}")
                if "NameError:" in err.args[0]:
                    import re

                    problem = re.search("NameError: (.*)\x1b", err.args[0])
                    if problem:
                        raise NameError(problem.group(1)) from err
                    problem = re.search("NameError: (.*)\n", err.args[0])
                    if problem:
                        raise NameError(problem.group(1)) from err
                raise
            except KeyError as err:
                # raise the inner key error which is more helpful
                context = getattr(err, "__context__", None)
                if context:
                    raise context
                else:
                    raise err

    def iload_raw(
        self,
        rg,
        runner=None,
        dtype=None,
        dot=None,
        mnl=None,
        pick_counted=False,
        logsums=False,
    ):
        assert isinstance(rg, DataTree)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=nb.NumbaExperimentalFeatureWarning
            )
            try:
                if runner is None:
                    if mnl is not None:
                        runner_ = self._imnl
                    elif dot is None:
                        runner_ = self._irunner
                    else:
                        runner_ = self._idotter
                else:
                    runner_ = runner
                try:
                    named_args = inspect.getfullargspec(runner_.py_func).args
                except AttributeError:
                    named_args = inspect.getfullargspec(runner_).args
                arguments = []
                for arg in named_args:
                    if arg in {
                        "dtype",
                        "dotarray",
                        "argshape",
                        "random_draws",
                        "pick_counted",
                        "logsums",
                    }:
                        continue
                    argument = rg.get_named_array(arg)
                    # aux_vars get passed through as is, not forced to be arrays
                    if arg.startswith("__aux_var"):
                        arguments.append(argument)
                    else:
                        if argument.dtype.kind == "O":
                            argument = argument.astype("unicode")
                        arguments.append(np.asarray(argument))
                kwargs = {}
                if dtype is not None:
                    kwargs["dtype"] = dtype
                if dot is not None:
                    kwargs["dotarray"] = np.asarray(dot)
                if mnl is not None:
                    kwargs["random_draws"] = mnl
                    kwargs["pick_counted"] = pick_counted
                    kwargs["logsums"] = logsums
                tree_root_dims = rg.root_dataset.dims
                argshape = [
                    tree_root_dims[i]
                    for i in presorted(tree_root_dims, self.dim_order, self.dim_exclude)
                ]
                return runner_(np.asarray(argshape), *arguments, **kwargs)
            except nb.TypingError as err:
                _raw_functions = getattr(self, "_raw_functions", {})
                logger.error(f"nb.TypingError in {len(_raw_functions)} functions")
                for k, v in _raw_functions.items():
                    logger.error(f"{k} = {v[0]} = {v[1]}")
                if "NameError:" in err.args[0]:
                    import re

                    problem = re.search("NameError: (.*)\x1b", err.args[0])
                    if problem:
                        raise NameError(problem.group(1)) from err
                    problem = re.search("NameError: (.*)\n", err.args[0])
                    if problem:
                        raise NameError(problem.group(1)) from err
                raise
            except KeyError as err:
                # raise the inner key error which is more helpful
                context = getattr(err, "__context__", None)
                if context:
                    raise context
                else:
                    raise err

    def _load(
        self,
        source=None,
        as_dataframe=False,
        as_dataarray=False,
        as_table=False,
        runner=None,
        dtype=None,
        dot=None,
        mnl_draws=None,
        pick_counted=False,
        compile_watch=False,
        logsums=False,
    ):
        """
        Compute the flow outputs.

        Parameters
        ----------
        source : DataTree, optional
            This is the source of the data for this flow. If not provided, the
            tree used to initialize this flow is used.
        as_dataframe : bool, default False
            Return the loaded data as a pandas.DataFrame. Must not be used in
            conjunction with the `dot` argument.
        as_dataarray : bool, default False
            Return the loaded data as a xarray.DataArray.
        as_table : bool, default False
            Return the loaded data as a sharrow.Table (a subclass of pyarrow.Table).
        runner : Callable, optional
            Overload the prepared function with a different callable. Recommended
            for advanced usage only.
        dtype : str or dtype
            Override the default dtype for the result. May trigger re-compilation
            of the underlying code.
        dot : array-like, optional
            An array of coefficients. If provided, the function returns the
            dot-product of the computed expressions and this array of coefficients,
            but without ever materializing the array of computed expression values
            in memory, achiving significant performance gains.
        mnl_draws : array-like, optional
            An array of random values in the unit interval. If provided, `dot` must
            also be provided. The dot-product is treated as the utility function
            for a multinomial logit model, and these draws are used to simulate
            choices from the implied probabilities.
        compile_watch : bool, default False
            Watch for compiled code.
        logsums : bool, default False
            Also return logsums when making draws from MNL models.
        """
        if compile_watch:
            compile_watch = time.time()
        if (as_dataframe or as_table) and dot is not None:
            raise ValueError("cannot format output other than as array if using dot")
        if source is None:
            source = self.tree
        if dtype is None and dot is not None:
            dtype = dot.dtype
        dot_collapse = False
        result_p = None
        pick_count = None
        out_logsum = None
        if dot is not None and dot.ndim == 1:
            dot = np.expand_dims(dot, -1)
            dot_collapse = True
        mnl_collapse = False
        if mnl_draws is not None and mnl_draws.ndim == 1:
            mnl_draws = np.expand_dims(mnl_draws, -1)
            mnl_collapse = True
        if not source.relationships_are_digitized:
            source = source.digitize_relationships()
        if source.relationships_are_digitized:
            if mnl_draws is None:
                result = self.iload_raw(source, runner=runner, dtype=dtype, dot=dot)
            else:
                result, result_p, pick_count, out_logsum = self.iload_raw(
                    source,
                    runner=runner,
                    dtype=dtype,
                    dot=dot,
                    mnl=mnl_draws,
                    pick_counted=pick_counted,
                    logsums=logsums,
                )
        else:
            raise RuntimeError("please digitize")
        if as_dataframe:
            index = getattr(source.root_dataset, "index", None)
            result = pd.DataFrame(
                result, index=index, columns=list(self._raw_functions.keys())
            )
        elif as_table:
            result = Table(
                {k: result[:, n] for n, k in enumerate(self._raw_functions.keys())}
            )
        elif as_dataarray:
            use_dims = list(
                presorted(source.root_dataset.dims, self.dim_order, self.dim_exclude)
            )
            if dot is None:
                result = xr.DataArray(
                    result,
                    dims=use_dims + ["expressions"],
                    coords={
                        k: v
                        for k, v in source.root_dataset.coords.items()
                        if k in use_dims
                    },
                )
                result.coords["expressions"] = self.function_names
            elif dot_collapse and mnl_draws is None:
                result = xr.DataArray(
                    np.squeeze(result, -1),
                    dims=use_dims,
                    coords=source.root_dataset.coords,
                )
            elif mnl_collapse:
                if isinstance(dot, xr.DataArray):
                    plus_dims = list(dot.dims[1:])
                else:
                    plus_dims = []
                result = xr.DataArray(
                    np.squeeze(result, -1),
                    dims=use_dims[:-1] + plus_dims,
                    coords=source.root_dataset.coords,
                )
                result_p = xr.DataArray(
                    np.squeeze(result_p, -1),
                    dims=use_dims[:-1] + plus_dims,
                    coords=source.root_dataset.coords,
                )
                if pick_count is not None:
                    pick_count = xr.DataArray(
                        np.squeeze(pick_count, -1),
                        dims=use_dims[:-1] + plus_dims,
                        coords=source.root_dataset.coords,
                    )
                for plus_dim in plus_dims:
                    if plus_dim in dot.coords:
                        result.coords[plus_dim] = dot.coords[plus_dim]
                        result_p.coords[plus_dim] = dot.coords[plus_dim]
                        if pick_count is not None:
                            pick_count.coords[plus_dim] = dot.coords[plus_dim]
            elif isinstance(dot, xr.DataArray):
                plus_dims = dot.dims[1:]
                result = xr.DataArray(
                    result,
                    dims=use_dims + list(plus_dims),
                    coords=source.root_dataset.coords,
                )
                for plus_dim in plus_dims:
                    if plus_dim in dot.coords:
                        result.coords[plus_dim] = dot.coords[plus_dim]
            else:
                dot_ = xr.DataArray(dot)
                plus_dims = dot_.dims[1:]
                result = xr.DataArray(
                    result,
                    dims=use_dims + list(plus_dims),
                    coords=source.root_dataset.coords,
                )
        elif dot_collapse and mnl_draws is None:
            result = np.squeeze(result, -1)
        elif mnl_collapse:
            result = np.squeeze(result, -1)
            result_p = np.squeeze(result_p, -1)
        if compile_watch:
            self.compiled_recently = False
            for i in os.walk(os.path.join(self.cache_dir, self.name)):
                for f in i[2]:
                    fi = os.path.join(i[0], f)
                    try:
                        t = os.path.getmtime(fi)
                    except FileNotFoundError:
                        # something is actively happening in this directory
                        self.compiled_recently = True
                        logger.warning(
                            f"unidentified activity (file deletion) detected for {self.name}"
                        )
                        break
                    if t > compile_watch:
                        self.compiled_recently = True
                        logger.warning(f"compilation activity detected for {self.name}")
                        break
                if self.compiled_recently:
                    break
        else:
            try:
                del self.compiled_recently
            except AttributeError:
                pass
        if result_p is not None:
            if logsums:
                return result, result_p, pick_count, out_logsum
            if pick_counted:
                return result, result_p, pick_count
            else:
                return result, result_p
        return result

    def load(self, source=None, dtype=None, compile_watch=False):
        """
        Compute the flow outputs as a numpy array.

        Parameters
        ----------
        source : DataTree, optional
            This is the source of the data for this flow. If not provided, the
            tree used to initialize this flow is used.
        dtype : str or dtype
            Override the default dtype for the result. May trigger re-compilation
            of the underlying code.
        compile_watch : bool, default False
            Set the `compiled_recently` flag on this flow to True if any file
            modification activity is observed in the cache directory.

        Returns
        -------
        numpy.array
        """
        return self._load(source=source, dtype=dtype, compile_watch=compile_watch)

    def load_dataframe(self, source=None, dtype=None, compile_watch=False):
        """
        Compute the flow outputs as a pandas.DataFrame.

        Parameters
        ----------
        source : DataTree, optional
            This is the source of the data for this flow. If not provided, the
            tree used to initialize this flow is used.
        dtype : str or dtype
            Override the default dtype for the result. May trigger re-compilation
            of the underlying code.
        compile_watch : bool, default False
            Set the `compiled_recently` flag on this flow to True if any file
            modification activity is observed in the cache directory.

        Returns
        -------
        pandas.DataFrame
        """
        return self._load(
            source=source, dtype=dtype, as_dataframe=True, compile_watch=compile_watch
        )

    def load_dataarray(self, source=None, dtype=None, compile_watch=False):
        """
        Compute the flow outputs as a xarray.DataArray.

        Parameters
        ----------
        source : DataTree, optional
            This is the source of the data for this flow. If not provided, the
            tree used to initialize this flow is used.
        dtype : str or dtype
            Override the default dtype for the result. May trigger re-compilation
            of the underlying code.
        compile_watch : bool, default False
            Set the `compiled_recently` flag on this flow to True if any file
            modification activity is observed in the cache directory.

        Returns
        -------
        xarray.DataArray
        """
        return self._load(
            source=source, dtype=dtype, as_dataarray=True, compile_watch=compile_watch
        )

    def dot(self, coefficients, source=None, dtype=None, compile_watch=False):
        """
        Compute the dot-product of expression results and coefficients.

        Parameters
        ----------
        coefficients : array-like
            This function will return the dot-product of the computed expressions
            and this array of coefficients, but without ever materializing the
            array of computed expression values in memory, achieving significant
            performance gains.
        source : DataTree, optional
            This is the source of the data for this flow. If not provided, the
            tree used to initialize this flow is used.
        dtype : str or dtype
            Override the default dtype for the result. May trigger re-compilation
            of the underlying code.
        compile_watch : bool, default False
            Set the `compiled_recently` flag on this flow to True if any file
            modification activity is observed in the cache directory.

        Returns
        -------
        numpy.ndarray
        """
        return self._load(
            source,
            dot=coefficients,
            dtype=dtype,
            compile_watch=compile_watch,
        )

    def dot_dataarray(self, coefficients, source=None, dtype=None, compile_watch=False):
        """
        Compute the dot-product of expression results and coefficients.

        Parameters
        ----------
        coefficients : DataArray
            This function will return the dot-product of the computed expressions
            and this array of coefficients, but without ever materializing the
            array of computed expression values in memory, achieving significant
            performance gains.
        source : DataTree, optional
            This is the source of the data for this flow. If not provided, the
            tree used to initialize this flow is used.
        dtype : str or dtype
            Override the default dtype for the result. May trigger re-compilation
            of the underlying code.
        compile_watch : bool, default False
            Set the `compiled_recently` flag on this flow to True if any file
            modification activity is observed in the cache directory.

        Returns
        -------
        xarray.DataArray
        """
        return self._load(
            source,
            dot=coefficients,
            dtype=dtype,
            as_dataarray=True,
            compile_watch=compile_watch,
        )

    def mnl_draws(
        self,
        coefficients,
        draws,
        source=None,
        pick_counted=False,
        logsums=False,
        dtype=None,
        compile_watch=False,
    ):
        """
        Make random simulated choices for a multinomial logit model.

        Parameters
        ----------
        coefficients : array-like
            These coefficients are used is in `dot` to compute the dot-product
            of the computed expressions, and this result is treated as the utility
            function for a multinomial logit model.
        draws : array-like
            A one or two dimensional array of random values in the unit interval.
            If one dimensional, then it must have length equal to the first
            dimension of the base `shape` of `source`, and a single draw will be
            applied for each row in that dimension.  If two dimensional, the first
            dimension must match as above, and the second dimension determines the
            number of draws applied for each row in the first dimension.
        source : DataTree, optional
            This is the source of the data for this flow. If not provided, the
            tree used to initialize this flow is used.
        pick_counted : bool, default False
            Whether to tally multiple repeated choices with a pick count.
        logsums : bool, default False
            Whether to also return logsums.
        dtype : str or dtype
            Override the default dtype for the probability. May trigger re-compilation
            of the underlying code.  The choices and pick counts (if included)
            are always integers.
        compile_watch : bool, default False
            Set the `compiled_recently` flag on this flow to True if any file
            modification activity is observed in the cache directory.

        Returns
        -------
        choices : array[int32]
            The positions of the simulated choices.
        probs : array[dtype]
            The probability that was associated with each simulated choice.
        pick_count : array[int32], optional
            A count of how many times this choice was chosen, only included
            if `pick_counted` is True.
        """
        return self._load(
            source=source,
            dot=coefficients,
            mnl_draws=draws,
            dtype=dtype,
            pick_counted=pick_counted,
            compile_watch=compile_watch,
            logsums=logsums,
        )

    @property
    def function_names(self):
        return list(self._raw_functions.keys())

    @function_names.setter
    def function_names(self, x):
        for name in x:
            if name not in self._raw_functions:
                self._raw_functions[name] = (None, None, set(), [])

    def _spill(self, all_name_tokens=None):
        cmds = [self.tree._spill(all_name_tokens)]
        cmds.append("\n")
        cmds.append(f"output_name_positions = {self.output_name_positions!r}")
        cmds.append(f"function_names = {self.function_names!r}")
        return "\n".join(cmds)

    def show_code(self, linenos="inline"):
        """
        Display the underlying Python code constructed for this flow.

        This convenience function is provided primarily to display the underlying
        source code in a Jupyter notebook, for debugging and educational purposes.

        Parameters
        ----------
        linenos : {'inline', 'table'}
            This argument is passed to the pygments HtmlFormatter.
            If set to ``'table'``, output line numbers as a table with two cells,
            one containing the line numbers, the other the whole code.  This is
            copy-and-paste-friendly, but may cause alignment problems with some
            browsers or fonts.  If set to ``'inline'``, the line numbers will be
            integrated in the ``<pre>`` tag that contains the code.

        Returns
        -------
        IPython.display.HTML
        """
        from IPython.display import HTML
        from pygments import highlight
        from pygments.formatters.html import HtmlFormatter
        from pygments.lexers.python import PythonLexer

        codefile = os.path.join(self.cache_dir, self.name, "__init__.py")
        with open(codefile, "rt") as f_code:
            code = f_code.read()
        pretty = highlight(code, PythonLexer(), HtmlFormatter(linenos=linenos))
        css = HtmlFormatter().get_style_defs(".highlight")
        bbedit_url = f"x-bbedit://open?url=file://{codefile}"
        bb_link = f'<a href="{bbedit_url}">{codefile}</a>'
        return HTML(f"<style>{css}</style><p>{bb_link}</p>{pretty}")
