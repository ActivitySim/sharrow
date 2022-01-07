import io
import os
import textwrap
import sys
import importlib
import inspect
import re
import hashlib
import base64

import dask
import numpy as np
import xarray as xr
import numba as nb
import pandas as pd
import pyarrow as pa
import dask.array as da
import warnings
import logging
import time
from collections.abc import Sequence


from .aster import expression_for_numba, extract_all_name_tokens, extract_names_2
from .maths import piece, hard_sigmoid, transpose_leading, clip
from .table import Table
from .filewrite import rewrite, blacken
from .shared_memory import *
from .dataset import Dataset
from .relationships import DataTree
from . import __version__

logger = logging.getLogger("sharrow")

well_known_names = {
    'nb', 'np', 'pd', 'xr', 'pa',
    'log', 'exp', 'log1p', 'expm1', 'max', 'min',
    'piece', 'hard_sigmoid', 'transpose_leading', 'clip',
}


def one_based(n):
    return pd.RangeIndex(1, n+1)


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
    cleaned = re.sub('\W|^(?=\d)','_', s)
    if cleaned != s or len(cleaned) > 120:
        # digest size 15 creates a 24 character base32 string
        h = base64.b32encode(
            hashlib.blake2b(s.encode(), digest_size=15).digest()
        ).decode()
        cleaned = f"{cleaned[:90]}_{h}"
    return cleaned


def presorted(sortable, presort):
    queue = set(sortable)
    for j in presort:
        if j in queue:
            yield j
            queue.remove(j)
    for i in sorted(queue):
        yield i


def _flip_flop_def(v):
    if isinstance(v, str) and "# sharrow:" in v:
        return v.split("# sharrow:", 1)[1].strip()
    else:
        return v

well_known_names |= {'_args', '_inputs', '_outputs', }
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
            return pd.RangeIndex(idx[0], idx[0]+idx.size)
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

MNL_2D_TEMPLATE = """
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



@nb.jit(cache=True, parallel=True, error_model='{error_model}', boundscheck={boundscheck}, nopython={nopython}, fastmath={fastmath})
def mnl_transform(
    argshape, 
    {joined_namespace_names}
    dtype=np.{dtype},
    dotarray=None,
    random_draws=None,
    pick_counted=False,
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
    if argshape[0] > 1000:
        for j0 in nb.prange(argshape[0]):
          partial = np.zeros(argshape[1], dtype=dtype)
          for j1 in range(argshape[1]):
            intermediate = np.zeros({len_self_raw_functions}, dtype=dtype)
            {meta_code_stack_dot}
            partial[j1] = np.exp(np.dot(intermediate, dotarray))[0]
          local_sum = np.sum(partial)
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
          partial /= local_sum
          if pick_counted:
            _sample_choices_maker_counted(partial, random_draws[j0], result[j0], result_p[j0], pick_count[j0])
          else:
            _sample_choices_maker(partial, random_draws[j0], result[j0], result_p[j0])
    return result, result_p, pick_count
"""






class RFlow:

    def __new__(
            cls,
            shared_data,
            defs,
            error_model='numpy',
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
    ):
        assert isinstance(shared_data, DataTree)
        shared_data.digitize_relationships(inplace=True)

        self = super().__new__(cls)
        # clean defs with hidden values
        defs = {k: _flip_flop_def(v) for k, v in defs.items()}

        # start init up to flow_hash
        self.__initialize_1(
            shared_data,
            defs,
            cache_dir=cache_dir,
            extra_hash_data=extra_hash_data,
            hashing_level=hashing_level,
        )
        # return from library if available
        if flow_library is not None and self.flow_hash in flow_library:
            logger.info(f"flow exists in library: {self.flow_hash}")
            result = flow_library[self.flow_hash]
            result.shared_data = shared_data
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
            shared_data,
            defs,
            cache_dir=None,
            extra_hash_data=(),
            error_model='numpy',
            boundscheck=False,
            nopython=True,
            fastmath=True,
            hashing_level=1,
    ):
        """
        Initialize up to the flow_hash

        Parameters
        ----------
        shared_data : SharedData
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
        if cache_dir is None:
            import tempfile
            self.temp_cache_dir = tempfile.TemporaryDirectory()
            self.cache_dir = self.temp_cache_dir.name
        else:
            self.cache_dir = cache_dir

        self.shared_data = shared_data
        self._raw_functions = {}
        self._secondary_flows = {}

        all_raw_names = set()
        all_name_tokens = set()
        for k, expr in defs.items():
            plain_names, attribute_pairs, subscript_pairs = extract_names_2(expr)
            all_raw_names |= plain_names
            if self.shared_data.root_node_name:
                all_raw_names |= attribute_pairs.get(self.shared_data.root_node_name, set())
                all_raw_names |= subscript_pairs.get(self.shared_data.root_node_name, set())

        index_slots = {i: n for n, i in enumerate(sorted(self.shared_data.dims))}
        self.arg_name_positions = index_slots
        self.arg_names = sorted(self.shared_data.dims)
        self.output_name_positions = {}

        self._used_extra_vars = {}
        if self.shared_data.extra_vars:
            for k, v in self.shared_data.extra_vars.items():
                if k in all_raw_names:
                    self._used_extra_vars[k] = v

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
        _flow_hash_push("---DataTree---")
        for k in self.arg_names:
            _flow_hash_push(f"arg:{k}")
        for k in self.shared_data._hash_features():
            if self._hashing_level > 0 or not k.startswith("relationship:"):
                _flow_hash_push(k)
        if self._hashing_level > 1:
            _flow_hash_push("---namespace_names---")
            for k in sorted(self._namespace_names):
                if k.startswith("__base__"):
                    continue
                _flow_hash_push(k)
                parts = k.split("__")
                if len(parts) > 2:
                    try:
                        digital_encoding = self.shared_data.subspaces[parts[1]]["__".join(parts[2:])].attrs['digital_encoding']
                    except (AttributeError, KeyError) as err:
                        pass
                    else:
                        if digital_encoding:
                            for de_k in sorted(digital_encoding.keys()):
                                de_v = digital_encoding[de_k]
                                if de_k == 'dictionary':
                                    self.encoding_dictionaries[k] = de_v
                                _flow_hash_push((k, 'digital_encoding', de_k, de_v))
        for k in extra_hash_data:
            _flow_hash_push(k)

        _flow_hash_push(f"{boundscheck=}")
        _flow_hash_push(f"{error_model=}")
        _flow_hash_push(f"{fastmath=}")

        self.flow_hash = base64.b32encode(flow_hash.digest()).decode()
        self.flow_hash_audit = "]\n# [".join(flow_hash_audit)

    def _index_slots(self):
        return {i: n for n, i in enumerate(sorted(self.shared_data.dims))}

    def init_sub_funcs(
            self,
            defs,
            error_model='numpy',
            boundscheck=False,
            nopython=True,
            fastmath=True,
    ):
        func_code = ""
        all_name_tokens = set()
        index_slots = {i: n for n, i in enumerate(sorted(self.shared_data.dims))}
        self.arg_name_positions = index_slots
        candidate_names = self.shared_data.namespace_names()

        meta_data = {}

        if self.shared_data.relationships_are_digitized:
            for spacename, spacearrays in self.shared_data.subspaces.items():
                dim_slots = {}
                for k1 in spacearrays.keys():
                    try:
                        toks = self.shared_data._arg_tokenizer(spacename, spacearray=spacearrays._variables[k1])
                    except:
                        toks = self.shared_data._arg_tokenizer(spacename, spacearray=spacearrays[k1])
                    dim_slots[k1] = toks
                try:
                    digital_encodings = spacearrays.digital_encodings
                except AttributeError:
                    digital_encodings = {}
                meta_data[spacename] = (dim_slots, digital_encodings)

        else:
            for spacename, spacearrays in self.shared_data.subspaces.items():
                dim_slots = {}
                for k1 in spacearrays.keys():
                    try:
                        _dims = spacearrays._variables[k1].dims
                    except:
                        _dims = spacearrays[k1].dims
                    dim_slots[k1] = [index_slots[z] for z in _dims]
                try:
                    digital_encodings = spacearrays.digital_encodings
                except AttributeError:
                    digital_encodings = {}
                meta_data[spacename] = (dim_slots, digital_encodings)


        # write individual function files for each expression
        for n, (k, expr) in enumerate(defs.items()):
            expr = str(expr).lstrip()
            init_expr = expr
            for spacename, spacearrays in self.shared_data.subspaces.items():
                if spacename == '':
                    expr = expression_for_numba(
                        expr,
                        spacename,
                        (),
                        {},  # input name positions not used
                        rawalias=self.shared_data.root_node_name or "____",
                    )
                else:
                    # dim_slots = {}
                    # for k1 in spacearrays.keys():
                    #     try:
                    #         _dims = spacearrays._variables[k1].dims
                    #     except:
                    #         _dims = spacearrays[k1].dims
                    #     dim_slots[k1] = [index_slots[z] for z in _dims]
                    #
                    # try:
                    #     digital_encodings = spacearrays.digital_encodings
                    # except AttributeError:
                    #     digital_encodings = {}
                    dim_slots, digital_encodings = meta_data[spacename]
                    try:
                        expr = expression_for_numba(
                            expr,
                            spacename,
                            dim_slots,
                            dim_slots,
                            digital_encodings=digital_encodings,
                        )
                    except KeyError:
                        # check if we can resolve this name on any other subspace
                        other_way = False
                        for other_spacename in self.shared_data.subspaces:
                            if other_spacename == spacename:
                                continue
                            dim_slots, digital_encodings = meta_data[other_spacename]
                            try:
                                expr = expression_for_numba(
                                    expr,
                                    spacename,
                                    dim_slots,
                                    dim_slots,
                                    digital_encodings=digital_encodings,
                                    prefer_name=other_spacename,
                                )
                            except KeyError:
                                pass
                            else:
                                other_way = True
                                break
                        if not other_way:
                            raise

            # now find instances where an identifier is previously created in this flow.
            expr = expression_for_numba(
                expr,
                '',
                (),
                self.output_name_positions,
                '_outputs',
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
                extra_imports="\n".join([
                    "from .extra_funcs import *" if self.shared_data.extra_funcs else "",
                    "from .extra_vars import *" if self._used_extra_vars else ""
                ]),
                boundscheck=boundscheck,
                nopython=nopython,
                fastmath=fastmath,
                init_expr=init_expr if k == init_expr else f"{k}: {init_expr}",
            )
            self._raw_functions[k] = (init_expr, expr, f_name_tokens, argtokens)
            self.output_name_positions[k] = n

        return blacken(func_code), all_name_tokens


    def __initialize_2(
            self,
            defs,
            error_model='numpy',
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
        shared_data : DataTree
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
                        digital_encoding = self.shared_data.subspaces[parts[1]]["__".join(parts[2:])].attrs['digital_encoding']
                    except (AttributeError, KeyError) as err:
                        pass
                    else:
                        if digital_encoding:
                            for de_k in sorted(digital_encoding.keys()):
                                de_v = digital_encoding[de_k]
                                if de_k == 'dictionary':
                                    self.encoding_dictionaries[k] = de_v


        # assign flow name based on hash unless otherwise given
        if name is None:
            name = f"flow_{self.flow_hash}"
        self.name = name

        # create the package directory for the flow if it does not exist
        os.makedirs(self.cache_dir, exist_ok=True)

        # if an existing __init__ file matches the hash, just use it
        init_file = os.path.join(self.cache_dir, self.name, f"__init__.py")
        if os.path.isfile(init_file):
            with open(init_file, 'rt') as f:
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
            }

            func_code = self._func_code

            # write extra_funcs file, if there are any extra_funcs
            if self.shared_data.extra_funcs:
                try:
                    import cloudpickle as pickle
                except ModuleNotFoundError:
                    import pickle
                dependencies.add("import pickle")
                func_code += "\n\n# extra_funcs\n"
                for x_func in self.shared_data.extra_funcs:
                    func_code += f"\n\n{x_func.__name__} = pickle.loads({repr(pickle.dumps(x_func))})\n"

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
                        buffer.write(f"{x_name} = pickle.loads({repr(pickle.dumps(x_var))})\n")
                        dependencies.add("import pickle")
                with io.StringIO() as x_code:
                    x_code.write(f"\n")
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
                    buffer.write(f"__encoding_dict{x_name} = pickle.loads({repr(pickle.dumps(x_dict))})\n")
                with io.StringIO() as x_code:
                    x_code.write(f"\n")
                    x_code.write(buffer.getvalue())
                    func_code += "\n\n# encoding dictionaries\n"
                    func_code += x_code.getvalue()

            # write the master module for this flow
            os.makedirs(os.path.join(self.cache_dir, self.name), exist_ok=True)
            with rewrite(os.path.join(self.cache_dir, self.name, f"__init__.py"), 'wt') as f_code:

                f_code.write(textwrap.dedent(f"""
                # this module generated automatically using sharrow version {__version__}
                # generation time: {time.strftime('%d %B %Y %I:%M:%S %p')}
                """)[1:])

                if readme:
                    f_code.write(
                        textwrap.indent(
                            textwrap.dedent(readme),
                            '# ',
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
                f_code.write(f"\n\n{func_code}")

                f_code.write("\n\n# machinery code\n\n")

                if self.shared_data.relationships_are_digitized:

                    root_dims = sorted(self.shared_data.root_dataset.dims)
                    n_root_dims = len(root_dims)

                    if n_root_dims == 1:
                        js = "j0"
                    elif n_root_dims == 2:
                        js = "j0, j1"

                    meta_code = []
                    meta_code_dot = []
                    for n, k in enumerate(self._raw_functions):
                        f_name_tokens = self._raw_functions[k][2]
                        f_arg_tokens = self._raw_functions[k][3]
                        f_name_tokens = ", ".join(sorted(f_name_tokens))
                        f_args_j = ", ".join([f"j{argn[-1]}" for argn in f_arg_tokens])
                        if f_args_j:
                            f_args_j += ", "
                        meta_code.append(f"result[{js}, {n}] = {clean(k)}({f_args_j}result[{js}], {f_name_tokens})")
                        meta_code_dot.append(f"intermediate[{n}] = {clean(k)}({f_args_j}intermediate, {f_name_tokens})")
                    meta_code_stack = textwrap.indent("\n".join(meta_code), ' ' * 12).lstrip()
                    meta_code_stack_dot = textwrap.indent("\n".join(meta_code_dot), ' ' * 12).lstrip()
                    len_self_raw_functions = len(self._raw_functions)
                    joined_namespace_names = "\n    ".join(f'{nn},' for nn in self._namespace_names)
                    linefeed = "\n                           "
                    if n_root_dims == 1:
                        meta_template = IRUNNER_1D_TEMPLATE.format(**locals())
                        meta_template_dot = IDOTTER_1D_TEMPLATE.format(**locals())
                        line_template = ILINER_1D_TEMPLATE.format(**locals())
                        mnl_template = ""
                    elif n_root_dims == 2:
                        meta_template = IRUNNER_2D_TEMPLATE.format(**locals())
                        meta_template_dot = IDOTTER_2D_TEMPLATE.format(**locals())
                        line_template = ILINER_2D_TEMPLATE.format(**locals())
                        mnl_template = MNL_2D_TEMPLATE.format(**locals())
                    else:
                        raise ValueError(f"invalid n_root_dims {n_root_dims}")

                else:

                    raise RuntimeError("deprecated")

                    line_template = ""
                    mnl_template = ""
                    meta_code = []
                    meta_code_dot = []
                    meta_args = ", ".join([f"_arg{n:02}" for n in self.arg_name_positions.values()])
                    meta_args_j = ", ".join([f"_arg{n:02}[j]" for n in self.arg_name_positions.values()])
                    # meta_args_explode = textwrap.indent(
                    #     "\n".join([f"_arg{n:02} = argarray[:,{n}]" for n in self.arg_name_positions.values()]),
                    #     ' ' * 20,
                    # ).lstrip()
                    for n, k in enumerate(self._raw_functions):
                        f_name_tokens = self._raw_functions[k][2]
                        f_arg_tokens = self._raw_functions[k][3]
                        f_name_tokens = ", ".join(sorted(f_name_tokens))
                        f_args_j = ", ".join([f"{argn}[j]" for argn in f_arg_tokens])
                        if f_args_j:
                            f_args_j += ", "
                        meta_code.append \
                            (f"result[j, {n}] = {clean(k)}({f_args_j}result[j], {f_name_tokens})")
                        meta_code_dot.append \
                            (f"intermediate[{n}] = {clean(k)}({f_args_j}intermediate, {f_name_tokens})")
                    meta_code_stack = textwrap.indent("\n".join(meta_code), ' '*32).lstrip()
                    meta_code_stack_dot = textwrap.indent("\n".join(meta_code_dot), ' '*36).lstrip()
                    linefeed = "\n                           "
                    meta_template = f"""
                    @nb.jit(cache=True, parallel=True, error_model='{error_model}', boundscheck={boundscheck}, nopython={nopython}, fastmath={fastmath})
                    def runner({meta_args}, 
                               {"".join(f"{j}, {linefeed}" for j in self._namespace_names)}dtype=np.{dtype}, min_shape_0=0,
                    ):
                        out_size = max(_arg00.shape[0], min_shape_0)
                        if out_size != _arg00.shape[0]:
                            result = np.zeros((out_size, {len(self._raw_functions)}), dtype=dtype)
                        else:
                            result = np.empty((out_size, {len(self._raw_functions)}), dtype=dtype)
                        if out_size > 1000:
                            for j in nb.prange(out_size):
                                {meta_code_stack}
                        else:
                            for j in range(out_size):
                                {meta_code_stack}
                        return result
                    """
                    if parallel:
                        meta_template_dot = f"""
                        @nb.jit(cache=True, parallel=True, error_model='{error_model}', boundscheck={boundscheck}, nopython={nopython}, fastmath={fastmath})
                        def dotter({meta_args}, 
                                   {"".join(f"{j}, {linefeed}" for j in self._namespace_names)}dtype=np.{dtype}, min_shape_0=0, dotarray=None, 
                        ):
                            out_size = max(_arg00.shape[0], min_shape_0)
                            if dotarray is None:
                                raise ValueError("dotarray cannot be None")
                            assert dotarray.ndim == 2
                            result = np.zeros((out_size, dotarray.shape[1]), dtype=dtype)
                            if out_size > 1000:
                                for j in nb.prange(out_size):
                                    intermediate = np.zeros({len(self._raw_functions)}, dtype=dtype)
                                    {meta_code_stack_dot}
                                    np.dot(intermediate, dotarray, out=result[j,:])
                            else:
                                intermediate = np.zeros({len(self._raw_functions)}, dtype=dtype)
                                for j in range(out_size):
                                    {meta_code_stack_dot}
                                    np.dot(intermediate, dotarray, out=result[j,:])
                            return result
                        """
                    else:
                        meta_template_dot = f"""
                            @nb.jit(cache=True, parallel=False, error_model='{error_model}', boundscheck={boundscheck}, nopython={nopython}, fastmath={fastmath})
                            def dotter({meta_args}, 
                                       {"".join(f"{j}, {linefeed}" for j in self._namespace_names)}dtype=np.{dtype}, min_shape_0=0, dotarray=None, 
                            ):
                                out_size = max(_arg00.shape[0], min_shape_0)
                                if dotarray is None:
                                    raise ValueError("dotarray cannot be None")
                                assert dotarray.ndim == 2
                                result = np.zeros((out_size, dotarray.shape[1]), dtype=dtype)
                                intermediate = np.zeros({len(self._raw_functions)}, dtype=dtype)
                                for j in range(out_size):
                                    {meta_code_stack_dot}
                                    np.dot(intermediate, dotarray, out=result[j,:])
                                return result
                        """


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
                f_code.write("# Greetings, tinkerer!  The `flow_hash` included here is a safety \n"
                             "# measure to prevent unknowing users creating a mess by modifying \n"
                             "# the code in this module so that it no longer matches the expected \n"
                             "# variable definitions. If you want to modify this code, you should \n"
                             "# delete this hash to allow the code to run without any checks, but \n"
                             "# you do so at your own risk. \n")
                f_code.write(f"flow_hash = {self.flow_hash!r}\n")

        if str(self.cache_dir) not in sys.path:
            logger.debug(f"inserting {self.cache_dir} into sys.path")
            sys.path.insert(0, str(self.cache_dir))
        importlib.invalidate_caches()
        logger.debug(f"importing {self.name}")
        module = importlib.import_module(self.name)
        sys.path = sys.path[1:]
        self._runner = getattr(module, 'runner', None)
        self._dotter = getattr(module, 'dotter', None)
        self._irunner = getattr(module, 'irunner', None)
        self._imnl = getattr(module, 'mnl_transform', None)
        self._idotter = getattr(module, 'idotter', None)
        if not writing:
            self.function_names = module.function_names
            self.output_name_positions = module.output_name_positions

    def load_raw(self, rg, args, runner=None, dtype=None, dot=None):
        assert isinstance(rg, DataTree)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=nb.NumbaExperimentalFeatureWarning)
            assembled_args = [args.get(k) for k in self.arg_name_positions.keys()]
            for aa in assembled_args:
                if aa.dtype.kind != 'i':
                    warnings.warn(
                        "position arguments are not all integers"
                    )
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
                    if arg in {'dtype', 'dotarray', 'inputarray', 'argarray'}:
                        continue
                    if arg.startswith('_arg'):
                        continue
                    arguments.append(np.asarray(rg.get_named_array(arg)))
                kwargs = {}
                if dtype is not None:
                    kwargs['dtype'] = dtype
                # else:
                #     kwargs['dtype'] = np.float64
                if dot is not None:
                    kwargs['dotarray'] = dot
                #logger.debug(f"load_raw calling runner with {assembled_args.shape=}, {assembled_inputs.shape=}")
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

    def iload_raw(self, rg, runner=None, dtype=None, dot=None, mnl=None, pick_counted=False):
        assert isinstance(rg, DataTree)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=nb.NumbaExperimentalFeatureWarning)
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
                named_args = inspect.getfullargspec(runner_.py_func).args
                arguments = []
                for arg in named_args:
                    if arg in {'dtype', 'dotarray', 'argshape', 'random_draws', 'pick_counted'}:
                        continue
                    arguments.append(np.asarray(rg.get_named_array(arg)))
                kwargs = {}
                if dtype is not None:
                    kwargs['dtype'] = dtype
                if dot is not None:
                    kwargs['dotarray'] = dot
                if mnl is not None:
                    kwargs['random_draws'] = mnl
                    kwargs['pick_counted'] = pick_counted
                tree_root_dims = rg.root_dataset.dims
                argshape = [tree_root_dims[i] for i in sorted(tree_root_dims)]
                #logger.debug(f"load_raw calling runner with {assembled_args.shape=}, {assembled_inputs.shape=}")
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

    def load(
            self,
            source=None,
            as_dataframe=False,
            as_dataarray=False,
            as_table=False,
            runner=None,
            dtype=None,
            dot=None,
            return_indexes=False,
            use_indexes_cache=True,
            mnl_draws=None,
            pick_counted=False,
            dim_order=None,
    ):
        """
        Compute the flow outputs.

        Parameters
        ----------
        source : DataTree, optional
            This is the source of the data for this flow. If not provided, the
            last available source is used.
        as_dataframe
        as_table : bool

        Returns
        -------

        """
        if (as_dataframe or as_table) and dot is not None:
            raise ValueError("cannot format output other than as array if using dot")
        if source is None:
            source = self.shared_data
        if dtype is None and dot is not None:
            dtype = dot.dtype
        dot_collapse = False
        result_p = None
        pick_count = None
        if dot is not None and dot.ndim == 1:
            dot = np.expand_dims(dot, -1)
            dot_collapse = True
        if not source.relationships_are_digitized:
            source = source.digitize_relationships()
        if source.relationships_are_digitized:
            if mnl_draws is None:
                result = self.iload_raw(source, runner=runner, dtype=dtype, dot=dot)
            else:
                result, result_p, pick_count = self.iload_raw(
                    source, runner=runner, dtype=dtype, dot=dot, mnl=mnl_draws,
                    pick_counted=pick_counted,
                )
            indexes_dict = None
        else:
            raise RuntimeError("please digitize")
            indexes_dict = source.get_indexes(use_cache=use_indexes_cache)
            # TODO only compute and use required indexes
            result = self.load_raw(source, indexes_dict, runner=runner, dtype=dtype, dot=dot)
        if as_dataframe:
            index = getattr(source.root_dataset, 'index', None)
            result = pd.DataFrame(result, index=index, columns=list(self._raw_functions.keys()))
        elif as_table:
            result = Table({k: result[:, n] for n, k in enumerate(self._raw_functions.keys())})
        elif as_dataarray:
            if dot is None:
                result = xr.DataArray(
                    result,
                    dims=sorted(source.root_dataset.dims) + ["expressions"],
                    coords=source.root_dataset.coords,
                )
                result.coords["expressions"] = self.function_names
            elif dot_collapse:
                result = xr.DataArray(
                    np.squeeze(result, -1),
                    dims=sorted(source.root_dataset.dims),
                    coords=source.root_dataset.coords,
                )
            else:
                raise NotImplementedError("cannot format as DataArray with multi-dimensional dot array")
        if return_indexes:
            return result, indexes_dict
        if result_p is not None:
            if pick_counted:
                return result, result_p, pick_count
            else:
                return result, result_p
        return result

    def merge(self, source, dtype=None):
        """
        Merge the data created by this flow into the source.

        Parameters
        ----------
        source : Dataset or Table or DataFrame
        dtype : str or dtype
            The loaded data will be generated with this dtype.

        Returns
        -------
        merged : Dataset or Table or DataFrame
            The same data type as `source` is returned.
        """
        assert isinstance(source, (xr.Dataset, pa.Table, pd.DataFrame, Table))
        new_cols = self.load(source, dtype=dtype)
        if isinstance(source, (pa.Table, Table)):
            for n, k in enumerate(self._raw_functions.keys()):
                source = source.append_column(k, [new_cols[:, n]])
        else:
            for n, k in enumerate(self._raw_functions.keys()):
                source[k] = new_cols[:, n]
        return source

    @property
    def function_names(self):
        return list(self._raw_functions.keys())

    @function_names.setter
    def function_names(self, x):
        for name in x:
            if name not in self._raw_functions:
                self._raw_functions[name] = (None, None, set(), [])

    def _spill(self, all_name_tokens=None):
        cmds = [self.shared_data._spill(all_name_tokens)]
        cmds.append("\n")
        cmds.append(f"output_name_positions = {self.output_name_positions!r}")
        cmds.append(f"function_names = {self.function_names!r}")
        return "\n".join(cmds)

    def show_code(self, linenos='inline'):
        from pygments import highlight
        from pygments.lexers.python import PythonLexer
        from pygments.formatters.html import HtmlFormatter
        from IPython.display import HTML
        codefile = os.path.join(self.cache_dir, self.name, f"__init__.py")
        with open(codefile, 'rt') as f_code:
            code = f_code.read()
        pretty = highlight(code, PythonLexer(), HtmlFormatter(linenos=linenos))
        css = HtmlFormatter().get_style_defs('.highlight')
        bbedit_url = f"x-bbedit://open?url=file://{codefile}"
        bb_link = f'<a href="{bbedit_url}">{codefile}</a>'
        return HTML(f"<style>{css}</style><p>{bb_link}</p>{pretty}")

