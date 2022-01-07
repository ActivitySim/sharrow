import ast
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import networkx as nx
import pyarrow as pa
from collections.abc import Sequence
from .dataset import Dataset, Table
from .maths import piece, hard_sigmoid, transpose_leading, clip
from .shared_memory import *
from . import __version__

logger = logging.getLogger("sharrow")

well_known_names = {
    'nb', 'np', 'pd', 'xr', 'pa',
    'log', 'exp', 'log1p', 'expm1', 'max', 'min',
    'piece', 'hard_sigmoid', 'transpose_leading', 'clip',
}

def _require_string(x):
    if not isinstance(x, str):
        raise ValueError('must be string')
    return x


def _iat(source, *, _names=None, _load=False, _index_name=None, **idxs):
    loaders = {}
    if _index_name is None:
        _index_name = "index"
    for k, v in idxs.items():
        if v.ndim == 1:
            loaders[k] = xr.DataArray(v, dims=[_index_name])
        else:
            loaders[k] = xr.DataArray(v, dims=[f"{_index_name}{n}" for n in range(v.ndim)])
    if _names:
        ds = source[_names]
    else:
        ds = source
    if _load:
        ds = ds.load()
    return ds.isel(**loaders)


def _at(source, *, _names=None, _load=False, _index_name=None, **idxs):
    loaders = {}
    if _index_name is None:
        _index_name = "index"
    for k, v in idxs.items():
        if v.ndim == 1:
            loaders[k] = xr.DataArray(v, dims=[_index_name])
        else:
            loaders[k] = xr.DataArray(v, dims=[f"{_index_name}{n}" for n in range(v.ndim)])
    if _names:
        ds = source[_names]
    else:
        ds = source
    if _load:
        ds = ds.load()
    return ds.sel(**loaders)


def gather(source, indexes):
    """
    Extract values by label on the coordinates indicated by columns of a DataFrame.

    Parameters
    ----------
    source : xarray.DataArray or xarray.Dataset
        The source of the values to extract.
    indexes : Mapping[str, array-like]
        The keys of `indexes` (if given as a dataframe, the column names)
        should match the named dimensions of `source`.  The resulting extracted
        data will have a shape one row per row of `df`, and columns matching
        the data variables in `source`, and each value is looked up by the labels.

    Returns
    -------
    pd.DataFrame
    """
    result = _at(source, **indexes).reset_coords(drop=True)
    return result


def igather(source, positions):
    """
    Extract values by position on the coordinates indicated by columns of a DataFrame.

    Parameters
    ----------
    source : xarray.DataArray or xarray.Dataset
    positions : pd.DataFrame or Mapping[str, array-like]
        The columns (or keys) of `df` should match the named dimensions of
        this Dataset.  The resulting extracted DataFrame will have one row
        per row of `df`, columns matching the data variables in this dataset,
        and each value is looked up by the positions.

    Returns
    -------
    pd.DataFrame
    """
    result = _iat(source, **positions).reset_coords(drop=True)
    return result


def xgather(source, positions, indexes):
    if len(indexes) == 0:
        return igather(source, positions)
    elif len(positions) == 0:
        return gather(source, indexes)
    else:
        return gather(igather(source, positions), indexes)




class Relationship:

    def __init__(
            self,
            parent_data,
            parent_name,
            child_data,
            child_name,
            indexing='label',
    ):
        self.parent_data = _require_string(parent_data)
        self.parent_name = _require_string(parent_name)
        self.child_data = _require_string(child_data)
        self.child_name = _require_string(child_name)
        if indexing not in {'label', 'position'}:
            raise ValueError("indexing must be by label or position")
        self.indexing = indexing

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return repr(self) == repr(other)

    def __repr__(self):
        return f"<Relationship by {self.indexing}: {self.parent_data}[{self.parent_name!r}] -> {self.child_data}[{self.child_name!r}]>"

    def attrs(self):
        return dict(
            parent_name=self.parent_name,
            child_name=self.child_name,
            indexing=self.indexing,
        )


class DataTree:

    DatasetType = Dataset

    def __init__(
            self,
            graph=None,
            root_node_name=None,
            extra_funcs=(),
            extra_vars=None,
            cache_dir=None,
            relationships=(),
            force_digitization=False,
            **kwargs,
    ):
        if isinstance(graph, Dataset):
            raise ValueError("datasets must be given as keyword arguments")

        # raw init
        if graph is None:
            graph = nx.MultiDiGraph()
        self._graph = graph
        self._root_node_name = None
        self.force_digitization = force_digitization

        # defined init
        if root_node_name is not None and root_node_name in kwargs:
            self.add_dataset(root_node_name, kwargs[root_node_name])
        self.root_node_name = root_node_name
        self.extra_funcs = extra_funcs
        self.extra_vars = extra_vars or {}
        self.cache_dir = cache_dir
        self._cached_indexes = {}
        for k, v in kwargs.items():
            if root_node_name is not None and k == root_node_name:
                continue
            self.add_dataset(k, v)
        for r in relationships:
            self.add_relationship(r)
        if force_digitization:
            self.digitize_relationships(inplace=True)

    def __shallow_copy_extras(self):
        return dict(
            extra_funcs=self.extra_funcs,
            extra_vars=self.extra_vars,
            cache_dir=self.cache_dir,
            force_digitization=self.force_digitization,
        )

    def __repr__(self):
        s = f"<{self.__module__}.{self.__class__.__name__}>"
        if len(self._graph.nodes):
            s += "\n datasets:"
            if self.root_node_name:
                s += f"\n - {self.root_node_name}"
            for k in self._graph.nodes:
                if k == self.root_node_name:
                    continue
                s += f"\n - {k}"
        else:
            s += "\n datasets: none"
        if len(self._graph.edges):
            s += "\n relationships:"
            for e in self._graph.edges:
                s += f"\n - {self._get_relationship(e)!r}".replace("<Relationship ", "").rstrip(">")
        else:
            s += "\n relationships: none"
        return s

    def _hash_features(self):
        h = []
        if len(self._graph.nodes):
            if self.root_node_name:
                h.append(f"dataset:{self.root_node_name}")
            for k in self._graph.nodes:
                if k == self.root_node_name:
                    continue
                h.append(f"dataset:{k}")
        else:
            h.append(f"datasets:none")
        if len(self._graph.edges):
            for e in self._graph.edges:
                r = f"relationship:{self._get_relationship(e)!r}".replace("<Relationship ", "").rstrip(">")
                h.append(r)
        else:
            h.append("relationships:none")
        return h

    @property
    def root_node_name(self):
        if self._root_node_name is None:
            for nodename in self._graph.nodes:
                if self._graph.in_degree(nodename) == 0:
                    self._root_node_name = nodename
                    break
        return self._root_node_name

    @root_node_name.setter
    def root_node_name(self, name):
        if name is None:
            self._root_node_name = None
            return
        if not isinstance(name, str):
            raise TypeError(f'root_node_name must be str not {type(name)}')
        if name not in self._graph.nodes:
            raise KeyError(name)
        self._root_node_name = name

    def add_relationship(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], Relationship):
            r = args[0]
        elif len(args) == 1 and isinstance(args[0], str):
            s = args[0]
            if "->" in s:
                parent, child = s.split("->", 1)
                i = "position"
            elif "@":
                parent, child = s.split("@", 1)
                i = "label"
            p1, p2 = parent.split(".", 1)
            c1, c2 = child.split(".", 1)
            p1 = p1.strip()
            p2 = p2.strip()
            c1 = c1.strip()
            c2 = c2.strip()
            r = Relationship(
                parent_data=p1, parent_name=p2,
                child_data=c1, child_name=c2,
                indexing=i,
            )
        else:
            r = Relationship(*args, **kwargs)

        # check for existing relationships, don't duplicate
        for e in self._graph.edges:
            r2 = self._get_relationship(e)
            if r == r2:
                return

        # confirm correct pointer
        r.parent_data = self.finditem(r.parent_name, maybe_in=r.parent_data)

        self._graph.add_edge(r.parent_data, r.child_data, **r.attrs())
        if self.force_digitization:
            self.digitize_relationships(inplace=True)

    def get_relationship(self, parent, child):
        attrs = self._graph.edges[parent, child]
        return Relationship(parent_data=parent, child_data=child, **attrs)

    def add_dataset(self, name, dataset, relationships=(), as_root=False):
        self._graph.add_node(name, dataset=self.DatasetType.construct(dataset))
        if self.root_node_name is None or as_root:
            self.root_node_name = name
        if isinstance(relationships, str):
            relationships = [relationships]
        for r in relationships:
            # TODO validate relationships before adding.
            self.add_relationship(r)
        if self.force_digitization:
            self.digitize_relationships(inplace=True)

    def add_items(self, items):

        from collections.abc import Sequence, Mapping
        if isinstance(items, Sequence):
            for i in items:
                self.add_items(i)
        elif isinstance(items, Mapping):
            if 'name' in items and 'dataset' in items:
                self.add_dataset(items['name'], items['dataset'])
                preload = True
            else:
                preload = False
            for k, v in items.items():
                if preload and k in {'name', 'dataset'}:
                    continue
                if k == "relationships":
                    for r in v:
                        self.add_relationship(r)
                else:
                    self.add_dataset(k, v)
        else:
            raise ValueError("add_items requires Sequence or Mapping")


    @property
    def root_node(self):
        return self._graph.nodes[self.root_node_name]

    @property
    def root_dataset(self):
        return self._graph.nodes[self.root_node_name]['dataset']

    @root_dataset.setter
    def root_dataset(self, x):
        from .dataset import Dataset
        if not isinstance(x, Dataset):
            x = self.DatasetType.construct(x)
        self._graph.nodes[self.root_node_name]['dataset'] = x

    def _get_relationship(self, edge):
        return Relationship(parent_data=edge[0], child_data=edge[1], **self._graph.edges[edge])

    def __getitem__(self, item):
        if isinstance(item, (list, tuple)):
            from .dataset import Dataset
            return Dataset({k: self[k] for k in item})
        try:
            return self._getitem(item)
        except KeyError as err:
            return self._getitem(item, include_blank_dims=True)

    def finditem(self, item, maybe_in=None):
        if maybe_in is not None and maybe_in in self._graph.nodes:
            dataset = self._graph.nodes[maybe_in].get('dataset', {})
            if item in dataset:
                return maybe_in
        return self._getitem(item, just_node_name=True)

    def _getitem(self, item, include_blank_dims=False, only_dims=False, just_node_name=False):

        if isinstance(item, (list, tuple)):
            from .dataset import Dataset
            return Dataset({k: self[k] for k in item})

        if "." in item:
            item_in, item = item.split(".", 1)
        else:
            item_in = None

        queue = [self.root_node_name]
        examined = set()
        while len(queue):
            current_node = queue.pop(0)
            if current_node in examined:
                continue
            dataset = self._graph.nodes[current_node].get('dataset', {})
            by_name = item in dataset and not only_dims
            by_dims = not by_name and include_blank_dims and (item in dataset.dims)
            if (by_name or by_dims) and (item_in is None or item_in == current_node):
                if just_node_name:
                    return current_node
                if current_node == self.root_node_name:
                    if by_dims:
                        return xr.DataArray(pd.RangeIndex(dataset.dims[item]), dims=item)
                    else:
                        return dataset[item]
                else:
                    _positions = {}
                    _labels = {}
                    if by_dims:
                        if item in dataset.variables:
                            coords = {item: dataset.variables[item]}
                        else:
                            coords = None
                        result = xr.DataArray(
                            pd.RangeIndex(dataset.dims[item]),
                            dims=item,
                            coords=coords,
                        )
                    else:
                        result = dataset[item]
                    dims_in_result = set(result.dims)
                    for path in nx.algorithms.simple_paths.all_simple_edge_paths(self._graph, self.root_node_name, current_node):
                        path_dim = self._graph.edges[path[-1]].get('child_name')
                        if path_dim not in dims_in_result:
                            continue
                        # path_indexing = self._graph.edges[path[-1]].get('indexing')
                        t1 = None
                        # intermediate nodes on path
                        for (e, e_next) in zip(path[:-1], path[1:]):
                            r = self._get_relationship(e)
                            r_next = self._get_relationship(e_next)
                            if t1 is None:
                                t1 = self._graph.nodes[r.parent_data].get('dataset')
                            t2 = self._graph.nodes[r.child_data].get('dataset')[[r_next.parent_name]]
                            if r.indexing == 'label':
                                t1 = t2.sel({r.child_name: t1[r.parent_name].to_numpy()})
                            else:  # by position
                                t1 = t2.isel({r.child_name: t1[r.parent_name].to_numpy()})
                        # final node in path
                        e = path[-1]
                        r = Relationship(parent_data=e[0], child_data=e[1], **self._graph.edges[e])
                        if t1 is None:
                            t1 = self._graph.nodes[r.parent_data].get('dataset')
                        if r.indexing == 'label':
                            _labels[r.child_name] = t1[r.parent_name].to_numpy()
                        else:  # by position
                            _idx = t1[r.parent_name].to_numpy()
                            if not np.issubdtype(_idx.dtype, np.integer):
                                _idx = _idx.astype(np.int64)
                            _positions[r.child_name] = _idx

                    y = xgather(result, _positions, _labels)
                    if len(result.dims) == 1 and len(y.dims) == 1:
                        y = y.rename({y.dims[0]: result.dims[0]})
                    elif len(dims_in_result) == len(y.dims):
                        y = y.rename({_i:_j for _i,_j in zip(y.dims, result.dims)})
                    return y
            else:
                examined.add(current_node)
                for _, next_up in self._graph.out_edges(current_node):
                    if next_up not in examined:
                        queue.append(next_up)

        raise KeyError(item)

    @property
    def subspaces(self):
        spaces = {}
        for k in self._graph.nodes:
            s = self._graph.nodes[k].get('dataset', None)
            if s is not None:
                spaces[k] = s
        return spaces

    def subspaces_iter(self):
        for k in self._graph.nodes:
            s = self._graph.nodes[k].get('dataset', None)
            if s is not None:
                yield (k, s)

    # @property
    # def namespace(self):
    #     namespace = globals()
    #     namespace['piece'] = piece
    #     namespace['hard_sigmoid'] = hard_sigmoid
    #     namespace['transpose_leading'] = transpose_leading
    #     namespace['clip'] = clip
    #     namespace['log'] = np.log
    #     namespace['exp'] = np.exp
    #     namespace['log1p'] = np.log1p
    #     namespace['expm1'] = np.expm1
    #     for f in self.extra_funcs:
    #         namespace[f.__name__] = f
    #     for spacename, spacearrays in self.subspaces_iter():
    #         for k, arr in spacearrays.items():
    #             namespace[f"__{spacename or 'base'}__{k}"] = arr.data
    #     return namespace

    def namespace_names(self):
        namespace = set()
        for spacename, spacearrays in self.subspaces_iter():
            for k, arr in spacearrays.coords.items():
                namespace.add(f"__{spacename or 'base'}__{k}")
            for k, arr in spacearrays.items():
                namespace.add(f"__{spacename or 'base'}__{k}")
        return namespace

    @property
    def dims(self):
        dims = {}
        for k, v in self.subspaces_iter():
            for name, length in v.dims.items():
                if name in dims:
                    if dims[name] != length:
                        raise ValueError("inconsistent dimensions\n"+self.dims_detail())
                else:
                    dims[name] = length
        return xr.core.utils.Frozen(dims)

    def dims_detail(self):
        s = ""
        for k, v in self.subspaces_iter():
            s += f"\n{k}:"
            for name, length in v.dims.items():
                s += f"\n - {name}: {length}"
        return s[1:]

    def drop_dims(self, dims, inplace=False, ignore_missing_dims=True):
        if inplace:
            obj = self
        else:
            obj = self.copy()
        if not ignore_missing_dims:
            obj.root_dataset = obj.root_dataset.drop_dims(dims)
        else:
            if isinstance(dims, str):
                dims = [dims]
            for d in dims:
                if d in obj.root_dataset.dims:
                    obj.root_dataset = obj.root_dataset.drop_dims(d)
        return obj

    def get_indexes(
            self,
            position_only=True,
            as_dict=True,
            replacements=None,
            use_cache=True,
            check_shapes=True,
    ):
        if use_cache and (position_only, as_dict) in self._cached_indexes:
            return self._cached_indexes[(position_only, as_dict)]
        if not position_only:
            raise NotImplementedError
        dims = [d for d in self.dims if d[-1:]!="_" or (d[-1:]=="_" and d[:-1] not in self.dims)]
        if replacements is not None:
            obj = self.replace_datasets(replacements)
        else:
            obj = self
        result = {}
        result_shape = None
        for k in sorted(dims):
            result_k = obj._getitem(k, include_blank_dims=True, only_dims=True)
            if result_shape is None:
                result_shape = result_k.shape
            if result_shape != result_k.shape:
                if check_shapes:
                    raise ValueError(f"inconsistent index shapes {result_k.shape} v {result_shape} (probably an error on {k} or {sorted(dims)[0]})")
            result[k] = result_k

        if as_dict:
            result = {k: v.to_numpy() for k, v in result.items()}
        else:
            result = Dataset(result)
        if use_cache:
            self._cached_indexes[(position_only, as_dict)] = result
        return result

    def replace_datasets(self, other=None, validate=True, **kwargs):
        replacements = {}
        if other is not None:
            replacements.update(other)
        replacements.update(kwargs)
        graph = self._graph.copy()
        for k in replacements:
            if k not in graph.nodes:
                raise KeyError(k)
            x = self.DatasetType.construct(replacements[k])
            if validate:
                if x.dims != graph.nodes[k]['dataset'].dims:
                    # when replacement dimensions do not match, check for
                    # any upstream nodes that reference this dataset by
                    # position... which will potentially be problematic.
                    for e in self._graph.edges:
                        if e[1] == k:
                            indexing = self._graph.edges[e].get("indexing")
                            if indexing == 'position':
                                raise ValueError(
                                    f"dimensions mismatch on "
                                    f"positionally-referenced dataset {k}: "
                                    f"receiving {x.dims} "
                                    f"expected {graph.nodes[k]['dataset'].dims}"
                                )
            graph.nodes[k]['dataset'] = x
        return type(self)(graph, self.root_node_name, **self.__shallow_copy_extras())

    def setup_flow(
            self,
            definition_spec,
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
            hashing_level=1,
    ):
        """

        Parameters
        ----------
        definition_spec : Dict[str,str]
            Gives the names and definitions for the columns to
            create in our generated table.

        Returns
        -------
        TableGroupProcessor
        """
        from .flows import RFlow
        return RFlow(
            self,
            definition_spec,
            cache_dir=cache_dir or self.cache_dir,
            name=name,
            dtype=dtype,
            boundscheck=boundscheck,
            nopython=nopython,
            fastmath=fastmath,
            parallel=parallel,
            readme=readme,
            flow_library=flow_library,
            extra_hash_data=extra_hash_data,
            hashing_level=hashing_level,
        )

    def _spill(self, all_name_tokens=()):
        """
        Write backup code for sharrow-lite.

        Parameters
        ----------
        all_name_tokens

        Returns
        -------

        """
        cmds = []
        return "\n".join(cmds)

    def get_named_array(self, mangled_name):
        if mangled_name[:2] != "__":
            raise KeyError(mangled_name)
        name1, name2 = mangled_name[2:].split("__", 1)
        dataset = self._graph.nodes[name1].get("dataset")
        return dataset[name2].to_numpy()

    _BY_OFFSET = "digitizedOffset"

    def digitize_relationships(self, inplace=False):

        if inplace:
            obj = self
        else:
            obj = self.copy()

        for e in obj._graph.edges:
            r = obj._get_relationship(e)
            if r.indexing == 'label':
                p_dataset = obj._graph.nodes[r.parent_data].get('dataset', None)
                c_dataset = obj._graph.nodes[r.child_data].get('dataset', None)

                upstream = p_dataset[r.parent_name]
                downstream = c_dataset[r.child_name]

                # vectorize version
                mapper = {i: j for (j, i) in enumerate(downstream.to_numpy())}
                offsets = xr.apply_ufunc(np.vectorize(mapper.get), upstream)

                # candidate name for write back
                r_parent_name_new = f"{self._BY_OFFSET}{r.parent_name}_{r.child_data}_{r.child_name}"

                # it is common to have mirrored offsets in various dimensions.
                # we'd like to retain only the same data in memory once, so we'll
                # check if these offsets match any existing ones and if so just
                # point to that memory.
                for k in p_dataset:
                    if isinstance(k, str) and k.startswith(self._BY_OFFSET):
                        if p_dataset[k].equals(offsets):
                            # we found a match, so we'll assign this name to
                            # the match's memory storage instead of replicating it.
                            obj._graph.nodes[r.parent_data]['dataset'] = p_dataset.assign({r_parent_name_new: p_dataset[k]})
                            # r_parent_name_new = k
                            break
                else:
                    # no existing offset arrays match, make this new one
                    obj._graph.nodes[r.parent_data]['dataset'] = p_dataset.assign({r_parent_name_new: offsets})
                obj._graph.edges[e].update(dict(
                    parent_name=r_parent_name_new,
                    indexing='position',
                ))

        if not inplace:
            return obj

    @property
    def relationships_are_digitized(self):
        for e in self._graph.edges:
            r = self._get_relationship(e)
            if r.indexing != 'position':
                return False
        return True

    def _arg_tokenizer(self, spacename, spacearray):

        if spacename == self.root_node_name:
            root_dataset = self.root_dataset
            root_dims = sorted(root_dataset.dims)
            if isinstance(spacearray, str):
                from_dims = root_dataset[spacearray].dims
            else:
                from_dims = spacearray.dims
            return tuple(
                ast.parse(f"_arg{root_dims.index(dim):02}", mode='eval').body
                for dim in from_dims
            )

        if isinstance(spacearray, str):
            from_dims = self._graph.nodes[spacename]['dataset'][spacearray].dims
        else:
            from_dims = spacearray.dims

        tokens = []

        for dimname in from_dims:
            for e in self._graph.in_edges(spacename, keys=True):
                this_dim_name = self._graph.edges[e]['child_name']
                if dimname != this_dim_name:
                    continue
                parent_name = self._graph.edges[e]['parent_name']
                parent_data = e[0]

                upside_ast = self._arg_tokenizer(parent_data, parent_name)
                try:
                    upside = ", ".join(ast.unparse(t) for t in upside_ast)
                except:
                    for t in upside_ast:
                        print(f"t:{t}")
                    raise
                tokens.append(f"__{parent_data}__{parent_name}[{upside}]")

        result = []
        for t in tokens:
            result.append(ast.parse(t, mode='eval').body)
        return tuple(result)

    @property
    def coords(self):
        return self.root_dataset.coords

    def copy(self):
        return type(self)(
            self._graph.copy(),
            self.root_node_name,
            **self.__shallow_copy_extras()
        )
