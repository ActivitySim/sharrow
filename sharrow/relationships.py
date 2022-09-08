import ast
import logging
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr

from .dataset import Dataset, construct

try:
    from ast import unparse
except ImportError:
    from astunparse import unparse as _unparse

    unparse = lambda *args: _unparse(*args).strip("\n")


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


def _require_string(x):
    if not isinstance(x, str):
        raise ValueError("must be string")
    return x


def _iat(source, *, _names=None, _load=False, _index_name=None, **idxs):
    loaders = {}
    if _index_name is None:
        _index_name = "index"
    for k, v in idxs.items():
        if v.ndim == 1:
            loaders[k] = xr.DataArray(v, dims=[_index_name])
        else:
            loaders[k] = xr.DataArray(
                v, dims=[f"{_index_name}{n}" for n in range(v.ndim)]
            )
    if _names:
        ds = source[_names]
    else:
        ds = source
    if _load:
        ds = ds._load()
    return ds.isel(**loaders)


def _at(source, *, _names=None, _load=False, _index_name=None, **idxs):
    loaders = {}
    if _index_name is None:
        _index_name = "index"
    for k, v in idxs.items():
        if v.ndim == 1:
            loaders[k] = xr.DataArray(v, dims=[_index_name])
        else:
            loaders[k] = xr.DataArray(
                v, dims=[f"{_index_name}{n}" for n in range(v.ndim)]
            )
    if _names:
        ds = source[_names]
    else:
        ds = source
    if _load:
        ds = ds._load()
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
    """
    Defines a linkage between datasets in a `DataTree`.
    """

    def __init__(
        self,
        parent_data,
        parent_name,
        child_data,
        child_name,
        indexing="label",
        analog=None,
    ):
        self.parent_data = _require_string(parent_data)
        """str: Name of the parent dataset."""

        self.parent_name = _require_string(parent_name)
        """str: Variable in the parent dataset that references the child dimension."""

        self.child_data = _require_string(child_data)
        """str: Name of the child dataset."""

        self.child_name = _require_string(child_name)
        """str: Dimension in the child dataset that is used by this relationship."""

        if indexing not in {"label", "position"}:
            raise ValueError("indexing must be by label or position")
        self.indexing = indexing
        """str: How the target dimension is used, either by 'label' or 'position'."""

        self.analog = analog
        """str: Original variable that defined label-based relationship before digitization."""

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

    @classmethod
    def from_string(cls, s):
        """
        Construct a `Relationship` from a string.

        Parameters
        ----------
        s : str
            The relationship definition.
            To create a label-based relationship, the string should look like
            "ParentNode.variable_name @ ChildNode.dimension_name".  To create
            a position-based relationship, give
            "ParentNode.variable_name -> ChildNode.dimension_name".

        Returns
        -------
        Relationship
        """
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
        return cls(
            parent_data=p1,
            parent_name=p2,
            child_data=c1,
            child_name=c2,
            indexing=i,
        )


class DataTree:
    """
    A tree representing linked datasets, from which data can flow.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
    root_node_name : str
        The name of the node at the root of the tree.
    extra_funcs : Tuple[Callable]
        Additional functions that can be called by Flow objects created
        using this DataTree.  These functions should have defined `__name__`
        attributes, so they can be called in expressions.
    extra_vars : Mapping[str,Any], optional
        Additional named constants that can be referenced by expressions in
        Flow objects created using this DataTree.
    cache_dir : Path-like, optional
        The default directory where Flow objects are created.
    relationships : Iterable[str or Relationship]
        The relationship definitions used to define this tree.  All dataset
        nodes named in these relationships should also be included as
        keyword arguments for this constructor.
    force_digitization : bool, default False
        Whether to automatically digitize all relationships (converting them
        from label-based to position-based).  Digitization is required to
        evaluate Flows, but doing so automatically on construction may be
        inefficient.
    dim_order : Tuple[str], optional
        The order of dimensions to use in Flow outputs.  Generally only needed
        if there are multiple dimensions in the root dataset.
    aux_vars : Mapping[str,Any], optional
        Additional named arrays or numba-typable variables that can be
        referenced by expressions in Flow objects created using this DataTree.
    """

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
        dim_order=None,
        aux_vars=None,
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
        self.dim_order = dim_order
        self.dim_exclude = set()

        # defined init
        if root_node_name is not None and root_node_name in kwargs:
            self.add_dataset(root_node_name, kwargs[root_node_name])
        self.root_node_name = root_node_name
        self.extra_funcs = extra_funcs
        self.extra_vars = extra_vars or {}
        self.aux_vars = aux_vars or {}
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

        # These filters are applied to incoming datasets when using `replace_datasets`.
        self.replacement_filters = {}
        """Dict[Str,Callable]: Filters that are automatically applied to data on replacement.

        When individual datasets are replaced in the tree, the incoming dataset is
        passed through the filter with a matching name-key (if it exists).  The filter
        should be a function that accepts one argument (the incoming dataset) and returns
        one value (the dataset to save in the tree).  These filters can be used to ensure
        data quality, e.g. renaming variables, ensuring particular data types, etc.
        """

        self.subspace_fallbacks = {}
        """Dict[Str:List[Str]]: Allowable fallback subspace lookups.

        When a named variable is not found in a given subspace, the default result is
        raising a KeyError. But, if fallbacks are defined for a given subspace, the
        fallbacks are searched in order for the desired variable.
        """

    @property
    def shape(self):
        """Tuple[int]: base shape of arrays that will be loaded when using this DataTree."""
        if self.dim_order:
            dim_order = self.dim_order
        else:
            from .flows import presorted

            dim_order = presorted(self.root_dataset.dims, self.dim_order)
        return tuple(
            self.root_dataset.dims[i] for i in dim_order if i not in self.dim_exclude
        )

    def __shallow_copy_extras(self):
        return dict(
            extra_funcs=self.extra_funcs,
            extra_vars=self.extra_vars,
            aux_vars=self.aux_vars,
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
                s += f"\n - {self._get_relationship(e)!r}".replace(
                    "<Relationship ", ""
                ).rstrip(">")
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
            h.append("datasets:none")
        if len(self._graph.edges):
            for e in self._graph.edges:
                r = f"relationship:{self._get_relationship(e)!r}".replace(
                    "<Relationship ", ""
                ).rstrip(">")
                h.append(r)
        else:
            h.append("relationships:none")
        h.append(f"dim_order:{self.dim_order}")
        return h

    @property
    def root_node_name(self):
        """str: The root node for this data tree, which is only ever a parent."""
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
            raise TypeError(f"root_node_name must be str not {type(name)}")
        if name not in self._graph.nodes:
            raise KeyError(name)
        self._root_node_name = name

    def add_relationship(self, *args, **kwargs):
        """
        Add a relationship to this DataTree.

        The new relationship will point from a variable in one dataset
        to a dimension of another dataset in this tree.  Both the parent
        and the child datasets should already have been added.

        Parameters
        ----------
        *args, **kwargs
            All arguments are passed through to the `Relationship`
            contructor, unless only a single `str` argument is provided,
            in which case the `Relationship.from_string` class constructor
            is used.
        """
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
                parent_data=p1,
                parent_name=p2,
                child_data=c1,
                child_name=c2,
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
        """
        Add a new Dataset node to this DataTree.

        Parameters
        ----------
        name : str
        dataset : Dataset or pandas.DataFrame
            Will be coerced into a `Dataset` object if it is not already
            in that format, using a no-copy process if possible.
        relationships : Tuple[str or Relationship]
            Also add these relationships.
        as_root : bool, default False
            Set this new node as the root of the tree, displacing any existing
            root.
        """
        self._graph.add_node(name, dataset=construct(dataset))
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

        from collections.abc import Mapping, Sequence

        if isinstance(items, Sequence):
            for i in items:
                self.add_items(i)
        elif isinstance(items, Mapping):
            if "name" in items and "dataset" in items:
                self.add_dataset(items["name"], items["dataset"])
                preload = True
            else:
                preload = False
            for k, v in items.items():
                if preload and k in {"name", "dataset"}:
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
        return self._graph.nodes[self.root_node_name]["dataset"]

    @root_dataset.setter
    def root_dataset(self, x):
        from .dataset import Dataset

        if not isinstance(x, Dataset):
            x = construct(x)
        if self.root_node_name in self.replacement_filters:
            x = self.replacement_filters[self.root_node_name](x)
        self._graph.nodes[self.root_node_name]["dataset"] = x

    def _get_relationship(self, edge):
        return Relationship(
            parent_data=edge[0], child_data=edge[1], **self._graph.edges[edge]
        )

    def __getitem__(self, item):
        if isinstance(item, (list, tuple)):
            from .dataset import Dataset

            return Dataset({k: self[k] for k in item})
        try:
            return self._getitem(item)
        except KeyError:
            return self._getitem(item, include_blank_dims=True)

    def finditem(self, item, maybe_in=None):
        if maybe_in is not None and maybe_in in self._graph.nodes:
            dataset = self._graph.nodes[maybe_in].get("dataset", {})
            if item in dataset:
                return maybe_in
        return self._getitem(item, just_node_name=True)

    def _getitem(
        self, item, include_blank_dims=False, only_dims=False, just_node_name=False
    ):

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
            dataset = self._graph.nodes[current_node].get("dataset", {})
            try:
                by_name = item in dataset and not only_dims
            except TypeError:
                by_name = False
            try:
                by_dims = not by_name and include_blank_dims and (item in dataset.dims)
            except TypeError:
                by_dims = False
            if (by_name or by_dims) and (item_in is None or item_in == current_node):
                if just_node_name:
                    return current_node
                if current_node == self.root_node_name:
                    if by_dims:
                        return xr.DataArray(
                            pd.RangeIndex(dataset.dims[item]), dims=item
                        )
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
                    for path in nx.algorithms.simple_paths.all_simple_edge_paths(
                        self._graph, self.root_node_name, current_node
                    ):
                        path_dim = self._graph.edges[path[-1]].get("child_name")
                        if path_dim not in dims_in_result:
                            continue
                        # path_indexing = self._graph.edges[path[-1]].get('indexing')
                        t1 = None
                        # intermediate nodes on path
                        for (e, e_next) in zip(path[:-1], path[1:]):
                            r = self._get_relationship(e)
                            r_next = self._get_relationship(e_next)
                            if t1 is None:
                                t1 = self._graph.nodes[r.parent_data].get("dataset")
                            t2 = self._graph.nodes[r.child_data].get("dataset")[
                                [r_next.parent_name]
                            ]
                            if r.indexing == "label":
                                t1 = t2.sel(
                                    {r.child_name: t1[r.parent_name].to_numpy()}
                                )
                            else:  # by position
                                t1 = t2.isel(
                                    {r.child_name: t1[r.parent_name].to_numpy()}
                                )
                        # final node in path
                        e = path[-1]
                        r = Relationship(
                            parent_data=e[0], child_data=e[1], **self._graph.edges[e]
                        )
                        if t1 is None:
                            t1 = self._graph.nodes[r.parent_data].get("dataset")
                        if r.indexing == "label":
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
                        y = y.rename({_i: _j for _i, _j in zip(y.dims, result.dims)})
                    return y
            else:
                examined.add(current_node)
                for _, next_up in self._graph.out_edges(current_node):
                    if next_up not in examined:
                        queue.append(next_up)

        raise KeyError(item)

    def get_expr(self, expression, engine="sharrow", allow_native=True):
        """
        Access or evaluate an expression.

        Parameters
        ----------
        expression : str
        engine : {'sharrow', 'numexpr'}
            The engine used to resolve expressions.
        allow_native : bool, default True
            If the expression is an array in a dataset of this tree, return
            that array directly.  Set to false to force evaluation, which
            will also ensure proper broadcasting consistent with this data tree.

        Returns
        -------
        DataArray
        """
        try:
            if allow_native:
                result = self[expression]
            else:
                raise KeyError
        except (KeyError, IndexError):
            if engine == "sharrow":
                result = (
                    self.setup_flow({expression: expression})
                    .load_dataarray()
                    .isel(expressions=0)
                )
            elif engine == "numexpr":
                from xarray import DataArray

                result = DataArray(
                    pd.eval(expression, resolvers=[self], engine="numexpr"),
                )
        return result

    @property
    def subspaces(self):
        """Mapping[str,Dataset] : Direct access to node Dataset objects by name."""
        spaces = {}
        for k in self._graph.nodes:
            s = self._graph.nodes[k].get("dataset", None)
            if s is not None:
                spaces[k] = s
        return spaces

    def subspaces_iter(self):
        for k in self._graph.nodes:
            s = self._graph.nodes[k].get("dataset", None)
            if s is not None:
                yield (k, s)

    def namespace_names(self):
        namespace = set()
        for spacename, spacearrays in self.subspaces_iter():
            for k, arr in spacearrays.coords.items():
                namespace.add(f"__{spacename or 'base'}__{k}")
            for k, arr in spacearrays.items():
                if k.startswith("_s_"):
                    namespace.add(f"__{spacename or 'base'}__{k}__indptr")
                    namespace.add(f"__{spacename or 'base'}__{k}__indices")
                    namespace.add(f"__{spacename or 'base'}__{k}__data")
                else:
                    namespace.add(f"__{spacename or 'base'}__{k}")
        return namespace

    @property
    def dims(self):
        """
        Mapping from dimension names to lengths across all dataset nodes.
        """
        dims = {}
        for k, v in self.subspaces_iter():
            for name, length in v.dims.items():
                if name in dims:
                    if dims[name] != length:
                        raise ValueError(
                            "inconsistent dimensions\n" + self.dims_detail()
                        )
                else:
                    dims[name] = length
        return xr.core.utils.Frozen(dims)

    def dims_detail(self):
        """
        Report on the names and sizes of dimensions in all Dataset nodes.

        Returns
        -------
        str
        """
        s = ""
        for k, v in self.subspaces_iter():
            s += f"\n{k}:"
            for name, length in v.dims.items():
                s += f"\n - {name}: {length}"
        return s[1:]

    def drop_dims(self, dims, inplace=False, ignore_missing_dims=True):
        """
        Drop dimensions from root Dataset node.

        Parameters
        ----------
        dims : str or Iterable[str]
            One or more named dimensions to drop.
        inplace : bool, default False
            Whether to drop dimensions in-place.
        ignore_missing_dims : bool, default True
            Simply ignore any dimensions that are not present.

        Returns
        -------
        DataTree
            Returns self if dropping inplace, otherwise returns a copy
            with dimensions dropped.
        """

        if isinstance(dims, str):
            dims = [dims]
        if inplace:
            obj = self
        else:
            obj = self.copy()

        if not ignore_missing_dims:
            new_root_dataset = obj.root_dataset.drop_dims(dims)
        else:
            new_root_dataset = obj.root_dataset
            for d in dims:
                if d in obj.root_dataset.dims:
                    new_root_dataset = new_root_dataset.drop_dims(d)

        # remove subspaces that rely on dropped dim
        boot_queue = set()
        booted = set()
        for (up, dn, n), e in obj._graph.edges.items():
            if up == obj.root_node_name:
                _analog = e.get("analog", "<missing>")
                if _analog in dims:
                    boot_queue.add(dn)
                if _analog != "<missing>" and _analog not in new_root_dataset:
                    boot_queue.add(dn)
                if e.get("parent_name", "<missing>") in dims:
                    boot_queue.add(dn)
                if e.get("parent_name", "<missing>") not in new_root_dataset:
                    boot_queue.add(dn)

        while boot_queue:
            b = boot_queue.pop()
            booted.add(b)
            for (up, dn, n), e in obj._graph.edges.items():
                if up == b:
                    boot_queue.add(dn)

        edges_to_remove = [e for e in obj._graph.edges if e[1] in booted]
        obj._graph.remove_edges_from(edges_to_remove)
        obj._graph.remove_nodes_from(booted)

        obj.root_dataset = new_root_dataset
        obj.dim_order = tuple(x for x in self.dim_order if x not in dims)
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
        dims = [
            d
            for d in self.dims
            if d[-1:] != "_" or (d[-1:] == "_" and d[:-1] not in self.dims)
        ]
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
                    raise ValueError(
                        f"inconsistent index shapes {result_k.shape} v {result_shape} (probably an error on {k} or {sorted(dims)[0]})"
                    )
            result[k] = result_k

        if as_dict:
            result = {k: v.to_numpy() for k, v in result.items()}
        else:
            result = Dataset(result)
        if use_cache:
            self._cached_indexes[(position_only, as_dict)] = result
        return result

    def replace_datasets(self, other=None, validate=True, redigitize=True, **kwargs):
        """
        Replace one or more datasets in the nodes of this tree.

        Parameters
        ----------
        other : Mapping[str,Dataset]
            A dictionary of replacement datasets.
        validate : bool, default True
            Raise an error when replacing downstream datasets that
            are referenced by position, unless the replacement is identically
            sized.  If validation is deactivated, and an incompatible dataset
            is placed in this tree, flows that rely on that relationship will
            give erroneous results or crash with a segfault.
        redigitize : bool, default True
            Automatically re-digitize relationships that are label-based and
            were previously digitized.
        **kwargs : Mapping[str,Dataset]
            Alternative format to `other`.

        Returns
        -------
        DataTree
            A new DataTree with data replacements completed.
        """
        replacements = {}
        if other is not None:
            replacements.update(other)
        replacements.update(kwargs)
        graph = self._graph.copy()
        for k in replacements:
            if k not in graph.nodes:
                raise KeyError(k)
            x = construct(replacements[k])
            if validate:
                if x.dims != graph.nodes[k]["dataset"].dims:
                    # when replacement dimensions do not match, check for
                    # any upstream nodes that reference this dataset by
                    # position... which will potentially be problematic.
                    for e in self._graph.edges:
                        if e[1] == k:
                            indexing = self._graph.edges[e].get("indexing")
                            if indexing == "position":
                                raise ValueError(
                                    f"dimensions mismatch on "
                                    f"positionally-referenced dataset {k}: "
                                    f"receiving {x.dims} "
                                    f"expected {graph.nodes[k]['dataset'].dims}"
                                )
            if k in self.replacement_filters:
                x = self.replacement_filters[k](x)
            graph.nodes[k]["dataset"] = x
        result = type(self)(graph, self.root_node_name, **self.__shallow_copy_extras())
        if redigitize:
            result.digitize_relationships(inplace=True)
        return result

    def setup_flow(
        self,
        definition_spec,
        *,
        cache_dir=None,
        name=None,
        dtype="float32",
        boundscheck=False,
        error_model="numpy",
        nopython=True,
        fastmath=True,
        parallel=True,
        readme=None,
        flow_library=None,
        extra_hash_data=(),
        write_hash_audit=True,
        hashing_level=1,
        dim_exclude=None,
    ):
        """
        Set up a new Flow for analysis using the structure of this DataTree.

        Parameters
        ----------
        definition_spec : Dict[str,str]
            Gives the names and expressions that define the variables to
            create in this new `Flow`.
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
        error_model : {'numpy', 'python'}, default 'numpy'
            The error_model option controls the divide-by-zero behavior. Setting
            it to ‘python’ causes divide-by-zero to raise exception like
            CPython. Setting it to ‘numpy’ causes divide-by-zero to set the
            result to +/-inf or nan.
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
            hash conflicts for similar flows.  Level 2 adds information about
            names used in expressions and digital encodings to the flow hash,
            which prevents conflicts but requires more pre-computation to generate
            the hash.
        dim_exclude : Collection[str], optional
            Exclude these root dataset dimensions from this flow.

        Returns
        -------
        Flow
        """
        from .flows import Flow

        return Flow(
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
            error_model=error_model,
            write_hash_audit=write_hash_audit,
            dim_order=self.dim_order,
            dim_exclude=dim_exclude,
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
        if name1 == "aux_var":
            return self.aux_vars[name2]
        dataset = self._graph.nodes[name1].get("dataset")
        if name2.startswith("_s_"):
            if name2.endswith("__data"):
                return dataset[name2[:-6]].data.data
            elif name2.endswith("__indptr"):
                return dataset[name2[:-8]].data.indptr
            elif name2.endswith("__indices"):
                return dataset[name2[:-9]].data.indices
        return dataset[name2].to_numpy()

    _BY_OFFSET = "digitizedOffset"

    def digitize_relationships(self, inplace=False, redigitize=True):
        """
        Convert all label-based relationships into position-based.

        Parameters
        ----------
        inplace : bool, default False
        redigitize : bool, default True
            Re-compute position-based relationships from labels, even
            if the relationship had previously been digitized.

        Returns
        -------
        DataTree or None
            Only returns a copy if not digitizing in-place.
        """

        if inplace:
            obj = self
        else:
            obj = self.copy()

        for e in obj._graph.edges:
            r = obj._get_relationship(e)
            if redigitize and r.analog:
                p_dataset = obj._graph.nodes[r.parent_data].get("dataset", None)
                if p_dataset is not None:
                    if r.parent_name not in p_dataset:
                        r.indexing = "label"
                        r.parent_name = r.analog
            if r.indexing == "label":
                p_dataset = obj._graph.nodes[r.parent_data].get("dataset", None)
                c_dataset = obj._graph.nodes[r.child_data].get("dataset", None)

                upstream = p_dataset[r.parent_name]
                downstream = c_dataset[r.child_name]

                # vectorize version
                mapper = {i: j for (j, i) in enumerate(downstream.to_numpy())}
                offsets = xr.apply_ufunc(np.vectorize(mapper.get), upstream)
                if offsets.dtype.kind != "i":
                    warnings.warn(
                        f"detected missing values in digitizing {r.parent_data}.{r.parent_name}",
                    )

                # candidate name for write back
                r_parent_name_new = (
                    f"{self._BY_OFFSET}{r.parent_name}_{r.child_data}_{r.child_name}"
                )

                # it is common to have mirrored offsets in various dimensions.
                # we'd like to retain only the same data in memory once, so we'll
                # check if these offsets match any existing ones and if so just
                # point to that memory.
                for k in p_dataset:
                    if isinstance(k, str) and k.startswith(self._BY_OFFSET):
                        if p_dataset[k].equals(offsets):
                            # we found a match, so we'll assign this name to
                            # the match's memory storage instead of replicating it.
                            obj._graph.nodes[r.parent_data][
                                "dataset"
                            ] = p_dataset.assign({r_parent_name_new: p_dataset[k]})
                            # r_parent_name_new = k
                            break
                else:
                    # no existing offset arrays match, make this new one
                    obj._graph.nodes[r.parent_data]["dataset"] = p_dataset.assign(
                        {r_parent_name_new: offsets}
                    )
                obj._graph.edges[e].update(
                    dict(
                        parent_name=r_parent_name_new,
                        indexing="position",
                        analog=r.parent_name,
                    )
                )

        if not inplace:
            return obj

    @property
    def relationships_are_digitized(self):
        """bool : Whether all relationships are digital (by position)."""
        for e in self._graph.edges:
            r = self._get_relationship(e)
            if r.indexing != "position":
                return False
        return True

    def _arg_tokenizer(
        self, spacename, spacearray, spacearrayname, exclude_dims=None, blends=None
    ):

        if blends is None:
            blends = {}

        if spacename == self.root_node_name:
            root_dataset = self.root_dataset
            from .flows import presorted

            root_dims = list(presorted(root_dataset.dims, self.dim_order, exclude_dims))
            if isinstance(spacearray, str):
                from_dims = root_dataset[spacearray].dims
            else:
                from_dims = spacearray.dims
            return (
                tuple(
                    ast.parse(f"_arg{root_dims.index(dim):02}", mode="eval").body
                    for dim in from_dims
                ),
                blends,
            )

        if isinstance(spacearray, str):
            spacearray_ = self._graph.nodes[spacename]["dataset"][spacearray]
        else:
            spacearray_ = spacearray

        from_dims = spacearray_.dims
        offset_source = spacearray_.attrs.get("digital_encoding", {}).get(
            "offset_source", None
        )
        if offset_source is not None:
            from_dims = self._graph.nodes[spacename]["dataset"][offset_source].dims

        tokens = []

        n_missing_tokens = 0
        for dimname in from_dims:
            found_token = False
            for e in self._graph.in_edges(spacename, keys=True):
                this_dim_name = self._graph.edges[e]["child_name"]
                retarget = None
                if dimname != this_dim_name:
                    retarget = self._graph.nodes[spacename][
                        "dataset"
                    ].redirection.target(this_dim_name)
                    if dimname != retarget:
                        continue
                parent_name = self._graph.edges[e]["parent_name"]
                parent_data = e[0]

                upside_ast, blends_ = self._arg_tokenizer(
                    parent_data,
                    parent_name,
                    spacearrayname=spacearrayname,
                    exclude_dims=exclude_dims,
                    blends=blends,
                )
                try:
                    upside = ", ".join(unparse(t) for t in upside_ast)
                except:  # noqa: E722
                    for t in upside_ast:
                        print(f"t:{t}")
                    raise

                # check for redirection target
                if retarget is not None:
                    tokens.append(
                        f"__{spacename}___digitized_{retarget}_of_{this_dim_name}[__{parent_data}__{parent_name}[{upside}]]"
                    )
                else:
                    tokens.append(f"__{parent_data}__{parent_name}[{upside}]")
                found_token = True
                break
            if not found_token:
                if dimname in self.subspaces[spacename].indexes:
                    ix = self.subspaces[spacename].indexes[dimname]
                    ix = {i: n for n, i in enumerate(ix)}
                    tokens.append(ix)
                    n_missing_tokens += 1
                elif dimname.endswith("_indices") or dimname.endswith("_indptr"):
                    tokens.append(None)
                    # this dimension corresponds to a blender

        if n_missing_tokens > 1:
            raise ValueError("at most one missing dimension is allowed")
        result = []
        for t in tokens:
            if isinstance(t, str):
                # print(f"TOKENIZE: {spacename=} {spacearray=} {t}")
                result.append(ast.parse(t, mode="eval").body)
            else:
                result.append(t)
        return tuple(result), blends

    @property
    def coords(self):
        return self.root_dataset.coords

    def get_index(self, dim):
        for spacename, subspace in self.subspaces.items():
            if dim in subspace.coords:
                return subspace.indexes[dim]

    def copy(self):
        return type(self)(
            self._graph.copy(), self.root_node_name, **self.__shallow_copy_extras()
        )
