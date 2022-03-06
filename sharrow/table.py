import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf
import yaml

from .aster import Asterize


def _getsize(pa_array):
    if pa_array.null_count:
        return 1_000_000
        # raise ValueError("getsize does not play nicely with nulls yet")
    try:
        return pa_array.nbytes * 8 // pa_array.type.bit_width
    except ValueError:
        # Non-fixed width type
        return 1_000_000


class Table:
    def __new__(cls, init, *args, **kwargs):
        if init is None:
            return None
        if isinstance(init, cls):
            return init
        return super().__new__(cls)

    def __init__(self, init, nthreads=None):
        if isinstance(init, Table):
            self._table = init._table
        elif isinstance(init, pd.DataFrame):
            self._table = pa.Table.from_pandas(init, nthreads=nthreads)
        elif isinstance(init, dict):
            self._table = pa.Table.from_pydict(init)
        elif isinstance(init, pa.Table):
            self._table = init
        elif isinstance(init, int):
            self._table = pa.table([])
            self._tentative_len = init
        else:
            raise TypeError(f"init is invalid type {type(init)}")
        self._revise = Asterize()

    def __repr__(self):
        r = repr(self._table)
        return r.replace("pyarrow.Table", "sharrow.Table")

    def append_column(self, name, value, overwrite=True):
        """
        Append a column to the Table.

        Unlike a regular pyarrow Table, by default this will remove any
        existing column with the same name.

        Parameters
        ----------
        name : str
        value : array-like
        overwrite : bool, default True

        Returns
        -------
        self
        """
        if len(self._table):
            if overwrite:
                try:
                    position = self._table.column_names.index(name)
                except ValueError:
                    pass
                else:
                    self._table = self._table.remove_column(position)
            self._table = self._table.append_column(name, value)
        else:
            self._table = pa.table({name: value})
        return self

    def __getattr__(self, item):
        return getattr(self._table, item)

    def __getitem__(self, item):
        return self._table[item]

    def __setitem__(self, item, value):
        if not isinstance(value, np.ndarray):
            value = np.asarray(value)
        result = np.broadcast_to(value, (self._table.num_rows))
        self._table = self._table.append_column(item, [result])

    def __len__(self):
        return len(self._table) or getattr(self, "_tentative_len", 0)

    def concat(self, others):
        tables = [self._table]
        if isinstance(others, Table):
            others = [others]
        for other in others:
            if isinstance(other, Table):
                other = other._table
            tables.append(other)
        self._table = pa.concat_tables(tables)

    @classmethod
    def from_pydict(cls, mapping):
        return cls(mapping)

    def to_feather(self, dest):
        """
        Write this table to Feather format.

        Parameters
        ----------
        dest : str
            Local destination path.
        compression : string, default None
            Can be one of {"zstd", "lz4", "uncompressed"}. The default of None uses
            LZ4 for V2 files if it is available, otherwise uncompressed.
        compression_level : int, default None
            Use a compression level particular to the chosen compressor. If None
            use the default compression level
        chunksize : int, default None
            For V2 files, the internal maximum size of Arrow RecordBatch chunks
            when writing the Arrow IPC file format. None means use the default,
            which is currently 64K
        version : int, default 2
            Feather file version. Version 2 is the current. Version 1 is the more
            limited legacy format
        """
        import pyarrow.feather as pf

        return pf.write_feather(
            self._table,
            dest,
            compression=None,
            compression_level=None,
            chunksize=None,
            version=2,
        )

    @classmethod
    def from_feather(cls, source, columns=None, memory_map=True):
        import pyarrow.feather as pf

        t = pf.read_table(source, columns=columns, memory_map=memory_map)
        return cls(t)

    def __or__(self, other):
        result = self.__class__(self._table)
        for k in other.column_names:
            result.append_column(k, other[k])
        return result

    def __dir__(self):
        return dir(self._table) + [
            "eval",
            "concat",
        ]

    def eval(
        self, expression, inplace=False, target=None, local_dict=None, nopython=True
    ):
        import numexpr as ne

        if isinstance(expression, str):
            try:
                num = float(expression.split("#")[0].strip())
            except:  # noqa: E722
                pass
            else:
                if not inplace:
                    return np.full(len(self), num)
                else:
                    if not target:
                        raise ValueError("cannot assign inplace without a target name")
                    self._table = self._table.append_column(
                        target, np.full(len(self), num)
                    )
                    return
            if target is not False:
                ex_target, ex_value = self._revise(expression)
                target = target or ex_target
            else:
                ex_value = expression
            if inplace and not target:
                target = expression
                # raise ValueError("cannot operate inplace without an assignment or target name")
            try:
                if self._table.num_rows > self._table.num_columns * 1000:
                    table = self._table
                    chunk_start = 0
                    result_chunks = []
                    while chunk_start < len(table):
                        chunk_len = min(
                            [
                                _getsize(i.slice(chunk_start).chunks[0])
                                for i in table.itercolumns()
                            ]
                        )
                        if chunk_start == 0:
                            j = ne.evaluate(
                                ex_value,
                                self._table.slice(chunk_start, chunk_len),
                                local_dict or globals(),
                            )
                        else:
                            j = ne.re_evaluate(
                                self._table.slice(chunk_start, chunk_len),
                            )
                        result_chunks.append(j)
                        chunk_start = chunk_start + chunk_len
                    expression_val = pa.chunked_array(result_chunks)
                else:
                    # wide or small tables, just go for broke
                    expression_val = ne.evaluate(
                        ex_value,
                        self._table,
                        local_dict or globals(),
                    )
            except Exception as err:
                if nopython:
                    raise type(err)(f"({expression}) {err!s}") from err
                expression_val = eval(
                    ex_value,
                    self._globals,
                    self._table,
                )
            if not inplace:
                return np.asarray(expression_val)
            try:
                if not isinstance(expression_val, pa.ChunkedArray):
                    expression_val = [expression_val]
                self._table = self._table.append_column(target, expression_val)
            except Exception as err:
                raise type(err)(f"({expression}) {err!s}") from err
        else:
            if not target:
                raise ValueError("cannot assign inplace without a target name")
            if not isinstance(expression, np.ndarray):
                expression = np.asarray(expression)
            try:
                result = np.broadcast_to(expression, (self._table.num_rows))
                if not inplace:
                    return result
                self._table = self._table.append_column(target, [result])
            except Exception as err:
                raise type(err)(f"target={target}") from err

    @classmethod
    def from_quilt(cls, path, blockname=None):
        if not os.path.isdir(path):
            raise NotADirectoryError(path)
        if blockname is not None:
            if isinstance(blockname, int):
                stopper = blockname
            else:
                qlog = os.path.join(path, "quilt.log")
                with open(qlog, "rt") as logreader:
                    existing_info = yaml.safe_load(logreader)
                for stopper, block in enumerate(existing_info):
                    if block.get("name", None) == blockname:
                        break
                else:
                    raise KeyError(blockname)
        else:
            stopper = 1e99
        n = 0
        rowfile = lambda n: os.path.join(path, f"block.{n:03d}.rows")
        colfile = lambda n: os.path.join(path, f"block.{n:03d}.cols")
        builder = None
        look = True
        while look and n <= stopper:
            look = False
            if os.path.exists(rowfile(n)):
                i = pf.read_table(rowfile(n))
                if builder is None:
                    builder = cls(i)
                else:
                    builder.concat([i])
                look = True
            if os.path.exists(colfile(n)):
                i = pf.read_table(colfile(n))
                if builder is None:
                    builder = cls(i)
                else:
                    builder = builder | i
                look = True
            if look:
                n += 1
        if builder is not None:
            metadata = builder.schema.metadata
            metadata[b"quilt_number"] = f"{n}".encode("utf8")
            return builder.replace_schema_metadata(metadata)
        return None

    def to_quilt(self, path, blockname=None):
        if not os.path.exists(path):
            os.makedirs(path)
        qlog = os.path.join(path, "quilt.log")

        if not os.path.exists(qlog):
            ex_rows = 0
            ex_cols = []
            max_block = -1
        else:
            with open(qlog, "rt") as logreader:
                existing_info = yaml.safe_load(logreader)
            ex_rows = sum(block.get("rows", 0) for block in existing_info)
            ex_cols = sum((block.get("cols", []) for block in existing_info), [])
            max_block = max(block["block"] for block in existing_info)

        with open(qlog, "a") as f:
            quilt_number = max_block + 1
            f.write(f"- block: {quilt_number}\n")
            if blockname is not None:
                f.write(f"  name: {blockname}\n")
            if quilt_number == 0:
                blob_file = f"block.{quilt_number:03d}.rows"
                pf.write_feather(
                    self._table,
                    os.path.join(path, blob_file),
                )
                f.write(f"  rows: {len(self)}\n")
                f.write("  cols:\n")
                for ci in self.column_names:
                    f.write(f"    - {ci}\n")
            else:
                add_rows = self[ex_rows:]
                if len(add_rows):
                    if len(ex_cols) == 0:
                        row_block = add_rows
                    else:
                        c1 = [i for i in self.column_names if i in ex_cols]
                        row_block = add_rows.select(c1)
                    blob_file = f"block.{quilt_number:03d}.rows"
                    pf.write_feather(
                        row_block,
                        os.path.join(path, blob_file),
                    )
                    f.write(f"  rows: {len(add_rows)}\n")
                c2 = [i for i in self.column_names if i not in ex_cols]
                if len(c2):
                    col_block = self.select(c2)
                    blob_file = f"block.{quilt_number:03d}.cols"
                    pf.write_feather(
                        col_block,
                        os.path.join(path, blob_file),
                    )
                    f.write("  cols:\n")
                    for ci in c2:
                        f.write(f"    - {ci}\n")


def concat_tables(tables):
    _tables = []
    for t in tables:
        if isinstance(t, Table):
            t = t._table
        _tables.append(t)
    return Table(pa.concat_tables(_tables))
