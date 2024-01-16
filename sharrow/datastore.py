from __future__ import annotations

import datetime
import os
import shutil
from collections.abc import Collection
from pathlib import Path

import pandas as pd
import xarray as xr
import yaml

from .dataset import construct, from_zarr_with_attr
from .relationships import DataTree, Relationship


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()


class ReadOnlyError(ValueError):
    """Object is read-only."""


def _read_parquet(filename, index_col=None) -> xr.Dataset:
    import pyarrow.parquet as pq

    from sharrow.dataset import from_table

    content = pq.read_table(filename)
    if index_col is not None:
        index = content.column(index_col)
        content = content.drop([index_col])
    else:
        index = None
    x = from_table(content, index=index, index_name=index_col or "index")
    return x


class DataStore:
    metadata_filename: str = "metadata.yaml"
    checkpoint_subdir: str = "checkpoints"
    LATEST = "_"
    _BY_OFFSET = "digitizedOffset"

    def __init__(self, directory: Path | None, mode="a", storage_format: str = "zarr"):
        self._directory = Path(directory) if directory else directory
        self._mode = mode
        self._checkpoints = {}
        self._checkpoint_order = []
        self._tree = DataTree(root_node_name=False)
        self._keep_digitized = False
        assert storage_format in {"zarr", "parquet", "hdf5"}
        self._storage_format = storage_format
        try:
            self.read_metadata()
        except FileNotFoundError:
            pass

    @property
    def directory(self) -> Path:
        if self._directory is None:
            raise NotADirectoryError("no directory set")
        return self._directory

    def __setitem__(self, key: str, value: xr.Dataset | pd.DataFrame):
        assert isinstance(key, str)
        if self._mode == "r":
            raise ReadOnlyError
        if isinstance(value, xr.Dataset):
            self._set_dataset(key, value)
        elif isinstance(value, pd.DataFrame):
            self._set_dataset(key, construct(value))
        else:
            raise TypeError(f"cannot put {type(value)}")

    def __getitem__(self, key: str):
        assert isinstance(key, str)
        return self.get_dataset(key)

    def clone(self, mode="a"):
        """
        Create a clone of this DataStore.

        The clone has the same active datasets as the original (and the data
        shares the same memory) but it does not retain the checkpoint metadata
        and is not connected to the same checkpoint store.

        Returns
        -------
        DataStore
        """
        duplicate = self.__class__(None, mode=mode)
        duplicate._tree = self._tree
        return duplicate

    def _set_dataset(
        self,
        name: str,
        data: xr.Dataset,
        last_checkpoint: str = None,
    ) -> None:
        if self._mode == "r":
            raise ReadOnlyError
        data_vars = {}
        coords = {}
        for k, v in data.coords.items():
            coords[k] = v.assign_attrs(last_checkpoint=last_checkpoint)
        for k, v in data.items():
            if k in coords:
                continue
            data_vars[k] = v.assign_attrs(last_checkpoint=last_checkpoint)
        data = xr.Dataset(data_vars=data_vars, coords=coords, attrs=data.attrs)
        self._tree.add_dataset(name, data)

    def _update_dataset(
        self,
        name: str,
        data: xr.Dataset,
        last_checkpoint=None,
    ) -> xr.Dataset:
        if self._mode == "r":
            raise ReadOnlyError
        if not isinstance(data, xr.Dataset):
            raise TypeError(type(data))
        partial_update = self._tree.get_subspace(name, default_empty=True)
        for k, v in data.items():
            if k in data.coords:
                continue
            assert v.name == k
            partial_update = self._update_dataarray(
                name, v, last_checkpoint, partial_update=partial_update
            )
        for k, v in data.coords.items():
            assert v.name == k
            partial_update = self._update_dataarray(
                name, v, last_checkpoint, as_coord=True, partial_update=partial_update
            )
        return partial_update

    def _update_dataarray(
        self,
        name: str,
        data: xr.DataArray,
        last_checkpoint=None,
        as_coord=False,
        partial_update=None,
    ) -> xr.Dataset:
        if self._mode == "r":
            raise ReadOnlyError
        if partial_update is None:
            base_data = self._tree.get_subspace(name, default_empty=True)
        else:
            base_data = partial_update
        if isinstance(data, xr.DataArray):
            if as_coord:
                updated_dataset = base_data.assign_coords(
                    {data.name: data.assign_attrs(last_checkpoint=last_checkpoint)}
                )
            else:
                updated_dataset = base_data.assign(
                    {data.name: data.assign_attrs(last_checkpoint=last_checkpoint)}
                )
            self._tree = self._tree.replace_datasets(
                {name: updated_dataset}, redigitize=self._keep_digitized
            )
            return updated_dataset
        else:
            raise TypeError(type(data))

    def update(
        self,
        name: str,
        obj: xr.Dataset | xr.DataArray,
        last_checkpoint: str = None,
    ) -> None:
        """
        Make a partial update of an existing named dataset.

        Parameters
        ----------
        name : str
        obj : Dataset or DataArray
        last_checkpoint : str or None
            Set the "last_checkpoint" attribute on all updated variables to this
            value.  Users should typically leave this as "None", which flags the
            checkpointing algorithm to write this data to disk the next time a
            checkpoint is written.
        """
        if isinstance(obj, xr.Dataset):
            self._update_dataset(name, obj, last_checkpoint=last_checkpoint)
        elif isinstance(obj, xr.DataArray):
            self._update_dataarray(name, obj, last_checkpoint=last_checkpoint)
        else:
            raise TypeError(type(obj))

    def set_data(
        self,
        name: str,
        data: xr.Dataset | pd.DataFrame,
        relationships: str | Relationship | Collection[str | Relationship] = None,
    ) -> None:
        """
        Set the content of a named dataset.

        This completely overwrites any existing data with the same name.

        Parameters
        ----------
        name : str
        data : Dataset or DataFrame
        relationships : str or Relationship or list thereof
        """
        self.__setitem__(name, data)
        if relationships is not None:
            if isinstance(relationships, (str, Relationship)):
                relationships = [relationships]
            for r in relationships:
                self._tree.add_relationship(r)

    def get_dataset(self, name: str, columns: Collection[str] = None) -> xr.Dataset:
        """
        Retrieve some or all of a named dataset.

        Parameters
        ----------
        name : str
        columns : Collection[str], optional
            Get only these variables of the dataset.
        """
        if columns is None:
            return self._tree.get_subspace(name)
        else:
            return xr.Dataset({c: self._tree[f"{name}.{c}"] for c in columns})

    def get_dataframe(self, name: str, columns: Collection[str] = None) -> pd.DataFrame:
        """
        Retrieve some or all of a named dataset, as a pandas DataFrame.

        This completely overwrites any existing data with the same name.

        Parameters
        ----------
        name : str
        columns : Collection[str], optional
            Get only these variables of the dataset.
        """
        dataset = self.get_dataset(name, columns)
        return dataset.single_dim.to_pandas()

    def _to_be_checkpointed(self) -> dict[str, xr.Dataset]:
        result = {}
        for table_name, table_data in self._tree.subspaces_iter():
            # any data elements that were created without a
            # last_checkpoint attr get one now
            for _k, v in table_data.variables.items():
                if "last_checkpoint" not in v.attrs:
                    v.attrs["last_checkpoint"] = None
            # collect everything not checkpointed
            uncheckpointed = table_data.filter_by_attrs(last_checkpoint=None)
            if uncheckpointed:
                result[table_name] = uncheckpointed
        return result

    def _zarr_subdir(self, table_name, checkpoint_name):
        return self.directory.joinpath(table_name, checkpoint_name).with_suffix(".zarr")

    def _parquet_name(self, table_name, checkpoint_name):
        return self.directory.joinpath(table_name, checkpoint_name).with_suffix(
            ".parquet"
        )

    def make_checkpoint(self, checkpoint_name: str, overwrite: bool = True):
        """
        Write data to disk.

        Only new data (since the last time a checkpoint was made) is actually
        written out.

        Parameters
        ----------
        checkpoint_name : str
        overwrite : bool, default True
        """
        if self._mode == "r":
            raise ReadOnlyError
        to_be_checkpointed = self._to_be_checkpointed()
        new_checkpoint = {
            "timestamp": timestamp(),
            "tables": {},
            "relationships": [],
        }
        # remove checkpoint name from ordered list if it already exists
        while checkpoint_name in self._checkpoint_order:
            self._checkpoint_order.remove(checkpoint_name)
        # add checkpoint name at end ordered list
        self._checkpoint_order.append(checkpoint_name)
        for table_name, table_data in to_be_checkpointed.items():
            if self._storage_format == "parquet" and len(table_data.dims) == 1:
                target = self._parquet_name(table_name, checkpoint_name)
                if overwrite and target.is_file():
                    os.unlink(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                table_data.single_dim.to_parquet(str(target))
            elif self._storage_format == "zarr" or (
                self._storage_format == "parquet" and len(table_data.dims) > 1
            ):
                # zarr is used if ndim > 1
                target = self._zarr_subdir(table_name, checkpoint_name)
                if overwrite and target.is_dir():
                    shutil.rmtree(target)
                target.mkdir(parents=True, exist_ok=True)
                table_data.to_zarr_with_attr(target)
            elif self._storage_format == "hdf5":
                raise NotImplementedError
            else:
                raise ValueError(
                    f"cannot write with storage format {self._storage_format!r}"
                )
            self.update(table_name, table_data, last_checkpoint=checkpoint_name)
        for table_name, table_data in self._tree.subspaces_iter():
            inventory = {"data_vars": {}, "coords": {}}
            for varname, vardata in table_data.items():
                inventory["data_vars"][varname] = {
                    "last_checkpoint": vardata.attrs.get("last_checkpoint", "MISSING"),
                    "dtype": str(vardata.dtype),
                }
            for varname, vardata in table_data.coords.items():
                _cp = checkpoint_name
                # coords in every checkpoint with any content
                if table_name not in to_be_checkpointed:
                    _cp = vardata.attrs.get("last_checkpoint", "MISSING")
                inventory["coords"][varname] = {
                    "last_checkpoint": _cp,
                    "dtype": str(vardata.dtype),
                }
            new_checkpoint["tables"][table_name] = inventory
        for r in self._tree.list_relationships():
            new_checkpoint["relationships"].append(r.to_dict())
        self._checkpoints[checkpoint_name] = new_checkpoint
        self._write_checkpoint(checkpoint_name, new_checkpoint)
        self._write_metadata()

    def _write_checkpoint(self, name, checkpoint):
        if self._mode == "r":
            raise ReadOnlyError
        checkpoint_metadata_target = self.directory.joinpath(
            self.checkpoint_subdir, f"{name}.yaml"
        )
        if checkpoint_metadata_target.exists():
            n = 1
            while checkpoint_metadata_target.with_suffix(f".{n}.yaml").exists():
                n += 1
            os.rename(
                checkpoint_metadata_target,
                checkpoint_metadata_target.with_suffix(f".{n}.yaml"),
            )
        checkpoint_metadata_target.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_metadata_target, "w") as f:
            yaml.safe_dump(checkpoint, f)

    def _write_metadata(self):
        if self._mode == "r":
            raise ReadOnlyError
        metadata_target = self.directory.joinpath(self.metadata_filename)
        if metadata_target.exists():
            n = 1
            while metadata_target.with_suffix(f".{n}.yaml").exists():
                n += 1
            os.rename(metadata_target, metadata_target.with_suffix(f".{n}.yaml"))
        with open(metadata_target, "w") as f:
            metadata = dict(
                datastore_format_version=1,
                checkpoint_order=self._checkpoint_order,
            )
            yaml.safe_dump(metadata, f)

    def read_metadata(self, checkpoints=None):
        """
        Read storage metadata.

        Parameters
        ----------
        checkpoints : str | list[str], optional
            Read only these checkpoints.  If not provided, only the latest
            checkpoint metadata is read. Set to "*" to read all.
        """
        with open(self.directory.joinpath(self.metadata_filename)) as f:
            metadata = yaml.safe_load(f)
        datastore_format_version = metadata.get("datastore_format_version", "missing")
        if datastore_format_version == 1:
            self._checkpoint_order = metadata["checkpoint_order"]
        else:
            raise NotImplementedError(f"{datastore_format_version=}")
        if checkpoints is None or checkpoints == self.LATEST:
            checkpoints = [self._checkpoint_order[-1]]
        elif isinstance(checkpoints, str):
            if checkpoints == "*":
                checkpoints = list(self._checkpoint_order)
            else:
                checkpoints = [checkpoints]
        for c in checkpoints:
            with open(
                self.directory.joinpath(self.checkpoint_subdir, f"{c}.yaml")
            ) as f:
                self._checkpoints[c] = yaml.safe_load(f)

    def restore_checkpoint(self, checkpoint_name: str):
        if checkpoint_name not in self._checkpoints:
            try:
                self.read_metadata(checkpoint_name)
            except FileNotFoundError:
                raise KeyError(checkpoint_name) from None
        checkpoint = self._checkpoints[checkpoint_name]
        self._tree = DataTree(root_node_name=False)
        for table_name, table_def in checkpoint["tables"].items():
            if table_name == "timestamp":
                continue
            t = xr.Dataset()
            opened_targets = {}
            coords = table_def.get("coords", {})
            if len(coords) == 1:
                index_name = list(coords)[0]
            else:
                index_name = None
            for coord_name, coord_def in coords.items():
                target = self._zarr_subdir(table_name, coord_def["last_checkpoint"])
                if target.exists():
                    if target not in opened_targets:
                        opened_targets[target] = from_zarr_with_attr(target)
                else:
                    # zarr not found, try parquet
                    target2 = self._parquet_name(
                        table_name, coord_def["last_checkpoint"]
                    )
                    if target2.exists():
                        if target not in opened_targets:
                            opened_targets[target] = _read_parquet(target2, index_name)
                    else:
                        raise FileNotFoundError(target)
                t = t.assign_coords({coord_name: opened_targets[target][coord_name]})
            data_vars = table_def.get("data_vars", {})
            for var_name, var_def in data_vars.items():
                if var_def["last_checkpoint"] == "MISSING":
                    raise ValueError(f"missing checkpoint for {table_name}.{var_name}")
                target = self._zarr_subdir(table_name, var_def["last_checkpoint"])
                if target.exists():
                    if target not in opened_targets:
                        opened_targets[target] = from_zarr_with_attr(target)
                else:
                    # zarr not found, try parquet
                    target2 = self._parquet_name(table_name, var_def["last_checkpoint"])
                    if target2.exists():
                        if target not in opened_targets:
                            opened_targets[target] = _read_parquet(target2, index_name)
                    else:
                        raise FileNotFoundError(target)
                t = t.assign({var_name: opened_targets[target][var_name]})
            self._tree.add_dataset(table_name, t)
        for r in checkpoint["relationships"]:
            self._tree.add_relationship(Relationship(**r))

    def add_relationship(self, relationship: str | Relationship):
        self._tree.add_relationship(relationship)

    def digitize_relationships(self, redigitize=True):
        """
        Convert all label-based relationships into position-based.

        Parameters
        ----------
        redigitize : bool, default True
            Re-compute position-based relationships from labels, even
            if the relationship had previously been digitized.
        """
        self._keep_digitized = True
        self._tree.digitize_relationships(inplace=True, redigitize=redigitize)

    @property
    def relationships_are_digitized(self) -> bool:
        """Bool : Whether all relationships are digital (by position)."""
        return self._tree.relationships_are_digitized
