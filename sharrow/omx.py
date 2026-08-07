import logging
from pathlib import Path

import h5py

logger = logging.getLogger("sharrow.omx")


def _initialize_omx_file(source: h5py.File, target: h5py.File) -> None:
    """Copy OMX metadata and ensure the standard HDF5 groups exist."""
    for name, value in source.attrs.items():
        target.attrs[name] = value
    for group_name in ("data", "lookup"):
        source_group = source[group_name]
        target_group = target.require_group(group_name)
        for name, value in source_group.attrs.items():
            target_group.attrs[name] = value


def _copy_dataset(
    source_group: h5py.Group, target_group: h5py.Group, name: str
) -> None:
    """Copy one HDF5 dataset, replacing a same-named target if present."""
    if name in target_group:
        del target_group[name]
    source_group.copy(name, target_group, name=name)


def split_omx(source_file, dest_directory, global_lookups=False, n_chunks=None):
    """Split the matrices in an OMX file across smaller OMX files.

    Parameters
    ----------
    source_file : str or path-like
        OMX-format HDF5 file to split.
    dest_directory : str or path-like
        Directory in which to write the split files.
    global_lookups : bool, default False
        If true, copy all lookups into every matrix output. Otherwise, write
        each lookup to its own OMX file.
    n_chunks : int, optional
        Number of output matrix files. By default, write one file per matrix.
    """
    if n_chunks is not None and n_chunks < 1:
        raise ValueError("n_chunks must be a positive integer")

    source_path = Path(source_file)
    destination = Path(dest_directory)
    destination.mkdir(parents=True, exist_ok=True)

    with h5py.File(source_path, "r") as source:
        matrix_names = list(source["data"].keys())
        if n_chunks is not None:
            chunk_names = [
                f"{source_path.stem}-chunk{number}.omx" for number in range(n_chunks)
            ]
        else:
            chunk_names = [f"{name}.omx" for name in matrix_names]

        output_paths = []
        for number, matrix_name in enumerate(matrix_names):
            output_path = destination / chunk_names[number % len(chunk_names)]
            output_paths.append(output_path)
            logger.info(f"writing {matrix_name} to {output_path}")
            with h5py.File(output_path, "a") as target:
                _initialize_omx_file(source, target)
                _copy_dataset(source["data"], target["data"], matrix_name)

        if global_lookups:
            for output_path in dict.fromkeys(output_paths):
                with h5py.File(output_path, "a") as target:
                    for lookup_name in source["lookup"]:
                        _copy_dataset(source["lookup"], target["lookup"], lookup_name)
        else:
            for lookup_name in source["lookup"]:
                output_path = destination / f"_{lookup_name}.omx"
                logger.info(f"writing {output_path}")
                with h5py.File(output_path, "w") as target:
                    _initialize_omx_file(source, target)
                    _copy_dataset(source["lookup"], target["lookup"], lookup_name)
