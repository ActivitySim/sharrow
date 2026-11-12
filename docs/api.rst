API Reference
=============

--------------------------------------------------------------------------------
DataTree
--------------------------------------------------------------------------------
.. autoclass:: sharrow.DataTree

Attributes
~~~~~~~~~~
.. autoattribute:: sharrow.DataTree.root_node_name
.. autoattribute:: sharrow.DataTree.subspaces
.. autoattribute:: sharrow.DataTree.relationships_are_digitized
.. autoattribute:: sharrow.DataTree.replacement_filters

Setup Flow
~~~~~~~~~~
.. automethod:: sharrow.DataTree.setup_flow

Datasets
~~~~~~~~
.. automethod:: sharrow.DataTree.add_dataset
.. automethod:: sharrow.DataTree.add_relationship
.. automethod:: sharrow.DataTree.replace_datasets

Digitization
~~~~~~~~~~~~
.. automethod:: sharrow.DataTree.digitize_relationships



--------------------------------------------------------------------------------
Relationship
--------------------------------------------------------------------------------
.. autoclass:: sharrow.Relationship

Attributes
~~~~~~~~~~
.. autoattribute:: sharrow.Relationship.parent_data
.. autoattribute:: sharrow.Relationship.parent_name
.. autoattribute:: sharrow.Relationship.child_data
.. autoattribute:: sharrow.Relationship.child_name
.. autoattribute:: sharrow.Relationship.indexing
.. autoattribute:: sharrow.Relationship.analog



--------------------------------------------------------------------------------
Flow
--------------------------------------------------------------------------------
.. autoclass:: sharrow.Flow

Load
~~~~
.. automethod:: sharrow.Flow.load
.. automethod:: sharrow.Flow.load_dataframe
.. automethod:: sharrow.Flow.load_dataarray

Dot
~~~
.. automethod:: sharrow.Flow.dot
.. automethod:: sharrow.Flow.dot_dataarray

Logit
~~~~~~~~~
.. automethod:: sharrow.Flow.logit_draws

Convenience
~~~~~~~~~~~
.. automethod:: sharrow.Flow.show_code



--------------------------------------------------------------------------------
Dataset
--------------------------------------------------------------------------------
Sharrow uses the :py:class:`xarray.Dataset` class extensively.  Refer to the
`xarray documentation <https://docs.xarray.dev/en/stable/>`_ for standard usage.
The attributes and methods documented here are added to :py:class:`xarray.Dataset`
when you import sharrow.

Constructors
~~~~~~~~~~~~

The sharrow library provides several constructors for :py:class:`Dataset` objects.
These functions can be found in the :py:mod:`sharrow.dataset` module.

.. autofunction:: sharrow.dataset.construct
.. autofunction:: sharrow.dataset.from_table
.. autofunction:: sharrow.dataset.from_omx
.. autofunction:: sharrow.dataset.from_omx_3d
.. autofunction:: sharrow.dataset.from_parquet_3d
.. autofunction:: sharrow.dataset.from_zarr
.. autofunction:: sharrow.dataset.from_named_objects

OMX Loading Portability
~~~~~~~~~~~~~~~~~~~~~~~

Sharrow uses OpenMatrix 0.4 or newer, whose file objects are backed by h5py.
Paths, OpenMatrix handles, and plain h5py handles are supported. Sharrow's
bounded parallel chunk decoder operates directly on their HDF5 datasets,
including direct reads into shared memory and dtype conversion during decoding.
Caller-owned handles remain open.

Standard gzip/shuffle OMX files work with Sharrow's base installation. Files
that use Blosc, Zstandard, or other optional HDF5 filters may require the
additional filter packages::

    pip install "sharrow[hdf5-plugins]"

These third-party filter packages do not currently publish Windows ARM64
wheels, so Windows on ARM uses the standard HDF5 filters supported by h5py.

All selected OMX matrices must match the shape declared or inferred from the
first source file. Inconsistent matrices now raise an error during construction,
before loading or reserving shared storage. The error identifies the matrix,
source file, actual shape, and expected shape. Correct stale ``SHAPE`` metadata
and verify coordinates rather than relying on earlier permissive behavior.
``ignore`` can exclude an unrelated matrix, but does not repair incorrect
metadata for matrices that must be loaded.

Lazy loading defaults to one Dask task per matrix (time-period page), so
selecting a period does not load the other periods. Use
``task_granularity="variable"`` explicitly when loading whole variables and
reducing task overhead matters more than selective reads. Eager and direct
shared/memmap loads continue to fill their final destination arrays directly.

``reload_from_omx_3d`` loads files serially by default, with parallel chunk
decoding. Existing callers do not acquire multiprocessing requirements.
Pass ``workers>1`` to opt into file-level parallel reloads.

The ``shared`` and ``memmap`` loading modes can read several source files
concurrently. On Windows, Sharrow uses file-level threads so these modes work
from ordinary scripts, notebooks, and frozen applications without a
``__main__`` multiprocessing guard or Windows' process-pool size limit. On
other platforms, Sharrow uses spawned worker processes to avoid inheriting
open HDF5 state. Calls made at module scope in a standalone script should
therefore use the standard guarded entry point::

    if __name__ == "__main__":
        skims = sharrow.dataset.from_omx_3d(
            skim_files,
            time_periods=["EA", "AM", "MD", "PM", "EV"],
            load="shared",
        )

Process-based file workers divide the CPU budget between their decoder pools
instead of each creating a machine-sized pool. Chunk queue limits are still
per process, and at least one decoder is needed per active worker; this is not
a global byte limit. Windows file threads share one decoder pool.

Set ``workers=1`` for deterministic serial I/O on every platform. Successful
``memmap`` loads retain their backing data and metadata files intentionally.
Releasing a memmap flushes it and relinquishes the accessor's ownership;
existing Dataset arrays and retained NumPy views remain valid. The mapping
closes when the last array reference is discarded. Before deleting the backing
files on Windows, discard the Dataset and all retained views::

    key = result.shm.shared_memory_key
    result.shm.release_shared_memory()
    del result  # also discard any retained DataArrays or NumPy views
    gc.collect()  # import gc; Xarray accessor cycles may delay collection
    sharrow.Dataset.shm.delete_shared_memory_files(key)

File deletion is a separate, explicit operation; release never deletes memmap
files. Do not force-close the underlying mapping while views remain alive.

OMX to Zarr conversion preserves float64 values by default. Pass
``max_float_precision=32`` to ``sharrow.translate.omx_to_zarr`` only when the
precision reduction is intended. Output uses Zarr format 2 with spatial chunks
of at most 256 by 256 cells and one period per chunk. This bounds stored chunks
independently of the source loader's tasks; decoding still materializes a source
matrix, rather than streaming individual spatial tiles from OMX.

Editing
~~~~~~~
.. automethod:: sharrow.Dataset.ensure_integer

Indexing
~~~~~~~~
.. autoaccessor:: sharrow.Dataset.iloc
.. autoaccessor:: sharrow.Dataset.at
.. autoaccessormethod:: sharrow.Dataset.at.df
.. autoaccessor:: sharrow.Dataset.iat
.. autoaccessormethod:: sharrow.Dataset.iat.df


Shared Memory
~~~~~~~~~~~~~
Sharrow's shared memory system is consolidated into the :py:class:`Dataset.shm`
accessor.

.. autoaccessormethod:: sharrow.Dataset.shm.to_shared_memory
.. autoaccessormethod:: sharrow.Dataset.shm.from_shared_memory
.. autoaccessormethod:: sharrow.Dataset.shm.release_shared_memory
.. autoaccessormethod:: sharrow.Dataset.shm.preload_shared_memory_size
.. autoaccessorattribute:: sharrow.Dataset.shm.shared_memory_key
.. autoaccessorattribute:: sharrow.Dataset.shm.shared_memory_size
.. autoaccessorattribute:: sharrow.Dataset.shm.is_shared_memory

Digital Encoding
~~~~~~~~~~~~~~~~
Sharrow's digital encoding management is consolidated into the
:py:class:`Dataset.digital_encoding` accessor.

.. autoaccessormethod:: sharrow.Dataset.digital_encoding.info
.. autoaccessormethod:: sharrow.Dataset.digital_encoding.set
