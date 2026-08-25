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

Standard gzip/shuffle OMX files work with Sharrow's base installation. Files
that use Blosc, Zstandard, or other optional HDF5 filters may require the
additional filter packages::

    pip install "sharrow[hdf5-plugins]"

These third-party filter packages do not currently publish Windows ARM64
wheels, so Windows on ARM uses the standard HDF5 filters supported by h5py.

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

Set ``workers=1`` for deterministic serial I/O on every platform. Successful
``memmap`` loads retain their backing data and metadata files intentionally.
Call ``result.shm.release_shared_memory()`` before
``result.shm.delete_shared_memory_files(result.shm.shared_memory_key)`` when
the files are no longer needed.

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
