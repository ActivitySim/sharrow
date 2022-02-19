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
.. automethod:: sharrow.Flow.mnl_draws

Convenience
~~~~~~~~~~~
.. automethod:: sharrow.Flow.show_code



--------------------------------------------------------------------------------
Dataset
--------------------------------------------------------------------------------
.. autoclass:: sharrow.Dataset
    :show-inheritance:

Constructors
~~~~~~~~~~~~
.. automethod:: sharrow.Dataset.construct
.. automethod:: sharrow.Dataset.from_table
.. automethod:: sharrow.Dataset.from_omx
.. automethod:: sharrow.Dataset.from_omx_3d
.. automethod:: sharrow.Dataset.from_zarr
.. automethod:: sharrow.Dataset.from_named_objects

Editing
~~~~~~~
.. automethod:: sharrow.Dataset.update
.. automethod:: sharrow.Dataset.ensure_integer

Indexing
~~~~~~~~
.. automethod:: sharrow.Dataset.at
.. automethod:: sharrow.Dataset.iat
.. automethod:: sharrow.Dataset.at_df
.. automethod:: sharrow.Dataset.iat_df

Convenience
~~~~~~~~~~~
In many ways, a dataset with a single dimensions is like a pandas DataFrame,
with the one dimension giving the rows, and the variables as columns.  This
analogy eventually breaks down (DataFrame columns are ordered, Dataset
variables are not) but the similarities are enought that it's sometimes convenient
to have `loc` and `iloc` functionality enabled.  This only works for indexing on
the rows, but if there's only the one dimension the complexity of `sel` and `isel`
are not needed.

.. autoattribute:: sharrow.Dataset.loc
.. autoattribute:: sharrow.Dataset.iloc

Shared Memory
~~~~~~~~~~~~~
.. automethod:: sharrow.Dataset.to_shared_memory
.. automethod:: sharrow.Dataset.from_shared_memory
.. automethod:: sharrow.Dataset.release_shared_memory
.. automethod:: sharrow.Dataset.preload_shared_memory_size
.. autoattribute:: sharrow.Dataset.shared_memory_key
.. autoattribute:: sharrow.Dataset.shared_memory_size
.. autoattribute:: sharrow.Dataset.is_shared_memory

Digital Encoding
~~~~~~~~~~~~~~~~
.. autoattribute:: sharrow.Dataset.digital_encodings
.. automethod:: sharrow.Dataset.set_digital_encoding
