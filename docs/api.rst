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

setup_flow
~~~~~~~~~~
.. automethod:: sharrow.DataTree.setup_flow

datasets
~~~~~~~~
.. automethod:: sharrow.DataTree.add_dataset
.. automethod:: sharrow.DataTree.add_relationship
.. automethod:: sharrow.DataTree.replace_datasets

digitization
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

load
~~~~
.. automethod:: sharrow.Flow.load
.. automethod:: sharrow.Flow.load_dataframe
.. automethod:: sharrow.Flow.load_dataarray

dot
~~~
.. automethod:: sharrow.Flow.dot
.. automethod:: sharrow.Flow.dot_dataarray

mnl_draws
~~~~~~~~~
.. automethod:: sharrow.Flow.mnl_draws




--------------------------------------------------------------------------------
Dataset
--------------------------------------------------------------------------------
.. autoclass:: sharrow.Dataset
