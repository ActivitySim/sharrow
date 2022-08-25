from xarray import DataArray

from . import dataset, example_data, selectors, shared_memory, sparse
from .dataset import Dataset
from .digital_encoding import array_decode, array_encode
from .flows import Flow
from .relationships import DataTree, Relationship
from .table import Table, concat_tables

try:
    from ._version import __version_tuple__
    from ._version import version as __version__
except ImportError:
    # Package is not "installed", parse git tag at runtime
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version(__package__)
    except PackageNotFoundError:
        # package is not installed
        __version__ = "999.999"

    __version_tuple__ = __version__.split(".")
