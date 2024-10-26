from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .relationships import DataTree


class DataTreeBranch:
    """Access variables in a subspace of a tree.

    This class enables the use of the python and numexpr engines to access
    dotted data in the tree.
    """

    def __init__(self, tree: DataTree, branch: str):
        self.tree = tree
        self.branch = branch

    def __getitem__(self, key):
        return self.tree[self.branch + "." + key]

    def __setitem__(self, key, value):
        self.tree[self.branch + "." + key] = value

    def __getattr__(self, item):
        dataset = self.tree.subspaces[self.branch]
        if item in dataset:
            return self.tree[self.branch + "." + item]
        else:
            raise AttributeError(f"{item} not found in {self.branch}")