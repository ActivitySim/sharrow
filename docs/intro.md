# Getting Started

Sharrow is a Python library that enables several different kinds of data to flow
together in the same lane.  It leverages the power of numba and xarray to
compile expression specification files (like those used in ActivitySim) into
optimized, runnable code.


## Installation

Sharrow will soon be available through conda-forge.

```shell
conda install sharrow -c conda-forge
```

You can also install from the source code using setuptools and pip.  A number of
the dependencies are not pure Python packages, and so it's highly recommended that
you create an environment containing all those dependencies first, and then install
sharrow itself.  To do so with conda, you can run:

```shell
conda install -c conda-forge numpy pandas xarray numba pyarrow numexpr filelock
```

Then clone the [repository](https://github.com/camsys/sharrow), and then from
the root directory run

```shell
pip install -e .
```

## Testing

Sharrow includes unit tests both in the `sharrow/tests` directory and embedded
in the user documentation under `docs`.

To run the test suite after installing sharrow, install (via pypi or conda) pytest and nbmake,
and run `pytest` in the root directory of the sharrow repository.


## Building the Documentation

The docs for sharrow are built using [Jupyter Book](https://jupyterbook.org).
You can install Jupyter Book [via `pip`](https://pip.pypa.io/en/stable/):

```shell
pip install -U jupyter-book
```
or via [`conda-forge`](https://conda-forge.org/):

```shell
conda install -c conda-forge jupyter-book
```

Then to build the docs, in the root directory of the sharrow repository run

```shell
jb build docs
```
