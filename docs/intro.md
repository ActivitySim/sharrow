# Getting Started

Sharrow is a Python library that enables several different kinds of data to flow
together in the same lane.  It leverages the power of numba and xarray to
compile expression specification files (like those used in ActivitySim) into
optimized, runnable code.

## ActivitySim

Sharrow is a project of the [ActivitySim](https://activitysim.github.io/) consortium.

The mission of the ActivitySim Consortium is to create and maintain advanced,
open-source, activity-based travel behavior modeling software based on best
software development practices for distribution at no charge to the public.

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

## License

Sharrow is available under the open source [3-Clause BSD License](https://opensource.org/licenses/BSD-3-Clause).
