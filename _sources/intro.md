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

Alternatively, you can install sharrow plus all the dependencies (including
additional optional dependencies for development and testing) in a conda environment,
using the `envs/development.yml` environment to create a `sh-dev` environment:

```shell
conda env create -f envs/development.yml
```

## Testing

Sharrow includes unit tests both in the `sharrow/tests` directory and embedded
in the user documentation under `docs`.

To run the test suite after installing sharrow, install (via pypi or conda) pytest and nbmake,
and run `pytest` in the root directory of the sharrow repository.


## Code Formatting

Sharrow uses several tools to ensure a consistent code format throughout the project:

- [Black](https://black.readthedocs.io/en/stable/) for standardized code formatting,
- [Flake8](http://flake8.pycqa.org/en/latest/) for general code quality,
- [isort](https://github.com/timothycrosley/isort) for standardized order in imports, and
- [nbstripout](https://github.com/kynan/nbstripout) to ensure notebooks are committed
  to the GitHub repository without bulky outputs included.

We highly recommend that you setup [pre-commit hooks](https://pre-commit.com/)
to automatically run all the above tools every time you make a git commit. This
can be done by running:

```shell
pre-commit install
```

from the root of the sharrow repository. You can skip the pre-commit checks
with `git commit --no-verify`.


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
