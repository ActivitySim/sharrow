# Getting Started

Sharrow is a Python library that enables several different kinds of data to flow
together in the same lane.  It leverages the power of numba and xarray to
compile expression specification files (like those used in ActivitySim) into
optimized, runnable code.


## Installation

Install Sharrow from [PyPI](https://pypi.org/project/sharrow/) in your `uv` project:

```shell
uv add sharrow
```

`uv` resolves and installs Sharrow and its dependencies.

## Development Installation

To work from source, clone the [repository](https://github.com/activitysim/sharrow)
and, from its root directory, create a development environment:

```shell
uv sync
```

## Testing

Sharrow includes unit tests both in the `sharrow/tests` directory and embedded
in the user documentation under `docs`.

To run the test suite, install the development dependencies and run the following
from the root directory of the Sharrow repository:

```shell
uv sync
uv run pytest
```


## Code Formatting

Sharrow uses several tools to ensure a consistent code format throughout the project:

- [Ruff](https://docs.astral.sh/ruff/) for standardized code formatting, import
  sorting, and code quality,
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

The docs for sharrow are built using [Jupyter Book](https://jupyterbook.org). Install
it with `uv`:

```shell
uv tool install jupyter-book
```

Then to build the docs, in the root directory of the sharrow repository run

```shell
jb build docs
```
