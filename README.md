# sharrow
numba for ActivitySim-style spec files

## Building a Wheel

To build a wheel for sharrow, you need to have `build` installed. You can
install it with `python -m pip install build`.

Then run the builder:

```shell
python -m build .
```


## Building the documentation

Building the documentation for sharrow requires JupyterBook.

```shell
jupyterbook build docs
```

## Developer Note

This repository's continuous integration testing will use `ruff` to check code
quality.  There is a pre-commit hook that will run `ruff` on all staged files
to ensure that they pass the quality checks.  To install and use this hook,
run the following commands:

```shell
python -m pip install pre-commit  # if needed
pre-commit install
```

Then, when you try to make a commit, your code will be checked locally to ensure
that your code passes the quality checks.  If you want to run the checks manually,
you can do so with the following command:

```shell
pre-commit run --all-files
```

If you don't use `pre-commit`, a service will run the checks for you when you
open a pull request, and make fixes to your code when possible.
