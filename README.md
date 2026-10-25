# sharrow
numba for ActivitySim-style spec files

## Building a Wheel

Build a wheel for sharrow with [uv](https://docs.astral.sh/uv/) 0.13.4 or newer:

```shell
uv build
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
uv tool install pre-commit  # if needed
pre-commit install
```

If you installed pre-commit before 21 October 2026, upgrade it (for example with
`uv tool upgrade pre-commit`) so that it uses virtualenv 21.14.12 or newer.
Older virtualenv versions set up hook environments with a pip that trusts the
compromised eMudhra certificate authority (see the [incident notice](https://discuss.python.org/t/information-for-python-users-regarding-emudhra-emsign-ca-certificate-security-incident/109576)).

Then, when you try to make a commit, your code will be checked locally to ensure
that your code passes the quality checks.  If you want to run the checks manually,
you can do so with the following command:

```shell
pre-commit run --all-files
```

If you don't use `pre-commit`, a service will run the checks for you when you
open a pull request, and make fixes to your code when possible.
