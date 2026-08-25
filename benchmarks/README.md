# Sharrow OMX benchmarks

`benchmark_mtc.py` is a standalone, cross-platform benchmark for the full-scale
MTC `skims.omx`. It needs only Python and [uv](https://docs.astral.sh/uv/).
The script installs its own Python dependencies and creates isolated cached
environments for the release and development versions of Sharrow. It does not
use `git`, `curl`, or a system archive utility.

Run the default comparison with either command:

```shell
uv run benchmarks/benchmark_mtc.py
python benchmarks/benchmark_mtc.py
```

The script downloads `data_full.tar.zst` from the MTC v1.3.4 release, validates
its published SHA256 checksum, and extracts it with `wring`. Downloads,
extracted data, exact GitHub source snapshots, and uv environments are cached in
the platform's user cache directory. Set `SHARROW_BENCHMARK_CACHE` or pass
`--cache-dir` to put the cache elsewhere.

By default, the benchmark compares the latest ActivitySim/Sharrow GitHub release
with:

```text
https://github.com/driftlesslabs/sharrow/tree/codex/replace-openmatrix-with-h5py
```

Select another development source using a branch name, an `OWNER/REPO@REF`
identifier, or a GitHub tree URL:

```shell
uv run benchmarks/benchmark_mtc.py --development feature/my-branch
uv run benchmarks/benchmark_mtc.py --development ActivitySim/sharrow@my-branch
uv run benchmarks/benchmark_mtc.py --development https://github.com/OWNER/REPO/tree/BRANCH
```

The default run executes one warmup and three measured repetitions of lazy,
eager (when available), and ActivitySim-style shared-memory reloads. Every trial
runs in a fresh process. Peak memory is sampled across the complete process
tree, so it includes the final dataset, temporary allocations, and child
workers. Both Sharrow builds use the same pinned Python and dependency versions
so the comparison isolates their code changes and remains repeatable across
machines. Results and exact source commits are written to
`mtc-sharrow-benchmark.json` by default.

Useful options include:

```shell
# Quick comparison of the directly comparable lazy loaders
uv run benchmarks/benchmark_mtc.py --strategies lazy --warmups 0 --repetitions 1

# Use a previously available MTC skim file
uv run benchmarks/benchmark_mtc.py --omx /path/to/skims.omx

# Download data and build environments now, run trials later
uv run benchmarks/benchmark_mtc.py --prepare-only

# Force cache refreshes
uv run benchmarks/benchmark_mtc.py --refresh-data --rebuild-environments
```

The materialized dataset is about 6.8 GiB, while the release lazy loader can
peak near 17 GiB. A machine with at least 24 GiB of available memory and several
GiB of free cache space is recommended. Set `GITHUB_TOKEN` if unauthenticated
GitHub API rate limits are an issue.
