#!/usr/bin/env python3
"""Benchmark fully materialized Sharrow OMX loads in isolated processes.

The controller launches one fresh Python process per trial, samples the RSS of
the complete process tree, and combines that with the operating system's
high-water RSS for the worker. This captures the final Dataset as well as
temporary arrays created while HDF5 data is decoded, stacked, or cast.

Example
-------
python benchmarks/benchmark_omx_load.py \
    --omx /path/to/traffic_skims_EA.omx \
    --omx /path/to/traffic_skims_AM.omx \
    --case 'release-2.16.2|/tmp/sharrow-2.16.2|lazy' \
    --case 'current-lazy|/path/to/current/sharrow|lazy' \
    --case 'current-eager|/path/to/current/sharrow|eager' \
    --warmups 1 --repetitions 3 --output /tmp/omx-benchmark.json

Case strategies are ``lazy`` (construct the default Dask-backed Dataset, then
call ``load``), ``lazy-matrix`` (use one Dask task per physical matrix),
``shared-reload`` (the ActivitySim shared-memory workflow), and ``eager`` (use
the newer ``load="eager"`` path). Peak RSS excludes the operating system's file
cache and memory owned by unrelated processes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import psutil

try:
    import resource
except ImportError:  # Windows has no resource module; the controller samples RSS.
    resource = None

RESULT_PREFIX = "SHARROW_BENCHMARK_RESULT="


def _rss_high_water_bytes() -> int:
    """Return this process's maximum resident set size in bytes."""
    if resource is None:
        return psutil.Process().memory_info().rss
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, while Linux and most BSDs report KiB.
    if platform.system() == "Darwin":
        return int(value)
    return int(value * 1024)


def _dataset_fingerprint(dataset) -> str:
    """Hash array metadata and edge values without scanning or copying arrays."""
    digest = hashlib.sha256()
    for name in sorted(dataset.data_vars):
        array = dataset[name].data
        digest.update(name.encode())
        digest.update(repr((array.shape, str(array.dtype))).encode())
        if array.size:
            first = array[(0,) * array.ndim]
            last = array[tuple(size - 1 for size in array.shape)]
            digest.update(repr((first, last)).encode())
    return digest.hexdigest()


def _worker(args: argparse.Namespace) -> int:
    """Run one load trial and emit a machine-readable result."""
    import sharrow as sh

    process = psutil.Process()
    omx_paths = [str(Path(path).resolve()) for path in args.omx]
    omx_sources = omx_paths[0] if len(omx_paths) == 1 else omx_paths
    source_commit = subprocess.check_output(
        ["git", "-C", args.source_root, "rev-parse", "HEAD"], text=True
    ).strip()
    result = {
        "label": args.label,
        "strategy": args.strategy,
        "omx": omx_paths,
        "source_root": str(Path(args.source_root).resolve()),
        "source_commit": source_commit,
        "sharrow_file": str(Path(sh.__file__).resolve()),
        "sharrow_version": getattr(sh, "__version__", "unknown"),
        "baseline_rss_bytes": process.memory_info().rss,
    }
    started = time.perf_counter()
    try:
        kwargs = {
            "time_periods": args.periods.split(","),
            "max_float_precision": args.max_float_precision,
        }
        if args.strategy == "shared-reload":
            template = sh.dataset.from_omx_3d(omx_sources, **kwargs)
            shared_key = f"sharrow-benchmark-{os.getpid()}-{time.time_ns()}"
            dataset = template.shm.to_shared_memory(shared_key, mode="r", load=False)
            sh.dataset.reload_from_omx_3d(dataset, omx_paths)
        elif args.strategy == "eager":
            kwargs["load"] = "eager"
            dataset = sh.dataset.from_omx_3d(omx_sources, **kwargs)
        elif args.strategy == "lazy-matrix":
            kwargs["task_granularity"] = "matrix"
            dataset = sh.dataset.from_omx_3d(omx_sources, **kwargs)
        else:
            dataset = sh.dataset.from_omx_3d(omx_sources, **kwargs)
        if args.strategy in {"lazy", "lazy-matrix"}:
            dataset.load()
        result.update(
            {
                "status": "ok",
                "elapsed_seconds": time.perf_counter() - started,
                "dataset_nbytes": int(dataset.nbytes),
                "final_rss_bytes": process.memory_info().rss,
                "variable_count": len(dataset.data_vars),
                "dimensions": {name: int(size) for name, size in dataset.sizes.items()},
                "fingerprint": _dataset_fingerprint(dataset),
            }
        )
    except Exception as error:  # noqa: BLE001 - failures are benchmark results
        result.update(
            {
                "status": "error",
                "elapsed_seconds": time.perf_counter() - started,
                "final_rss_bytes": process.memory_info().rss,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        result["worker_peak_rss_bytes"] = _rss_high_water_bytes()

    print(f"{RESULT_PREFIX}{json.dumps(result, sort_keys=True)}", flush=True)
    # Give the external sampler one final opportunity to observe the loaded
    # Dataset before process teardown releases its resident pages.
    time.sleep(args.hold_seconds)
    return 0


def _process_tree_rss(process: psutil.Process) -> int:
    """Sum current RSS across a worker and all of its live descendants."""
    try:
        descendants = process.children(recursive=True)
    except (psutil.Error, OSError):
        descendants = []
    total = 0
    for item in [process, *descendants]:
        try:
            total += item.memory_info().rss
        except (psutil.Error, OSError):
            continue
    return total


def _parse_case(value: str) -> dict[str, str]:
    """Parse a LABEL|SOURCE_ROOT|STRATEGY command-line case."""
    try:
        label, source_root, strategy = value.split("|", 2)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "case must be LABEL|SOURCE_ROOT|STRATEGY"
        ) from error
    if strategy not in {"lazy", "lazy-matrix", "shared-reload", "eager"}:
        raise argparse.ArgumentTypeError(
            "case strategy must be lazy, lazy-matrix, shared-reload, or eager"
        )
    root = Path(source_root).expanduser().resolve()
    if not (root / "sharrow" / "__init__.py").is_file():
        raise argparse.ArgumentTypeError(f"not a Sharrow source tree: {root}")
    return {"label": label, "source_root": str(root), "strategy": strategy}


def _extract_result(stdout: str) -> dict:
    """Extract the worker's JSON record from captured standard output."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    raise RuntimeError(f"worker produced no benchmark result:\n{stdout}")


def _run_trial(
    args: argparse.Namespace,
    case: dict[str, str],
    run_number: int,
    warmup: bool,
) -> dict:
    """Launch and externally monitor one isolated benchmark trial."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
    ]
    for omx in args.omx:
        command.extend(["--omx", str(Path(omx).resolve())])
    command.extend(
        [
            "--label",
            case["label"],
            "--source-root",
            case["source_root"],
            "--strategy",
            case["strategy"],
            "--periods",
            args.periods,
            "--max-float-precision",
            str(args.max_float_precision),
            "--hold-seconds",
            str(args.hold_seconds),
        ]
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [case["source_root"], environment.get("PYTHONPATH")])
    )
    child = subprocess.Popen(
        command,
        cwd=case["source_root"],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    monitored = psutil.Process(child.pid)
    peak_tree_rss = 0
    while child.poll() is None:
        peak_tree_rss = max(peak_tree_rss, _process_tree_rss(monitored))
        time.sleep(args.sample_interval)
    peak_tree_rss = max(peak_tree_rss, _process_tree_rss(monitored))
    stdout, stderr = child.communicate()
    result = _extract_result(stdout)
    result.update(
        {
            "run": run_number,
            "warmup": warmup,
            "returncode": child.returncode,
            "peak_tree_rss_bytes": peak_tree_rss,
            "peak_rss_bytes": max(
                peak_tree_rss, result.get("worker_peak_rss_bytes", 0)
            ),
            "stderr": stderr,
        }
    )
    return result


def _gib(value: int | float) -> float:
    """Convert bytes to GiB for compact terminal reporting."""
    return value / 1024**3


def _print_trial(result: dict) -> None:
    """Print one compact human-readable result row."""
    status = result["status"]
    peak = _gib(result["peak_rss_bytes"])
    message = (
        f"{result['label']:20} {result['strategy']:11} "
        f"run={result['run']} warmup={result['warmup']} status={status:5} "
        f"time={result['elapsed_seconds']:8.3f}s peak={peak:7.3f} GiB"
    )
    if status == "error":
        message += f" error={result['error_type']}: {result['error']}"
    print(message, flush=True)


def _summaries(results: list[dict]) -> list[dict]:
    """Compute medians and ranges for successful measured trials by case."""
    summaries = []
    labels = dict.fromkeys(result["label"] for result in results)
    for label in labels:
        trials = [
            result
            for result in results
            if result["label"] == label
            and not result["warmup"]
            and result["status"] == "ok"
        ]
        if not trials:
            continue
        elapsed = [result["elapsed_seconds"] for result in trials]
        peak = [result["peak_rss_bytes"] for result in trials]
        final_rss = [result["final_rss_bytes"] for result in trials]
        baseline = [result["baseline_rss_bytes"] for result in trials]
        summaries.append(
            {
                "label": label,
                "strategy": trials[0]["strategy"],
                "trials": len(trials),
                "median_elapsed_seconds": statistics.median(elapsed),
                "min_elapsed_seconds": min(elapsed),
                "max_elapsed_seconds": max(elapsed),
                "median_peak_rss_bytes": statistics.median(peak),
                "median_final_rss_bytes": statistics.median(final_rss),
                "median_baseline_rss_bytes": statistics.median(baseline),
                "median_peak_increase_bytes": statistics.median(
                    trial_peak - trial_baseline
                    for trial_peak, trial_baseline in zip(peak, baseline)
                ),
                "dataset_nbytes": trials[0]["dataset_nbytes"],
                "fingerprint": trials[0]["fingerprint"],
            }
        )
    return summaries


def _controller(args: argparse.Namespace) -> int:
    """Run warmups and measured trials, rotating order between repetitions."""
    omx_paths = [Path(path).expanduser().resolve() for path in args.omx]
    for omx in omx_paths:
        if not omx.is_file():
            raise FileNotFoundError(omx)

    results = []
    for warmup_number in range(1, args.warmups + 1):
        for case in args.case:
            result = _run_trial(args, case, warmup_number, warmup=True)
            results.append(result)
            _print_trial(result)

    # Rotate trial order so no implementation systematically benefits from
    # immediately following the cache-warming run.
    for run_number in range(1, args.repetitions + 1):
        offset = (run_number - 1) % len(args.case)
        ordered_cases = args.case[offset:] + args.case[:offset]
        for case in ordered_cases:
            result = _run_trial(args, case, run_number, warmup=False)
            results.append(result)
            _print_trial(result)

    payload = {
        "omx": [str(path) for path in omx_paths],
        "omx_size_bytes": sum(path.stat().st_size for path in omx_paths),
        "periods": args.periods.split(","),
        "max_float_precision": args.max_float_precision,
        "memory_metric": "peak RSS across the complete benchmark process tree",
        "sample_interval_seconds": args.sample_interval,
        "results": results,
        "summaries": _summaries(results),
    }
    print("\nSummary (medians of successful measured trials)")
    for summary in payload["summaries"]:
        print(
            f"{summary['label']:20} {summary['strategy']:11} "
            f"time={summary['median_elapsed_seconds']:8.3f}s "
            f"peak={_gib(summary['median_peak_rss_bytes']):7.3f} GiB "
            f"final={_gib(summary['median_final_rss_bytes']):7.3f} GiB "
            f"dataset={_gib(summary['dataset_nbytes']):7.3f} GiB"
        )

    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote {output}")
    return 0


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser used by the controller and worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--omx",
        action="append",
        required=True,
        help="OMX file to load; repeat to load several files as one Dataset",
    )
    parser.add_argument(
        "--case",
        action="append",
        type=_parse_case,
        default=[],
        help="benchmark case as LABEL|SOURCE_ROOT|STRATEGY",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--periods", default="EA,AM,MD,PM,EV")
    parser.add_argument("--max-float-precision", type=int, default=32)
    parser.add_argument("--sample-interval", type=float, default=0.005)
    parser.add_argument("--hold-seconds", type=float, default=0.25)
    parser.add_argument("--output")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--label", help=argparse.SUPPRESS)
    parser.add_argument("--source-root", help=argparse.SUPPRESS)
    parser.add_argument(
        "--strategy",
        choices=("lazy", "lazy-matrix", "shared-reload", "eager"),
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> int:
    """Dispatch to the isolated worker or multi-case controller."""
    args = _parser().parse_args()
    if args.worker:
        return _worker(args)
    if not args.case:
        raise SystemExit("at least one --case is required")
    if args.warmups < 0 or args.repetitions < 1:
        raise SystemExit("warmups must be nonnegative and repetitions positive")
    return _controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
