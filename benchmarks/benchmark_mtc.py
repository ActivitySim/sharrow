#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "psutil>=5.9",
#   "wring==0.0.6",
# ]
# ///
"""Reproduce Sharrow OMX load benchmarks with the full-scale MTC skims.

This one-file benchmark downloads and caches the MTC ``data_full`` archive,
extracts it with wring, resolves the latest ActivitySim/Sharrow GitHub release
and a development branch to exact commits, and installs each source snapshot in
an isolated uv environment. No system tools such as git, curl, or tar are used.

Run with either of these commands::

    uv run benchmark_mtc.py
    python benchmark_mtc.py

The default development source is::

    https://github.com/driftlesslabs/sharrow/tree/codex/replace-openmatrix-with-h5py

Use ``--development BRANCH``, ``--development OWNER/REPO@BRANCH``, or another
GitHub tree URL to benchmark a different development branch.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import types
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

DATA_URL = (
    "https://github.com/ActivitySim/activitysim-prototype-mtc/"
    "releases/download/v1.3.4/data_full.tar.zst"
)
DATA_SHA256 = "b402506a61055e2d38621416dd9a5c7e3cf7517c0a9ae5869f6d760c03284ef3"
DEFAULT_DEVELOPMENT = (
    "https://github.com/driftlesslabs/sharrow/tree/codex/replace-openmatrix-with-h5py"
)
GITHUB_API = "https://api.github.com"
RESULT_PREFIX = "SHARROW_MTC_BENCHMARK_RESULT="
GIB = 1024**3

# Install identical runtime dependencies in both benchmark environments. The
# source package itself is installed without dependencies because release
# 2.16.2 made optional HDF5 plugins mandatory in its metadata even though the
# gzip-compressed MTC OMX file does not use them. Lightweight import stubs in
# the worker preserve that release's code path without requiring binary plugins
# that are unavailable on some Windows architectures.
BENCHMARK_DEPENDENCIES = (
    "numpy==2.2.6",
    "pandas==2.3.3",
    "pyarrow==23.0.1",
    "h5py==3.16.0",
    "xarray==2025.6.1",
    "numba==0.65.0",
    "numexpr==2.14.1",
    "filelock==3.25.2",
    "dask==2023.11.0",
    "networkx==3.4.2",
    "psutil==7.2.2",
)


def _bootstrap_with_uv() -> int | None:
    """Re-run the controller through uv when script dependencies are absent."""
    if "--worker" in sys.argv:
        return None
    required = ("psutil", "wring")
    if all(importlib.util.find_spec(name) is not None for name in required):
        return None
    if os.environ.get("SHARROW_BENCHMARK_BOOTSTRAPPED") == "1":
        missing = [name for name in required if importlib.util.find_spec(name) is None]
        raise RuntimeError(f"uv did not install required packages: {missing}")
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required and was not found on PATH")
    environment = os.environ.copy()
    environment["SHARROW_BENCHMARK_BOOTSTRAPPED"] = "1"
    command = [
        uv,
        "run",
        "--no-project",
        "--script",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    return subprocess.call(command, env=environment)


_bootstrap_result = _bootstrap_with_uv()
if _bootstrap_result is not None:
    raise SystemExit(_bootstrap_result)

import psutil  # noqa: E402  (available after the optional uv bootstrap)

try:
    import resource
except ImportError:  # Windows has no resource module; the controller samples RSS.
    resource = None


@dataclass(frozen=True)
class SourceSpec:
    """An immutable GitHub source snapshot used to build one environment."""

    kind: str
    label: str
    owner: str
    repository: str
    ref: str
    revision: str
    archive_url: str
    requested: str
    version: str

    @property
    def cache_key(self) -> str:
        """Return a short filesystem-safe identity for this source snapshot."""
        prefix = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{self.owner}-{self.repository}")
        return f"{self.kind}-{prefix}-{self.revision[:16]}"


@dataclass(frozen=True)
class BenchmarkCase:
    """One isolated source/environment and loading-strategy combination."""

    label: str
    source_kind: str
    strategy: str
    python: str
    source: SourceSpec


def _default_cache_dir() -> Path:
    """Return a conventional per-user cache directory on each operating system."""
    override = os.environ.get("SHARROW_BENCHMARK_CACHE")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches"
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "sharrow-mtc-benchmark"


def _github_headers(authenticated: bool = True) -> dict[str, str]:
    """Build public GitHub API headers, optionally using GITHUB_TOKEN."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "sharrow-mtc-benchmark",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if authenticated and token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _github_json(path: str) -> dict | list:
    """Read one GitHub API object with useful rate-limit diagnostics."""
    request = urllib.request.Request(
        f"{GITHUB_API}{path}", headers=_github_headers(authenticated=True)
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        rate_limit = error.headers.get("X-RateLimit-Remaining") == "0"
        hint = " Set GITHUB_TOKEN to raise the API rate limit." if rate_limit else ""
        raise RuntimeError(
            f"GitHub API request failed ({error.code}) for {path}.{hint}"
        ) from error


def _resolve_commit(owner: str, repository: str, ref: str) -> str:
    """Resolve a GitHub branch, tag, or SHA to an exact commit hash."""
    encoded_ref = urllib.parse.quote(ref, safe="")
    payload = _github_json(f"/repos/{owner}/{repository}/commits/{encoded_ref}")
    if not isinstance(payload, dict) or not re.fullmatch(
        r"[0-9a-fA-F]{40}", str(payload.get("sha", ""))
    ):
        raise RuntimeError(f"GitHub returned no commit for {owner}/{repository}@{ref}")
    return str(payload["sha"]).lower()


def _source_archive_url(owner: str, repository: str, revision: str) -> str:
    """Return a public archive URL that does not require a local git client."""
    return f"https://github.com/{owner}/{repository}/archive/{revision}.tar.gz"


def _release_source(selector: str) -> SourceSpec:
    """Resolve the latest release, or an explicitly requested Sharrow tag."""
    owner, repository = "ActivitySim", "sharrow"
    if selector == "latest":
        payload = _github_json(f"/repos/{owner}/{repository}/releases/latest")
        if not isinstance(payload, dict) or not payload.get("tag_name"):
            raise RuntimeError("the latest Sharrow GitHub release has no tag")
        ref = str(payload["tag_name"])
    else:
        ref = selector
    revision = _resolve_commit(owner, repository, ref)
    version = ref[1:] if ref.startswith("v") else ref
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)*(?:[A-Za-z0-9_.+-]*)", version):
        version = "0.0"
    return SourceSpec(
        kind="release",
        label=f"Sharrow {ref}",
        owner=owner,
        repository=repository,
        ref=ref,
        revision=revision,
        archive_url=_source_archive_url(owner, repository, revision),
        requested=selector,
        version=version,
    )


def _parse_development_selector(selector: str) -> tuple[str, str, str]:
    """Parse a branch name, OWNER/REPO@REF, or GitHub tree/commit URL."""
    default_owner, default_repository = "driftlesslabs", "sharrow"
    parsed = urllib.parse.urlparse(selector)
    if parsed.scheme in {"http", "https"}:
        if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
            raise ValueError("development URLs must refer to github.com")
        parts = [urllib.parse.unquote(part) for part in parsed.path.split("/") if part]
        if len(parts) < 4 or parts[2] not in {"tree", "commit"}:
            raise ValueError(
                "development URL must look like https://github.com/OWNER/REPO/tree/REF"
            )
        return parts[0], parts[1].removesuffix(".git"), "/".join(parts[3:])

    if "@" in selector:
        repository_name, ref = selector.rsplit("@", 1)
        repository_parts = repository_name.split("/", 1)
        if len(repository_parts) != 2 or not all(repository_parts) or not ref:
            raise ValueError("development selector must be OWNER/REPO@REF")
        return repository_parts[0], repository_parts[1].removesuffix(".git"), ref

    if not selector:
        raise ValueError("development branch must not be empty")
    return default_owner, default_repository, selector


def _development_source(selector: str) -> SourceSpec:
    """Resolve a named or linked development branch to an exact commit."""
    owner, repository, ref = _parse_development_selector(selector)
    revision = _resolve_commit(owner, repository, ref)
    return SourceSpec(
        kind="development",
        label=f"{owner}/{repository}@{ref}",
        owner=owner,
        repository=repository,
        ref=ref,
        revision=revision,
        archive_url=_source_archive_url(owner, repository, revision),
        requested=selector,
        version=f"9999.0.dev0+g{revision[:12]}",
    )


def _sha256(path: Path) -> str:
    """Hash a potentially large file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_once(url: str, destination: Path) -> None:
    """Download one URL, resuming a partial response when the server permits."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = _github_headers(authenticated=False)
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        status = getattr(response, "status", None)
        append = offset > 0 and status == 206
        mode = "ab" if append else "wb"
        downloaded = offset if append else 0
        content_length = response.headers.get("Content-Length")
        total = downloaded + int(content_length) if content_length else None
        last_update = 0.0
        with partial.open(mode) as stream:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                stream.write(block)
                downloaded += len(block)
                now = time.monotonic()
                if now - last_update >= 0.5:
                    if total:
                        progress = f"{downloaded / GIB:.2f}/{total / GIB:.2f} GiB"
                    else:
                        progress = f"{downloaded / GIB:.2f} GiB"
                    print(
                        f"\rDownloading {destination.name}: {progress}",
                        end="",
                        flush=True,
                    )
                    last_update = now
        print()
    partial.replace(destination)


def _download(url: str, destination: Path, expected_sha256: str | None = None) -> Path:
    """Return a cached download, verifying and retrying known content hashes."""
    if destination.is_file():
        if expected_sha256 is None or _sha256(destination) == expected_sha256:
            print(f"Using cached download: {destination}")
            return destination
        print(f"Discarding cached file with an invalid checksum: {destination}")
        destination.unlink()

    for attempt in range(2):
        _download_once(url, destination)
        if expected_sha256 is None or _sha256(destination) == expected_sha256:
            return destination
        destination.unlink(missing_ok=True)
        destination.with_name(destination.name + ".part").unlink(missing_ok=True)
        if attempt == 0:
            print("Checksum mismatch; retrying the download from the beginning.")
    raise RuntimeError(f"SHA256 mismatch for {url}; expected {expected_sha256}")


def _extract_with_wring(archive: Path, destination: Path) -> None:
    """Load wring's tar.zst extractor without its unrelated OMX extras.

    Wring 0.0.6 imports its optional OMX module from ``wring.__init__``, which
    otherwise makes the archive tool require NumPy and PyTables. Loading the
    packaged ``tar_zst`` submodule directly keeps extraction portable while
    still using wring's cross-platform implementation.
    """
    package_spec = importlib.util.find_spec("wring")
    if package_spec is None or not package_spec.submodule_search_locations:
        raise RuntimeError("wring is not installed in the controller environment")
    if "wring" not in sys.modules:
        package = types.ModuleType("wring")
        package.__package__ = "wring"
        package.__path__ = list(package_spec.submodule_search_locations)
        package.__spec__ = package_spec
        sys.modules["wring"] = package
    tar_zst = importlib.import_module("wring.tar_zst")
    tar_zst.extract_zst(archive, destination)


def _prepare_mtc_data(cache_dir: Path, refresh: bool = False) -> Path:
    """Download, verify, and transactionally extract the full MTC data archive."""
    data_root = cache_dir / "data" / "v1.3.4"
    archive = data_root / "data_full.tar.zst"
    extracted = data_root / "extracted"
    marker = extracted / ".sharrow-benchmark-data.json"
    if refresh:
        archive.unlink(missing_ok=True)
        archive.with_name(archive.name + ".part").unlink(missing_ok=True)
        if extracted.exists():
            shutil.rmtree(extracted)

    if marker.is_file():
        try:
            metadata = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            metadata = {}
        else:
            candidate = extracted / metadata.get("skims_relative_path", "")
            if metadata.get("sha256") == DATA_SHA256 and candidate.is_file():
                print(f"Using cached MTC skims: {candidate}")
                return candidate

    archive = _download(DATA_URL, archive, DATA_SHA256)
    temporary = data_root / f"extracting-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        # wring supplies the cross-platform zstandard/tar extraction workflow.
        _extract_with_wring(archive, temporary)
        candidates = sorted(temporary.rglob("skims.omx"))
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected one skims.omx in the MTC archive, found {len(candidates)}"
            )
        relative = candidates[0].relative_to(temporary)
        if extracted.exists():
            shutil.rmtree(extracted)
        temporary.replace(extracted)
        marker.write_text(
            json.dumps(
                {"sha256": DATA_SHA256, "skims_relative_path": str(relative)},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    skims = extracted / relative
    print(f"Cached MTC skims: {skims}")
    return skims


def _environment_python(environment: Path) -> Path:
    """Return a virtual environment's interpreter path on Windows or POSIX."""
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def _run_checked(command: list[str], **kwargs) -> None:
    """Print and run an environment-setup command, failing immediately."""
    print(f"+ {subprocess.list2cmdline([str(item) for item in command])}")
    subprocess.run([str(item) for item in command], check=True, **kwargs)


def _isolated_environment() -> dict[str, str]:
    """Prevent a local checkout or user site from contaminating a trial."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _source_archive(source: SourceSpec, cache_dir: Path) -> Path:
    """Download and cache an exact GitHub source archive."""
    archive = cache_dir / "sources" / f"{source.cache_key}.tar.gz"
    return _download(source.archive_url, archive)


def _probe_environment(python: Path, cache_dir: Path) -> dict:
    """Import the installed Sharrow build and report supported load modes."""
    command = [str(python), str(Path(__file__).resolve()), "--worker", "--probe"]
    completed = subprocess.run(
        command,
        check=True,
        cwd=cache_dir,
        env=_isolated_environment(),
        capture_output=True,
        text=True,
    )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    raise RuntimeError(
        "installed Sharrow environment could not be probed:\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def _ensure_environment(
    source: SourceSpec,
    cache_dir: Path,
    python_request: str,
    rebuild: bool,
) -> tuple[Path, dict]:
    """Build or reuse a uv environment for one exact Sharrow source snapshot."""
    environment = cache_dir / "environments" / source.cache_key
    interpreter = _environment_python(environment)
    marker = environment / ".sharrow-benchmark-environment.json"
    expected = {
        "source": asdict(source),
        "python_request": python_request,
        "dependencies": list(BENCHMARK_DEPENDENCIES),
    }
    if not rebuild and interpreter.is_file() and marker.is_file():
        try:
            cached_metadata = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cached_metadata = None
        if cached_metadata == expected:
            print(f"Using cached environment: {environment}")
            return interpreter, _probe_environment(interpreter, cache_dir)

    if environment.exists():
        shutil.rmtree(environment)
    environment.parent.mkdir(parents=True, exist_ok=True)
    archive = _source_archive(source, cache_dir)
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required and was not found on PATH")
    _run_checked(
        [uv, "venv", "--python", python_request, str(environment)], cwd=cache_dir
    )
    interpreter = _environment_python(environment)
    _run_checked(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(interpreter),
            *BENCHMARK_DEPENDENCIES,
        ],
        cwd=cache_dir,
    )
    install_environment = os.environ.copy()
    install_environment["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SHARROW"] = source.version
    _run_checked(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(interpreter),
            "--no-deps",
            "--reinstall",
            str(archive),
        ],
        cwd=cache_dir,
        env=install_environment,
    )
    probe = _probe_environment(interpreter, cache_dir)
    marker.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    return interpreter, probe


def _install_optional_filter_stubs() -> None:
    """Allow gzip-only release benchmarks without unavailable binary plugins."""
    for name in ("blosc2", "hdf5plugin"):
        try:
            importlib.import_module(name)
        except (ImportError, OSError):
            module = types.ModuleType(name)
            module.__doc__ = "Unavailable optional filter stub for the MTC benchmark."
            sys.modules[name] = module


def _rss_high_water_bytes() -> int:
    """Return this process's maximum resident set size in bytes."""
    if resource is None:
        return psutil.Process().memory_info().rss
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return int(value)
    return int(value * 1024)


def _dataset_fingerprint(dataset) -> str:
    """Hash array metadata and edge values without copying full skim arrays."""
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


class _H5pyOmxCompatibilityHandle:
    """Present the legacy OMX shape/root protocol over an h5py file.

    The MTC file predates the optional root ``SHAPE`` attribute expected by
    release 2.16.2's direct h5py-handle path. This adapter exposes the same
    small protocol as ``openmatrix.File`` while leaving all matrix reads on the
    h5py fast path, avoiding a PyTables/OpenMatrix runtime dependency.
    """

    def __init__(self, handle, filename: str):
        self.filename = filename
        self.root = handle
        first_matrix = next(iter(handle["data"].values()))
        self._shape = tuple(int(size) for size in first_matrix.shape)

    def shape(self) -> tuple[int, ...]:
        """Return the common OMX matrix shape."""
        return self._shape


def _worker_probe() -> int:
    """Emit installed-version and API-capability metadata for the controller."""
    _install_optional_filter_stubs()
    import sharrow as sh

    parameters = inspect.signature(sh.dataset.from_omx_3d).parameters
    payload = {
        "status": "ok",
        "sharrow_file": str(Path(sh.__file__).resolve()),
        "sharrow_version": getattr(sh, "__version__", "unknown"),
        "supports_eager": "load" in parameters,
        "supports_shared_reload": hasattr(sh.dataset, "reload_from_omx_3d"),
        "python_version": platform.python_version(),
    }
    print(f"{RESULT_PREFIX}{json.dumps(payload, sort_keys=True)}", flush=True)
    return 0


def _worker(args: argparse.Namespace) -> int:
    """Load the MTC OMX file once and emit a machine-readable trial record."""
    if args.probe:
        return _worker_probe()

    import h5py

    _install_optional_filter_stubs()
    import sharrow as sh

    process = psutil.Process()
    omx_path = str(Path(args.omx).resolve())
    result = {
        "label": args.label,
        "source_kind": args.source_kind,
        "source_revision": args.source_revision,
        "strategy": args.strategy,
        "omx": omx_path,
        "sharrow_file": str(Path(sh.__file__).resolve()),
        "sharrow_version": getattr(sh, "__version__", "unknown"),
        "python_version": platform.python_version(),
        "baseline_rss_bytes": process.memory_info().rss,
    }
    dataset = None
    started = time.perf_counter()
    try:
        kwargs = {
            "time_periods": args.periods.split(","),
            "max_float_precision": args.max_float_precision,
        }
        if args.strategy == "shared-reload":
            with h5py.File(omx_path, "r") as handle:
                source = _H5pyOmxCompatibilityHandle(handle, omx_path)
                template = sh.dataset.from_omx_3d(source, **kwargs)
            shared_key = f"sharrow-mtc-benchmark-{os.getpid()}-{time.time_ns()}"
            dataset = template.shm.to_shared_memory(shared_key, mode="r+", load=False)
            sh.dataset.reload_from_omx_3d(dataset, [omx_path])
        elif args.strategy == "eager":
            kwargs["load"] = "eager"
            with h5py.File(omx_path, "r") as handle:
                source = _H5pyOmxCompatibilityHandle(handle, omx_path)
                dataset = sh.dataset.from_omx_3d(source, **kwargs)
        else:
            with h5py.File(omx_path, "r") as handle:
                source = _H5pyOmxCompatibilityHandle(handle, omx_path)
                dataset = sh.dataset.from_omx_3d(source, **kwargs)
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
    except Exception as error:  # noqa: BLE001 - failures belong in benchmark JSON.
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
        if args.strategy == "shared-reload" and dataset is not None:
            try:
                dataset.shm.release_shared_memory()
            except Exception:  # noqa: BLE001 - retain the primary trial result.
                result["cleanup_traceback"] = traceback.format_exc()

    print(f"{RESULT_PREFIX}{json.dumps(result, sort_keys=True)}", flush=True)
    time.sleep(args.hold_seconds)
    return 0


def _process_tree_rss(process: psutil.Process) -> int:
    """Sum current RSS across a worker and all live descendants."""
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


def _extract_worker_result(stdout: str) -> dict:
    """Extract the final JSON record from worker output."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    raise RuntimeError("worker produced no benchmark result")


def _run_trial(
    args: argparse.Namespace,
    case: BenchmarkCase,
    cache_dir: Path,
    omx: Path,
    run_number: int,
    warmup: bool,
) -> dict:
    """Launch and externally monitor one isolated benchmark trial."""
    command = [
        case.python,
        str(Path(__file__).resolve()),
        "--worker",
        "--omx",
        str(omx),
        "--label",
        case.label,
        "--source-kind",
        case.source_kind,
        "--source-revision",
        case.source.revision,
        "--strategy",
        case.strategy,
        "--periods",
        args.periods,
        "--max-float-precision",
        str(args.max_float_precision),
        "--hold-seconds",
        str(args.hold_seconds),
    ]
    child = subprocess.Popen(
        command,
        cwd=cache_dir,
        env=_isolated_environment(),
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
    try:
        result = _extract_worker_result(stdout)
    except Exception as error:  # noqa: BLE001 - preserve process diagnostics.
        result = {
            "label": case.label,
            "source_kind": case.source_kind,
            "source_revision": case.source.revision,
            "strategy": case.strategy,
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_seconds": 0.0,
            "final_rss_bytes": 0,
            "worker_peak_rss_bytes": 0,
            "stdout": stdout,
        }
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
    """Convert bytes to GiB for terminal reporting."""
    return value / GIB


def _print_trial(result: dict) -> None:
    """Print one compact benchmark trial."""
    message = (
        f"{result['label']:24} {result['strategy']:13} "
        f"run={result['run']} warmup={result['warmup']} "
        f"status={result['status']:5} time={result['elapsed_seconds']:8.3f}s "
        f"peak={_gib(result['peak_rss_bytes']):7.3f} GiB"
    )
    if result["status"] == "error":
        message += f" error={result['error_type']}: {result['error']}"
    print(message, flush=True)


def _summaries(results: list[dict]) -> list[dict]:
    """Compute medians and ranges for successful measured trials."""
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
        elapsed = [trial["elapsed_seconds"] for trial in trials]
        peak = [trial["peak_rss_bytes"] for trial in trials]
        summaries.append(
            {
                "label": label,
                "source_kind": trials[0]["source_kind"],
                "source_revision": trials[0]["source_revision"],
                "strategy": trials[0]["strategy"],
                "trials": len(trials),
                "median_elapsed_seconds": statistics.median(elapsed),
                "min_elapsed_seconds": min(elapsed),
                "max_elapsed_seconds": max(elapsed),
                "median_peak_rss_bytes": statistics.median(peak),
                "median_final_rss_bytes": statistics.median(
                    trial["final_rss_bytes"] for trial in trials
                ),
                "median_baseline_rss_bytes": statistics.median(
                    trial["baseline_rss_bytes"] for trial in trials
                ),
                "median_peak_increase_bytes": statistics.median(
                    trial["peak_rss_bytes"] - trial["baseline_rss_bytes"]
                    for trial in trials
                ),
                "dataset_nbytes": trials[0]["dataset_nbytes"],
                "fingerprint": trials[0]["fingerprint"],
            }
        )
    return summaries


def _comparisons(summaries: list[dict]) -> list[dict]:
    """Compare release and development medians for matching strategies."""
    comparisons = []
    for strategy in dict.fromkeys(summary["strategy"] for summary in summaries):
        release = next(
            (
                summary
                for summary in summaries
                if summary["strategy"] == strategy
                and summary["source_kind"] == "release"
            ),
            None,
        )
        development = next(
            (
                summary
                for summary in summaries
                if summary["strategy"] == strategy
                and summary["source_kind"] == "development"
            ),
            None,
        )
        if release and development:
            comparisons.append(
                {
                    "strategy": strategy,
                    "development_speedup": release["median_elapsed_seconds"]
                    / development["median_elapsed_seconds"],
                    "development_peak_rss_ratio": development["median_peak_rss_bytes"]
                    / release["median_peak_rss_bytes"],
                }
            )
    return comparisons


def _parse_strategies(value: str) -> list[str]:
    """Normalize the public lazy/eager/shared strategy list."""
    aliases = {"lazy": "lazy", "eager": "eager", "shared": "shared-reload"}
    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = [item for item in requested if item not in aliases]
    if invalid or not requested:
        raise argparse.ArgumentTypeError(
            "strategies must be a comma-separated subset of lazy,eager,shared"
        )
    return list(dict.fromkeys(aliases[item] for item in requested))


def _build_cases(
    strategies: list[str],
    release: SourceSpec,
    release_python: Path,
    release_probe: dict,
    development: SourceSpec,
    development_python: Path,
    development_probe: dict,
) -> list[BenchmarkCase]:
    """Create only the source/strategy combinations supported by each API."""
    cases = []
    source_rows = (
        ("release", release, release_python, release_probe),
        ("development", development, development_python, development_probe),
    )
    for strategy in strategies:
        for source_kind, source, python, probe in source_rows:
            if strategy == "eager" and not probe["supports_eager"]:
                print(f"Skipping eager mode for {source.label}: API not available")
                continue
            if strategy == "shared-reload" and not probe["supports_shared_reload"]:
                print(f"Skipping shared mode for {source.label}: API not available")
                continue
            cases.append(
                BenchmarkCase(
                    label=f"{source_kind}-{strategy.replace('-reload', '')}",
                    source_kind=source_kind,
                    strategy=strategy,
                    python=str(python),
                    source=source,
                )
            )
    return cases


def _controller(args: argparse.Namespace) -> int:
    """Prepare inputs/environments, execute trials, and write the final report."""
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    if args.omx:
        omx = Path(args.omx).expanduser().resolve()
        if not omx.is_file():
            raise FileNotFoundError(omx)
        data_provenance = {"kind": "local", "omx": str(omx)}
    else:
        omx = _prepare_mtc_data(cache_dir, refresh=args.refresh_data)
        data_provenance = {
            "kind": "download",
            "url": DATA_URL,
            "sha256": DATA_SHA256,
            "omx": str(omx),
        }

    available = psutil.virtual_memory().available
    if available < 20 * GIB:
        print(
            f"Warning: only {_gib(available):.1f} GiB of memory is currently "
            "available; the release lazy load may peak near 17 GiB."
        )

    print("Resolving exact Sharrow source revisions...")
    release = _release_source(args.release)
    development = _development_source(args.development)
    print(f"Release:    {release.label} ({release.revision})")
    print(f"Development: {development.label} ({development.revision})")

    release_python, release_probe = _ensure_environment(
        release, cache_dir, args.python, args.rebuild_environments
    )
    development_python, development_probe = _ensure_environment(
        development, cache_dir, args.python, args.rebuild_environments
    )
    print(f"Release environment:    {release_probe['sharrow_file']}")
    print(f"Development environment: {development_probe['sharrow_file']}")
    if args.prepare_only:
        print("Preparation complete; benchmark trials were not requested.")
        return 0

    cases = _build_cases(
        args.strategies,
        release,
        release_python,
        release_probe,
        development,
        development_python,
        development_probe,
    )
    if not cases:
        raise RuntimeError("none of the requested strategies is supported")

    results = []
    for warmup_number in range(1, args.warmups + 1):
        for case in cases:
            result = _run_trial(args, case, cache_dir, omx, warmup_number, warmup=True)
            results.append(result)
            _print_trial(result)

    # Rotate order between repetitions to distribute filesystem-cache effects.
    for run_number in range(1, args.repetitions + 1):
        offset = (run_number - 1) % len(cases)
        for case in cases[offset:] + cases[:offset]:
            result = _run_trial(args, case, cache_dir, omx, run_number, warmup=False)
            results.append(result)
            _print_trial(result)

    summaries = _summaries(results)
    comparisons = _comparisons(summaries)
    successful_fingerprints = {
        result["fingerprint"] for result in results if result["status"] == "ok"
    }
    fingerprints_match = len(successful_fingerprints) == 1
    payload = {
        "benchmark": "Sharrow full-scale MTC OMX loading",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "total_memory_bytes": psutil.virtual_memory().total,
            "controller_python": platform.python_version(),
            "benchmark_python_request": args.python,
            "uv_version": subprocess.check_output(
                [shutil.which("uv") or "uv", "--version"], text=True
            ).strip(),
        },
        "data": data_provenance,
        "omx_size_bytes": omx.stat().st_size,
        "periods": args.periods.split(","),
        "max_float_precision": args.max_float_precision,
        "memory_metric": "peak RSS across the complete benchmark process tree",
        "sample_interval_seconds": args.sample_interval,
        "sources": {
            "release": asdict(release),
            "development": asdict(development),
        },
        "environment_probes": {
            "release": release_probe,
            "development": development_probe,
        },
        "fingerprints_match": fingerprints_match,
        "results": results,
        "summaries": summaries,
        "comparisons": comparisons,
    }

    print("\nSummary (medians of measured trials)")
    for summary in summaries:
        print(
            f"{summary['label']:24} {summary['strategy']:13} "
            f"time={summary['median_elapsed_seconds']:8.3f}s "
            f"peak={_gib(summary['median_peak_rss_bytes']):7.3f} GiB "
            f"dataset={_gib(summary['dataset_nbytes']):7.3f} GiB"
        )
    for comparison in comparisons:
        print(
            f"{comparison['strategy']:24} development speedup="
            f"{comparison['development_speedup']:.2f}x, peak RSS="
            f"{comparison['development_peak_rss_ratio']:.1%} of release"
        )

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {output}")

    failures = [result for result in results if result["status"] != "ok"]
    if failures:
        print(f"Benchmark completed with {len(failures)} failed trial(s).")
        return 1
    if not fingerprints_match:
        print("Benchmark outputs did not have matching fingerprints.")
        return 2
    return 0


def _parser() -> argparse.ArgumentParser:
    """Build the public controller and private worker command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development",
        default=DEFAULT_DEVELOPMENT,
        help=(
            "development branch name, OWNER/REPO@REF, or GitHub tree URL "
            f"(default: {DEFAULT_DEVELOPMENT})"
        ),
    )
    parser.add_argument(
        "--release",
        default="latest",
        help="ActivitySim/Sharrow release tag, or 'latest' (default: latest)",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(_default_cache_dir()),
        help="download, source, and environment cache directory",
    )
    parser.add_argument(
        "--omx",
        help="use an existing skims.omx instead of downloading the MTC archive",
    )
    parser.add_argument(
        "--python",
        default="3.12",
        help="Python version used in both benchmark environments (default: 3.12)",
    )
    parser.add_argument(
        "--strategies",
        type=_parse_strategies,
        default=_parse_strategies("lazy,eager,shared"),
        help="comma-separated subset of lazy,eager,shared",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--periods", default="EA,AM,MD,PM,EV")
    parser.add_argument("--max-float-precision", type=int, default=32)
    parser.add_argument("--sample-interval", type=float, default=0.005)
    parser.add_argument("--hold-seconds", type=float, default=0.25)
    parser.add_argument("--output", default="mtc-sharrow-benchmark.json")
    parser.add_argument(
        "--refresh-data", action="store_true", help="redownload and re-extract MTC data"
    )
    parser.add_argument(
        "--rebuild-environments",
        action="store_true",
        help="recreate cached release and development environments",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="prepare data and environments without running load trials",
    )

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--label", help=argparse.SUPPRESS)
    parser.add_argument("--source-kind", help=argparse.SUPPRESS)
    parser.add_argument("--source-revision", help=argparse.SUPPRESS)
    parser.add_argument(
        "--strategy",
        choices=("lazy", "eager", "shared-reload"),
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> int:
    """Dispatch to an isolated load worker or the benchmark controller."""
    args = _parser().parse_args()
    if args.worker:
        return _worker(args)
    if args.warmups < 0 or args.repetitions < 1:
        raise SystemExit("warmups must be nonnegative and repetitions positive")
    if args.sample_interval <= 0 or args.hold_seconds < 0:
        raise SystemExit("sample interval must be positive and hold time nonnegative")
    return _controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
