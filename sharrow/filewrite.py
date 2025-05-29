import base64
import contextlib
import filecmp
import os
import uuid
from pathlib import Path

from filelock import FileLock, Timeout

# by default, blackening is disabled for performance reasons,
# but it can be enabled by setting the environment variable SHARROW_BLACKEN=1
SHARROW_BLACKEN = os.environ.get("SHARROW_BLACKEN", "0") == "1"


def blacken(code):
    if not SHARROW_BLACKEN:
        return code

    # Ruff is much faster than black, so we try to use it first.
    import subprocess

    which_ruff = subprocess.run(["which", "ruff"], capture_output=True, text=True)
    which_ruff.check_returncode()
    ruff_bin = which_ruff.stdout.strip()
    if ruff_bin:
        try:
            result = subprocess.run(
                [ruff_bin, "format", "-"],
                input=code,
                text=True,
                capture_output=True,
                check=True,
                cwd=Path.cwd(),
            )
            result.check_returncode()
            return result.stdout
        except subprocess.CalledProcessError as err:
            import warnings

            warnings.warn(f"error in ruffen: {err!r}", stacklevel=2)
            return code
    else:
        import warnings

        warnings.warn("ruff not found, trying black instead", stacklevel=2)

    try:
        import black
    except ImportError:
        return code
    mode = black.FileMode()
    fast = False
    try:
        return black.format_file_contents(code, fast=fast, mode=mode)
    except black.NothingChanged:
        return code
    except Exception as err:
        import warnings

        warnings.warn(f"error in blacken: {err!r}", stacklevel=2)
        return code


@contextlib.contextmanager
def rewrite(pth, mode="wt"):
    if not os.path.exists(pth):
        with open(pth, mode) as f:
            yield f
    else:
        unique_id = base64.b32encode(uuid.uuid4().bytes)[:26].decode()
        pth_, pth_ext = os.path.splitext(pth)
        temp_file = pth_ + "_temp_" + unique_id + pth_ext
        with open(temp_file, mode) as f:
            yield f
        lock_path = pth + ".lock"
        lock = FileLock(lock_path, timeout=15)
        try:
            with lock.acquire():
                if filecmp.cmp(pth, temp_file, shallow=False):
                    # no changes, delete temp_file
                    os.remove(temp_file)
                else:
                    # changes, delete original and move tempfile
                    os.remove(pth)
                    os.rename(temp_file, pth)
        except Timeout:
            pass
