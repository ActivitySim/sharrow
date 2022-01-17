import base64
import contextlib
import filecmp
import os
import uuid

from filelock import FileLock, Timeout


def blacken(code):
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

        warnings.warn(f"error in blacken: {err!r}")
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
