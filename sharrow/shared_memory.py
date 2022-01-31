import atexit
import hashlib
import logging
import os
import pickle
from multiprocessing.shared_memory import ShareableList, SharedMemory

import numpy as np

__GLOBAL_MEMORY_ARRAYS = {}
__GLOBAL_MEMORY_LISTS = {}

logger = logging.getLogger("sharrow.shared_memory")


def si_units(x, kind="B", digits=3, shift=1000):

    #       nano micro milli    kilo mega giga tera peta exa  zeta yotta
    tiers = ["n", "µ", "m", "", "K", "M", "G", "T", "P", "E", "Z", "Y"]

    tier = 3
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x > 0:
        while x > shift and tier < len(tiers):
            x /= shift
            tier += 1
        while x < 1 and tier >= 0:
            x *= shift
            tier -= 1
    return f"{sign}{round(x,digits)} {tiers[tier]}{kind}"


def _hexhash(t, size=10, prefix="sharr"):
    h = hashlib.blake2b(str(t).encode(), digest_size=size).hexdigest()
    return prefix + h


def release_shared_memory(key=None):
    if key:
        v = __GLOBAL_MEMORY_ARRAYS.pop(key, None)
        if v:
            v.close()
            v.unlink()
        v = __GLOBAL_MEMORY_LISTS.pop(key, None)
        if v:
            v.shm.close()
            v.shm.unlink()
    else:
        while __GLOBAL_MEMORY_ARRAYS:
            k, v = __GLOBAL_MEMORY_ARRAYS.popitem()
            v.close()
            v.unlink()
        while __GLOBAL_MEMORY_LISTS:
            k, v = __GLOBAL_MEMORY_LISTS.popitem()
            v.shm.close()
            v.shm.unlink()


atexit.register(release_shared_memory)


def persist():
    atexit.unregister(release_shared_memory)


def create_shared_memory_array(key, size):
    logger.info(f"create_shared_memory_array({key}, )")

    if key.startswith("memmap:"):
        backing = key
    else:
        backing = "shared_memory"

    if backing == "shared_memory":
        logger.info(f"create_shared_memory_array:{key} ({si_units(size)})")
        h = _hexhash(f"sharrow__{key}")
        if h in __GLOBAL_MEMORY_ARRAYS:
            raise FileExistsError(f"sharrow_shared_memory_array:{key}")
        try:
            result = SharedMemory(
                name=h,
                create=True,
                size=size,
            )
        except FileExistsError:
            raise FileExistsError(f"sharrow_shared_memory_array:{key}")
        __GLOBAL_MEMORY_ARRAYS[key] = result
        return result

    if backing.startswith("memmap:"):
        logger.info(f"create_shared_memory_array:{key} ({backing}, {si_units(size)})")
        mmap_filename = backing[7:]
        if os.path.isfile(mmap_filename):
            result = np.memmap(
                mmap_filename,
                mode="r+",
            )
            if result.size != size:
                raise ValueError(f"file size mismatch, want {size} found {result.size}")
            return result
        else:
            return np.memmap(
                mmap_filename,
                mode="w+",
                shape=size,
            )

    raise ValueError(f"unknown backing {backing!r}")


def open_shared_memory_array(key, mode="r+"):
    logger.info(f"open_shared_memory_array:{key}")
    if key.startswith("memmap:"):
        backing = key
    else:
        backing = "shared_memory"
    if backing == "shared_memory":
        h = _hexhash(f"sharrow__{key}")
        try:
            result = SharedMemory(
                name=h,
                create=False,
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"sharrow_shared_memory_array:{key}")
        else:
            logger.info(
                f"shared memory array from ephemeral memory, {si_units(result.size)}"
            )
            return result

    if backing.startswith("memmap:"):
        mmap_filename = backing[7:]
        result = np.memmap(
            mmap_filename,
            mode=mode,
            shape=None,
        )
        logger.info(f"shared memory array from memmap, {si_units(result.size)}")
        return result

    raise ValueError(f"unknown backing {backing!r}")


def create_shared_list(content, key):
    if key.startswith("memmap:"):
        mmap_filename = key[7:]
        os.makedirs(os.path.dirname(mmap_filename), exist_ok=True)
        with open(mmap_filename + ".meta.pkl", mode="wb") as f:
            pickle.dump(content, f)
    else:
        logger.info(f"create_shared_list:{key}")
        h = _hexhash(f"sharrow__list__{key}")
        if h in __GLOBAL_MEMORY_LISTS:
            raise FileExistsError(f"sharrow_shared_memory_list:{key}")
        try:
            result = ShareableList(
                content,
                name=h,
            )
        except FileExistsError:
            raise FileExistsError(f"sharrow_shared_memory_list:{key}")
        __GLOBAL_MEMORY_LISTS[key] = result
        return result


def read_shared_list(key):
    if key.startswith("memmap:"):
        mmap_filename = key[7:]
        with open(mmap_filename + ".meta.pkl", mode="rb") as f:
            return pickle.load(f)
    else:
        logger.info(f"read_shared_list:{key}")
        try:
            sl = ShareableList(name=_hexhash(f"sharrow__list__{key}"))
        except FileNotFoundError:
            raise FileNotFoundError(f"sharrow_shared_memory_list:{key}")
        else:
            return sl


def get_shared_list_nbytes(key):
    if key.startswith("memmap:"):
        return 0
    h = _hexhash(f"sharrow__list__{key}")
    try:
        shm = SharedMemory(name=h, create=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"sharrow_shared_memory_list:{key}")
    else:
        return shm.size


def delete_shared_memory_files(key):
    if key.startswith("memmap:"):
        logger.info(f"delete_shared_memory_files:{key}")
        mmap_filename = key[7:]
        if os.path.isfile(mmap_filename):
            os.unlink(mmap_filename)
        if os.path.isfile(mmap_filename + ".meta.pkl"):
            os.unlink(mmap_filename + ".meta.pkl")
