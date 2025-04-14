import atexit
import hashlib
import logging
import os
import pickle
import time

import dask
import dask.array as da
import numpy as np
import xarray as xr

try:
    from multiprocessing.shared_memory import ShareableList, SharedMemory
except ImportError:
    ShareableList, SharedMemory = None, None

try:
    import sparse
except ImportError:
    sparse = None

__GLOBAL_MEMORY_ARRAYS = {}
__GLOBAL_MEMORY_LISTS = {}

logger = logging.getLogger("sharrow.shared_memory")


def si_units(x, kind="B", digits=3, shift=1000):
    #       nano micro milli    kilo mega giga tera peta exa  zeta yotta
    tiers = ["n", "µ", "m", "", "K", "M", "G", "T", "P", "E", "Z", "Y"]

    tier = 3
    try:
        sign = "-" if x < 0 else ""
    except TypeError:
        # x is not a number, just return it
        return x
    x = abs(x)
    if x > 0:
        while x > shift and tier < len(tiers):
            x /= shift
            tier += 1
        while x < 1 and tier >= 0:
            x *= shift
            tier -= 1
    return f"{sign}{round(x, digits)} {tiers[tier]}{kind}"


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
            try:
                v.unlink()
            except FileNotFoundError:
                pass
        while __GLOBAL_MEMORY_LISTS:
            k, v = __GLOBAL_MEMORY_LISTS.popitem()
            v.shm.close()
            try:
                v.shm.unlink()
            except FileNotFoundError:
                pass


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
            raise FileExistsError(f"sharrow_shared_memory_array:{key}") from None
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
            raise FileNotFoundError(f"sharrow_shared_memory_array:{key}") from None
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
            raise FileExistsError(f"sharrow_shared_memory_list:{key}") from None
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
            raise FileNotFoundError(f"sharrow_shared_memory_list:{key}") from None
        else:
            return sl


def get_shared_list_nbytes(key):
    if key.startswith("memmap:"):
        return 0
    h = _hexhash(f"sharrow__list__{key}")
    try:
        shm = SharedMemory(name=h, create=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"sharrow_shared_memory_list:{key}") from None
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


@xr.register_dataset_accessor("shm")
class SharedMemDatasetAccessor:
    _parent_class = xr.Dataset

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._shared_memory_key_ = None
        self._shared_memory_objs_ = []
        self._shared_memory_owned_ = False

    def _repr_html_(self):
        html = self._obj._repr_html_()
        html = html.replace("xarray.Dataset", "xarray.Dataset.shm")
        return html

    def __repr__(self):
        r = self._obj.__repr__()
        r = r.replace("xarray.Dataset", "xarray.Dataset.shm")
        return r

    def release_shared_memory(self):
        """Release shared memory allocated to this Dataset."""
        release_shared_memory(self._shared_memory_key_)

    @staticmethod
    def delete_shared_memory_files(key):
        delete_shared_memory_files(key)

    def to_shared_memory(
        self,
        key=None,
        mode="r+",
        _dupe=True,
        dask_scheduler="threads",
        pre_init=False,
        load=True,
    ):
        """
        Load this Dataset into shared memory.

        The returned Dataset object references the shared memory and is the
        "owner" of this data.  When this object is destroyed, the data backing
        it may also be freed, which can result in a segfault or other unfortunate
        condition if that memory is still accessed from elsewhere.

        Parameters
        ----------
        key : str
            An identifying key for this shared memory.  Use the same key
            in `from_shared_memory` to recreate this Dataset elsewhere.
        mode : {‘r+’, ‘r’, ‘w+’, ‘c’}, optional
            This method returns a copy of the Dataset in shared memory.
            If memmapped, that copy can be opened in various modes.
            See numpy.memmap() for details.
        dask_scheduler : str, default 'threads'
            The scheduler to use when loading dask arrays into shared memory.
            Typically, this is "threads" for multithreaded reads or "synchronous"
            for single-threaded reads. See dask.compute() for details.
        pre_init : bool, default False
            If True, the shared memory buffer will be pre-initialized with zeros.
            This is generally not necessary, but can be useful for debugging.
        load : bool, default True
            If True, load the data into shared memory immediately, using dask.
            If False, defer loading until later. Deferred tasks are stored in
            the `shm.tasks` attribute of the resulting Dataset object, but do not
            necessarily need to be run if data can be loaded using alternative
            methods (e.g. `sharrow.dataset.reload_from_omx_3d`).

        Returns
        -------
        Dataset
        """
        logger.info(f"sharrow.Dataset.to_shared_memory({key})")
        if key is None:
            import random

            key = random.randbytes(4).hex()
        self._shared_memory_key_ = key
        self._shared_memory_owned_ = False
        self._shared_memory_objs_ = []

        wrappers = []
        sizes = []
        names = []
        position = 0

        def emit(k, a, is_coord):
            nonlocal names, wrappers, sizes, position
            if sparse is not None and isinstance(a.data, sparse.GCXS):
                logger.debug(f"preparing sparse array {a.name}")
                wrappers.append(
                    {
                        "sparse": True,
                        "dims": a.dims,
                        "name": a.name,
                        "attrs": a.attrs,
                        "dtype": a.dtype,
                        "shape": a.shape,
                        "coord": is_coord,
                        "nbytes": a.data.nbytes,
                        "position": position,
                        "data.nbytes": a.data.data.nbytes,
                        "indices.nbytes": a.data.indices.nbytes,
                        "indptr.nbytes": a.data.indptr.nbytes,
                        "data.dtype": a.data.data.dtype,
                        "indices.dtype": a.data.indices.dtype,
                        "indptr.dtype": a.data.indptr.dtype,
                    }
                )
                a_nbytes = a.data.nbytes
            else:
                logger.debug(f"preparing dense array {a.name}")
                wrappers.append(
                    {
                        "dims": a.dims,
                        "name": a.name,
                        "attrs": a.attrs,
                        "dtype": a.dtype,
                        "shape": a.shape,
                        "coord": is_coord,
                        "nbytes": a.nbytes,
                        "position": position,
                    }
                )
                a_nbytes = a.nbytes

            sizes.append(a_nbytes)
            names.append(k)
            position += a_nbytes

        for k, a in self._obj.coords.items():
            emit(k, a, True)
        for k in self._obj.variables:
            if k in names:
                continue
            a = self._obj[k]
            emit(k, a, False)

        mem = create_shared_memory_array(key, size=position)

        logger.debug("declaring shared memory buffer")
        if key.startswith("memmap:"):
            buffer = memoryview(mem)
        else:
            buffer = mem.buf

        if pre_init:
            logger.debug("pre-initializing shared memory buffer")
            # gross init with all zeros
            buffer[:] = b"\0" * len(buffer)

        tasks = []
        task_names = []
        for w in wrappers:
            _is_sparse = w.get("sparse", False)
            _size = w["nbytes"]
            _name = w["name"]
            _pos = w["position"]
            a = self._obj[_name]
            if _is_sparse:
                logger.info(f"running load task: {_name} ({si_units(_size)})")
                ad = a.data
                _size_d = w["data.nbytes"]
                _size_i = w["indices.nbytes"]
                _size_p = w["indptr.nbytes"]
                mem_arr_d = np.ndarray(
                    shape=(_size_d // ad.data.dtype.itemsize,),
                    dtype=ad.data.dtype,
                    buffer=buffer[_pos : _pos + _size_d],
                )
                mem_arr_i = np.ndarray(
                    shape=(_size_i // ad.indices.dtype.itemsize,),
                    dtype=ad.indices.dtype,
                    buffer=buffer[_pos + _size_d : _pos + _size_d + _size_i],
                )
                mem_arr_p = np.ndarray(
                    shape=(_size_p // ad.indptr.dtype.itemsize,),
                    dtype=ad.indptr.dtype,
                    buffer=buffer[
                        _pos + _size_d + _size_i : _pos + _size_d + _size_i + _size_p
                    ],
                )
                mem_arr_d[:] = ad.data[:]
                mem_arr_i[:] = ad.indices[:]
                mem_arr_p[:] = ad.indptr[:]
            else:
                logger.debug(f"preparing load task: {_name} ({si_units(_size)})")
                mem_arr = np.ndarray(
                    shape=a.shape, dtype=a.dtype, buffer=buffer[_pos : _pos + _size]
                )
                if isinstance(a, xr.DataArray) and isinstance(a.data, da.Array):
                    tasks.append(da.store(a.data, mem_arr, lock=False, compute=False))
                    task_names.append(_name)
                else:
                    mem_arr[:] = a[:]
        if tasks and load:
            self.tasks = tasks
            self.task_names = task_names
            self.run_tasks(dask_scheduler=dask_scheduler)

        if key.startswith("memmap:"):
            mem.flush()

        logger.info("storing metadata in shared memory")
        create_shared_list(
            [pickle.dumps(self._obj.attrs)] + [pickle.dumps(i) for i in wrappers], key
        )
        result = type(self).from_shared_memory(key, own_data=mem, mode=mode)
        if tasks and not load:
            # attach incompleted tasks to the result
            result.shm.tasks = tasks
            result.shm.task_names = task_names
        result.shm._buffer = buffer
        result.shm._position = position
        return result

    def run_tasks(self, dask_scheduler="threads"):
        """Run any deferred dask tasks."""
        if not hasattr(self, "tasks"):
            return
        else:
            tasks = self.tasks
        if not hasattr(self, "task_names"):
            task_names = ["untitled" for _ in tasks]
        else:
            task_names = self.task_names
        t = time.time()
        logger.info(f"running {len(tasks)} dask data load tasks")
        if dask_scheduler == "synchronous":
            for task, task_name in zip(tasks, task_names):
                logger.info(f"running load task: {task_name}")
                dask.compute(task, scheduler=dask_scheduler)
        else:
            dask.compute(tasks, scheduler=dask_scheduler)
        logger.info(f"completed tasks in {time.time() - t:.3f} seconds")
        del self.tasks

    @property
    def shared_memory_key(self):
        try:
            return self._shared_memory_key_
        except AttributeError:
            raise ValueError("this dataset is not in shared memory") from None

    @classmethod
    def from_shared_memory(cls, key, own_data=False, mode="r+"):
        """
        Connect to an existing Dataset in shared memory.

        Parameters
        ----------
        key : str
            An identifying key for this shared memory.  Use the same key
            in `from_shared_memory` to recreate this Dataset elsewhere.
        own_data : bool or memmap, default False
            The returned Dataset object references the shared memory but is
            not the "owner" of this data unless this flag is set.  Pass a
            memmap to reuse an existing buffer.

        Returns
        -------
        Dataset
        """
        import pickle

        from xarray import DataArray

        _shared_memory_objs_ = []

        shr_list = read_shared_list(key)
        try:
            _shared_memory_objs_.append(shr_list.shm)
        except AttributeError:
            # for memmap, list is loaded from pickle, not shared ram
            pass

        if own_data and own_data is not True:
            mem = own_data
            own_data = True
        else:
            mem = open_shared_memory_array(key, mode=mode)
        _shared_memory_objs_.append(mem)
        if key.startswith("memmap:"):
            buffer = memoryview(mem)
        else:
            buffer = mem.buf

        content = {}

        for _n, w in enumerate(shr_list):
            if _n == 0:
                continue
            t = pickle.loads(w)
            shape = t.pop("shape")
            dtype = t.pop("dtype")
            name = t.pop("name")
            coord = t.pop("coord", False)  # noqa: F841
            position = t.pop("position")
            nbytes = t.pop("nbytes")
            is_sparse = t.pop("sparse", False)
            if is_sparse:
                if sparse is None:
                    raise ImportError("sparse is not installed")
                _size_d = t.pop("data.nbytes")
                _size_i = t.pop("indices.nbytes")
                _size_p = t.pop("indptr.nbytes")
                _dtype_d = t.pop("data.dtype")
                _dtype_i = t.pop("indices.dtype")
                _dtype_p = t.pop("indptr.dtype")

                mem_arr_d = np.ndarray(
                    _size_d // _dtype_d.itemsize,
                    dtype=_dtype_d,
                    buffer=buffer[position : position + _size_d],
                )
                mem_arr_i = np.ndarray(
                    _size_i // _dtype_i.itemsize,
                    dtype=_dtype_i,
                    buffer=buffer[position + _size_d : position + _size_d + _size_i],
                )
                mem_arr_p = np.ndarray(
                    _size_p // _dtype_p.itemsize,
                    dtype=_dtype_p,
                    buffer=buffer[
                        position + _size_d + _size_i : position
                        + _size_d
                        + _size_i
                        + _size_p
                    ],
                )
                mem_arr = sparse.GCXS(
                    (
                        mem_arr_d,
                        mem_arr_i,
                        mem_arr_p,
                    ),
                    shape=shape,
                    compressed_axes=(0,),
                )
            else:
                mem_arr = np.ndarray(
                    shape, dtype=dtype, buffer=buffer[position : position + nbytes]
                )
            content[name] = DataArray(mem_arr, **t)

        obj = cls._parent_class(content)
        obj.attrs = pickle.loads(shr_list[0])
        obj.shm._shared_memory_key_ = key
        obj.shm._shared_memory_owned_ = own_data
        obj.shm._shared_memory_objs_ = _shared_memory_objs_
        return obj

    @property
    def shared_memory_size(self):
        """Int : Size (in bytes) in shared memory, raises ValueError if not shared."""
        try:
            return sum(i.size for i in self._shared_memory_objs_)
        except AttributeError:
            raise ValueError("this dataset is not in shared memory") from None

    @property
    def is_shared_memory(self):
        """Bool : Whether this Dataset is in shared memory."""
        try:
            return sum(i.size for i in self._shared_memory_objs_) > 0
        except AttributeError:
            return False

    @staticmethod
    def preload_shared_memory_size(key):
        """
        Compute the size in bytes of a shared Dataset without actually loading it.

        Parameters
        ----------
        key : str
            The identifying key for this shared memory.

        Returns
        -------
        int
        """
        memsize = 0
        try:
            n = get_shared_list_nbytes(key)
        except FileNotFoundError:
            pass
        else:
            memsize += n
        try:
            mem = open_shared_memory_array(key, mode="r")
        except FileNotFoundError:
            pass
        else:
            memsize += mem.size
        return memsize

    def __getattr__(self, item):
        return getattr(self._obj, item)

    def __getitem__(self, item):
        return self._obj[item]
