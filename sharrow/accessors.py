import xarray as xr


def register_dataset_method(func):
    def wrapper(dataset):
        def f(*args, **kwargs):
            return func(dataset, *args, **kwargs)

        return f

    wrapper.__doc__ = func.__doc__
    return xr.register_dataset_accessor(func.__name__)(wrapper)


def register_dataarray_method(func):
    def wrapper(dataarray):
        def f(*args, **kwargs):
            return func(dataarray, *args, **kwargs)

        return f

    wrapper.__doc__ = func.__doc__
    return xr.register_dataarray_accessor(func.__name__)(wrapper)


def register_dataarray_staticmethod(func):
    return xr.register_dataarray_accessor(func.__name__)(func)


def register_dataset_staticmethod(func):
    return xr.register_dataset_accessor(func.__name__)(func)
