# sharrow

The sharrow package is an extension of [numba](https://numba.pydata.org/), and
offers access to data formatting and a just-in-time compiler specifically for
converting ActivitySim-style “specification” files into optimized, runnable
functions that “can approach the speeds of C or FORTRAN”. The idea is to pay the
cost of compiling these specification files only once, and then re-use the
optimized results many times.

This system depends only on widely used free open-source libraries, including
[numba](https://numba.pydata.org/) and [xarray](https://xarray.pydata.org/) and
is tested to be compatible with Windows, Linux, and macOS. Most importantly,
using sharrow is very user friendly and requires almost no knowledge of the
underlying mechanisms that make it speedy.  The first time a data flow is run on
a new machine, the run will be slow as the compiler is invoked to generate
machine-specific optimized code. The compiled version of this code is cached to
disk, so that all future runs of the same model with the same spec files (while
allowing for revisions to input data and parameter values) can use the
pre-compiled functions, resulting in a massive speed boost.

The documentation published here is primarily geared towards software developers,
as regular users probably won't need to understand much of how sharrow works in
order to benefit from it, once it is embedded in other tools like ActivitySim.


## ActivitySim

Sharrow is a project of the [ActivitySim](https://activitysim.github.io/) consortium.

The mission of the ActivitySim Consortium is to create and maintain advanced,
open-source, activity-based travel behavior modeling software based on best
software development practices for distribution at no charge to the public.

![AMPORF](_static/ampo.png#floatleft)

ActivitySim is administered by the [Association of Metropolitan Planning
Organizations (AMPO) Research Foundation](https://research.ampo.org), a Federal 501(c)(3)
organization.


## History

Sharrow was originally developed by [Cambridge Systematics](https://www.camsys.com)
as an accelerator for evaluating utility expressions.  It was released as open source
and copyright transferred to the AMPO Research Foundation in 2022 as part of ActivitySim's
Phase 7 development work.


## License

Copyright (c) 2022, Association of Metropolitan Planning Organizations Research Foundation

Sharrow is made available under the open source [3-Clause BSD License](https://opensource.org/licenses/BSD-3-Clause).
