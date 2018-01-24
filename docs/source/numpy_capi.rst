.. Copyright (c) 2016, Johan Mabille and Sylvain Corlay

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Importing numpy C API
=====================

Importing the C API module of numpy requires more code than just including a header. ``xtensor-python`` simplifies a lot
this import, however some actions are still required in the user code.

Extension module with a single file
-----------------------------------

When writing an extension module that is self-contained in a single file, its author should pay attention to the following
points:

- ``FORCE_IMPORT_ARRAY`` must be defined before including any header of ``xtensor-python``.
- ``xt::import_numpy()`` must be called in the function initializing the module.

Thus the basic skeleton of the module looks like:

.. code::

    #include "pybind11/pybind11.h"
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pyarray.hpp"

    PYBIND11_MODULE(plugin_name, m)
    {
        xt::import_numpy();
        //...
    }


Extension module with multiple files
------------------------------------

If the extension module contain many source files that include ``xtensor-python`` header files, the previous points are still
required. However, the source files that don't contain the initializing code of the module can directly include ``xtensor-python``
header files.


Using other extension modules
-----------------------------

Including an header of ``xtensor-python`` actually defines ``PY_ARRAY_UNIQUE_SYMBOL`` to ``xtensor_python_ARRAY_API``. This might
be problematic if you import another library that defines its own ``PY_ARRAY_UNIQUE_SYMBOL``, or if you define yours. If so,
you can override the behavior of ``xtensor-python`` by explicitly defining ``PY_ARRAY_UNIQUE_SYMBOL`` prior to including any
``stenxor-python`` header:

.. code::

    // in every source file
    #define PY_ARRAY_UNIQUE_SYMBOL my_uniqe_array_api
    #include "xtensor-python/pyarray.hpp"



