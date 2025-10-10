.. Copyright (c) 2016, Johan Mabille and Sylvain Corlay

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Basic Usage
===========

Example 1: Use an algorithm of the C++ library on a numpy array inplace
-----------------------------------------------------------------------

**C++ code**

.. code::

    #include <numeric>                        // Standard library import for std::accumulate
    #include "pybind11/pybind11.h"            // Pybind11 import to define Python bindings
    #include "xtensor/core/xmath.hpp"              // xtensor import for the C++ universal functions
    #define FORCE_IMPORT_ARRAY                // numpy C api loading
    #include "xtensor-python/pyarray.hpp"     // Numpy bindings

    double sum_of_sines(xt::pyarray<double>& m)
    {
        auto sines = xt::sin(m);  // sines does not actually hold values.
        return std::accumulate(sines.cbegin(), sines.cend(), 0.0);
    }

    PYBIND11_MODULE(xtensor_python_test, m)
    {
        xt::import_numpy();
        m.doc() = "Test module for xtensor python bindings";

        m.def("sum_of_sines", sum_of_sines, "Sum the sines of the input values");
    }

**Python code:**

.. code::

    import numpy as np
    import xtensor_python_test as xt

    a = np.arange(15).reshape(3, 5)
    s = xt.sum_of_sines(v)
    s

**Outputs**

.. code::

    1.2853996391883833


Example 2: Create a numpy-style universal function from a C++ scalar function
-----------------------------------------------------------------------------

**C++ code**

.. code::

    #include "pybind11/pybind11.h"
    #define FORCE_IMPORT_ARRAY
    #include "xtensor-python/pyvectorize.hpp"
    #include <numeric>
    #include <cmath>

    namespace py = pybind11;

    double scalar_func(double i, double j)
    {
        return std::sin(i) - std::cos(j);
    }

    PYBIND11_MODULE(xtensor_python_test, m)
    {
        xt::import_numpy();
        m.doc() = "Test module for xtensor python bindings";

        m.def("vectorized_func", xt::pyvectorize(scalar_func), "");
    }

**Python code:**

.. code::

    import numpy as np
    import xtensor_python_test as xt

    x = np.arange(15).reshape(3, 5)
    y = [1, 2, 3, 4, 5]
    z = xt.vectorized_func(x, y)
    z

**Outputs**

.. code::

    [[-0.540302,  1.257618,  1.89929 ,  0.794764, -1.040465],
     [-1.499227,  0.136731,  1.646979,  1.643002,  0.128456],
     [-1.084323, -0.583843,  0.45342 ,  1.073811,  0.706945]]

