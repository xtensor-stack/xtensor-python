.. Copyright (c) 2016, Johan Mabille and Sylvain Corlay

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Usage
=====

Basic Usage
-----------

Example 1: Use an algorithm of the C++ library on a numpy array inplace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**C++ code**

.. code::

    #include <numeric>                        // Standard library import for std::accumulate
    #include "pybind11/pybind11.h"            // Pybind11 import to define Python bindings
    #include "xtensor/xmath.hpp"              // xtensor import for the C++ universal functions
    #include "xtensor-python/pyarray.hpp"     // Numpy bindings

    double sum_of_sines(xt::pyarray<double> &m)
    {
        auto sines = xt::sin(m);  // sines does not actually hold any value, which are only computed upon access
        return std::accumulate(sines.begin(), sines.end(), 0.0);
    }

    PYBIND11_PLUGIN(xtensor_python_test)
    {
        pybind11::module m("xtensor_python_test", "Test module for xtensor python bindings");

        m.def("sum_of_sines", sum_of_sines, "Computes the sum of the sines of the values of the input array");

        return m.ptr();
    }

**Python code:**

.. code::

    Python Code

    import numpy as np
    import xtensor_python_test as xt

    a = np.arange(15).reshape(3, 5)
    s = xt.sum_of_sines(v)
    s

**Outputs**

.. code::

    1.2853996391883833


Example 2: Create a universal function from a C++ scalar function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**C++ code**

.. code::

    #include "pybind11/pybind11.h"
    #include "xtensor-python/pyvectorize.hpp"
    #include <numeric>
    #include <cmath>

    namespace py = pybind11;

    double scalar_func(double i, double j)
    {
        return std::sin(i) - std::cos(j);
    }

    PYBIND11_PLUGIN(xtensor_python_test)
    {
        py::module m("xtensor_python_test", "Test module for xtensor python bindings");

        m.def("vectorized_func", xt::pyvectorize(scalar_func), "");

        return m.ptr();
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


Getting started with xtensor-cookiecutter
-----------------------------------------

`xtensor-cookiecutter`_ helps extension authors create Python extension modules making use of xtensor.

It takes care of the initial work of generating a project skeleton with

- A complete ``setup.py`` compiling the extension module
- A few examples included in the resulting project including

    - A universal function defined from C++
    - A function making use of an algorithm from the STL on a numpy array
    - Unit tests
    - The generation of the HTML documentation with sphinx

Usage
^^^^^

Install cookiecutter_

.. code::

    pip install cookiecutter

After installing cookiecutter, use the xtensor-cookiecutter_:

.. code::

    cookiecutter https://github.com/QuantStack/xtensor-cookiecutter.git

As xtensor-cookiecutter runs, you will be asked for basic information about
your custom extension project. You will be prompted for the following
information:

- ``author_name``: your name or the name of your organization,
- ``author_email`` : your project's contact email,
- ``github_project_name``: name of the GitHub repository for your project,
- ``github_organization_name``: name of the GithHub organization for your project,
- ``python_package_name``: name of the Python package created by your extension,
- ``cpp_namespace``: name for the cpp namespace holding the implementation of your extension,
- ``project_short_description``: a short description for your project.
  
This will produce a directory containing all the required content for a minimal extension
project making use of xtensor with all the required boilerplate for package management,
together with a few basic examples.

.. _xtensor-cookicutter: https://github.com/QuantStack/xtensor-cookiecutter
.. _cookiecutter: https://github.com/audreyr/cookiecutter
