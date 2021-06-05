
****************
(CMake) Examples
****************

Basic example (from readme)
===========================

Consider the following C++ code:

:download:`main.cpp <examples/readme_example_1/main.cpp>`

.. literalinclude:: examples/readme_example_1/main.cpp
   :language: cpp

There are several options to build the module,
whereby we will CMake here with the following ``CMakeLists.txt``:

:download:`CMakeLists.txt <examples/readme_example_1/CMakeLists.txt>`

.. literalinclude:: examples/readme_example_1/CMakeLists.txt
   :language: cmake

Then we can test the module:

:download:`example.py <examples/readme_example_1/example.py>`

.. literalinclude:: examples/readme_example_1/example.py
   :language: cmake

.. note::

    Since we did not install the module,
    we should compile and run the example from the same folder.
    To install, please consult
    `this pybind11 / CMake example <https://github.com/pybind/cmake_example>`_.


Type restriction with SFINAE
============================

.. seealso::

    `Medium post by Johan Mabille <https://medium.com/@johan.mabille/designing-language-bindings-with-xtensor-f32aa0f20db>`__
    This example covers "Option 4".

In this example we will design a module with a function that accepts an ``xt::xtensor`` as argument,
but in such a way that an ``xt::pyxtensor`` can be accepted in the Python module.
This is done by having a templated function

.. code-block:: cpp

    template <class T>
    void times_dimension(T& t);

As this might be a bit too permissive for your liking, we will show you how to limit the
scope to *xtensor* types, and allow other overloads using the principle of SFINAE
(Substitution Failure Is Not An Error).
In particular:

:download:`mymodule.hpp <examples/sfinae/mymodule.hpp>`

.. literalinclude:: examples/sfinae/mymodule.hpp
   :language: cpp

Consequently from C++, the interaction with the module's function is trivial

:download:`main.cpp <examples/sfinae/main.cpp>`

.. literalinclude:: examples/sfinae/main.cpp
   :language: cpp

For the Python module we just have to specify the template to be
``xt::pyarray`` or ``xt::pytensor``. E.g.

:download:`src/python.cpp <examples/sfinae/python.cpp>`

.. literalinclude:: examples/sfinae/python.cpp
   :language: cpp

We will again use CMake to compile, with the following ``CMakeLists.txt``:

:download:`CMakeLists.txt <examples/sfinae/CMakeLists.txt>`

.. literalinclude:: examples/sfinae/CMakeLists.txt
   :language: cmake

Then we can test the module:

:download:`example.py <examples/readme_example_1/example.py>`

.. literalinclude:: examples/readme_example_1/example.py
   :language: cmake

.. note::

    Since we did not install the module,
    we should compile and run the example from the same folder.
    To install, please consult
    `this pybind11 / CMake example <https://github.com/pybind/cmake_example>`_.


Fall-back cast
==============

The previous example showed you how to design your module to be flexible in accepting data.
From C++ we used ``xt::xarray<double>``,
whereas for the Python API we used ``xt::pyarray<double>`` to operate directly on the memory
of a NumPy array from Python (without copying the data).

Sometimes, you might not have the flexibility to design your module's methods
with template parameters.
This might occur when you want to ``override`` functions
(though it is recommended to use CRTP to still use templates).
In this case we can still bind the module in Python using *xtensor-python*,
however, we have to copy the data from a (NumPy) array.
This means that although the following signatures are quite different when used from C++,
as follows:

1.  *Constant reference*: read from the data, without copying it.

    .. code-block:: cpp

         void foo(const xt::xarray<double>& a);

2.  *Reference*: read from and/or write to the data, without copying it.

    .. code-block:: cpp

         void foo(xt::xarray<double>& a);

3.   *Copy*: copy the data.

     .. code-block:: cpp

         void foo(xt::xarray<double> a);

The Python will all cases result in a copy to a temporary variable
(though the last signature will lead to a copy to a temporary variable, and another copy to ``a``).
On the one hand, this is more costly than when using ``xt::pyarray`` and ``xt::pyxtensor``,
on the other hand, it means that all changes you make to a reference, are made to the temporary
copy, and are thus lost.

Still, it might be a convenient way to create Python bindings, using a minimal effort.
Consider this example:

:download:`main.cpp <examples/copy_cast/main.cpp>`

.. literalinclude:: examples/copy_cast/main.cpp
   :language: cpp
