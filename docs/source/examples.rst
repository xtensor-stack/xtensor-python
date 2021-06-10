
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
whereby we will use *CMake* here with the following ``CMakeLists.txt``:

:download:`CMakeLists.txt <examples/readme_example_1/CMakeLists.txt>`

.. literalinclude:: examples/readme_example_1/CMakeLists.txt
   :language: cmake

.. tip::

    There is a potential pitfall here, centered around the fact that *CMake*
    has a 'new' *FindPython* and a 'classic' *FindPythonLibs*.
    We here use *FindPython* because of its ability to find the NumPy headers,
    that we need for *xtensor-python*.

    This has the consequence that when we want to force *CMake*
    to use a specific *Python* executable, we have to use something like

    .. code-block:: none

        cmake -Bbuild -DPython_EXECUTABLE=`which python`

    whereby it is crucial that one uses the correct case ``Python_EXECUTABLE``, as:

    .. code-block:: none

        Python_EXECUTABLE   <->   FindPython
        PYTHON_EXECUTABLE   <->   FindPythonLibs

    (remember that *CMake* is **case-sensitive**!).

    Now, since we use *FindPython* because of *xtensor-python* we also want *pybind11*
    to use *FindPython*
    (and not the classic *FindPythonLibs*,
    since we want to specify the *Python* executable only once).
    To this end we have to make sure to do things in the correct order, which is

    .. code-block:: cmake

        find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
        find_package(pybind11 REQUIRED CONFIG)

    (i.e. one finds *Python* **before** *pybind11*).
    See the `pybind11 documentation <https://pybind11.readthedocs.io/en/latest/cmake/index.html#new-findpython-mode>`_.

    In addition, be sure to use a quite recent *CMake* version,
    by starting your ``CMakeLists.txt`` for example with

    .. code-block:: cmake

        cmake_minimum_required(VERSION 3.18..3.20)

Then we can test the module:

:download:`example.py <examples/readme_example_1/example.py>`

.. literalinclude:: examples/readme_example_1/example.py
   :language: cmake

.. note::

    Since we did not install the module,
    we should compile and run the example from the same folder.
    To install, please consult
    `this *pybind11* / *CMake* example <https://github.com/pybind/cmake_example>`_.


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

We will again use *CMake* to compile, with the following ``CMakeLists.txt``:

:download:`CMakeLists.txt <examples/sfinae/CMakeLists.txt>`

.. literalinclude:: examples/sfinae/CMakeLists.txt
   :language: cmake

(see *CMake* tip above).

Then we can test the module:

:download:`example.py <examples/readme_example_1/example.py>`

.. literalinclude:: examples/readme_example_1/example.py
   :language: cmake

.. note::

    Since we did not install the module,
    we should compile and run the example from the same folder.
    To install, please consult
    `this *pybind11* / *CMake* example <https://github.com/pybind/cmake_example>`_.
    **Tip**: take care to modify that example with the correct *CMake* case ``Python_EXECUTABLE``.
