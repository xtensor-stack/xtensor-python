
********
Examples
********

.. seealso::

    `Medium post by Johan Mabille <https://medium.com/@johan.mabille/designing-language-bindings-with-xtensor-f32aa0f20db>`__

Option 4: type restriction with SFINAE
======================================

In this example we will design a module with a function that accepts an ``xt::xtensor`` as argument, but in such a way that an ``xt::pyxtensor`` can be accepted in the Python module. We will achieve this with the principle of SFINAE (Substitution Failure Is Not An Error).

Our example has the following file-structure.

.. code-block:: none

    examples/sfinae
       |- src
       |   |- foobar.hpp
       |   |- python.cpp
       |- CMakeLists.txt
       |- main.cpp

The module has one function that accepts an ``xt::xtensor`` as argument:

:download:`src/foobar.hpp <examples/sfinae/src/foobar.hpp>`

.. literalinclude:: examples/sfinae/src/foobar.hpp
   :language: cpp

Consequently from C++, the interaction with the module's function is trivial

:download:`main.cpp <examples/sfinae/main.cpp>`

.. literalinclude:: examples/sfinae/main.cpp
   :language: cpp

From the Python module, we will add an additional overload to the template

:download:`src/python.cpp <examples/sfinae/src/python.cpp>`

.. literalinclude:: examples/sfinae/src/python.cpp
   :language: cpp

There are several options to build the module. Using cmake we can use the following ``CMakeLists.txt``:

:download:`CMakeLists.txt <examples/sfinae/CMakeLists.txt>`

.. literalinclude:: examples/sfinae/CMakeLists.txt
   :language: cmake
