.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Compiler workarounds
====================

This page tracks the workarounds for the various compiler issues that we
encountered in the development. This is mostly of interest for developers
interested in contributing to xtensor-python.

GCC and ``std::allocator<long long>``
-------------------------------------

GCC sometimes fails to automatically instantiate the ``std::allocator``
class template for the types ``long long`` and ``unsigned long long``.
Those allocators are thus explicitly instantiated in the dummy function
``void long_long_allocator()`` in the file ``py_container.hpp``.
