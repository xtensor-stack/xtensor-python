.. Copyright (c) 2016, Johan Mabille and Sylvain Corlay

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Arrays, Tensors and NumPy Views
===============================

``xtensor-python`` provides two container types wrapping numpy arrays: ``pyarray`` and ``pytensor``. They are the counterparts
to ``xarray`` and ``xtensor`` containers.

pyarray
-------

Like ``xarray``, ``pyarray`` has a dynamic shape. This means that you can reshape the numpy array on the C++ side and see this
change reflected on the python side. ``pyarray`` doesn't make a copy of the shape or the strides, but reads them each time it
is needed. Therefore, if a reference on a ``pyarray`` is kept in the C++ code and the corresponding numpy array is then reshaped
in the python code, this modification will reflect in the ``pyarray``.

pytensor
--------

Like ``xtensor``, ``pytensor`` has a static stack-allocated shape. This means that the shape of the numpy array is copied into
the shape of the ``pytensor`` upon creation. As a consequence, reshapes are not reflected across languages. However, this drawback
is offset by a more effective computation of shape and broadcast.

NumPy views
-----------

If you are trying to call a xtensor function with a NumPy view, you might read the following:
"passing container with wrong strides for static layout". This means, that the *static layout*
of the pytensor does not match the layout of the NumPy array (or view) that you are using to
call the function.

One fix for this is to choose the ``dynamic`` layout for your pyarray or pytensor:

.. code::

    void my_function(xt::pyarray<double, xt::layout_type::dynamic> input)

When choosing the dynamic layout, you can pass any NumPy container (with Fortran order, or
an arbitrary view).
Note that iterators on a pyarray with dynamic layout are generally slower!

