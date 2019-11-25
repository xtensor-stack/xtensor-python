.. Copyright (c) 2016, Johan Mabille and Sylvain Corlay

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.


Build, test and benchmark
=========================

``xtensor-python`` build supports the following options:

- ``BUILD_TESTS``: enables the ``xtest`` and ``xbenchmark`` targets (see below).
- ``DOWNLOAD_GTEST``: downloads ``gtest`` and builds it locally instead of using a binary installation.
- ``GTEST_SRC_DIR``: indicates where to find the ``gtest`` sources instead of downloading them.

All these options are disabled by default. Enabling ``DOWNLOAD_GTEST`` or
setting ``GTEST_SRC_DIR`` enables ``BUILD_TESTS``.

If the ``BUILD_TESTS`` option is enabled, the following targets are available:

- xtest: builds an run the test suite.
- xbenchmark: builds and runs the benchmarks.

For instance, building the test suite of ``xtensor-python`` and downloading ``gtest`` automatically:

.. code::

    mkdir build
    cd build
    cmake -DDOWNLOAD_GTEST=ON ../
    make xtest

To run the benchmark:

.. code::

    make xbenchmark

To test the Python bindings:

.. code::

    cd ..
    pytest -s
