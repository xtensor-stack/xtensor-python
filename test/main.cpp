/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

// Required to avoid the error "std does not have memeber copysign"
#include <cmath>
#include <Python.h>

#include "pybind11/numpy.h"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"

#include "gtest/gtest.h"
#include <iostream>

int main(int argc, char* argv[])
{
    // Initialize all the things (google-test and Python interpreter)
    Py_Initialize();
    xt::import_numpy();
    ::testing::InitGoogleTest(&argc, argv);

    // Run test suite
    int ret = RUN_ALL_TESTS();

    // Closure of the Python interpreter
    Py_Finalize();
    return ret;
}

