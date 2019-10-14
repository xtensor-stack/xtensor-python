/***************************************************************************
* Copyright (c) Wolf Vollprecht, Johan Mabille and Sylvain Corlay          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

// Required to avoid the error "std does not have member copysign"
#include <cmath>

#include "gtest/gtest.h"

#include <pybind11/embed.h>

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"

namespace py = pybind11;

int main(int argc, char* argv[])
{
    // Initialize all the things (Python, numpy, gtest)
    py::scoped_interpreter guard{};
    xt::import_numpy();
    ::testing::InitGoogleTest(&argc, argv);

    // Run test suite
    int ret = RUN_ALL_TESTS();

    // Return test results
    return ret;
}

