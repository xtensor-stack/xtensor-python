/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <Python.h>

#include "gtest/gtest.h"

int main(int argc, char* argv[])
{
    // Initialize all the things (google-test and Python interpreter)
    Py_Initialize();
    ::testing::InitGoogleTest(&argc, argv);

    // Run test suite
    int ret = RUN_ALL_TESTS();

    // Closure of the Python interpreter
    Py_Finalize();
    return ret;
}

