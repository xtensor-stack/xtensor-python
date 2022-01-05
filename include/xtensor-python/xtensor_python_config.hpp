/***************************************************************************
* Copyright (c) Wolf Vollprecht, Johan Mabille and Sylvain Corlay          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_PYTHON_CONFIG_HPP
#define XTENSOR_PYTHON_CONFIG_HPP

#define XTENSOR_PYTHON_VERSION_MAJOR 0
#define XTENSOR_PYTHON_VERSION_MINOR 26
#define XTENSOR_PYTHON_VERSION_PATCH 0

#include "xtensor/xtensor_config.hpp"

#ifdef XTENSOR_PYTHON_ENABLE_DEBUG
#define XTENSOR_PYTHON_DEBUG(expr) XTENSOR_PYTHON_DEBUG_IMPL(expr, __FILE__, __LINE__)
#define XTENSOR_PYTHON_DEBUG_IMPL(expr, file, line)                         \
    if (!(expr))                                                            \
    {                                                                       \
        XTENSOR_THROW(std::runtime_error,                                   \
                      std::string(file) + ':' + std::to_string(line) +      \
                      ": assertion failed (" #expr ") \n\t");               \
    }
#else
#define XTENSOR_PYTHON_DEBUG(expr)
#endif

#endif
