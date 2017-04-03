/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "test_common.hpp"
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyvectorize.hpp"
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

namespace xt
{

    double f1(double a, double b)
    {
        return a + b;
    }

    using shape_type = std::vector<std::size_t>;

    TEST(pyvectorize, function)
    {
        auto vecf1 = pyvectorize(f1);
        shape_type shape = { 3, 2 };
        pyarray<double> a(shape, 1.5);
        pyarray<double> b(shape, 2.3);
        pyarray<double> c = vecf1(a, b);
        EXPECT_EQ(a(0, 0) + b(0, 0), c(0, 0));
    }

    TEST(pyvectorize, lambda)
    {
        auto vecf1 = pyvectorize([](double a, double b) { return a + b; });
        shape_type shape = { 3, 2 };
        pyarray<double> a(shape, 1.5);
        pyarray<double> b(shape, 2.3);
        pyarray<double> c = vecf1(a, b);
        EXPECT_EQ(a(0, 0) + b(0, 0), c(0, 0));
    }

    TEST(pyvectorize, complex)
    {
        using complex_t = std::complex<double>;
        shape_type shape = { 3, 2 };
        pyarray<complex_t> a(shape, complex_t(1.2, 2.5));
        auto f = [](const pyarray<complex_t>& t) {
            return std::make_tuple(pyvectorize([](complex_t x) { return std::abs(x); })(t),
                                   pyvectorize([](complex_t x) { return std::arg(x); })(t));
        };

        auto res = f(a);
        double expected_abs = std::abs(a(1, 1));
        double expected_arg = std::arg(a(1, 1));
        EXPECT_EQ(expected_abs, std::get<0>(res)(1, 1));
        EXPECT_EQ(expected_arg, std::get<1>(res)(1, 1));
    }

    TEST(pyvectorize, complex_pybind)
    {
        using complex_t = std::complex<double>;
        shape_type shape = { 3, 2 };
        pyarray<complex_t> a(shape, complex_t(1.2, 2.5));
        auto f = [](const pyarray<complex_t>& t) {
            return pybind11::make_tuple(pyvectorize([](complex_t x) { return std::abs(x); })(t),
                                        pyvectorize([](complex_t x) { return std::arg(x); })(t));
        };

        auto res = f(a);
        double expected_abs = std::abs(a(1, 1));
        double expected_arg = std::arg(a(1, 1));

        EXPECT_EQ(expected_abs, res[0].cast<pyarray<double>>()(1, 1));
        EXPECT_EQ(expected_arg, res[1].cast<pyarray<double>>()(1, 1));
    }
}
