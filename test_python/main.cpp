/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <numeric>

#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pyvectorize.hpp"

namespace py = pybind11;
using complex_t = std::complex<double>;

// Examples

double example1(xt::pyarray<double>& m)
{
    return m(0);
}

xt::pyarray<double> example2(xt::pyarray<double>& m)
{
    return m + 2;
}

// Readme Examples

double readme_example1(xt::pyarray<double>& m)
{
    auto sines = xt::sin(m);
    return std::accumulate(sines.cbegin(), sines.cend(), 0.0);
}

double readme_example2(double i, double j)
{
    return std::sin(i) -  std::cos(j);
}

auto complex_overload(const xt::pyarray<std::complex<double>>& a)
{
    return a;
}
auto no_complex_overload(const xt::pyarray<double>& a)
{
    return a;
}

auto complex_overload_reg(const std::complex<double>& a)
{
    return a;
}

auto no_complex_overload_reg(const double& a)
{
    return a;
}

// Vectorize Examples

int add(int i, int j)
{
    return i + j;
}

template <class T> std::string typestring() { return "Unknown"; }
template <> std::string typestring<uint8_t>() { return "uint8"; }
template <> std::string typestring<int8_t>() { return "int8"; }
template <> std::string typestring<uint16_t>() { return "uint16"; }
template <> std::string typestring<int16_t>() { return "int16"; }
template <> std::string typestring<uint32_t>() { return "uint32"; }
template <> std::string typestring<int32_t>() { return "int32"; }
template <> std::string typestring<uint64_t>() { return "uint64"; }
template <> std::string typestring<int64_t>() { return "int64"; }

template <class T>
inline std::string int_overload(xt::pyarray<T>& m)
{
    return typestring<T>();
}

void dump_numpy_constant()
{
    std::cout << "NPY_BOOL = " << NPY_BOOL << std::endl;
    std::cout << "NPY_BYTE = " << NPY_BYTE << std::endl;
    std::cout << "NPY_UBYTE = " << NPY_UBYTE << std::endl;
    std::cout << "NPY_INT8 = " << NPY_INT8 << std::endl;
    std::cout << "NPY_UINT8 = " << NPY_UINT8 << std::endl;
    std::cout << "NPY_SHORT = " << NPY_SHORT << std::endl;
    std::cout << "NPY_USHORT = " << NPY_USHORT << std::endl;
    std::cout << "NPY_INT16 = " << NPY_INT16 << std::endl;
    std::cout << "NPY_UINT16 = " << NPY_UINT16 << std::endl;
    std::cout << "NPY_INT = " << NPY_INT << std::endl;
    std::cout << "NPY_UINT = " << NPY_UINT << std::endl;
    std::cout << "NPY_INT32 = " << NPY_INT32 << std::endl;
    std::cout << "NPY_UINT32 = " << NPY_UINT32 << std::endl;
    std::cout << "NPY_LONG = " << NPY_LONG << std::endl;
    std::cout << "NPY_ULONG = " << NPY_ULONG << std::endl;
    std::cout << "NPY_LONGLONG = " << NPY_LONGLONG << std::endl;
    std::cout << "NPY_ULONGLONG = " << NPY_ULONGLONG << std::endl;
    std::cout << "NPY_INT64 = " << NPY_INT64 << std::endl;
    std::cout << "NPY_UINT64 = " << NPY_UINT64 << std::endl;
}

struct A
{
    double a;
    int b;
    char c;
    std::array<double, 3> x;
};

struct B
{
    double a;
    int b;
};

xt::pyarray<A> dtype_to_python()
{
    A a1{123, 321, 'a', {1, 2, 3}};
    A a2{111, 222, 'x', {5, 5, 5}};

    return xt::pyarray<A>({a1, a2});
}

xt::pyarray<B> dtype_from_python(xt::pyarray<B>& b)
{
    if (b(0).a != 1 || b(0).b != 'p' || b(1).a != 123 || b(1).b != 'c')
    {
        throw std::runtime_error("FAIL");
    }

    b(0).a = 123.;
    b(0).b = 'w';
    return b;
}

void char_array(xt::pyarray<char[20]>& carr)
{
    if (strcmp(carr(2), "python"))
    {
        throw std::runtime_error("TEST FAILED!");
    }
    std::fill(&carr(2)[0], &carr(2)[0] + 20, 0);
    carr(2)[0] = 'c';
    carr(2)[1] = '+';
    carr(2)[2] = '+';
    carr(2)[3] = '\0';
}

PYBIND11_MODULE(xtensor_python_test, m)
{
    xt::import_numpy();

    m.doc() = "Test module for xtensor python bindings";

    m.def("example1", example1);
    m.def("example2", example2);

    m.def("complex_overload", no_complex_overload);
    m.def("complex_overload", complex_overload);
    m.def("complex_overload_reg", no_complex_overload_reg);
    m.def("complex_overload_reg", complex_overload_reg);

    m.def("readme_example1", readme_example1);
    m.def("readme_example2", xt::pyvectorize(readme_example2));

    m.def("vectorize_example1", xt::pyvectorize(add));

    m.def("rect_to_polar", xt::pyvectorize([](complex_t x) { return std::abs(x); }));

    m.def("compare_shapes", [](const xt::pyarray<double>& a, const xt::pyarray<double>& b) {
        return a.shape() == b.shape();
    });

    m.def("int_overload", int_overload<uint8_t>);
    m.def("int_overload", int_overload<int8_t>);
    m.def("int_overload", int_overload<uint16_t>);
    m.def("int_overload", int_overload<int16_t>);
    m.def("int_overload", int_overload<uint32_t>);
    m.def("int_overload", int_overload<int32_t>);
    m.def("int_overload", int_overload<uint64_t>);
    m.def("int_overload", int_overload<int64_t>);

    m.def("dump_numpy_constant", dump_numpy_constant);

    // Register additional dtypes
    PYBIND11_NUMPY_DTYPE(A, a, b, c, x);
    PYBIND11_NUMPY_DTYPE(B, a, b);

    m.def("dtype_to_python", dtype_to_python);
    m.def("dtype_from_python", dtype_from_python);
    m.def("char_array", char_array);
}
