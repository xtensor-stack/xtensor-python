/***************************************************************************
* Copyright (c) Wolf Vollprecht, Johan Mabille and Sylvain Corlay          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <numeric>

#include "xtensor/core/xmath.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/containers/xfixed.hpp"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyvectorize.hpp"
#include "xtensor/containers/xadapt.hpp"
#include "xtensor/views/xstrided_view.hpp"

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

xt::xarray<int> example3_xarray(const xt::xarray<int>& m)
{
    return xt::transpose(m) + 2;
}

xt::xarray<int, xt::layout_type::column_major> example3_xarray_colmajor(
    const xt::xarray<int, xt::layout_type::column_major>& m)
{
    return xt::transpose(m) + 2;
}

xt::xtensor<int, 3> example3_xtensor3(const xt::xtensor<int, 3>& m)
{
    return xt::transpose(m) + 2;
}

xt::xtensor<int, 2> example3_xtensor2(const xt::xtensor<int, 2>& m)
{
    return xt::transpose(m) + 2;
}

xt::xtensor<int, 2, xt::layout_type::column_major> example3_xtensor2_colmajor(
    const xt::xtensor<int, 2, xt::layout_type::column_major>& m)
{
    return xt::transpose(m) + 2;
}

xt::xtensor_fixed<int, xt::xshape<4, 3, 2>> example3_xfixed3(const xt::xtensor_fixed<int, xt::xshape<2, 3, 4>>& m)
{
    return xt::transpose(m) + 2;
}

xt::xtensor_fixed<int, xt::xshape<3, 2>> example3_xfixed2(const xt::xtensor_fixed<int, xt::xshape<2, 3>>& m)
{
    return xt::transpose(m) + 2;
}

xt::xtensor_fixed<int, xt::xshape<3, 2>, xt::layout_type::column_major> example3_xfixed2_colmajor(
    const xt::xtensor_fixed<int, xt::xshape<2, 3>, xt::layout_type::column_major>& m)
{
    return xt::transpose(m) + 2;
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
//
// Operator examples
//
xt::pyarray<double> array_addition(const xt::pyarray<double>& m, const xt::pyarray<double>& n)
{
    return m + n;
}

xt::pyarray<double> array_subtraction(xt::pyarray<double>& m, xt::pyarray<double>& n)
{
    return m - n;
}

xt::pyarray<double> array_multiplication(xt::pyarray<double>& m, xt::pyarray<double>& n)
{
    return m * n;
}

xt::pyarray<double> array_division(xt::pyarray<double>& m, xt::pyarray<double>& n)
{
    return m / n;
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

class C
{
public:
    using array_type = xt::xarray<double, xt::layout_type::row_major>;
    C() : m_array{0, 0, 0, 0} {}
    array_type & array() { return m_array; }
private:
    array_type m_array;
};

struct test_native_casters
{
    using array_type = xt::xarray<double>;
    array_type a = xt::ones<double>({50, 50});

    const auto & get_array()
    {
        return a;
    }

    auto get_strided_view()
    {
        return xt::strided_view(a, {xt::range(0, 1), xt::range(0, 3, 2)});
    }

    auto get_array_adapter()
    {
        using shape_type = std::vector<size_t>;
        shape_type shape = {2, 2};
        shape_type stride = {3, 2};
        return xt::adapt(a.data(), 4, xt::no_ownership(), shape, stride);
    }

    auto get_tensor_adapter()
    {
        using shape_type = std::array<size_t, 2>;
        shape_type shape = {2, 2};
        shape_type stride = {3, 2};
        return xt::adapt(a.data(), 4, xt::no_ownership(), shape, stride);
    }

    auto get_owning_array_adapter()
    {
        size_t size = 100;
        int * data = new int[size];
        std::fill(data, data + size, 1);

        using shape_type = std::vector<size_t>;
        shape_type shape = {size};
        return xt::adapt(std::move(data), size, xt::acquire_ownership(), shape);
    }
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

void row_major_tensor(xt::pytensor<double, 3, xt::layout_type::row_major>& arg)
{
    if (!std::is_same<decltype(arg.begin()), double*>::value)
    {
        throw std::runtime_error("TEST FAILED");
    }
}

void col_major_array(xt::pyarray<double, xt::layout_type::column_major>& arg)
{
    if (!std::is_same<decltype(arg.template begin<xt::layout_type::column_major>()), double*>::value)
    {
        throw std::runtime_error("TEST FAILED");
    }
}

xt::pytensor<int, 0> xscalar(const xt::pytensor<int, 1>& arg)
{
    return xt::sum(arg);
}

template <class T>
using ndarray = xt::pyarray<T, xt::layout_type::row_major>;

void test_rm(ndarray<int>const& x)
{
    ndarray<int> y = x;
    ndarray<int> z = xt::zeros<int>({10});
}

PYBIND11_MODULE(xtensor_python_test, m)
{
    xt::import_numpy();

    m.doc() = "Test module for xtensor python bindings";

    m.def("example1", example1);
    m.def("example2", example2);
    m.def("example3_xarray", example3_xarray);
    m.def("example3_xarray_colmajor", example3_xarray_colmajor);
    m.def("example3_xtensor3", example3_xtensor3);
    m.def("example3_xtensor2", example3_xtensor2);
    m.def("example3_xtensor2_colmajor", example3_xtensor2_colmajor);
    m.def("example3_xfixed3", example3_xfixed3);
    m.def("example3_xfixed2", example3_xfixed2);
    m.def("example3_xfixed2_colmajor", example3_xfixed2_colmajor);

    m.def("complex_overload", no_complex_overload);
    m.def("complex_overload", complex_overload);
    m.def("complex_overload_reg", no_complex_overload_reg);
    m.def("complex_overload_reg", complex_overload_reg);

    m.def("readme_example1", readme_example1);
    m.def("readme_example2", xt::pyvectorize(readme_example2));

    m.def("array_addition", array_addition);
    m.def("array_subtraction", array_subtraction);
    m.def("array_multiplication", array_multiplication);
    m.def("array_division", array_division);

    m.def("vectorize_example1", xt::pyvectorize(add));

    m.def("rect_to_polar", xt::pyvectorize([](complex_t x) { return std::abs(x); }));

    m.def("compare_shapes", [](const xt::pyarray<double>& a, const xt::pyarray<double>& b) {
        return a.shape() == b.shape();
    });

    m.def("test_rm", test_rm);

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

    m.def("col_major_array", col_major_array);
    m.def("row_major_tensor", row_major_tensor);

    m.def("xscalar", xscalar);

    py::class_<C>(m, "C")
        .def(py::init<>())
        .def_property_readonly(
            "copy",
            [](C & self) { return self.array(); }
        )
        .def_property_readonly(
            "ref",
            [](C & self) -> C::array_type & { return self.array(); }
        )
    ;

    m.def("simple_array", [](xt::pyarray<int>) { return 1; } );
    m.def("simple_tensor", [](xt::pytensor<int, 1>) { return 2; } );

    m.def("diff_shape_overload", [](xt::pytensor<int, 1> a) { return 1; });
    m.def("diff_shape_overload", [](xt::pytensor<int, 2> a) { return 2; });

    py::class_<test_native_casters>(m, "test_native_casters")
            .def(py::init<>())
            .def("get_array", &test_native_casters::get_array, py::return_value_policy::reference_internal) // memory managed by the class instance
            .def("get_strided_view", &test_native_casters::get_strided_view, py::keep_alive<0, 1>())        // keep_alive<0, 1>() => do not free "self" before the returned view
            .def("get_array_adapter", &test_native_casters::get_array_adapter, py::keep_alive<0, 1>())      // keep_alive<0, 1>() => do not free "self" before the returned adapter
            .def("get_tensor_adapter", &test_native_casters::get_tensor_adapter, py::keep_alive<0, 1>())    // keep_alive<0, 1>() => do not free "self" before the returned adapter
            .def("get_owning_array_adapter", &test_native_casters::get_owning_array_adapter)                // auto memory management as the adapter owns its memory
            .def("view_keep_alive_member_function", [](test_native_casters & self, xt::pyarray<double> & a) // keep_alive<0, 2>() => do not free second parameter before the returned view
                    {return xt::reshape_view(a, {a.size(), });},
                    py::keep_alive<0, 2>());
}
