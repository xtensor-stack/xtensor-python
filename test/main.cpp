#include "pybind11/pybind11.h"
#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pyvectorize.hpp"
#include <iostream>
#include <numeric>

namespace py = pybind11;

// Examples

double example1(xt::pyarray<double> &m)
{
    return m(0);
}

xt::pyarray<double> example2(xt::pyarray<double> &m)
{
    return m + 2;
}

// Readme Examples

double readme_example1(xt::pyarray<double> &m)
{
    auto sines = xt::sin(m);
    return std::accumulate(sines.begin(), sines.end(), 0.0);
}

double readme_example2(double i, double j)
{
    return std::sin(i) -  std::cos(j);
}

// Vectorize Examples

int add(int i, int j)
{
    return i + j;
}

PYBIND11_PLUGIN(xtensor_python_test)
{
    py::module m("xtensor_python_test", "Test module for xtensor python bindings");

    m.def("example1", example1, "");
    m.def("example2", example2, "");

    m.def("readme_example1", readme_example1, "");
    m.def("readme_example2", xt::pyvectorize(readme_example2), "");

    m.def("vectorize_example1", xt::pyvectorize(add), "");

    return m.ptr();
}
