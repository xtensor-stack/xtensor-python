#include <pybind11/pybind11.h>
#include "xtensor/xarray.hpp"
#include "xtensor-python/pyarray.hpp"

namespace py = pybind11;

double test0(xt::pyarray<double> &m)
{
    return m(0);
}

xt::pyarray<double> test1(xt::pyarray<double> &m) {
    return m + 2;
}

PYBIND11_PLUGIN(xtensor_python_test)
{
    py::module m("xtensor_python_test", "Test module for xtensor python bindings");

    m.def("test0", test0, "");
    m.def("test1", test1, "");

    return m.ptr();
}
