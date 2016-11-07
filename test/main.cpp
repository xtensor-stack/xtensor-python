#include <pybind11/pybind11.h>
#include "xtensor/xarray.hpp"
#include "xtensor-python/pyarray.hpp"

namespace py = pybind11;

int test0(xt::pyarray<double> &m)
{
    return m(0);
}

PYBIND11_PLUGIN(xtensor_python_test)
{
    py::module m("xtensor_python_test", "Test module for xtensor python bindings");

    m.def("test0", test0, "");

    return m.ptr();
}
