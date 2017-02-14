#include "pybind11/pybind11.h"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor-python/pyarray.hpp"

#include <complex>

namespace py = pybind11;

PYBIND11_PLUGIN(xtensor_python_benchmark)
{
    py::module m("xtensor_python_benchmark", "Benchmark module for xtensor python bindings");

    m.def("sum_array", [](xt::pyarray<double> const& x) {
            double sum = 0;
            for(auto e : x)
                sum += e;
            return sum;
        }
    );

    return m.ptr();
}
