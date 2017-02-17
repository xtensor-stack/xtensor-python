#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
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

    m.def("pybind_sum_array", [](py::array_t<double> const& x) {
            double sum = 0;
            size_t size = x.size();
            const double* data = x.data(0);
            for(size_t i = 0; i < size; ++i)
                sum += data[i];
            return sum;
        }
    );

    return m.ptr();
}
