#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"

#include "numpy/arrayobject.h"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyvectorize.hpp"

using complex_t = std::complex<double>;

namespace py = pybind11;

PYBIND11_MODULE(benchmark_xtensor_python, m)
{
    if (_import_array() < 0)
    {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
    }

    m.doc() = "Benchmark module for xtensor python bindings";

    m.def(
        "sum_array",
        [](const xt::pyarray<double>& x)
        {
            double sum = 0;
            for (auto e : x)
            {
                sum += e;
            }
            return sum;
        }
    );

    m.def(
        "sum_tensor",
        [](const xt::pytensor<double, 1>& x)
        {
            double sum = 0;
            for (auto e : x)
            {
                sum += e;
            }
            return sum;
        }
    );

    m.def(
        "pybind_sum_array",
        [](const py::array_t<double>& x)
        {
            double sum = 0;
            size_t size = x.size();
            const double* data = x.data(0);
            for (size_t i = 0; i < size; ++i)
            {
                sum += data[i];
            }
            return sum;
        }
    );

    m.def(
        "rect_to_polar",
        [](const xt::pyarray<complex_t>& a)
        {
            return py::vectorize(
                [](complex_t x)
                {
                    return std::abs(x);
                }
            )(a);
        }
    );

    m.def(
        "pybind_rect_to_polar",
        [](py::array a)
        {
            if (py::isinstance<py::array_t<complex_t>>(a))
            {
                return py::vectorize(
                    [](complex_t x)
                    {
                        return std::abs(x);
                    }
                )(a);
            }
            else
            {
                throw py::type_error("rect_to_polar unhandled type");
            }
        }
    );
}
