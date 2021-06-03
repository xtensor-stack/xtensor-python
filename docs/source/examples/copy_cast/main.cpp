#include <numeric>
#include <xtensor.hpp>
#include <pybind11/pybind11.h>
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>

double sum_of_sines(xt::pyarray<double>& m)
{
    auto xt::sum(xt::sin(m))();
}

double sum_of_cosines(const xt::xtensor<double>& m)
{
    auto xt::sum(xt::cos(m))();
}

PYBIND11_MODULE(mymodule, m)
{
    xt::import_numpy();
    m.doc() = "Test module for xtensor python bindings";
    m.def("sum_of_sines", sum_of_sines, "Sum the sines of the input values");
    m.def("sum_of_cosines", sum_of_cosines, "Sum the cosines of the input values");
}
