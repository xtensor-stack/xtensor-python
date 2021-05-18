#include <numeric>
#include <xtensor.hpp>
#include <pybind11/pybind11.h>
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>

double sum_of_sines(xt::pyarray<double>& m)
{
    auto sines = xt::sin(m);  // sines does not actually hold values.
    return std::accumulate(sines.begin(), sines.end(), 0.0);
}

PYBIND11_MODULE(mymodule, m)
{
    xt::import_numpy();
    m.doc() = "Test module for xtensor python bindings";
    m.def("sum_of_sines", sum_of_sines, "Sum the sines of the input values");
}
