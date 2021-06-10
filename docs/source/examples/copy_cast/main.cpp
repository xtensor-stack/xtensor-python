#include <numeric>
#include <xtensor.hpp>
#include <pybind11/pybind11.h>
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>

template <class T>
double sum_of_sines(T& m)
{
    auto sines = xt::sin(m);  // sines does not actually hold values.
    return std::accumulate(sines.begin(), sines.end(), 0.0);
}

// In the Python API this a reference to a temporary variable
double sum_of_cosines(const xt::xarray<double>& m)
{
    auto cosines = xt::cos(m);  // cosines does not actually hold values.
    return std::accumulate(cosines.begin(), cosines.end(), 0.0);
}

PYBIND11_MODULE(mymodule, m)
{
    xt::import_numpy();
    m.doc() = "Test module for xtensor python bindings";
    m.def("sum_of_sines", sum_of_sines<xt::pyarray<double>>, "Sum the sines of the input values");
    m.def("sum_of_cosines", sum_of_cosines, "Sum the cosines of the input values");
}
