#include "mymodule.hpp"
#include <pybind11/pybind11.h>
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>

PYBIND11_MODULE(mymodule, m)
{
    xt::import_numpy();
    m.doc() = "Test module for xtensor python bindings";
    m.def("times_dimension", &mymodule::times_dimension<xt::pyarray<double>>);
}
