/**
 * @file
 * @copyright Copyright 2020. Tom de Geus. All rights reserved.
 * @license This project is released under the GNU Public License (MIT).
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pyarray.hpp>
#include <xtensor-python/pytensor.hpp>
#include <xtensor.hpp>

namespace py = pybind11;

/**
 * Overrides the `__name__` of a module.
 * Classes defined by pybind11 use the `__name__` of the module as of the time they are defined,
 * which affects the `__repr__` of the class type objects.
 */
class ScopedModuleNameOverride {
public:
    explicit ScopedModuleNameOverride(py::module m, std::string name) : module_(std::move(m))
    {
        original_name_ = module_.attr("__name__");
        module_.attr("__name__") = name;
    }
    ~ScopedModuleNameOverride()
    {
        module_.attr("__name__") = original_name_;
    }

private:
    py::module module_;
    py::object original_name_;
};

PYBIND11_MODULE(_xt, m)
{
    // Ensure members to display as `xt.X` (not `xt._xt.X`)
    ScopedModuleNameOverride name_override(m, "xt");

    xt::import_numpy();

    m.doc() = "Python bindings of xtensor";

    m.def("mean", [](const xt::pyarray<double>& a) -> xt::pyarray<double> { return xt::mean(a); });

    m.def("flip", [](const xt::pyarray<double>& a, ptrdiff_t axis) -> xt::pyarray<double> { return xt::flip(a, axis); });

    m.def("cos", [](const xt::pyarray<double>& a) -> xt::pyarray<double> { return xt::cos(a); });

    m.def("isin", [](const xt::pyarray<int>& a, const xt::pyarray<int>& b) -> xt::pyarray<bool> { return xt::isin(a, b); });
    m.def("in1d", [](const xt::pyarray<int>& a, const xt::pyarray<int>& b) -> xt::pyarray<bool> { return xt::in1d(a, b); });


}
