#include <pybind11/pybind11.h>
#define FORCE_IMPORT_ARRAY
#include <xtensor-python/pytensor.hpp>

namespace foobar {

template <class T, std::size_t N, xt::layout_type L>
struct is_tensor<xt::pytensor<T, N, L>> : std::true_type
{
};

}

PYBIND11_MODULE(foobar, m)
{
    xt::import_numpy();

    m.def("compute", &foobar::compute<xt::pytensor<double, 2>>);
}
