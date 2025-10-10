#include <xtensor/containers/xtensor.hpp>

namespace mymodule {

template <class T>
struct is_std_vector
{
    static const bool value = false;
};

template <class T>
struct is_std_vector<std::vector<T> >
{
    static const bool value = true;
};

// any xtensor object
template <class T, std::enable_if_t<xt::is_xexpression<T>::value, bool> = true>
void times_dimension(T& t)
{
    using value_type = typename T::value_type;
    t *= (value_type)(t.dimension());
}

// an std::vector
template <class T, std::enable_if_t<is_std_vector<T>::value, bool> = true>
void times_dimension(T& t)
{
    // do nothing
}

}
