#include <xtensor/xtensor.hpp>
#include <xtensor/xio.hpp>

namespace foobar {

template <class C>
struct is_tensor : std::false_type
{
};

template <class T, std::size_t N, xt::layout_type L, class Tag>
struct is_tensor<xt::xtensor<T, N, L, Tag>> : std::true_type
{
};

template <template<class> class C, class T>
using check_constraints = std::enable_if_t<C<T>::value, bool>;

template <class T, template <class> class C = is_tensor,
          check_constraints<C, T> = true>
void compute(T& t)
{
  using value_type = typename T::value_type;
  t *= (value_type)(2);
}

}
