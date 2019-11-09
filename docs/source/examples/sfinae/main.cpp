#include "src/foobar.hpp"

int main()
{
  xt::xtensor<size_t,2> a = xt::arange<size_t>(2 * 3).reshape({2, 3});
  foobar::compute(a);
  std::cout << a << std::endl;
  return 0;
}
