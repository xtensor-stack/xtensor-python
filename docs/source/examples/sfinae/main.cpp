#include "mymodule.hpp"
#include <xtensor/io/xio.hpp>

int main()
{
    xt::xtensor<size_t, 2> a = xt::arange<size_t>(2 * 3).reshape({2, 3});
    mymodule::times_dimension(a);
    std::cout << a << std::endl;
    return 0;
}
