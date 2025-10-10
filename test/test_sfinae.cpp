/***************************************************************************
* Copyright (c) Wolf Vollprecht, Johan Mabille and Sylvain Corlay          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <limits>

#include "gtest/gtest.h"
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyarray.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/containers/xtensor.hpp"

namespace xt
{
    template <class E, std::enable_if_t<!xt::has_fixed_rank_t<E>::value, int> = 0>
    inline bool sfinae_has_fixed_rank(E&&)
    {
        return false;
    }

    template <class E, std::enable_if_t<xt::has_fixed_rank_t<E>::value, int> = 0>
    inline bool sfinae_has_fixed_rank(E&&)
    {
        return true;
    }

    TEST(sfinae, fixed_rank)
    {
        xt::pyarray<size_t> a = {{9, 9, 9}, {9, 9, 9}};
        xt::pytensor<size_t, 1> b = {9, 9};
        xt::pytensor<size_t, 2> c = {{9, 9}, {9, 9}};

        EXPECT_TRUE(sfinae_has_fixed_rank(a) == false);
        EXPECT_TRUE(sfinae_has_fixed_rank(b) == true);
        EXPECT_TRUE(sfinae_has_fixed_rank(c) == true);
    }

    TEST(sfinae, get_rank)
    {
        xt::pytensor<double, 1> A = xt::zeros<double>({2});
        xt::pytensor<double, 2> B = xt::zeros<double>({2, 2});
        xt::pyarray<double> C = xt::zeros<double>({2, 2});

        EXPECT_TRUE(xt::get_rank<decltype(A)>::value == 1ul);
        EXPECT_TRUE(xt::get_rank<decltype(B)>::value == 2ul);
        EXPECT_TRUE(xt::get_rank<decltype(C)>::value == SIZE_MAX);
    }
}
