/***************************************************************************
* Copyright (c) Wolf Vollprecht, Johan Mabille and Sylvain Corlay          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include "xtensor-python/pyarray.hpp"



namespace xt
{
    namespace testing
    {
        class pyarray_traits: public ::testing::Test
        {
        protected:
        
            using dynamic_type = xt::pyarray<double>;
            using row_major_type = xt::pyarray<double, xt::layout_type::row_major>;
            using column_major_type = xt::pyarray<double, xt::layout_type::column_major>;

            dynamic_type d1 = {{0., 1.}, {0., 10.}, {0., 100.}};
            dynamic_type d2 = {{0., 2.}, {0., 20.}, {0., 200.}};

            row_major_type r1 = {{0., 1.}, {0., 10.}, {0., 100.}};
            row_major_type r2 = {{0., 2.}, {0., 20.}, {0., 200.}};

            column_major_type c1 = {{0., 1.}, {0., 10.}, {0., 100.}};
            column_major_type c2 = {{0., 2.}, {0., 20.}, {0., 200.}};

            template <class T>
            bool test_has_strides(T const&)
            {
                return xt::has_strides<T>::value;
            }

            template <class T>
            xt::layout_type test_result_layout(T const& a1, T const& a2)
            {
                auto tmp1 = pow(sin((a2 - a1) / 2.), 2.);
                auto tmp2 = cos(a1);
                return (tmp1 + tmp2).layout();
            }

            template <class T>
            bool test_linear_assign(T const& a1, T const& a2)
            {
                auto tmp1 = pow(sin((a2 - a1) / 2.), 2.);
                auto tmp2 = cos(a1);
                T res = tmp1 + tmp2;
                return xt::xassign_traits<T, decltype(tmp1 + tmp2)>::linear_assign(res, tmp1 + tmp2, true);
            }

            template <class T>
            bool test_static_simd_linear_assign(T const& a1, T const& a2)
            {
                auto tmp1 = pow(sin((a2 - a1) / 2.), 2.);
                auto tmp2 = cos(a1);
                return xt::xassign_traits<T, decltype(tmp2)>::simd_linear_assign();
            }

            template <class T>
            bool test_dynamic_simd_linear_assign(T const& a1, T const& a2)
            {
                auto tmp1 = pow(sin((a2 - a1) / 2.), 2.);
                auto tmp2 = cos(a1);
                return xt::xassign_traits<T, decltype(tmp2)>::simd_linear_assign(a1, tmp2);
            }

            template <class T>
            bool test_linear_static_layout(T const& a1, T const& a2)
            {
                auto tmp1 = pow(sin((a2 - a1) / 2.), 2.);
                auto tmp2 = cos(a1);
                return xt::detail::linear_static_layout<decltype(tmp1), decltype(tmp2)>();
            }

            template <class T>
            bool test_contiguous_layout(T const& a1, T const& a2)
            {
                auto tmp1 = pow(sin((a2 - a1) / 2.), 2.);
                auto tmp2 = cos(a1);
                return decltype(tmp1)::contiguous_layout && decltype(tmp2)::contiguous_layout;
            }
        };

        TEST_F(pyarray_traits, result_layout)
        {
            EXPECT_TRUE(d1.layout() == layout_type::row_major);
            EXPECT_TRUE(test_result_layout(d1, d2) == layout_type::row_major);

            EXPECT_TRUE(r1.layout() == layout_type::row_major);
            EXPECT_TRUE(test_result_layout(r1, r2) == layout_type::row_major);

            EXPECT_TRUE(c1.layout() == layout_type::column_major);
            EXPECT_TRUE(test_result_layout(c1, c2) == layout_type::column_major);
        }

        TEST_F(pyarray_traits, has_strides)
        {
            EXPECT_TRUE(test_has_strides(d1));
            EXPECT_TRUE(test_has_strides(r1));
            EXPECT_TRUE(test_has_strides(c1));
        }

        TEST_F(pyarray_traits, has_linear_assign)
        {
            EXPECT_TRUE(d2.has_linear_assign(d1.strides()));
            EXPECT_TRUE(r2.has_linear_assign(r1.strides()));
            EXPECT_TRUE(c2.has_linear_assign(c1.strides()));
        }

        TEST_F(pyarray_traits, linear_assign)
        {
            EXPECT_TRUE(test_linear_assign(d1, d2));
            EXPECT_TRUE(test_linear_assign(r1, r2));
            EXPECT_TRUE(test_linear_assign(c1, c2));
        }

        TEST_F(pyarray_traits, static_simd_linear_assign)
        {
#ifdef XTENSOR_USE_XSIMD
            EXPECT_FALSE(test_static_simd_linear_assign(d1, d2));
            EXPECT_TRUE(test_static_simd_linear_assign(r1, r2));
            EXPECT_TRUE(test_static_simd_linear_assign(c1, c2));
#else
            EXPECT_FALSE(test_static_simd_linear_assign(d1, d2));
            EXPECT_FALSE(test_static_simd_linear_assign(r1, r2));
            EXPECT_FALSE(test_static_simd_linear_assign(c1, c2));
#endif
        }

        TEST_F(pyarray_traits, dynamic_simd_linear_assign)
        {
#ifdef XTENSOR_USE_XSIMD
            EXPECT_TRUE(test_dynamic_simd_linear_assign(d1, d2));
            EXPECT_TRUE(test_dynamic_simd_linear_assign(r1, r2));
            EXPECT_TRUE(test_dynamic_simd_linear_assign(c1, c2));
#else
            EXPECT_FALSE(test_dynamic_simd_linear_assign(d1, d2));
            EXPECT_FALSE(test_dynamic_simd_linear_assign(r1, r2));
            EXPECT_FALSE(test_dynamic_simd_linear_assign(c1, c2));
#endif
        }

        TEST_F(pyarray_traits, linear_static_layout)
        {
            EXPECT_FALSE(test_linear_static_layout(d1, d2));
            EXPECT_TRUE(test_linear_static_layout(r1, r2));
            EXPECT_TRUE(test_linear_static_layout(c1, c2));
        }

        TEST_F(pyarray_traits, contiguous_layout)
        {
            EXPECT_FALSE(test_contiguous_layout(d1, d2));
            EXPECT_TRUE(test_contiguous_layout(r1, r2));
            EXPECT_TRUE(test_contiguous_layout(c1, c2));
        }
    }
}
