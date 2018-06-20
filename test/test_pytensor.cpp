/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include "xtensor-python/pytensor.hpp"

#include "xtensor/xtensor.hpp"

#include "test_common.hpp"

namespace xt
{
    using container_type = std::array<npy_intp, 3>;

    TEST(pytensor, initializer_constructor)
    {
        pytensor<int, 3> t 
          {{{ 0,  1,  2}, 
            { 3,  4,  5}, 
            { 6,  7,  8}}, 
           {{ 9, 10, 11}, 
            {12, 13, 14}, 
            {15, 16, 17}}}; 
        EXPECT_EQ(t.dimension(), 3);
        EXPECT_EQ(t(0, 0, 1), 1);
        EXPECT_EQ(t.shape()[0], 2);
    }

    TEST(pytensor, shaped_constructor)
    {
        {
            SCOPED_TRACE("row_major constructor");
            row_major_result<container_type> rm;
            pytensor<int, 3> ra(rm.m_shape);
            compare_shape(ra, rm);
            EXPECT_EQ(layout_type::row_major, ra.layout());
        }

        {
            SCOPED_TRACE("column_major constructor");
            column_major_result<container_type> cm;
            pytensor<int, 3> ca(cm.m_shape, layout_type::column_major);
            compare_shape(ca, cm);
            EXPECT_EQ(layout_type::column_major, ca.layout());
        }
    }

    TEST(pytensor, strided_constructor)
    {
        central_major_result<container_type> cmr;
        pytensor<int, 3> cma(cmr.m_shape, cmr.m_strides);
        compare_shape(cma, cmr);
    }

    TEST(pytensor, valued_constructor)
    {
        {
            SCOPED_TRACE("row_major valued constructor");
            row_major_result<container_type> rm;
            int value = 2;
            pytensor<int, 3> ra(rm.m_shape, value);
            compare_shape(ra, rm);
            std::vector<int> vec(ra.size(), value);
            EXPECT_TRUE(std::equal(vec.cbegin(), vec.cend(), ra.storage().cbegin()));
        }

        {
            SCOPED_TRACE("column_major valued constructor");
            column_major_result<container_type> cm;
            int value = 2;
            pytensor<int, 3> ca(cm.m_shape, value, layout_type::column_major);
            compare_shape(ca, cm);
            std::vector<int> vec(ca.size(), value);
            EXPECT_TRUE(std::equal(vec.cbegin(), vec.cend(), ca.storage().cbegin()));
        }
    }

    TEST(pytensor, strided_valued_constructor)
    {
        central_major_result<container_type> cmr;
        int value = 2;
        pytensor<int, 3> cma(cmr.m_shape, cmr.m_strides, value);
        compare_shape(cma, cmr);
        std::vector<int> vec(cma.size(), value);
        EXPECT_TRUE(std::equal(vec.cbegin(), vec.cend(), cma.storage().cbegin()));
    }

    TEST(pytensor, copy_semantic)
    {
        central_major_result<container_type> res;
        int value = 2;
        pytensor<int, 3> a(res.m_shape, res.m_strides, value);

        {
            SCOPED_TRACE("copy constructor");
            pytensor<int, 3> b(a);
            compare_shape(a, b);
            EXPECT_EQ(a.storage(), b.storage());
            a.data()[0] += 1;
            EXPECT_NE(a.storage()[0], b.storage()[0]);
        }

        {
            SCOPED_TRACE("assignment operator");
            row_major_result<container_type> r;
            pytensor<int, 3> c(r.m_shape, 0);
            EXPECT_NE(a.data(), c.data());
            c = a;
            compare_shape(a, c);
            EXPECT_EQ(a.storage(), c.storage());
            a.data()[0] += 1;
            EXPECT_NE(a.storage()[0], c.storage()[0]);
        }
    }

    TEST(pytensor, move_semantic)
    {
        central_major_result<container_type> res;
        int value = 2;
        pytensor<int, 3> a(res.m_shape, res.m_strides, value);

        {
            SCOPED_TRACE("move constructor");
            pytensor<int, 3> tmp(a);
            pytensor<int, 3> b(std::move(tmp));
            compare_shape(a, b);
            EXPECT_EQ(a.storage(), b.storage());
        }

        {
            SCOPED_TRACE("move assignment");
            row_major_result<container_type> r;
            pytensor<int, 3> c(r.m_shape, 0);
            EXPECT_NE(a.storage(), c.storage());
            pytensor<int, 3> tmp(a);
            c = std::move(tmp);
            compare_shape(a, c);
            EXPECT_EQ(a.storage(), c.storage());
        }
    }

    TEST(pytensor, extended_constructor)
    {
        xt::xtensor<int, 2> a1 = { {1, 2}, {3, 4} };
        xt::xtensor<int, 2> a2 = { {1, 2}, {3, 4} };
        pytensor<int, 2> c = a1 + a2;
        EXPECT_EQ(c(0, 0), a1(0, 0) + a2(0, 0));
        EXPECT_EQ(c(0, 1), a1(0, 1) + a2(0, 1));
        EXPECT_EQ(c(1, 0), a1(1, 0) + a2(1, 0));
        EXPECT_EQ(c(1, 1), a1(1, 1) + a2(1, 1));
    }

    TEST(pytensor, resize)
    {
        pytensor<int, 3> a;
        test_resize<pytensor<int, 3>, container_type>(a);

        pytensor<int, 3> b = { { { 1, 2 },{ 3, 4 } } };
        a.resize(b.shape());
        EXPECT_EQ(a.shape(), b.shape());
    }

    TEST(pytensor, transpose)
    {
        pytensor<int, 3> a;
        test_transpose<pytensor<int, 3>, container_type>(a);
    }

    TEST(pytensor, access)
    {
        pytensor<int, 3> a;
        test_access<pytensor<int, 3>, container_type>(a);
    }

    TEST(pytensor, indexed_access)
    {
        pytensor<int, 3> a;
        test_indexed_access<pytensor<int, 3>, container_type>(a);
    }

    TEST(pytensor, broadcast_shape)
    {
        pytensor<int, 4> a;
        test_broadcast(a);
    }

    TEST(pytensor, iterator)
    {
        pytensor<int, 3> a;
        pytensor<int, 3> b;
        test_iterator<pytensor<int, 3>, pytensor<int, 3>, container_type>(a, b);

        pytensor<int, 3, layout_type::row_major> c;
        bool truthy = std::is_same<decltype(c.begin()), int*>::value;
        EXPECT_TRUE(truthy);
    }

    TEST(pytensor, zerod)
    {
        pytensor<int, 3> a;
        EXPECT_EQ(0, a());
    }

    TEST(pytensor, reshape)
    {
        pytensor<int, 2> a = {{1,2,3}, {4,5,6}};
        auto ptr = a.data();
        a.reshape({1, 6});
        EXPECT_EQ(ptr, a.data());
        EXPECT_THROW(a.reshape({6}), std::runtime_error);
    }
}
