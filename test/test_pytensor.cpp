/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include "test_common.hpp"

#include "xtensor-python/pytensor.hpp"

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
        }

        {
            SCOPED_TRACE("column_major constructor");
            column_major_result<container_type> cm;
            pytensor<int, 3> ca(cm.m_shape, layout::column_major);
            compare_shape(ca, cm);
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
            EXPECT_TRUE(std::equal(vec.cbegin(), vec.cend(), ra.data().cbegin()));
        }

        {
            SCOPED_TRACE("column_major valued constructor");
            column_major_result<container_type> cm;
            int value = 2;
            pytensor<int, 3> ca(cm.m_shape, value, layout::column_major);
            compare_shape(ca, cm);
            std::vector<int> vec(ca.size(), value);
            EXPECT_TRUE(std::equal(vec.cbegin(), vec.cend(), ca.data().cbegin()));
        }
    }

    TEST(pytensor, strided_valued_constructor)
    {
        central_major_result<container_type> cmr;
        int value = 2;
        pytensor<int, 3> cma(cmr.m_shape, cmr.m_strides, value);
        compare_shape(cma, cmr);
        std::vector<int> vec(cma.size(), value);
        EXPECT_TRUE(std::equal(vec.cbegin(), vec.cend(), cma.data().cbegin()));
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
            EXPECT_EQ(a.data(), b.data());
        }

        {
            SCOPED_TRACE("assignment operator");
            row_major_result<container_type> r;
            pytensor<int, 3> c(r.m_shape, 0);
            EXPECT_NE(a.data(), c.data());
            c = a;
            compare_shape(a, c);
            EXPECT_EQ(a.data(), c.data());
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
            EXPECT_EQ(a.data(), b.data());
        }

        {
            SCOPED_TRACE("move assignment");
            row_major_result<container_type> r;
            pytensor<int, 3> c(r.m_shape, 0);
            EXPECT_NE(a.data(), c.data());
            pytensor<int, 3> tmp(a);
            c = std::move(tmp);
            compare_shape(a, c);
            EXPECT_EQ(a.data(), c.data());
        }
    }

    TEST(pytensor, reshape)
    {
        pytensor<int, 3> a;
        test_reshape<pytensor<int, 3>, container_type>(a);
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
        test_iterator<pytensor<int, 3>, container_type>(a);
    }

    TEST(pytensor, zerod)
    {
        pytensor<int, 3> a;
        EXPECT_EQ(0, a());
    }
}
