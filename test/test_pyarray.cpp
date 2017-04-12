/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "test_common.hpp"
#include "xtensor-python/pyarray.hpp"

namespace xt
{
    using container_type = std::vector<npy_intp>;

    TEST(pyarray, initializer_constructor)
    {
        pyarray<int> t 
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

    TEST(pyarray, shaped_constructor)
    {
        {
            SCOPED_TRACE("row_major constructor");
            row_major_result<> rm;
            pyarray<int> ra(rm.m_shape);
            compare_shape(ra, rm);
        }
        
        {
            SCOPED_TRACE("column_major constructor");
            column_major_result<> cm;
            pyarray<int> ca(cm.m_shape, layout::column_major);
            compare_shape(ca, cm);
        }
    }

    TEST(pyarray, strided_constructor)
    {
        central_major_result<> cmr;
        pyarray<int> cma(cmr.m_shape, cmr.m_strides);
        compare_shape(cma, cmr);
    }

    TEST(pyarray, valued_constructor)
    {
        {
            SCOPED_TRACE("row_major valued constructor");
            row_major_result<> rm;
            int value = 2;
            pyarray<int> ra(rm.m_shape, value);
            compare_shape(ra, rm);
            std::vector<int> vec(ra.size(), value);
            EXPECT_TRUE(std::equal(vec.cbegin(), vec.cend(), ra.data().cbegin()));
        }

        {
            SCOPED_TRACE("column_major valued constructor");
            column_major_result<> cm;
            int value = 2;
            pyarray<int> ca(cm.m_shape, value, layout::column_major);
            compare_shape(ca, cm);
            std::vector<int> vec(ca.size(), value);
            EXPECT_TRUE(std::equal(vec.cbegin(), vec.cend(), ca.data().cbegin()));
        }
    }

    TEST(pyarray, strided_valued_constructor)
    {
        central_major_result<> cmr;
        int value = 2;
        pyarray<int> cma(cmr.m_shape, cmr.m_strides, value);
        compare_shape(cma, cmr);
        std::vector<int> vec(cma.size(), value);
        EXPECT_TRUE(std::equal(vec.cbegin(), vec.cend(), cma.data().cbegin()));
    }

    TEST(pyarray, copy_semantic)
    {
        central_major_result<> res;
        int value = 2;
        pyarray<int> a(res.m_shape, res.m_strides, value);
        
        {
            SCOPED_TRACE("copy constructor");
            pyarray<int> b(a);
            compare_shape(a, b);
            EXPECT_EQ(a.data(), b.data());
            a.data()[0] += 1;
            EXPECT_NE(a.data()[0], b.data()[0]);
        }

        {
            SCOPED_TRACE("assignment operator");
            row_major_result<> r;
            pyarray<int> c(r.m_shape, 0);
            EXPECT_NE(a.data(), c.data());
            c = a;
            compare_shape(a, c);
            EXPECT_EQ(a.data(), c.data());
            a.data()[0] += 1;
            EXPECT_NE(a.data()[0], c.data()[0]);
        }
    }

    TEST(pyarray, move_semantic)
    {
        central_major_result<> res;
        int value = 2;
        pyarray<int> a(res.m_shape, res.m_strides, value);

        {
            SCOPED_TRACE("move constructor");
            pyarray<int> tmp(a);
            pyarray<int> b(std::move(tmp));
            compare_shape(a, b);
            EXPECT_EQ(a.data(), b.data());
        }

        {
            SCOPED_TRACE("move assignment");
            row_major_result<> r;
            pyarray<int> c(r.m_shape, 0);
            EXPECT_NE(a.data(), c.data());
            pyarray<int> tmp(a);
            c = std::move(tmp);
            compare_shape(a, c);
            EXPECT_EQ(a.data(), c.data());
        }
    }

    TEST(pyarray, reshape)
    {
        pyarray<int> a;
        test_reshape(a);
    }
/*
    TEST(pyarray, transpose)
    {
        pyarray<int> a;
        test_transpose(a);
    }
*/
    TEST(pyarray, access)
    {
        pyarray<int> a;
        test_access(a);
    }

    TEST(pyarray, indexed_access)
    {
        pyarray<int> a;
        test_indexed_access(a);
    }

    TEST(pyarray, broadcast_shape)
    {
        pyarray<int> a;
        test_broadcast(a);
        test_broadcast2(a);
    }

    TEST(pyarray, iterator)
    {
        pyarray<int> a;
        test_iterator(a);
    }

    TEST(pyarray, initializer_list)
    {
        pyarray<int> a0(1);
        pyarray<int> a1({1, 2});
        pyarray<int> a2({{1, 2}, {2, 4}, {5, 6}});
        EXPECT_EQ(1, a0());
        EXPECT_EQ(2, a1(1));
        EXPECT_EQ(4, a2(1, 1));
    }
 
    TEST(pyarray, zerod)
    {
        pyarray<int> a;
        EXPECT_EQ(0, a());
    }
}
