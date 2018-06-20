/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include "xtensor-python/pyarray.hpp"

#include "xtensor/xarray.hpp"

#include "test_common.hpp"

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
            EXPECT_EQ(layout_type::row_major, ra.layout());
        }
        
        {
            SCOPED_TRACE("column_major constructor");
            column_major_result<> cm;
            pyarray<int> ca(cm.m_shape, layout_type::column_major);
            compare_shape(ca, cm);
            EXPECT_EQ(layout_type::column_major, ca.layout());
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
            EXPECT_TRUE(std::equal(vec.cbegin(), vec.cend(), ra.storage().cbegin()));
        }

        {
            SCOPED_TRACE("column_major valued constructor");
            column_major_result<> cm;
            int value = 2;
            pyarray<int> ca(cm.m_shape, value, layout_type::column_major);
            compare_shape(ca, cm);
            std::vector<int> vec(ca.size(), value);
            EXPECT_TRUE(std::equal(vec.cbegin(), vec.cend(), ca.storage().cbegin()));
        }
    }

    TEST(pyarray, strided_valued_constructor)
    {
        central_major_result<> cmr;
        int value = 2;
        pyarray<int> cma(cmr.m_shape, cmr.m_strides, value);
        compare_shape(cma, cmr);
        std::vector<int> vec(cma.size(), value);
        EXPECT_TRUE(std::equal(vec.cbegin(), vec.cend(), cma.storage().cbegin()));
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
            EXPECT_EQ(a.storage(), b.storage());
            a.data()[0] += 1;
            EXPECT_NE(a.storage()[0], b.storage()[0]);
        }

        {
            SCOPED_TRACE("assignment operator");
            row_major_result<> r;
            pyarray<int> c(r.m_shape, 0);
            EXPECT_NE(a.storage(), c.storage());
            c = a;
            compare_shape(a, c);
            EXPECT_EQ(a.storage(), c.storage());
            a.data()[0] += 1;
            EXPECT_NE(a.storage()[0], c.storage()[0]);
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
            EXPECT_EQ(a.storage(), b.storage());
        }

        {
            SCOPED_TRACE("move assignment");
            row_major_result<> r;
            pyarray<int> c(r.m_shape, 0);
            EXPECT_NE(a.storage(), c.storage());
            pyarray<int> tmp(a);
            c = std::move(tmp);
            compare_shape(a, c);
            EXPECT_EQ(a.storage(), c.storage());
        }
    }

    TEST(pyarray, extended_constructor)
    {
        xt::xarray<int> a1 = { { 1, 2 },{ 3, 4 } };
        xt::xarray<int> a2 = { { 1, 2 },{ 3, 4 } };
        pyarray<int> c = a1 + a2;
        EXPECT_EQ(c(0, 0), a1(0, 0) + a2(0, 0));
        EXPECT_EQ(c(0, 1), a1(0, 1) + a2(0, 1));
        EXPECT_EQ(c(1, 0), a1(1, 0) + a2(1, 0));
        EXPECT_EQ(c(1, 1), a1(1, 1) + a2(1, 1));
    }

    TEST(pyarray, resize)
    {
        pyarray<int> a;
        test_resize(a);

        pyarray<int> b = { {1, 2}, {3, 4} };
        a.resize(b.shape());
        EXPECT_EQ(a.shape(), b.shape());
    }

    TEST(pyarray, transpose)
    {
        pyarray<int> a;
        test_transpose(a);
    }

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
        pyarray<int> b;
        test_iterator(a, b);

        pyarray<int, layout_type::row_major> c;
        bool truthy = std::is_same<decltype(c.begin()), int*>::value;
        EXPECT_TRUE(truthy);
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

    TEST(pyarray, reshape)
    {
        pyarray<int> a = {{1,2,3}, {4,5,6}};
        auto ptr = a.data();
        a.reshape({1, 6});
        std::vector<std::size_t> sc1({1, 6});
        EXPECT_TRUE(std::equal(sc1.begin(), sc1.end(), a.shape().begin()) && a.shape().size() == 2);
        EXPECT_EQ(ptr, a.data());
        a.reshape({6});
        std::vector<std::size_t> sc2 = {6};
        EXPECT_TRUE(std::equal(sc2.begin(), sc2.end(), a.shape().begin()) && a.shape().size() == 1);
        EXPECT_EQ(ptr, a.data());
    }
}
