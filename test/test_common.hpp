/***************************************************************************
* Copyright (c) Wolf Vollprecht, Johan Mabille and Sylvain Corlay          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef TEST_COMMON_HPP
#define TEST_COMMON_HPP

#include "xtensor/core/xlayout.hpp"
#include "xtensor/misc/xmanipulation.hpp"

#include "xtl/xsequence.hpp"

namespace xt
{
    template <class T, class A>
    bool operator==(const uvector<T, A>& lhs, const std::vector<T, A>& rhs)
    {
        return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    template <class T, class A>
    bool operator==(const std::vector<T, A>& lhs, const uvector<T, A>& rhs)
    {
        return rhs == lhs;
    }

    template <class C = std::vector<std::size_t>>
    struct layout_result
    {
        using vector_type = uvector<int>;
        using size_type = typename C::value_type;
        using shape_type = C;
        using strides_type = get_strides_t<shape_type>;

        using assigner_type = std::vector<std::vector<vector_type>>;

        inline layout_result()
        {
            m_shape = {3, 2, 4};
            m_assigner.resize(m_shape[0]);
            for (size_type i = 0; i < m_shape[0]; ++i)
            {
                m_assigner[i].resize(m_shape[1]);
            }
            m_assigner[0][0] = {-1, 1, 2, 3};
            m_assigner[0][1] = {4, 5, 6, 7};
            m_assigner[1][0] = {8, 9, 10, 11};
            m_assigner[1][1] = {12, 13, 14, 15};
            m_assigner[2][0] = {16, 17, 18, 19};
            m_assigner[2][1] = {20, 21, 22, 23};
        }

        shape_type m_shape;
        strides_type m_strides;
        strides_type m_backstrides;
        vector_type m_data;
        layout_type m_layout;
        assigner_type m_assigner;

        inline size_type size() const { return m_data.size(); }
        inline const shape_type& shape() const { return m_shape; }
        inline const strides_type& strides() const { return m_strides; }
        inline const strides_type& backstrides() const { return m_backstrides; }
        inline layout_type layout() const { return m_layout; }
        inline const vector_type& data() const { return m_data; }
    };

    template <class C = std::vector<std::size_t>>
    struct row_major_result : layout_result<C>
    {
        inline row_major_result()
        {
            this->m_strides = {8, 4, 1};
            this->m_backstrides = {16, 4, 3};
            this->m_data = {-1, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                            10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                            20, 21, 22, 23};
            this->m_layout = layout_type::row_major;
        }
    };

    template <class C = std::vector<std::size_t>>
    struct column_major_result : layout_result<C>
    {
        inline column_major_result()
        {
            this->m_strides = {1, 3, 6};
            this->m_backstrides = {2, 3, 18};
            this->m_data = {-1, 8, 16, 4, 12, 20,
                             1, 9, 17, 5, 13, 21,
                             2, 10, 18, 6, 14, 22,
                             3, 11, 19, 7, 15, 23};
            this->m_layout = layout_type::column_major;
        }
    };

    template <class C = std::vector<std::size_t>>
    struct central_major_result : layout_result<C>
    {
        inline central_major_result()
        {
            this->m_strides = {8, 1, 2};
            this->m_backstrides = {16, 1, 6};
            this->m_data = {-1, 4, 1, 5, 2, 6, 3, 7,
                            8, 12, 9, 13, 10, 14, 11, 15,
                            16, 20, 17, 21, 18, 22, 19, 23};
            this->m_layout = layout_type::dynamic;
        }
    };

    template <class C = std::vector<std::size_t>>
    struct unit_shape_result
    {
        using vector_type = std::vector<int>;
        using size_type = typename C::value_type;
        using shape_type = C;
        using strides_type = C;

        using assigner_type = std::vector<std::vector<vector_type>>;

        inline unit_shape_result()
        {
            m_shape = {3, 1, 4};
            m_strides = {4, 0, 1};
            m_backstrides = {8, 0, 3};
            m_data = {-1, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19};
            m_layout = layout_type::dynamic;
            m_assigner.resize(m_shape[0]);
            for (std::size_t i = 0; i < std::size_t(m_shape[0]); ++i)
            {
                m_assigner[i].resize(m_shape[1]);
            }
            m_assigner[0][0] = {-1, 1, 2, 3};
            m_assigner[1][0] = {8, 9, 10, 11};
            m_assigner[2][0] = {16, 17, 18, 19};
        }

        shape_type m_shape;
        strides_type m_strides;
        strides_type m_backstrides;
        vector_type m_data;
        layout_type m_layout;
        assigner_type m_assigner;

        inline size_type size() const { return m_data.size(); }
        inline const shape_type& shape() const { return m_shape; }
        inline const strides_type& strides() const { return m_strides; }
        inline const strides_type& backstrides() const { return m_backstrides; }
        inline layout_type layout() const { return m_layout; }
        inline const vector_type& data() const { return m_data; }
    };

    template <class V, class R>
    void compare_shape(V& vec, const R& result, bool compare_layout = true)
    {
        EXPECT_TRUE(std::equal(vec.shape().cbegin(), vec.shape().cend(), result.shape().cbegin()));
        EXPECT_TRUE(std::equal(vec.strides().cbegin(), vec.strides().cend(), result.strides().cbegin()));
// TODO: check why this does not build on modern MSVC compilers
#ifndef WIN32
        EXPECT_TRUE(std::equal(vec.backstrides().cbegin(), vec.backstrides().cend(), result.backstrides().cbegin()));
#endif
        EXPECT_EQ(vec.size(), result.size());
        if (compare_layout)
        {
            EXPECT_EQ(vec.layout(), result.layout());
        }
    }

    template <class V, class C = std::vector<std::size_t>>
    void test_resize(V& vec)
    {
        {
            SCOPED_TRACE("row_major resize");
            row_major_result<C> rm;
            vec.resize(rm.m_shape, layout_type::row_major);
            compare_shape(vec, rm);
        }

        {
            SCOPED_TRACE("different types resize");
            row_major_result<C> rm;
            auto v_copy_a = vec;
            auto v_copy_b = vec;
            std::array<std::size_t, 3> ar = {3, 2, 4};
            std::vector<std::size_t> vr = {3, 2, 4};
            v_copy_a.resize(ar);
            compare_shape(v_copy_a, rm);
            v_copy_b.resize(vr);
            compare_shape(v_copy_b, rm);
        }

        {
            SCOPED_TRACE("column_major resize");
            column_major_result<C> cm;
            vec.resize(cm.m_shape, layout_type::column_major);
            compare_shape(vec, cm);
        }

        {
            SCOPED_TRACE("central_major resize");
            central_major_result<C> cem;
            vec.resize(cem.m_shape, cem.m_strides);
            compare_shape(vec, cem);
        }

        {
            SCOPED_TRACE("unit_shape resize");
            unit_shape_result<C> usr;
            vec.resize(usr.m_shape, layout_type::row_major);
            compare_shape(vec, usr, false);
            EXPECT_EQ(vec.layout(), layout_type::row_major);
        }
    }

    template <class V, class C = std::vector<std::size_t>>
    void test_transpose(V& vec)
    {
        using shape_type = typename V::shape_type;
        using strides_type = typename V::strides_type;
        {
            SCOPED_TRACE("transpose");
            shape_type shape_new = xtl::make_sequence<shape_type>(vec.dimension(), 0);
            std::copy(vec.shape().cbegin(), vec.shape().cend(), shape_new.begin());
            auto vt = transpose(vec);
            std::reverse(shape_new.begin(), shape_new.end());
            EXPECT_EQ(vt.shape(), shape_new);
            EXPECT_TRUE(std::equal(vt.shape().cbegin(), vt.shape().cend(), shape_new.cbegin()));
        }

        {
            SCOPED_TRACE("transpose with data");
            row_major_result<C> rm;
            vec.resize(rm.shape(), layout_type::row_major);

            assign_array(vec, rm.m_assigner);
            EXPECT_TRUE(std::equal(vec.storage().cbegin(), vec.storage().cend(), rm.m_data.cbegin()));

            auto vec_copy = vec;

            shape_type shape_new(rm.shape());
            auto vt = transpose(vec);
            std::reverse(shape_new.begin(), shape_new.end());
            EXPECT_EQ(vt.shape(), shape_new);
            EXPECT_TRUE(std::equal(vt.storage().cbegin(), vt.storage().cend(), rm.m_data.cbegin()));

            strides_type new_strides = {rm.m_strides[2],
                                        rm.m_strides[1],
                                        rm.m_strides[0]};
            EXPECT_EQ(vt.strides(), new_strides);

            strides_type new_backstrides = {rm.m_backstrides[2],
                                            rm.m_backstrides[1],
                                            rm.m_backstrides[0]};
            EXPECT_EQ(vt.backstrides(), new_backstrides);

            EXPECT_EQ(vec_copy(0, 0, 0), vt(0, 0, 0));
            EXPECT_EQ(vec_copy(0, 1, 0), vt(0, 1, 0));
            EXPECT_EQ(vec_copy(1, 1, 0), vt(0, 1, 1));
            EXPECT_EQ(vec_copy(1, 1, 2), vt(2, 1, 1));
        }

        {
            SCOPED_TRACE("transpose with permutation");
            row_major_result<C> rm;
            vec.resize(rm.shape(), layout_type::row_major);

            assign_array(vec, rm.m_assigner);
            EXPECT_TRUE(std::equal(vec.storage().cbegin(), vec.storage().cend(), rm.m_data.cbegin()));

            auto vec_copy = vec;

            shape_type a = xtl::make_sequence<shape_type>(vec.dimension(), 0);
            std::copy(vec.shape().cbegin(), vec.shape().cend(), a.begin());
            auto vt = transpose(vec, {1, 0, 2});
            shape_type shape_new = {a[1], a[0], a[2]};
            EXPECT_TRUE(std::equal(vt.shape().cbegin(), vt.shape().cend(), shape_new.begin()));
            EXPECT_TRUE(std::equal(vt.storage().cbegin(), vt.storage().cend(), rm.m_data.cbegin()));

            strides_type new_strides = {rm.m_strides[1],
                                        rm.m_strides[0],
                                        rm.m_strides[2]};
            EXPECT_EQ(vt.strides(), new_strides);

            // strides_type new_backstrides = {rm.m_backstrides[1],
            //                                 rm.m_backstrides[0],
            //                                 rm.m_backstrides[2]};
            // EXPECT_EQ(vt.backstrides(), new_backstrides);

            EXPECT_EQ(vec_copy(0, 0, 0), vt(0, 0, 0));
            EXPECT_EQ(vec_copy(0, 1, 0), vt(1, 0, 0));
            EXPECT_EQ(vec_copy(1, 1, 0), vt(1, 1, 0));
            EXPECT_EQ(vec_copy(1, 1, 2), vt(1, 1, 2));

            // Compilation check only
            std::vector<std::size_t> perm = {1, 0, 2};
            transpose(vec, perm);
        }

        {
            SCOPED_TRACE("transpose permutation throws");
            row_major_result<C> rm;
            vec.resize(rm.shape(), layout_type::row_major);

            EXPECT_THROW(transpose(vec, {1, 1, 0}, check_policy::full()), transpose_error);
            EXPECT_THROW(transpose(vec, {1, 0, 2, 3}, check_policy::full()), transpose_error);
            EXPECT_THROW(transpose(vec, {1, 2}, check_policy::full()), transpose_error);
            EXPECT_THROW(transpose(vec, {3, 0, 1}, check_policy::full()), transpose_error);
        }
    }

    template <class V1, class V2>
    void assign_array(V1& dst, const V2& src)
    {
        for (std::size_t i = 0; i < std::size_t(dst.shape()[0]); ++i)
        {
            for (std::size_t j = 0; j < std::size_t(dst.shape()[1]); ++j)
            {
                for (std::size_t k = 0; k < std::size_t(dst.shape()[2]); ++k)
                {
                    dst(i, j, k) = src[i][j][k];
                }
            }
        }
    }

    template <class V>
    void test_bound_check(V& vec)
    {
#ifdef XTENSOR_ENABLE_ASSERT
        EXPECT_ANY_THROW(vec(10, 10, 10));
#else
        (void)vec;
#endif
    }

    template <class V, class C = std::vector<std::size_t>>
    void test_access(V& vec)
    {
        {
            SCOPED_TRACE("row_major access");
            row_major_result<C> rm;
            vec.resize(rm.m_shape, layout_type::row_major);
            assign_array(vec, rm.m_assigner);
            EXPECT_TRUE(std::equal(vec.storage().cbegin(), vec.storage().cend(), rm.m_data.cbegin()));
            EXPECT_EQ(vec(0, 1, 1), vec(1, 1));
            EXPECT_EQ(vec(2, 1, 3), vec(2, 2, 2, 1, 3));
            test_bound_check(vec);
        }

        {
            SCOPED_TRACE("column_major access");
            column_major_result<C> cm;
            vec.resize(cm.m_shape, layout_type::column_major);
            assign_array(vec, cm.m_assigner);
            EXPECT_TRUE(std::equal(vec.storage().cbegin(), vec.storage().cend(), cm.m_data.cbegin()));
            EXPECT_EQ(vec(0, 1, 1), vec(1, 1));
            EXPECT_EQ(vec(2, 1, 3), vec(2, 2, 2, 1, 3));
            test_bound_check(vec);
        }

        {
            SCOPED_TRACE("central_major access");
            central_major_result<C> cem;
            vec.resize(cem.m_shape, cem.m_strides);
            assign_array(vec, cem.m_assigner);
            EXPECT_TRUE(std::equal(vec.storage().cbegin(), vec.storage().cend(), cem.m_data.cbegin()));
            EXPECT_EQ(vec(0, 1, 1), vec(1, 1));
            EXPECT_EQ(vec(2, 1, 3), vec(2, 2, 2, 1, 3));
            test_bound_check(vec);
        }

        {
            SCOPED_TRACE("unit_shape access");
            unit_shape_result<C> usr;
            vec.resize(usr.m_shape, layout_type::row_major);
            assign_array(vec, usr.m_assigner);
            EXPECT_TRUE(std::equal(vec.storage().cbegin(), vec.storage().cend(), usr.m_data.cbegin()));
            EXPECT_EQ(vec(0, 1, 0), vec(1, 0));
            EXPECT_EQ(vec(2, 0, 3), vec(2, 2, 2, 0, 3));
            test_bound_check(vec);
        }
    }

    template <class V, class C = std::vector<std::size_t>>
    void test_element(V& vec)
    {
        {
            SCOPED_TRACE("row_major access");
            row_major_result<C> rm;
            vec.resize(rm.m_shape, layout_type::row_major);
            assign_array(vec, rm.m_assigner);
            EXPECT_EQ(vec.data(), rm.m_data);
            std::vector<std::size_t> index1 = {0, 1, 1};
            std::vector<std::size_t> index2 = {1, 1};
            std::vector<std::size_t> index3 = {2, 1, 3};
            std::vector<std::size_t> index4 = {2, 2, 2, 1, 3};
            EXPECT_EQ(vec.element(index1.begin(), index1.end()), vec.element(index2.begin(), index2.end()));
            EXPECT_EQ(vec.element(index3.begin(), index3.end()), vec.element(index4.begin(), index4.end()));
            test_bound_check(vec);
        }

        {
            SCOPED_TRACE("column_major access");
            column_major_result<C> cm;
            vec.resize(cm.m_shape, layout_type::column_major);
            assign_array(vec, cm.m_assigner);
            EXPECT_EQ(vec.data(), cm.m_data);
            std::vector<std::size_t> index1 = {0, 1, 1};
            std::vector<std::size_t> index2 = {1, 1};
            std::vector<std::size_t> index3 = {2, 1, 3};
            std::vector<std::size_t> index4 = {2, 2, 2, 1, 3};
            EXPECT_EQ(vec.element(index1.begin(), index1.end()), vec.element(index2.begin(), index2.end()));
            EXPECT_EQ(vec.element(index3.begin(), index3.end()), vec.element(index4.begin(), index4.end()));
            test_bound_check(vec);
        }

        {
            SCOPED_TRACE("central_major access");
            central_major_result<C> cem;
            vec.resize(cem.m_shape, cem.m_strides);
            assign_array(vec, cem.m_assigner);
            EXPECT_EQ(vec.data(), cem.m_data);
            std::vector<std::size_t> index1 = {0, 1, 1};
            std::vector<std::size_t> index2 = {1, 1};
            std::vector<std::size_t> index3 = {2, 1, 3};
            std::vector<std::size_t> index4 = {2, 2, 2, 1, 3};
            EXPECT_EQ(vec.element(index1.begin(), index1.end()), vec.element(index2.begin(), index2.end()));
            EXPECT_EQ(vec.element(index3.begin(), index3.end()), vec.element(index4.begin(), index4.end()));
            test_bound_check(vec);
        }

        {
            SCOPED_TRACE("unit_shape access");
            unit_shape_result<C> usr;
            vec.resize(usr.m_shape, layout_type::row_major);
            assign_array(vec, usr.m_assigner);
            EXPECT_EQ(vec.data(), usr.m_data);
            std::vector<std::size_t> index1 = {0, 1, 0};
            std::vector<std::size_t> index2 = {1, 0};
            std::vector<std::size_t> index3 = {2, 0, 3};
            std::vector<std::size_t> index4 = {2, 2, 2, 0, 3};
            EXPECT_EQ(vec.element(index1.begin(), index1.end()), vec.element(index2.begin(), index2.end()));
            EXPECT_EQ(vec.element(index3.begin(), index3.end()), vec.element(index4.begin(), index4.end()));
            test_bound_check(vec);
        }
    }

    template <class V1, class V2>
    void indexed_assign_array(V1& dst, const V2& src)
    {
        xindex index(dst.dimension());
        for (std::size_t i = 0; i < std::size_t(dst.shape()[0]); ++i)
        {
            index[0] = i;
            for (std::size_t j = 0; j < std::size_t(dst.shape()[1]); ++j)
            {
                index[1] = j;
                for (std::size_t k = 0; k < std::size_t(dst.shape()[2]); ++k)
                {
                    index[2] = k;
                    dst[index] = src[i][j][k];
                }
            }
        }
    }

    template <class V, class C = std::vector<std::size_t>>
    void test_indexed_access(V& vec)
    {
        xindex index1 = {1, 1};
        xindex index2 = {2, 2, 2, 1, 3};
        {
            SCOPED_TRACE("row_major access");
            row_major_result<C> rm;
            vec.resize(rm.m_shape, layout_type::row_major);
            indexed_assign_array(vec, rm.m_assigner);
            EXPECT_TRUE(std::equal(vec.storage().cbegin(), vec.storage().cend(), rm.m_data.cbegin()));
            EXPECT_EQ(vec(0, 1, 1), vec[index1]);
            EXPECT_EQ(vec(2, 1, 3), vec[index2]);
        }

        {
            SCOPED_TRACE("column_major access");
            column_major_result<C> cm;
            vec.resize(cm.m_shape, layout_type::column_major);
            indexed_assign_array(vec, cm.m_assigner);
            EXPECT_TRUE(std::equal(vec.storage().cbegin(), vec.storage().cend(), cm.m_data.cbegin()));
            EXPECT_EQ(vec(0, 1, 1), vec[index1]);
            EXPECT_EQ(vec(2, 1, 3), vec[index2]);
        }

        {
            SCOPED_TRACE("central_major access");
            central_major_result<C> cem;
            vec.resize(cem.m_shape, cem.m_strides);
            indexed_assign_array(vec, cem.m_assigner);
            EXPECT_TRUE(std::equal(vec.storage().cbegin(), vec.storage().cend(), cem.m_data.cbegin()));
            EXPECT_EQ(vec(0, 1, 1), vec[index1]);
            EXPECT_EQ(vec(2, 1, 3), vec[index2]);
        }

        {
            SCOPED_TRACE("unit_shape access");
            unit_shape_result<C> usr;
            vec.resize(usr.m_shape, layout_type::row_major);
            indexed_assign_array(vec, usr.m_assigner);
            EXPECT_TRUE(std::equal(vec.storage().cbegin(), vec.storage().cend(), usr.m_data.cbegin()));
            xindex id1 = {1, 0};
            xindex id2 = {2, 2, 2, 0, 3};
            EXPECT_EQ(vec(0, 1, 0), vec[id1]);
            EXPECT_EQ(vec(2, 0, 3), vec[id2]);
        }
    }

    template <class V>
    void test_broadcast(V& vec)
    {
        using shape_type = typename V::shape_type;

        shape_type s = {3, 1, 4, 2};
        vec.resize(s);

        {
            SCOPED_TRACE("same shape");
            shape_type s1 = s;
            bool res = vec.broadcast_shape(s1);
            EXPECT_EQ(s1, s);
            EXPECT_TRUE(res);
        }

        {
            SCOPED_TRACE("different shape");
            shape_type s2 = {3, 5, 1, 2};
            shape_type s2r = {3, 5, 4, 2};
            bool res = vec.broadcast_shape(s2);
            EXPECT_EQ(s2, s2r);
            EXPECT_FALSE(res);
        }

        {
            SCOPED_TRACE("incompatible shapes");
            shape_type s4 = {2, 1, 3, 2};
            bool wit = false;
            try
            {
                vec.broadcast_shape(s4);
            }
            catch (broadcast_error&)
            {
                wit = true;
            }
            EXPECT_TRUE(wit);
        }
    }

    template <class V>
    void test_broadcast2(V& vec)
    {
        using shape_type = typename V::shape_type;

        shape_type s = {3, 1, 4, 2};
        vec.resize(s);

        {
            SCOPED_TRACE("different dimensions");
            shape_type s3 = {5, 3, 1, 4, 2};
            shape_type s3r = s3;
            bool res = vec.broadcast_shape(s3);
            EXPECT_EQ(s3, s3r);
            EXPECT_FALSE(res);
        }
    }

    template <class VRM, class VCM, class C = std::vector<std::size_t>>
    void test_iterator(VRM& vecrm, VCM& veccm)
    {
        {
            SCOPED_TRACE("row_major storage iterator");
            row_major_result<C> rm;
            vecrm.resize(rm.m_shape, layout_type::row_major);
            std::copy(rm.data().cbegin(), rm.data().cend(), vecrm.template begin<layout_type::row_major>());
            EXPECT_TRUE(std::equal(rm.data().cbegin(), rm.data().cend(), vecrm.storage().cbegin()));
            //EXPECT_EQ(vecrm.template end<layout_type::row_major>(), vecrm.data().end());
        }

        {
            SCOPED_TRACE("column_major storage iterator");
            column_major_result<C> cm;
            veccm.resize(cm.m_shape, layout_type::column_major);
            std::copy(cm.data().cbegin(), cm.data().cend(), veccm.template begin<layout_type::column_major>());
            EXPECT_TRUE(std::equal(cm.data().cbegin(), cm.data().cend(), veccm.storage().cbegin()));
            //EXPECT_EQ(veccm.template end<layout_type::column_major>(), veccm.data().end());
        }
    }

    template <class V, class C = std::vector<std::size_t>>
    void test_xiterator(V& vec)
    {
        row_major_result<C> rm;
        vec.resize(rm.m_shape, layout_type::row_major);
        indexed_assign_array(vec, rm.m_assigner);
        size_t nb_iter = vec.size() / 2;
        using shape_type = std::vector<size_t>;

        // broadcast_iterator
        {
            auto iter = vec.template begin<layout_type::row_major>();
            auto iter_end = vec.template end<layout_type::row_major>();
            for (size_t i = 0; i < nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(vec.data()[nb_iter], *iter);
            for (size_t i = 0; i < nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(iter, iter_end);
        }

        // shaped_xiterator
        {
            shape_type shape(rm.m_shape.size() + 1);
            std::copy(rm.m_shape.begin(), rm.m_shape.end(), shape.begin() + 1);
            shape[0] = 2;
            auto iter = vec.template begin<shape_type, layout_type::row_major>(shape);
            auto iter_end = vec.template end<shape_type, layout_type::row_major>(shape);
            for (size_t i = 0; i < 2 * nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(vec.data()[0], *iter);
            for (size_t i = 0; i < 2 * nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(iter, iter_end);
        }

        // column broadcast_iterator
        {
            auto iter = vec.template begin<layout_type::column_major>();
            auto iter_end = vec.template end<layout_type::column_major>();
            for (size_t i = 0; i < nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(vec(0, 0, 2), *iter);
            for (size_t i = 0; i < nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(iter, iter_end);
        }

        // column shaped_xiterator
        {
            shape_type shape(rm.m_shape.size() + 1);
            std::copy(rm.m_shape.begin(), rm.m_shape.end(), shape.begin() + 1);
            shape[0] = 2;
            auto iter = vec.template begin<shape_type, layout_type::column_major>(shape);
            auto iter_end = vec.template end<shape_type, layout_type::column_major>(shape);
            for (size_t i = 0; i < 2 * nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(vec(0, 0, 2), *iter);
            for (size_t i = 0; i < 2 * nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(iter, iter_end);
        }
    }

    template <class V, class C = std::vector<std::size_t>>
    void test_reverse_xiterator(V& vec)
    {
        row_major_result<C> rm;
        vec.resize(rm.m_shape, layout_type::row_major);
        indexed_assign_array(vec, rm.m_assigner);
        size_t nb_iter = vec.size() / 2;

        // broadcast_iterator
        {
            auto iter = vec.template rbegin<layout_type::row_major>();
            auto iter_end = vec.template rend<layout_type::row_major>();
            for (size_t i = 0; i < nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(vec.data()[nb_iter - 1], *iter);
            for (size_t i = 0; i < nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(iter, iter_end);
        }

        // shaped_xiterator
        {
            using shape_type = std::vector<size_t>;
            shape_type shape(rm.m_shape.size() + 1);
            std::copy(rm.m_shape.begin(), rm.m_shape.end(), shape.begin() + 1);
            shape[0] = 2;
            auto iter = vec.template rbegin<shape_type, layout_type::row_major>(shape);
            auto iter_end = vec.template rend<shape_type, layout_type::row_major>(shape);
            for (size_t i = 0; i < 2 * nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(vec.data()[2 * nb_iter - 1], *iter);
            for (size_t i = 0; i < 2 * nb_iter; ++i)
            {
                ++iter;
            }
            EXPECT_EQ(iter, iter_end);
        }
    }
}

#endif
