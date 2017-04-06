/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef PY_TENSOR_HPP
#define PY_TENSOR_HPP

#include <cstddef>
#include <array>
#include <algorithm>
#include "xtensor/xutils.hpp"
#include "xtensor/xsemantic.hpp"
#include "xtensor/xiterator.hpp"

#include "pycontainer.hpp"
#include "pybuffer_adaptor.hpp"

namespace xt
{
    template <class T, std::size_t N>
    class pytensor;
}

namespace pybind11
{
    namespace detail
    {
        template <class T, std::size_t N>
        struct handle_type_name<xt::pytensor<T, N>>
        {
            static PYBIND11_DESCR name()
            {
                return _("numpy.ndarray[") + make_caster<T>::name() + _("]");
            }
        };

        template <class T, std::size_t N>
        struct pyobject_caster<xt::pytensor<T, N>>
        {
            using type = xt::pytensor<T, N>;

            bool load(handle src, bool)
            {
                value = type::ensure(src);
                return static_cast<bool>(value);
            }

            static handle cast(const handle& src, return_value_policy, handle)
            {
                return src.inc_ref();
            }

            PYBIND11_TYPE_CASTER(type, handle_type_name<type>::name());
        };
    }
}

namespace xt
{

    template <class T, std::size_t N>
    struct xiterable_inner_types<pytensor<T, N>>
        : xcontainer_iterable_types<pytensor<T, N>>
    {
    };

    template <class T, std::size_t N>
    struct xcontainer_inner_types<pytensor<T, N>>
    {
        using container_type = pybuffer_adaptor<T>;
        using shape_type = std::array<npy_intp, N>;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = pytensor<T, N>;
    };

    template <class T, std::size_t N>
    class pytensor : public pycontainer<pytensor<T, N>>,
                     public xcontainer_semantic<pytensor<T, N>>
    {
    public:

        using self_type = pytensor<T, N>;
        using semantic_base = xcontainer_semantic<self_type>;
        using base_type = pycontainer<self_type>;
        using container_type = typename base_type::container_type;
        using value_type = typename base_type::value_type; 
        using reference = typename base_type::reference; 
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using size_type = typename base_type::size_type;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;
        using backstrides_type = typename base_type::backstrides_type;
        using inner_shape_type = typename base_type::inner_shape_type;
        using inner_strides_type = typename base_type::inner_strides_type;
        using inner_backstrides_type = typename base_type::inner_backstrides_type;

        pytensor();
        pytensor(const self_type&) = default;
        pytensor(self_type&&) = default;
        pytensor(nested_initializer_list_t<T, N> t);
        pytensor(pybind11::handle h, pybind11::object::borrowed_t);
        pytensor(pybind11::handle h, pybind11::object::stolen_t);
        pytensor(const pybind11::object& o);
        
        explicit pytensor(const shape_type& shape, layout l = layout::row_major);
        explicit pytensor(const shape_type& shape, const_reference value, layout l = layout::row_major);
        explicit pytensor(const shape_type& shape, const strides_type& strides, const_reference value);
        explicit pytensor(const shape_type& shape, const strides_type& strides);

        self_type& operator=(const self_type& e) = default;
        self_type& operator=(self_type&& e) = default;

        template <class E>
        pytensor(const xexpression<E>& e);

        template <class E>
        self_type& operator=(const xexpression<E>& e);

        using base_type::begin;
        using base_type::end;

        static self_type ensure(pybind11::handle h);
        static bool check_(pybind11::handle h);

    private:

        inner_shape_type m_shape;
        inner_strides_type m_strides;
        inner_backstrides_type m_backstrides;
        container_type m_data;

        void init_tensor(const shape_type& shape, const strides_type& strides);
        void init_from_python();

        inner_shape_type& shape_impl() noexcept;
        const inner_shape_type& shape_impl() const noexcept;
        inner_strides_type& strides_impl() noexcept;
        const inner_strides_type& strides_impl() const noexcept;
        inner_backstrides_type& backstrides_impl() noexcept;
        const inner_backstrides_type& backstrides_impl() const noexcept;

        container_type& data_impl() noexcept;
        const container_type& data_impl() const noexcept;

        friend class xcontainer<pytensor<T, N>>;
    };

    /***************************
     * pytensor implementation *
     ***************************/

    template <class T, std::size_t N>
    inline pytensor<T, N>::pytensor()
    {
        m_shape = make_sequence<shape_type>(N, size_type(1));
        m_strides = make_sequence<strides_type>(N, size_type(0));
        init_tensor(m_shape, m_strides);
        m_data[0] = T();
    }

    template <class T, std::size_t N>
    inline pytensor<T, N>::pytensor(nested_initializer_list_t<T, N> t)
    {
        base_type::reshape(xt::shape<shape_type>(t), layout::row_major);
        nested_copy(m_data.begin(), t);
    }

    template <class T, std::size_t N>
    inline pytensor<T, N>::pytensor(pybind11::handle h, pybind11::object::borrowed_t b)
        : base_type(h, b)
    {
        init_from_python();
    }

    template <class T, std::size_t N>
    inline pytensor<T, N>::pytensor(pybind11::handle h, pybind11::object::stolen_t s)
        : base_type(h, s)
    {
        init_from_python();
    }

    template <class T, std::size_t N>
    inline pytensor<T, N>::pytensor(const pybind11::object& o)
        : base_type(o)
    {
        init_from_python();
    }

    template <class T, std::size_t N>
    inline pytensor<T, N>::pytensor(const shape_type& shape, layout l)
    {
        compute_strides(shape, l, m_strides);
        init_tensor(shape, m_strides);
    }

    template <class T, std::size_t N>
    inline pytensor<T, N>::pytensor(const shape_type& shape,
                                    const_reference value,
                                    layout l)
    {
        compute_strides(shape, l, m_strides);
        init_tensor(shape, m_strides);
        std::fill(m_data.begin(), m_data.end(), value);
    }

    template <class T, std::size_t N>
    inline pytensor<T, N>::pytensor(const shape_type& shape,
                                    const strides_type& strides,
                                    const_reference value)
    {
        init_tensor(shape, strides);
        std::fill(m_data.begin(), m_data.end(), value);
    }

    template <class T, std::size_t N>
    inline pytensor<T, N>::pytensor(const shape_type& shape,
                                    const strides_type& strides)
    {
        init_tensor(shape, strides);
    }

    template <class T, std::size_t N>
    template <class E>
    inline pytensor<T, N>::pytensor(const xexpression<E>& e)
    {
        semantic_base::assign(e);
    }

    template <class T, std::size_t N>
    template <class E>
    inline auto pytensor<T, N>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }

    template <class T, std::size_t N>
    inline auto pytensor<T, N>::ensure(pybind11::handle h) -> self_type
    {
        return base_type::ensure(h);
    }

    template <class T, std::size_t N>
    inline bool pytensor<T, N>::check_(pybind11::handle h)
    {
        return base_type::check_(h);
    }
    
    template <class T, std::size_t N>
    inline void pytensor<T, N>::init_tensor(const shape_type& shape, const strides_type& strides)
    {
        npy_intp python_strides[N];
        std::transform(strides.begin(), strides.end(), python_strides,
                [](auto v) { return sizeof(T) * v; });
        int flags = NPY_ARRAY_ALIGNED;
        if (!std::is_const<T>::value)
        {
            flags |= NPY_ARRAY_WRITEABLE;
        }
        int type_num = detail::numpy_traits<T>::type_num;

        auto tmp = pybind11::reinterpret_steal<pybind11::object>(
                PyArray_New(&PyArray_Type, N, const_cast<npy_intp*>(shape.data()),
                            type_num, python_strides, nullptr, sizeof(T), flags, nullptr)
                );
        
        if (!tmp)
        {
            throw std::runtime_error("NumPy: unable to create ndarray");
        }

        this->m_ptr = tmp.release().ptr();
        m_shape = shape;
        m_strides = strides;
        adapt_strides(m_shape, m_strides, m_backstrides);
        m_data = container_type(reinterpret_cast<pointer>(PyArray_DATA(this->python_array())),
                                static_cast<size_type>(PyArray_SIZE(this->python_array())));
    }

    template <class T, std::size_t N>
    inline void pytensor<T, N>::init_from_python()
    {
        if (PyArray_NDIM(this->python_array()) != N)
        {
            throw std::runtime_error("NumPy: ndarray has incorrect number of dimensions");
        }

        std::copy(PyArray_DIMS(this->python_array()), PyArray_DIMS(this->python_array()) + N, m_shape.begin());
        std::transform(PyArray_STRIDES(this->python_array()), PyArray_STRIDES(this->python_array()) + N, m_strides.begin(),
                [](auto v) { return v / sizeof(T); });
        adapt_strides(m_shape, m_strides, m_backstrides);
        m_data = container_type(reinterpret_cast<pointer>(PyArray_DATA(this->python_array())),
                                static_cast<size_type>(PyArray_SIZE(this->python_array())));
    }

    template <class T, std::size_t N>
    inline auto pytensor<T, N>::shape_impl() noexcept -> inner_shape_type&
    {
        return m_shape;
    }

    template <class T, std::size_t N>
    inline auto pytensor<T, N>::shape_impl() const noexcept -> const inner_shape_type&
    {
        return m_shape;
    }

    template <class T, std::size_t N>
    inline auto pytensor<T, N>::strides_impl() noexcept -> inner_strides_type&
    {
        return m_strides;
    }

    template <class T, std::size_t N>
    inline auto pytensor<T, N>::strides_impl() const noexcept -> const inner_strides_type&
    {
        return m_strides;
    }

    template <class T, std::size_t N>
    inline auto pytensor<T, N>::backstrides_impl() noexcept -> inner_backstrides_type&
    {
        return m_backstrides;
    }

    template <class T, std::size_t N>
    inline auto pytensor<T, N>::backstrides_impl() const noexcept -> const inner_backstrides_type&
    {
        return m_backstrides;
    }

    template <class T, std::size_t N>
    inline auto pytensor<T, N>::data_impl() noexcept -> container_type&
    {
        return m_data;
    }

    template <class T, std::size_t N>
    inline auto pytensor<T, N>::data_impl() const noexcept -> const container_type&
    {
        return m_data;
    }
}

#endif

