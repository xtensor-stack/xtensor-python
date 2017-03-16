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
        : pycontainer_iterable_types<pytensor<T, N>>
    {
    };

    template <class T, std::size_t N>
    struct xcontainer_inner_types<pytensor<T, N>>
    {
        using container_type = pybuffer_adaptor<T>;
        using shape_type = std::array<npy_int, N>;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
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
        using pointer = typename base_type::pointer;
        using size_type = typename base_type::size_type;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;
        using backstrides_type = typename base_type::backstrides_type;
        using inner_shape_type = typename base_type::inner_shape_type;
        using inner_strides_type = typename base_type::inner_strides_type;

        pytensor() = default;
        pytensor(nested_initializer_list_t<T, N> t);
        pytensor(pybind11::handle h, pybind11::object::borrowed_t);
        pytensor(pybind11::handle h, pybind11::object::stolen_t);
        pytensor(const pybind11::object &o);
        
        explicit pytensor(const shape_type& shape, layout l = layout::row_major);
        pytensor(const shape_type& shape, const strides_type& strides);

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
        backstrides_type m_backstrides;
        container_type m_data;

        void init_tensor(const shape_type& shape, const strides_type& strides);
        void init_from_python();
        void compute_backstrides();

        const inner_shape_type& shape_impl() const;
        const inner_strides_type& strides_impl() const;
        const backstrides_type& backstrides_impl() const;

        container_type& data_impl();
        const container_type& data_impl() const;

        friend class pycontainer<pytensor<T, N>>;
    };

    /***************************
     * pytensor implementation *
     ***************************/

    template <class T, std::size_t N>
    inline pytensor<T, N>::pytensor(nested_initializer_list_t<T, N> t)
    {
        base_type::reshape(xt::shape<shape_type>(t), layout::row_major);
        nested_copy(m_data.begin(), t);
    }

    template <class T, std::size_t N>
    inline pytensor<T, N>::pytensor(pybind11::handle h, pybind11::object::borrowed_t)
        : base_type(h, pybind11::object::borrowed)
    {
        init_from_python();
    }

    template <class T, std::size_t N>
    inline pytensor<T, N>::pytensor(pybind11::handle h, pybind11::object::stolen_t)
        : base_type(h, pybind11::object::stolen)
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
        base_type::fill_default_strides(shape, l, m_strides);
        init_tensor(shape, m_strides);
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
        if(!std::is_const<T>::value)
        {
            flags |= NPY_ARRAY_WRITEABLE;
        }
        int type_num = detail::numpy_traits<T>::type_num;
        
        auto tmp = pybind11::reinterpret_steal<pybind11::object>(
                PyArray_New(&PyArray_Type, N, reinterpret_cast<npy_intp*>(shape.data()),
                            type_num, python_strides, nullptr, sizeof(T), flags, nullptr)
                );
        
        if(!tmp)
            throw std::runtime_error("NumPy: unable to create ndarray");

        this->m_ptr = tmp.release().ptr();
        m_shape = shape;
        m_strides = strides;
        compute_backstrides();
        m_data = container_type(reinterpret_cast<pointer>(PyArray_DATA(this->python_array())),
                                static_cast<size_type>(PyArray_SIZE(this->python_array())));
    }

    template <class T, std::size_t N>
    inline void pytensor<T, N>::init_from_python()
    {
        if(PyArray_NDIM(this->python_array()) != N)
            throw std::runtime_error("NumPy: ndarray has incorrect number of dimensions");

        std::copy(PyArray_DIMS(this->python_array()), PyArray_DIMS(this->python_array()) + N, m_shape.begin());
        std::transform(PyArray_STRIDES(this->python_array()), PyArray_STRIDES(this->python_array()) + N, m_strides.begin(),
                [](auto v) { return v / sizeof(T); });
        compute_backstrides();
        m_data = container_type(reinterpret_cast<pointer>(PyArray_DATA(this->python_array())),
                                static_cast<size_type>(PyArray_SIZE(this->python_array())));
    }

    template <class T, std::size_t N>
    inline void pytensor<T, N>::compute_backstrides()
    {
        for(size_type i = 0; i < m_shape.size(); ++i)
        {
            if(m_shape[i] == 1)
            {
                m_backstrides[i] = 0;
            }
            else
            {
                m_backstrides[i] = m_strides[i] * (m_shape[i] - 1);
            }
        }
    }

    template <class T, std::size_t N>
    inline auto pytensor<T, N>::shape_impl() const -> const inner_shape_type&
    {
        return m_shape;
    }

    template <class T, std::size_t N>
    inline auto pytensor<T, N>::strides_impl() const -> const inner_strides_type&
    {
        return m_strides;
    }

    template <class T, std::size_t N>
    inline auto pytensor<T, N>::backstrides_impl() const -> const backstrides_type&
    {
        return m_backstrides;
    }

    template <class T, std::size_t N>
    inline auto pytensor<T, N>::data_impl() -> container_type&
    {
        return m_data;
    }

    template <class T, std::size_t N>
    inline auto pytensor<T, N>::data_impl() const -> const container_type&
    {
        return m_data;
    }

}

#endif

