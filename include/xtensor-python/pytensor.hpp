/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef PY_TENSOR_HPP
#define PY_TENSOR_HPP

#include <algorithm>
#include <array>
#include <cstddef>

#include "xtensor/xbuffer_adaptor.hpp"
#include "xtensor/xiterator.hpp"
#include "xtensor/xsemantic.hpp"
#include "xtensor/xutils.hpp"

#include "pycontainer.hpp"
#include "pystrides_adaptor.hpp"
#include "xtensor_type_caster_base.hpp"

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

            bool load(handle src, bool convert)
            {
                if (!convert)
                {
                    if (!PyArray_Check(src.ptr()))
                    {
                        return false;
                    }
                    int type_num = xt::detail::numpy_traits<T>::type_num;
                    if(xt::detail::pyarray_type(reinterpret_cast<PyArrayObject*>(src.ptr())) != type_num)
                    {
                        return false;
                    }
                }

                value = type::ensure(src);
                return static_cast<bool>(value);
            }

            static handle cast(const handle& src, return_value_policy, handle)
            {
                return src.inc_ref();
            }

            PYBIND11_TYPE_CASTER(type, handle_type_name<type>::name());
        };

        // Type caster for casting ndarray to xexpression<pytensor>
        template <class T, std::size_t N>
        struct type_caster<xt::xexpression<xt::pytensor<T, N>>> : pyobject_caster<xt::pytensor<T, N>>
        {
            using Type = xt::xexpression<xt::pytensor<T, N>>;

            operator Type&()
            {
                return this->value;
            }

            operator const Type&()
            {
                return this->value;
            }
        };

        // Type caster for casting xt::xtensor to ndarray
        template <class T, std::size_t N>
        struct type_caster<xt::xtensor<T, N>> : xtensor_type_caster_base<xt::xtensor<T, N>>
        {
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
        using container_type = xbuffer_adaptor<T*>;
        using shape_type = std::array<npy_intp, N>;
        using strides_type = shape_type;
        using backstrides_type = shape_type;
        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = pytensor<T, N>;
        static constexpr layout_type layout = layout_type::dynamic;
    };

    /**
     * @class pytensor
     * @brief Multidimensional container providing the xtensor container semantics wrapping a numpy array.
     *
     * pytensor is similar to the xtensor container in that it has a static dimensionality.
     *
     * Unlike with the pyarray container, pytensor cannot be resized with a different number of dimensions
     * and resizes are not reflected on the Python side. However, pytensor has benefits compared to pyarray
     * in terms of performances. pytensor shapes are stack-allocated which makes iteration upon pytensor
     * faster than with pyarray.
     *
     * @tparam T The type of the element stored in the pyarray.
     * @sa pyarray
     */
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
        pytensor(nested_initializer_list_t<T, N> t);
        pytensor(pybind11::handle h, pybind11::object::borrowed_t);
        pytensor(pybind11::handle h, pybind11::object::stolen_t);
        pytensor(const pybind11::object& o);

        explicit pytensor(const shape_type& shape, layout_type l = layout_type::row_major);
        explicit pytensor(const shape_type& shape, const_reference value, layout_type l = layout_type::row_major);
        explicit pytensor(const shape_type& shape, const strides_type& strides, const_reference value);
        explicit pytensor(const shape_type& shape, const strides_type& strides);

        pytensor(const self_type& rhs);
        self_type& operator=(const self_type& rhs);

        pytensor(self_type&&) = default;
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

    /**
     * @name Constructors
     */
    //@{
    /**
     * Allocates an uninitialized pytensor that holds 1 element.
     */
    template <class T, std::size_t N>
    inline pytensor<T, N>::pytensor()
        : base_type()
    {
        m_shape = xtl::make_sequence<shape_type>(N, size_type(1));
        m_strides = xtl::make_sequence<strides_type>(N, size_type(0));
        init_tensor(m_shape, m_strides);
        m_data[0] = T();
    }

    /**
     * Allocates a pytensor with a nested initializer list.
     */
    template <class T, std::size_t N>
    inline pytensor<T, N>::pytensor(nested_initializer_list_t<T, N> t)
        : base_type()
    {
        base_type::resize(xt::shape<shape_type>(t), layout_type::row_major);
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

    /**
     * Allocates an uninitialized pytensor with the specified shape and
     * layout.
     * @param shape the shape of the pytensor
     * @param l the layout_type of the pytensor
     */
    template <class T, std::size_t N>
    inline pytensor<T, N>::pytensor(const shape_type& shape, layout_type l)
    {
        compute_strides(shape, l, m_strides);
        init_tensor(shape, m_strides);
    }

    /**
     * Allocates a pytensor with the specified shape and layout. Elements
     * are initialized to the specified value.
     * @param shape the shape of the pytensor
     * @param value the value of the elements
     * @param l the layout_type of the pytensor
     */
    template <class T, std::size_t N>
    inline pytensor<T, N>::pytensor(const shape_type& shape,
                                    const_reference value,
                                    layout_type l)
    {
        compute_strides(shape, l, m_strides);
        init_tensor(shape, m_strides);
        std::fill(m_data.begin(), m_data.end(), value);
    }

    /**
     * Allocates an uninitialized pytensor with the specified shape and strides.
     * Elements are initialized to the specified value.
     * @param shape the shape of the pytensor
     * @param strides the strides of the pytensor
     * @param value the value of the elements
     */
    template <class T, std::size_t N>
    inline pytensor<T, N>::pytensor(const shape_type& shape,
                                    const strides_type& strides,
                                    const_reference value)
    {
        init_tensor(shape, strides);
        std::fill(m_data.begin(), m_data.end(), value);
    }

    /**
     * Allocates an uninitialized pytensor with the specified shape and strides.
     * @param shape the shape of the pytensor
     * @param strides the strides of the pytensor
     */
    template <class T, std::size_t N>
    inline pytensor<T, N>::pytensor(const shape_type& shape,
                                    const strides_type& strides)
    {
        init_tensor(shape, strides);
    }
    //@}

    /**
     * @name Copy semantic
     */
    //@{
    /**
     * The copy constructor.
     */
    template <class T, std::size_t N>
    inline pytensor<T, N>::pytensor(const self_type& rhs)
        : base_type(), semantic_base()
    {
        init_tensor(rhs.shape(), rhs.strides());
        std::copy(rhs.data().cbegin(), rhs.data().cend(), this->data().begin());
    }

    /**
     * The assignment operator.
     */
    template <class T, std::size_t N>
    inline auto pytensor<T, N>::operator=(const self_type& rhs) -> self_type&
    {
        self_type tmp(rhs);
        *this = std::move(tmp);
        return *this;
    }
    //@}

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended copy constructor.
     */
    template <class T, std::size_t N>
    template <class E>
    inline pytensor<T, N>::pytensor(const xexpression<E>& e)
        : base_type()
    {
        shape_type shape = xtl::forward_sequence<shape_type>(e.derived_cast().shape());
        strides_type strides = xtl::make_sequence<strides_type>(N, size_type(0));
        compute_strides(shape, layout_type::row_major, strides);
        init_tensor(shape, strides);
        semantic_base::assign(e);
    }

    /**
     * The extended assignment operator.
     */
    template <class T, std::size_t N>
    template <class E>
    inline auto pytensor<T, N>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

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
                        type_num, python_strides, nullptr, sizeof(T), flags, nullptr));

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
                                this->get_min_stride() * static_cast<size_type>(PyArray_SIZE(this->python_array())));
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
