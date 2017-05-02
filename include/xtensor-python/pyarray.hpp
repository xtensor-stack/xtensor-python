/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef PY_ARRAY_HPP
#define PY_ARRAY_HPP

#include <cstddef>
#include <vector>
#include <algorithm>
#include "xtensor/xsemantic.hpp"
#include "xtensor/xiterator.hpp"

#include "pycontainer.hpp"
#include "pybuffer_adaptor.hpp"

namespace xt
{
    template <class T>
    class pyarray;
}

namespace pybind11
{
    namespace detail
    {
        template <class T>
        struct handle_type_name<xt::pyarray<T>>
        {
            static PYBIND11_DESCR name()
            {
                return _("numpy.ndarray[") + make_caster<T>::name() + _("]");
            }
        };

        template <typename T>
        struct pyobject_caster<xt::pyarray<T>>
        {
            using type = xt::pyarray<T>;

            bool load(handle src, bool convert)
            {
                if (!convert)
                {
                    if (!PyArray_Check(src.ptr()))
                    {
                        return false;
                    }
                    int type_num = xt::detail::numpy_traits<T>::type_num;
                    if (PyArray_TYPE(reinterpret_cast<PyArrayObject*>(src.ptr())) != type_num)
                    {
                        return false;
                    }
                }
                value = type::ensure(src);
                return static_cast<bool>(value);
            }

            static handle cast(const handle &src, return_value_policy, handle)
            {
                return src.inc_ref();
            }

            PYBIND11_TYPE_CASTER(type, handle_type_name<type>::name());
        };
    }
}

namespace xt
{

    template <class A>
    class pyarray_backstrides
    {
    public:

        using array_type = A;
        using value_type = typename array_type::size_type;
        using size_type = typename array_type::size_type;

        pyarray_backstrides() = default;
        pyarray_backstrides(const array_type& a);

        value_type operator[](size_type i) const;

        size_type size() const;

    private:

        const array_type* p_a;
    };

    template <class T>
    struct xiterable_inner_types<pyarray<T>>
        : xcontainer_iterable_types<pyarray<T>>
    {
    };

    template <class T>
    struct xcontainer_inner_types<pyarray<T>>
    {
        using container_type = pybuffer_adaptor<T>;
        using shape_type = std::vector<typename container_type::size_type>;
        using strides_type = shape_type;
        using backstrides_type = pyarray_backstrides<pyarray<T>>;
        using inner_shape_type = pybuffer_adaptor<std::size_t>;
        using inner_strides_type = pystrides_adaptor<sizeof(T)>;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = pyarray<T>;
    };

    /**
     * @class pyarray
     * @brief Multidimensional container providing the xtensor container semantics to a numpy array.
     *
     * pyarray is similar to the xarray container in that it has a dynamic dimensionality. Reshapes of
     * a pyarray container are reflected in the underlying numpy array.
     *
     * @tparam T The type of the element stored in the pyarray.
     * @sa pytensor
     */
    template <class T>
    class pyarray : public pycontainer<pyarray<T>>,
                    public xcontainer_semantic<pyarray<T>>
    {
    public:

        using self_type = pyarray<T>;
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

        pyarray();
        pyarray(const value_type& t);
        pyarray(nested_initializer_list_t<T, 1> t);
        pyarray(nested_initializer_list_t<T, 2> t);
        pyarray(nested_initializer_list_t<T, 3> t);
        pyarray(nested_initializer_list_t<T, 4> t);
        pyarray(nested_initializer_list_t<T, 5> t);

        pyarray(pybind11::handle h, pybind11::object::borrowed_t);
        pyarray(pybind11::handle h, pybind11::object::stolen_t);
        pyarray(const pybind11::object &o);
        
        explicit pyarray(const shape_type& shape, layout_type l = layout_type::row_major);
        explicit pyarray(const shape_type& shape, const_reference value, layout_type l = layout_type::row_major);
        explicit pyarray(const shape_type& shape, const strides_type& strides, const_reference value);
        explicit pyarray(const shape_type& shape, const strides_type& strides);

        pyarray(const self_type& rhs);
        self_type& operator=(const self_type& rhs);

        pyarray(self_type&&) = default;
        self_type& operator=(self_type&& e) = default;

        template <class E>
        pyarray(const xexpression<E>& e);

        template <class E>
        self_type& operator=(const xexpression<E>& e);

        using base_type::begin;
        using base_type::end;

        static self_type ensure(pybind11::handle h);
        static bool check_(pybind11::handle h);

    private:

        inner_shape_type m_shape;
        inner_strides_type m_strides;
        mutable inner_backstrides_type m_backstrides;
        container_type m_data;

        void init_array(const shape_type& shape, const strides_type& strides);
        void init_from_python();

        const inner_shape_type& shape_impl() const noexcept;
        const inner_strides_type& strides_impl() const noexcept;
        const inner_backstrides_type& backstrides_impl() const noexcept;

        container_type& data_impl() noexcept;
        const container_type& data_impl() const noexcept;

        friend class xcontainer<pyarray<T>>;
    };
    
    /**************************************
     * pyarray_backstrides implementation *
     **************************************/

    template <class A>
    inline pyarray_backstrides<A>::pyarray_backstrides(const array_type& a)
        : p_a(&a)
    {
    }

    template <class A>
    inline auto pyarray_backstrides<A>::size() const -> size_type
    {
        return p_a->dimension();
    }

    template <class A>
    inline auto pyarray_backstrides<A>::operator[](size_type i) const -> value_type
    {
        value_type sh = p_a->shape()[i];
        value_type res = sh == 1 ? 0 : (sh - 1) * p_a->strides()[i];
        return  res;
    }

    /**************************
     * pyarray implementation *
     **************************/

    /**
     * @name Constructors
     */
    //@{
    template <class T>
    inline pyarray<T>::pyarray()
        : base_type()
    {
        // TODO: avoid allocation
        shape_type shape = make_sequence<shape_type>(0, size_type(1));
        strides_type strides = make_sequence<strides_type>(0, size_type(0));
        init_array(shape, strides);
        m_data[0] = T();
    }

    /**
     * Allocates a pyarray with nested initializer lists.
     */
    template <class T>
    inline pyarray<T>::pyarray(const value_type& t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t), layout_type::row_major);
        nested_copy(m_data.begin(), t);
    }

    template <class T>
    inline pyarray<T>::pyarray(nested_initializer_list_t<T, 1> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t), layout_type::row_major);
        nested_copy(m_data.begin(), t);
    }

    template <class T>
    inline pyarray<T>::pyarray(nested_initializer_list_t<T, 2> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t), layout_type::row_major);
        nested_copy(m_data.begin(), t);
    }

    template <class T>
    inline pyarray<T>::pyarray(nested_initializer_list_t<T, 3> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t), layout_type::row_major);
        nested_copy(m_data.begin(), t);
    }

    template <class T>
    inline pyarray<T>::pyarray(nested_initializer_list_t<T, 4> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t), layout_type::row_major);
        nested_copy(m_data.begin(), t);
    }

    template <class T>
    inline pyarray<T>::pyarray(nested_initializer_list_t<T, 5> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t), layout_type::row_major);
        nested_copy(m_data.begin(), t);
    }

    template <class T>
    inline pyarray<T>::pyarray(pybind11::handle h, pybind11::object::borrowed_t b)
        : base_type(h, b)
    {
        init_from_python();
    }

    template <class T>
    inline pyarray<T>::pyarray(pybind11::handle h, pybind11::object::stolen_t s)
        : base_type(h, s)
    {
        init_from_python();
    }

    template <class T>
    inline pyarray<T>::pyarray(const pybind11::object &o)
        : base_type(o)
    {
        init_from_python();
    }

    /**
     * Allocates an uninitialized pyarray with the specified shape and
     * layout.
     * @param shape the shape of the pyarray
     * @param l the layout of the pyarray
     */
    template <class T>
    inline pyarray<T>::pyarray(const shape_type& shape, layout_type l)
        : base_type()
    {
        strides_type strides(shape.size());
        compute_strides(shape, l, strides);
        init_array(shape, strides);
    }

    /**
     * Allocates a pyarray with the specified shape and layout. Elements
     * are initialized to the specified value.
     * @param shape the shape of the pyarray
     * @param value the value of the elements
     * @param l the layout of the pyarray
     */
    template <class T>
    inline pyarray<T>::pyarray(const shape_type& shape, const_reference value, layout_type l)
        : base_type()
    {
        strides_type strides(shape.size());
        compute_strides(shape, l, strides);
        init_array(shape, strides);
        std::fill(m_data.begin(), m_data.end(), value);
    }
    
    /**
     * Allocates an uninitialized pyarray with the specified shape and strides.
     * Elements are initialized to the specified value.
     * @param shape the shape of the pyarray
     * @param strides the strides of the pyarray
     * @param value the value of the elements
     */
    template <class T>
    inline pyarray<T>::pyarray(const shape_type& shape, const strides_type& strides, const_reference value)
        : base_type()
    {
        init_array(shape, strides);
        std::fill(m_data.begin(), m_data.end(), value);
    }

    /**
     * Allocates an uninitialized pyarray with the specified shape and strides.
     * @param shape the shape of the pyarray
     * @param strides the strides of the pyarray
     */
    template <class T>
    inline pyarray<T>::pyarray(const shape_type& shape, const strides_type& strides)
        : base_type()
    {
        init_array(shape, strides);
    }
    //@}

    /**
     * @name Copy semantic
     */
    //@{
    /**
     * The copy constructor.
     */
    template <class T>
    inline pyarray<T>::pyarray(const self_type& rhs)
        : base_type()
    {
        auto tmp = pybind11::reinterpret_steal<pybind11::object>(
            PyArray_NewLikeArray(rhs.python_array(), NPY_KEEPORDER, nullptr, 1)
            );

        if (!tmp)
        {
            throw std::runtime_error("NumPy: unable to create ndarray");
        }

        this->m_ptr = tmp.release().ptr();
        init_from_python();
        std::copy(rhs.data().begin(), rhs.data().end(), this->data().begin());
    }

    /**
     * The assignment operator.
     */
    template <class T>
    inline auto pyarray<T>::operator=(const self_type& rhs) -> self_type&
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
    template <class T>
    template <class E>
    inline pyarray<T>::pyarray(const xexpression<E>& e)
        : base_type()
    {
        shape_type shape = forward_sequence<shape_type>(e.derived_cast().shape());
        strides_type strides = make_sequence<strides_type>(shape.size(), size_type(0));
        compute_strides(shape, layout_type::row_major, strides);
        init_array(shape, strides);
        semantic_base::assign(e);
    }

    /**
     * The extended assignment operator.
     */
    template <class T>
    template <class E>
    inline auto pyarray<T>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    template <class T>
    inline auto pyarray<T>::ensure(pybind11::handle h) -> self_type
    {
        return base_type::ensure(h);
    }

    template <class T>
    inline bool pyarray<T>::check_(pybind11::handle h)
    {
        return base_type::check_(h);
    }

    template <class T>
    inline void pyarray<T>::init_array(const shape_type& shape, const strides_type& strides)
    {
        strides_type adapted_strides(strides);

        std::transform(strides.begin(), strides.end(), adapted_strides.begin(),
                [](auto v) { return sizeof(T) * v; });

        int flags = NPY_ARRAY_ALIGNED;
        if (!std::is_const<T>::value)
        {
            flags |= NPY_ARRAY_WRITEABLE;
        }
        int type_num = detail::numpy_traits<T>::type_num;
        
        npy_intp* shape_data = reinterpret_cast<npy_intp*>(const_cast<size_type*>(shape.data()));
        npy_intp* strides_data = reinterpret_cast<npy_intp*>(adapted_strides.data());
        auto tmp = pybind11::reinterpret_steal<pybind11::object>(
                PyArray_New(&PyArray_Type, static_cast<int>(shape.size()), shape_data, type_num, strides_data,
                            nullptr, static_cast<int>(sizeof(T)), flags, nullptr)
                );
        
        if (!tmp)
        {
            throw std::runtime_error("NumPy: unable to create ndarray");
        }

        this->m_ptr = tmp.release().ptr();
        init_from_python();
    }

    template <class T>
    inline void pyarray<T>::init_from_python()
    {
        m_shape = inner_shape_type(reinterpret_cast<size_type*>(PyArray_SHAPE(this->python_array())),
                                   static_cast<size_type>(PyArray_NDIM(this->python_array())));
        m_strides = inner_strides_type(reinterpret_cast<size_type*>(PyArray_STRIDES(this->python_array())),
                                       static_cast<size_type>(PyArray_NDIM(this->python_array())));
        m_backstrides = backstrides_type(*this);
        const size_type & (*min) (const size_type&, const size_type&) = std::min<size_type>;
        size_type min_stride = std::accumulate(m_strides.cbegin(), m_strides.cend(), std::numeric_limits<size_type>::max(), min);
        m_data = container_type(reinterpret_cast<pointer>(PyArray_DATA(this->python_array())),
                                min_stride * static_cast<size_type>(PyArray_SIZE(this->python_array())));
    }

    template <class T>
    inline auto pyarray<T>::shape_impl() const noexcept -> const inner_shape_type&
    {
        return m_shape;
    }

    template <class T>
    inline auto pyarray<T>::strides_impl() const noexcept -> const inner_strides_type&
    {
        return m_strides;
    }

    template <class T>
    inline auto pyarray<T>::backstrides_impl() const noexcept -> const inner_backstrides_type&
    {
        // m_backstrides wraps the numpy array backstrides, which is a raw pointer.
        // The address of the raw pointer stored in the wrapper would be invalidated when the pyarray is copied.
        // Hence, we build a new backstrides object (cheap wrapper around the underlying pointer) upon access.
        m_backstrides = backstrides_type(*this);
        return m_backstrides;
    }

    template <class T>
    inline auto pyarray<T>::data_impl() noexcept -> container_type&
    {
        return m_data;
    }

    template <class T>
    inline auto pyarray<T>::data_impl() const noexcept -> const container_type&
    {
        return m_data;
    }
}

#endif


