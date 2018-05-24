/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef PY_ARRAY_HPP
#define PY_ARRAY_HPP

#include <algorithm>
#include <cstddef>
#include <vector>

#include "xtensor/xbuffer_adaptor.hpp"
#include "xtensor/xiterator.hpp"
#include "xtensor/xsemantic.hpp"

#include "pycontainer.hpp"
#include "pystrides_adaptor.hpp"
#include "xtensor_type_caster_base.hpp"

namespace xt
{
    template <class T, layout_type L = layout_type::dynamic>
    class pyarray;
}

namespace pybind11
{
    namespace detail
    {
        template <class T, xt::layout_type L>
        struct handle_type_name<xt::pyarray<T, L>>
        {
            static PYBIND11_DESCR name()
            {
                return _("numpy.ndarray[") + npy_format_descriptor<T>::name() + _("]");
            }
        };

        template <typename T, xt::layout_type L>
        struct pyobject_caster<xt::pyarray<T, L>>
        {
            using type = xt::pyarray<T, L>;

            bool load(handle src, bool convert)
            {
                if (!convert)
                {
                    if (!xt::detail::check_array<T>(src))
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

        // Type caster for casting ndarray to xexpression<pyarray>
        template <typename T, xt::layout_type L>
        struct type_caster<xt::xexpression<xt::pyarray<T, L>>> : pyobject_caster<xt::pyarray<T, L>>
        {
            using Type = xt::xexpression<xt::pyarray<T, L>>;

            operator Type&()
            {
                return this->value;
            }

            operator const Type&()
            {
                return this->value;
            }
        };

        // Type caster for casting xarray to ndarray
        template <class T, xt::layout_type L>
        struct type_caster<xt::xarray<T, L>> : xtensor_type_caster_base<xt::xarray<T, L>>
        {
        };
    }
}

namespace xt
{

    /**************************
     * pybackstrides_iterator *
     **************************/

    template <class B>
    class pybackstrides_iterator
    {
    public:

        using self_type = pybackstrides_iterator<B>;

        using value_type = typename B::value_type;
        using pointer = const value_type*;
        using reference = value_type;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::random_access_iterator_tag;

        inline pybackstrides_iterator(const B* b, std::size_t offset)
            : p_b(b), m_offset(offset)
        {
        }

        inline reference operator*() const
        {
            return p_b->operator[](m_offset);
        }

        inline pointer operator->() const
        {
            // Returning the address of a temporary
            value_type res = p_b->operator[](m_offset);
            return &res;
        }

        inline reference operator[](difference_type n) const
        {
            return p_b->operator[](m_offset + n);
        }

        inline self_type& operator++()
        {
            ++m_offset;
            return *this;
        }

        inline self_type& operator--()
        {
            --m_offset;
            return *this;
        }

        inline self_type operator++(int)
        {
            self_type tmp(*this);
            ++m_offset;
            return tmp;
        }

        inline self_type operator--(int)
        {
            self_type tmp(*this);
            --m_offset;
            return tmp;
        }

        inline self_type& operator+=(difference_type n)
        {
            m_offset += n;
            return *this;
        }

        inline self_type& operator-=(difference_type n)
        {
            m_offset -= n;
            return *this;
        }

        inline self_type operator+(difference_type n) const
        {
            return self_type(p_b, m_offset + n);
        }

        inline self_type operator-(difference_type n) const
        {
            return self_type(p_b, m_offset - n);
        }

        inline self_type operator-(const self_type& rhs) const
        {
            self_type tmp(*this);
            tmp -= (m_offset - rhs.m_offset);
            return tmp;
        }

        inline std::size_t offset() const
        {
            return m_offset;
        }

    private:

        const B* p_b;
        std::size_t m_offset;
    };

    template <class B>
    inline bool operator==(const pybackstrides_iterator<B>& lhs,
                           const pybackstrides_iterator<B>& rhs)
    {
        return lhs.offset() == rhs.offset();
    }

    template <class B>
    inline bool operator!=(const pybackstrides_iterator<B>& lhs,
                           const pybackstrides_iterator<B>& rhs)
    {
        return !(lhs == rhs);
    }

    template <class B>
    inline bool operator<(const pybackstrides_iterator<B>& lhs,
                          const pybackstrides_iterator<B>& rhs)
    {
        return lhs.offset() < rhs.offset();
    }

    template <class B>
    inline bool operator<=(const pybackstrides_iterator<B>& lhs,
                           const pybackstrides_iterator<B>& rhs)
    {
        return (lhs < rhs) || (lhs == rhs);
    }

    template <class B>
    inline bool operator>(const pybackstrides_iterator<B>& lhs,
                          const pybackstrides_iterator<B>& rhs)
    {
        return !(lhs <= rhs);
    }

    template <class B>
    inline bool operator>=(const pybackstrides_iterator<B>& lhs,
                           const pybackstrides_iterator<B>& rhs)
    {
        return !(lhs < rhs);
    }

    template <class A>
    class pyarray_backstrides
    {
    public:

        using self_type = pyarray_backstrides<A>;
        using array_type = A;
        using value_type = typename array_type::size_type;
        using const_reference = value_type;
        using const_pointer = const value_type*;
        using size_type = typename array_type::size_type;
        using difference_type = typename array_type::difference_type;

        using const_iterator = pybackstrides_iterator<self_type>;

        pyarray_backstrides() = default;
        pyarray_backstrides(const array_type& a);

        bool empty() const;
        size_type size() const;

        value_type operator[](size_type i) const;

        const_reference front() const;
        const_reference back() const;

        const_iterator begin() const;
        const_iterator end() const;
        const_iterator cbegin() const;
        const_iterator cend() const;

    private:

        const array_type* p_a;
    };

    template <class T, layout_type L>
    struct xiterable_inner_types<pyarray<T, L>>
        : xcontainer_iterable_types<pyarray<T, L>>
    {
    };

    template <class T, layout_type L>
    struct xcontainer_inner_types<pyarray<T, L>>
    {
        using storage_type = xbuffer_adaptor<T*>;
        using shape_type = std::vector<typename storage_type::size_type>;
        using strides_type = shape_type;
        using backstrides_type = pyarray_backstrides<pyarray<T, L>>;
        using inner_shape_type = xbuffer_adaptor<std::size_t*>;
        using inner_strides_type = pystrides_adaptor<sizeof(T)>;
        using inner_backstrides_type = backstrides_type;
        using temporary_type = pyarray<T, L>;
        static constexpr layout_type layout = L;
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
    template <class T, layout_type L>
    class pyarray : public pycontainer<pyarray<T, L>>,
                    public xcontainer_semantic<pyarray<T, L>>
    {
    public:

        using self_type = pyarray<T, L>;
        using semantic_base = xcontainer_semantic<self_type>;
        using base_type = pycontainer<self_type>;
        using storage_type = typename base_type::storage_type;
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
        pyarray(const pybind11::object& o);

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
        storage_type m_storage;

        void init_array(const shape_type& shape, const strides_type& strides);
        void init_from_python();

        const inner_shape_type& shape_impl() const noexcept;
        const inner_strides_type& strides_impl() const noexcept;
        const inner_backstrides_type& backstrides_impl() const noexcept;

        storage_type& storage_impl() noexcept;
        const storage_type& storage_impl() const noexcept;

        friend class xcontainer<pyarray<T, L>>;
        friend class pycontainer<pyarray<T, L>>;
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
    inline bool pyarray_backstrides<A>::empty() const
    {
        return p_a->dimension() == 0;
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
        return res;
    }

    template <class A>
    inline auto pyarray_backstrides<A>::front() const -> const_reference
    {
        value_type sh = p_a->shape()[0];
        value_type res = sh == 1 ? 0 : (sh - 1) * p_a->strides()[0];
        return res;
    }

    template <class A>
    inline auto pyarray_backstrides<A>::back() const -> const_reference
    {
        auto index = p_a->size() - 1;
        value_type sh = p_a->shape()[index];
        value_type res = sh == 1 ? 0 : (sh - 1) * p_a->strides()[index];
        return res;
    }

    template <class A>
    inline auto pyarray_backstrides<A>::begin() const -> const_iterator
    {
        return cbegin();
    }

    template <class A>
    inline auto pyarray_backstrides<A>::end() const -> const_iterator
    {
        return cend();
    }

    template <class A>
    inline auto pyarray_backstrides<A>::cbegin() const -> const_iterator
    {
        return const_iterator(this, 0);
    }

    template <class A>
    inline auto pyarray_backstrides<A>::cend() const -> const_iterator
    {
        return const_iterator(this, size());
    }

    /**************************
     * pyarray implementation *
     **************************/

    /**
     * @name Constructors
     */
    //@{
    template <class T, layout_type L>
    inline pyarray<T, L>::pyarray()
        : base_type()
    {
        // TODO: avoid allocation
        shape_type shape = xtl::make_sequence<shape_type>(0, size_type(1));
        strides_type strides = xtl::make_sequence<strides_type>(0, size_type(0));
        init_array(shape, strides);
        detail::default_initialize(m_storage);
    }

    /**
     * Allocates a pyarray with nested initializer lists.
     */
    template <class T, layout_type L>
    inline pyarray<T, L>::pyarray(const value_type& t)
        : base_type()
    {
        base_type::resize(xt::shape<shape_type>(t), layout_type::row_major);
        nested_copy(m_storage.begin(), t);
    }

    template <class T, layout_type L>
    inline pyarray<T, L>::pyarray(nested_initializer_list_t<T, 1> t)
        : base_type()
    {
        base_type::resize(xt::shape<shape_type>(t), layout_type::row_major);
        nested_copy(m_storage.begin(), t);
    }

    template <class T, layout_type L>
    inline pyarray<T, L>::pyarray(nested_initializer_list_t<T, 2> t)
        : base_type()
    {
        base_type::resize(xt::shape<shape_type>(t), layout_type::row_major);
        nested_copy(m_storage.begin(), t);
    }

    template <class T, layout_type L>
    inline pyarray<T, L>::pyarray(nested_initializer_list_t<T, 3> t)
        : base_type()
    {
        base_type::resize(xt::shape<shape_type>(t), layout_type::row_major);
        nested_copy(m_storage.begin(), t);
    }

    template <class T, layout_type L>
    inline pyarray<T, L>::pyarray(nested_initializer_list_t<T, 4> t)
        : base_type()
    {
        base_type::resize(xt::shape<shape_type>(t), layout_type::row_major);
        nested_copy(m_storage.begin(), t);
    }

    template <class T, layout_type L>
    inline pyarray<T, L>::pyarray(nested_initializer_list_t<T, 5> t)
        : base_type()
    {
        base_type::resize(xt::shape<shape_type>(t), layout_type::row_major);
        nested_copy(m_storage.begin(), t);
    }

    template <class T, layout_type L>
    inline pyarray<T, L>::pyarray(pybind11::handle h, pybind11::object::borrowed_t b)
        : base_type(h, b)
    {
        init_from_python();
    }

    template <class T, layout_type L>
    inline pyarray<T, L>::pyarray(pybind11::handle h, pybind11::object::stolen_t s)
        : base_type(h, s)
    {
        init_from_python();
    }

    template <class T, layout_type L>
    inline pyarray<T, L>::pyarray(const pybind11::object& o)
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
    template <class T, layout_type L>
    inline pyarray<T, L>::pyarray(const shape_type& shape, layout_type l)
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
    template <class T, layout_type L>
    inline pyarray<T, L>::pyarray(const shape_type& shape, const_reference value, layout_type l)
        : base_type()
    {
        strides_type strides(shape.size());
        compute_strides(shape, l, strides);
        init_array(shape, strides);
        std::fill(m_storage.begin(), m_storage.end(), value);
    }

    /**
     * Allocates an uninitialized pyarray with the specified shape and strides.
     * Elements are initialized to the specified value.
     * @param shape the shape of the pyarray
     * @param strides the strides of the pyarray
     * @param value the value of the elements
     */
    template <class T, layout_type L>
    inline pyarray<T, L>::pyarray(const shape_type& shape, const strides_type& strides, const_reference value)
        : base_type()
    {
        init_array(shape, strides);
        std::fill(m_storage.begin(), m_storage.end(), value);
    }

    /**
     * Allocates an uninitialized pyarray with the specified shape and strides.
     * @param shape the shape of the pyarray
     * @param strides the strides of the pyarray
     */
    template <class T, layout_type L>
    inline pyarray<T, L>::pyarray(const shape_type& shape, const strides_type& strides)
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
    template <class T, layout_type L>
    inline pyarray<T, L>::pyarray(const self_type& rhs)
        : base_type(), semantic_base(rhs)
    {
        auto tmp = pybind11::reinterpret_steal<pybind11::object>(
            PyArray_NewLikeArray(rhs.python_array(), NPY_KEEPORDER, nullptr, 1));

        if (!tmp)
        {
            throw std::runtime_error("NumPy: unable to create ndarray");
        }

        this->m_ptr = tmp.release().ptr();
        init_from_python();
        std::copy(rhs.storage().cbegin(), rhs.storage().cend(), this->storage().begin());
    }

    /**
     * The assignment operator.
     */
    template <class T, layout_type L>
    inline auto pyarray<T, L>::operator=(const self_type& rhs) -> self_type&
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
    template <class T, layout_type L>
    template <class E>
    inline pyarray<T, L>::pyarray(const xexpression<E>& e)
        : base_type()
    {
        // TODO: prevent intermediary shape allocation
        shape_type shape = xtl::forward_sequence<shape_type>(e.derived_cast().shape());
        strides_type strides = xtl::make_sequence<strides_type>(shape.size(), size_type(0));
        compute_strides(shape, layout_type::row_major, strides);
        init_array(shape, strides);
        semantic_base::assign(e);
    }

    /**
     * The extended assignment operator.
     */
    template <class T, layout_type L>
    template <class E>
    inline auto pyarray<T, L>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    template <class T, layout_type L>
    inline auto pyarray<T, L>::ensure(pybind11::handle h) -> self_type
    {
        return base_type::ensure(h);
    }

    template <class T, layout_type L>
    inline bool pyarray<T, L>::check_(pybind11::handle h)
    {
        return base_type::check_(h);
    }

    template <class T, layout_type L>
    inline void pyarray<T, L>::init_array(const shape_type& shape, const strides_type& strides)
    {
        strides_type adapted_strides(strides);

        std::transform(strides.begin(), strides.end(), adapted_strides.begin(),
                       [](auto v) { return sizeof(T) * v; });

        int flags = NPY_ARRAY_ALIGNED;
        if (!std::is_const<T>::value)
        {
            flags |= NPY_ARRAY_WRITEABLE;
        }

        auto dtype = pybind11::detail::npy_format_descriptor<T>::dtype();

        npy_intp* shape_data = reinterpret_cast<npy_intp*>(const_cast<size_type*>(shape.data()));
        npy_intp* strides_data = reinterpret_cast<npy_intp*>(adapted_strides.data());

        auto tmp = pybind11::reinterpret_steal<pybind11::object>(
            PyArray_NewFromDescr(&PyArray_Type, (PyArray_Descr*) dtype.release().ptr(), static_cast<int>(shape.size()), shape_data, strides_data,
                        nullptr, flags, nullptr));

        if (!tmp)
        {
            throw std::runtime_error("NumPy: unable to create ndarray");
        }

        this->m_ptr = tmp.release().ptr();
        init_from_python();
    }

    template <class T, layout_type L>
    inline void pyarray<T, L>::init_from_python()
    {
        m_shape = inner_shape_type(reinterpret_cast<size_type*>(PyArray_SHAPE(this->python_array())),
                                   static_cast<size_type>(PyArray_NDIM(this->python_array())));
        m_strides = inner_strides_type(reinterpret_cast<size_type*>(PyArray_STRIDES(this->python_array())),
                                       static_cast<size_type>(PyArray_NDIM(this->python_array())));
        m_backstrides = backstrides_type(*this);
        m_storage = storage_type(reinterpret_cast<pointer>(PyArray_DATA(this->python_array())),
                                 this->get_min_stride() * static_cast<size_type>(PyArray_SIZE(this->python_array())));
    }

    template <class T, layout_type L>
    inline auto pyarray<T, L>::shape_impl() const noexcept -> const inner_shape_type&
    {
        return m_shape;
    }

    template <class T, layout_type L>
    inline auto pyarray<T, L>::strides_impl() const noexcept -> const inner_strides_type&
    {
        return m_strides;
    }

    template <class T, layout_type L>
    inline auto pyarray<T, L>::backstrides_impl() const noexcept -> const inner_backstrides_type&
    {
        // m_backstrides wraps the numpy array backstrides, which is a raw pointer.
        // The address of the raw pointer stored in the wrapper would be invalidated when the pyarray is copied.
        // Hence, we build a new backstrides object (cheap wrapper around the underlying pointer) upon access.
        m_backstrides = backstrides_type(*this);
        return m_backstrides;
    }

    template <class T, layout_type L>
    inline auto pyarray<T, L>::storage_impl() noexcept -> storage_type&
    {
        return m_storage;
    }

    template <class T, layout_type L>
    inline auto pyarray<T, L>::storage_impl() const noexcept -> const storage_type&
    {
        return m_storage;
    }
}

#endif
