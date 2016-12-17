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
#include <algorithm>
#include <vector>

#include "pybind11/numpy.h"
#include "pybind11_backport.hpp"

#include "xtensor/xexpression.hpp"
#include "xtensor/xsemantic.hpp"
#include "xtensor/xiterator.hpp"

namespace xt
{

    using pybind_array = pybind11::backport::array;
    using buffer_info = pybind11::buffer_info;

    /***********************
     * pyarray declaration *
     ***********************/

    template <class T, int ExtraFlags>
    class pyarray;

    template <class T, int ExtraFlags>
    struct xcontainer_inner_types<pyarray<T, ExtraFlags>>
    {
        using temporary_type = pyarray<T, ExtraFlags>;
    };

    template <class A>
    class pyarray_backstrides
    {

    public:

        using array_type = A;
        using value_type = typename array_type::size_type;
        using size_type = typename array_type::size_type;

        pyarray_backstrides(const A& a);

        value_type operator[](size_type i) const;

    private:

        const pybind_array* p_a;
    };

    /**
     * @class pyarray
     * @brief Wrapper on the Python buffer protocol.
     */
    template <class T, int ExtraFlags = pybind_array::forcecast>
    class pyarray : public pybind_array,
                    public xcontainer_semantic<pyarray<T, ExtraFlags>>
    {

    public:

        using self_type = pyarray<T, ExtraFlags>;
        using base_type = pybind_array;
        using semantic_base = xcontainer_semantic<self_type>;
        using value_type = T;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;

        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        using storage_iterator = T*;
        using const_storage_iterator = const T*;

        using shape_type = std::vector<size_type>;
        using strides_type = std::vector<size_type>;
        using backstrides_type = pyarray_backstrides<self_type>;
        
        using stepper = xstepper<self_type>;
        using const_stepper = xstepper<const self_type>;
        using iterator = xiterator<stepper, shape_type>;
        using const_iterator = xiterator<const_stepper, shape_type>;

        using closure_type = const self_type&;

        PYBIND11_OBJECT_CVT(pyarray, pybind_array, is_non_null, m_ptr = ensure_(m_ptr));

        pyarray();

        explicit pyarray(const buffer_info& info);

        pyarray(const shape_type& shape,
                const strides_type& strides, 
                const T* ptr = nullptr,
                handle base = handle());

        explicit pyarray(const shape_type& shape, 
                         const T* ptr = nullptr,
                         handle base = handle());

        explicit pyarray(size_type count,
                         const T* ptr = nullptr,
                         handle base = handle());

        size_type dimension() const;
        const shape_type& shape() const;
        const strides_type& strides() const;
        backstrides_type backstrides() const;

        void reshape(const shape_type& shape);
        void reshape(const shape_type& shape, layout l);
        void reshape(const shape_type& shape, const strides_type& strides);

        template<typename... Args>
        reference operator()(Args... args);

        template<typename... Args>
        const_reference operator()(Args... args) const;

        reference operator[](const xindex& index);
        const_reference operator[](const xindex& index) const;

        template<typename... Args>
        pointer data(Args... args);

        template<typename... Args>
        const_pointer data(Args... args) const;

        template <class S>
        bool broadcast_shape(S& shape) const;

        template <class S>
        bool is_trivial_broadcast(const S& strides) const;

        iterator begin();
        iterator end();

        const_iterator begin() const;
        const_iterator end() const;
        const_iterator cbegin() const;
        const_iterator cend() const;

        template <class S>
        xiterator<stepper, S> xbegin(const S& shape);
        template <class S>
        xiterator<stepper, S> xend(const S& shape);
        template <class S>
        xiterator<const_stepper, S> xbegin(const S& shape) const;
        template <class S>
        xiterator<const_stepper, S> xend(const S& shape) const;
        template <class S>
        xiterator<const_stepper, S> cxbegin(const S& shape) const;
        template <class S>
        xiterator<const_stepper, S> cxend(const S& shape) const;

        stepper stepper_begin(const shape_type& shape);
        stepper stepper_end(const shape_type& shape);

        const_stepper stepper_begin(const shape_type& shape) const;
        const_stepper stepper_end(const shape_type& shape) const;

        storage_iterator storage_begin();
        storage_iterator storage_end();
        const_storage_iterator storage_begin() const;
        const_storage_iterator storage_end() const;
        const_storage_iterator storage_cbegin() const;
        const_storage_iterator storage_cend() const;

        template <class E>
        pyarray(const xexpression<E>& e);

        template <class E>
        pyarray& operator=(const xexpression<E>& e);

    private:

        template<typename... Args>
        size_type index_at(Args... args) const;

        size_type data_offset(const xindex& index) const;

        static constexpr size_type itemsize();

        static bool is_non_null(PyObject* ptr);

        static PyObject *ensure_(PyObject* ptr);

        mutable shape_type m_shape;
        mutable strides_type m_strides;

    };
    
    /**************************************
     * pyarray_backstrides implementation *
     **************************************/

    template <class A>
    inline pyarray_backstrides<A>::pyarray_backstrides(const A& a)
        : p_a(&a)
    {
    }

    template <class A>
    inline auto pyarray_backstrides<A>::operator[](size_type i) const -> value_type
    {
        value_type sh = p_a->shape()[i];
        value_type res = sh == 1 ? 0 : (sh - 1) * p_a->strides()[i] / sizeof(typename A::value_type);
        return  res;
    }

    /**************************
     * pyarray implementation *
     **************************/

    template <class T, int ExtraFlags>
    inline pyarray<T, ExtraFlags>::pyarray()
         : pybind_array()
    {
    }

    template <class T, int ExtraFlags>
    inline pyarray<T, ExtraFlags>::pyarray(const buffer_info& info)
        : pybind_array(info)
    {
    }

    template <class T, int ExtraFlags>
    inline pyarray<T, ExtraFlags>::pyarray(const shape_type& shape,
                                           const strides_type& strides, 
                                           const T *ptr,
                                           handle base)
        : pybind_array(shape, strides, ptr, base)
    {
    }

    template <class T, int ExtraFlags>
    inline pyarray<T, ExtraFlags>::pyarray(const shape_type& shape, 
                                           const T* ptr,
                                           handle base)
        : pybind_array(shape, ptr, base)
    {
    }

    template <class T, int ExtraFlags>
    inline pyarray<T, ExtraFlags>::pyarray(size_type count,
                                           const T* ptr,
                                           handle base)
        : pybind_array(count, ptr, base)
    {
    }

    template <class T, int ExtraFlags>
    inline auto pyarray<T, ExtraFlags>::dimension() const -> size_type
    {
        return pybind_array::ndim();
    }

    template <class T, int ExtraFlags>
    inline auto pyarray<T, ExtraFlags>::shape() const -> const shape_type&
    {
        // Until we have the CRTP on shape types, we copy the shape.
        m_shape.resize(dimension());
        std::copy(pybind_array::shape(), pybind_array::shape() + dimension(), m_shape.begin());
        return m_shape;
    }

    template <class T, int ExtraFlags>
    inline auto pyarray<T, ExtraFlags>::strides() const -> const strides_type&
    {
        m_strides.resize(dimension());
        std::transform(pybind_array::strides(), pybind_array::strides() + dimension(), m_strides.begin(),
            [](size_type str) { return str / sizeof(value_type); });
        return m_strides;
    }

    template <class T, int ExtraFlags>
    inline auto pyarray<T, ExtraFlags>::backstrides() const -> backstrides_type
    {
        backstrides_type tmp(*this);
        return tmp;
    }

    template <class T, int ExtraFlags>
    void pyarray<T, ExtraFlags>::reshape(const shape_type& shape)
    {
        if (!m_ptr || shape.size() != dimension() || !std::equal(shape.begin(), shape.end(), pybind_array::shape()))
        {
            reshape(shape, layout::row_major);
        }
    }

    template <class T, int ExtraFlags>
    void pyarray<T, ExtraFlags>::reshape(const shape_type& shape, layout l)
    {
        strides_type strides(shape.size());
        size_type data_size = sizeof(value_type);
        if (l == layout::row_major)
        {
            for (size_type i = strides.size(); i != 0; --i)
            {
                strides[i - 1] = data_size;
                data_size = strides[i - 1] * shape[i - 1];
                if (shape[i - 1] == 1)
                {
                    strides[i - 1] = 0;
                }
            }
        }
        else
        {
            for (size_type i = 0; i < strides.size(); ++i)
            {
                strides[i] = data_size;
                data_size = strides[i] * shape[i];
                if (shape[i] == 1)
                {
                    strides[i] = 0;
                }
            }
        }
        reshape(shape, strides);
    }

    template <class T, int ExtraFlags>
    void pyarray<T, ExtraFlags>::reshape(const shape_type& shape, const strides_type& strides)
    {
        self_type tmp(shape, strides);
        *this = std::move(tmp);
    }

    template <class T, int ExtraFlags>
    template<typename... Args> 
    inline auto pyarray<T, ExtraFlags>::operator()(Args... args) -> reference
    {
        return *(static_cast<pointer>(pybind_array::mutable_data()) + pybind_array::byte_offset(args...) / itemsize());
    }

    template <class T, int ExtraFlags>
    template<typename... Args> 
    inline auto pyarray<T, ExtraFlags>::operator()(Args... args) const -> const_reference
    {
        return *(static_cast<const_pointer>(pybind_array::data()) + pybind_array::byte_offset(args...) / itemsize());
    }

    template <class T, int ExtraFlags>
    inline auto pyarray<T, ExtraFlags>::operator[](const xindex& index) -> reference
    {
        return *(static_cast<pointer>(pybind_array::mutable_data()) + data_offset(index));
    }

    template <class T, int ExtraFlags>
    inline auto pyarray<T, ExtraFlags>::operator[](const xindex& index) const -> const_reference
    {
        return *(static_cast<const_pointer>(pybind_array::data()) + data_offset(index));
    }

    template <class T, int ExtraFlags>
    template<typename... Args> 
    inline auto pyarray<T, ExtraFlags>::data(Args... args) -> pointer
    {
        return static_cast<pointer>(pybind_array::mutable_data(args...));
    }

    template <class T, int ExtraFlags>
    template<typename... Args>
    inline auto pyarray<T, ExtraFlags>::data(Args... args) const -> const_pointer
    {
        return static_cast<const T*>(pybind_array::data(args...));
    }

    template <class T, int ExtraFlags>
    template <class S>
    bool pyarray<T, ExtraFlags>::broadcast_shape(S& shape) const
    {
        return xt::broadcast_shape(this->shape(), shape);
    }

    template <class T, int ExtraFlags>
    template <class S>
    bool pyarray<T, ExtraFlags>::is_trivial_broadcast(const S& strides) const
    {
        return strides.size() == dimension() &&
            std::equal(strides.begin(), strides.end(), this->strides().begin());
    }

    template <class T, int ExtraFlags>
    inline auto pyarray<T, ExtraFlags>::begin() -> iterator
    {
        return xbegin(shape());
    }

    template <class T, int ExtraFlags>
    inline auto pyarray<T, ExtraFlags>::end() -> iterator
    {
        return xend(shape());
    }

    template <class T, int ExtraFlags>
    inline auto pyarray<T, ExtraFlags>::begin() const -> const_iterator
    {
        return xbegin(shape());
    }

    template <class T, int ExtraFlags>
    inline auto pyarray<T, ExtraFlags>::end() const -> const_iterator
    {
        return xend(shape());
    }

    template <class T, int ExtraFlags>
    inline auto pyarray<T, ExtraFlags>::cbegin() const -> const_iterator
    {
        return begin();
    }

    template <class T, int ExtraFlags>
    inline auto pyarray<T, ExtraFlags>::cend() const -> const_iterator
    {
        return end();
    }

    template <class T, int ExtraFlags>
    template <class S>
    inline auto pyarray<T, ExtraFlags>::xbegin(const S& shape) -> xiterator<stepper, S>
    {
        return xiterator<stepper, S>(stepper_begin(shape), shape);
    }

    template <class T, int ExtraFlags>
    template <class S>
    inline auto pyarray<T, ExtraFlags>::xend(const S& shape) -> xiterator<stepper, S>
    {
        return xiterator<stepper, S>(stepper_end(shape), shape);
    }

    template <class T, int ExtraFlags>
    template <class S>
    inline auto pyarray<T, ExtraFlags>::xbegin(const S& shape) const -> xiterator<const_stepper, S>
    {
        return xiterator<const_stepper, S>(stepper_begin(shape), shape);
    }

    template <class T, int ExtraFlags>
    template <class S>
    inline auto pyarray<T, ExtraFlags>::xend(const S& shape) const -> xiterator<const_stepper, S>
    {
        return xiterator<const_stepper, S>(stepper_end(shape), shape);
    }

    template <class T, int ExtraFlags>
    template <class S>
    inline auto pyarray<T, ExtraFlags>::cxbegin(const S& shape) const -> xiterator<const_stepper, S>
    {
        return xbegin(shape);
    }

    template <class T, int ExtraFlags>
    template <class S>
    inline auto pyarray<T, ExtraFlags>::cxend(const S& shape) const -> xiterator<const_stepper, S>
    {
        return xend(shape);
    }

    template <class T, int ExtraFlags>
    inline auto pyarray<T, ExtraFlags>::stepper_begin(const shape_type& shape) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(this, storage_begin(), offset);
    }

    template <class T, int ExtraFlags>
    inline auto pyarray<T, ExtraFlags>::stepper_end(const shape_type& shape) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(this, storage_end(), offset);
    }

    template <class T, int ExtraFlags>
    inline auto pyarray<T, ExtraFlags>::stepper_begin(const shape_type& shape) const -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, storage_begin(), offset);
    }

    template <class T, int ExtraFlags>
    inline auto pyarray<T, ExtraFlags>::stepper_end(const shape_type& shape) const -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, storage_end(), offset);
    }

    template <class T, int ExtraFlags>
    inline auto pyarray<T, ExtraFlags>::storage_begin() -> storage_iterator
    {
        return reinterpret_cast<storage_iterator>(pybind11::backport::array_proxy(m_ptr)->data);
    }

    template <class T, int ExtraFlags>
    inline auto pyarray<T, ExtraFlags>::storage_end() -> storage_iterator
    {
        return storage_begin() + pybind_array::size();
    }

    template <class T, int ExtraFlags>
    inline auto pyarray<T, ExtraFlags>::storage_begin() const -> const_storage_iterator
    {
        return reinterpret_cast<const_storage_iterator>(pybind11::backport::array_proxy(m_ptr)->data);
    }

    template <class T, int ExtraFlags>
    inline auto pyarray<T, ExtraFlags>::storage_end() const -> const_storage_iterator
    {
        return storage_begin() + pybind_array::size();
    }

    template <class T, int ExtraFlags>
    inline auto pyarray<T, ExtraFlags>::storage_cbegin() const -> const_storage_iterator
    {
        return reinterpret_cast<const_storage_iterator>(pybind11::backport::array_proxy(m_ptr)->data);
    }

    template <class T, int ExtraFlags>
    inline auto pyarray<T, ExtraFlags>::storage_cend() const -> const_storage_iterator
    {
        return storage_begin() + pybind_array::size();
    }

    template <class T, int ExtraFlags>
    template <class E>
    inline pyarray<T, ExtraFlags>::pyarray(const xexpression<E>& e)
         : pybind_array()
    {
        semantic_base::assign(e);
    }

    template <class T, int ExtraFlags>
    template <class E>
    inline auto pyarray<T, ExtraFlags>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }

    // Private methods

    template <class T, int ExtraFlags>
    template<typename... Args> 
    inline auto pyarray<T, ExtraFlags>::index_at(Args... args) const -> size_type
    {
        return pybind_array::byte_offset(args...) / itemsize();
    }

    template <class T, int ExtraFlags>
    inline auto pyarray<T, ExtraFlags>::data_offset(const xindex& index) const -> size_type
    {
        const strides_type& str = strides();
        auto iter = index.begin();
        iter += index.size() - str.size();
        return std::inner_product(str.begin(), str.end(), iter, size_type(0)) / itemsize();
    }

    template <class T, int ExtraFlags>
    constexpr auto pyarray<T, ExtraFlags>::itemsize() -> size_type
    {
        return sizeof(value_type);
    }

    template <class T, int ExtraFlags>
    inline bool pyarray<T, ExtraFlags>::is_non_null(PyObject* ptr)
    {
        return ptr != nullptr;
    }

    template <class T, int ExtraFlags>
    inline PyObject* pyarray<T, ExtraFlags>::ensure_(PyObject* ptr)
    {
        if (ptr == nullptr)
        {
            return nullptr;
        }
        API& api = lookup_api();
        PyObject* descr = api.PyArray_DescrFromType_(pybind11::detail::npy_format_descriptor<T>::value);
        PyObject* result = api.PyArray_FromAny_(ptr, descr, 0, 0, API::NPY_ENSURE_ARRAY_ | ExtraFlags, nullptr);
        if (!result)
        {
            PyErr_Clear();
        }
        Py_DECREF(ptr);
        return result;
    }

}

#endif


