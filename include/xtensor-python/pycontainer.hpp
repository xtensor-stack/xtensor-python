/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef PY_CONTAINER_HPP
#define PY_OCNTAINER_HPP

#include <functional>
#include <numeric>
#include <cmath>
#include "pybind11/pybind11.h"
#include "pybind11/common.h"
#include "xtensor/xtensor_forward.hpp"
#include "xtensor/xiterator.hpp"

namespace xt
{

    template <class D>
    class pycontainer : public pybind11::object
    {

    public:

        using derived_type = D;

        using inner_types = xcontainer_inner_types<D>;
        using container_type = typename inner_types::container_type;
        using value_type = typename container_type::value_type;
        using reference = typename container_type::reference;
        using const_reference = typename container_type::const_reference;
        using pointer = typename container_type::pointer;
        using const_pointer = typename container_type::const_pointer;
        using size_type = typename container_type::size_type;
        using difference_type = typename container_type::difference_type;

        using shape_type = typename inner_types::shape_type;
        using strides_type = typename inner_types::strides_type;
        using backstrides_type = typename inner_types::backstrides_type;

        using iterator = typename container_type::iterator;
        using const_iterator = typename container_type::const_iterator;

        using stepper = xstepper<D>;
        using const_stepper = xstepper<const D>;

        using broadcast_iterator = xiterator<stepper, shape_type*>;
        using const_broadcast_iterator = xiterator<const_stepper, shape_type*>;

        size_type size() const;
        size_type dimension() const;

        const shape_type& shape() const;
        const strides_type& strides() const;
        const backstrides_type& backstrides() const;

        template <class... Args>
        reference operator()(Args... args);

        template <class... Args>
        const_reference operator()(Args... args) const;

        reference operator[](const xindex& index);
        const_reference operator[](const xindex& index) const;

        template <class It>
        reference element(It first, It last);

        template <class It>
        const_reference element(It first, It last) const;

        container_type& data();
        const container_type& data() const;

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

        broadcast_iterator xbegin();
        broadcast_iterator xend();

        const_broadcast_iterator xbegin() const;
        const_broadcast_iterator xend() const;
        const_broadcast_iterator cxbegin() const;
        const_broadcast_iterator cxend() const;

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

        template <class S>
        stepper stepper_begin(const S& shape);
        template <class S>
        stepper stepper_end(const S& shape);

        template <class S>
        const_stepper stepper_begin(const S& shape) const;
        template <class S>
        const_stepper stepper_end(const S& shape) const;

    protected:

        pycontainer() = default;
        ~pycontainer() = default;

        pycontainer(pybind11::handle h, borrowed_t);
        pycontainer(pybind11::handle h, stolen_t);

        pycontainer(const pycontainer&) = default;
        pycontainer& operator=(const pycontainer&) = default;

        pycontainer(pycontainer&&) = default;
        pycontainer& operator=(pycontainer&&) = default;

        void fill_default_strides(const shape_type& shape,
                                  strides_type& strides);

        static PyObject* raw_array_t(PyObject* ptr);

    private:

        template <size_t dim = 0>
        size_type data_offset(const strides_type&) const;

        template <size_t dim, class... Args>
        size_type data_offset(const strides_type& strides, size_type i, Args... args) const;

        template <class It>
        size_type element_offset(const strides_type& strides, It first, It last) const;
    };

    namespace detail
    {
        // TODO : switch on pybind11::is_fmt_numeric when it is available
        template <typename T, typename SFINAE = void>
        struct is_fmt_numeric
        {
            static constexpr bool value = false;
        };
        
        constexpr int log2(size_t n, int k = 0)
        {
            return (n <= 1) ? k : log2(n >> 1, k + 1);
        } 
        
        template <typename T>
        struct is_fmt_numeric<T, std::enable_if_t<std::is_arithmetic<T>::value>>
        {
            static constexpr bool value = true;
            static constexpr int index = std::is_same<T, bool>::value ? 0 : 1 + (
                std::is_integral<T>::value ? log2(sizeof(T)) * 2 + std::is_unsigned<T>::value : 8 + (
                    std::is_same<T, double>::value ? 1 : std::is_same<T, long double>::value ? 2 : 0));
        };

        template <class T>
        struct numpy_traits
        {
        private:

            constexpr static const int value_list[15] = {
                NPY_BOOL,
                NPY_BYTE,   NPY_UBYTE,   NPY_SHORT,      NPY_USHORT,
                NPY_INT,    NPY_UINT,    NPY_LONGLONG,   NPY_ULONGLONG,
                NPY_FLOAT,  NPY_DOUBLE,  NPY_LONGDOUBLE,
                NPY_CFLOAT, NPY_CDOUBLE, NPY_CLONGDOUBLE
            };

        public:

            using value_type = std::remove_const_t<T>;

            static constexpr int type_num = value_list[is_fmt_numeric<value_type>::index];
        };
    }

    /******************************
     * pycontainer implementation *
     ******************************/

    template <class D>
    inline pycontainer<D>::pycontainer(pybind11::handle h, borrowed_t)
        : pybind11::object(h, borrowed)
    {
    }

    template <class D>
    inline pycontainer<D>::pycontainer(pybind11::handle h, stolen_t)
        : pybind11::object(h, stolen)
    {
    }
    
    template <class D>
    inline void pycontainer<D>::fill_default_strides(const shape_type& shape, strides_type& strides)
    {
        size_type data_size = 1;
        for(size_type i = strides.size(); i != 0; --i)
        {
            strides[i - 1] = data_size;
            data_size = strides[i - 1] * shape[i - 1];
        }
    }

    template <class D>
    inline PyObject* pycontainer<D>::raw_array_t(PyObject* ptr)
    {
        if(ptr == nullptr)
            return nullptr;

        int type_num = detail::numpy_traits<value_type>::type_num;
        return PyArray_FromAny(ptr, PyArray_DescrFromType(type_num), 0, 0, NPY_ARRAY_ENSUREARRAY, nullptr);
    }

    template <class D>
    template <size_t dim>
    inline auto pycontainer<D>::data_offset(const strides_type&) const -> size_type
    {
        return 0;
    }

    template <class D>
    template <size_t dim, class... Args>
    inline auto pycontainer<D>::data_offset(const strides_type& strides, size_type i, Args... args) const -> size_type
    {
        return i * strides[dim] + data_offset<dim + 1>(args...);
    }

    template <class D>
    template <class It>
    inline auto pycontainer<D>::element_offset(const strides_type& strides, It, It last) const -> size_type
    {
        It first = last;
        first -= strides.size();
        return std::inner_product(strides.begin(), strides.end(), first, size_type(0));
    }

    template <class D>
    inline auto pycontainer<D>::size() const -> size_type
    {
        return data().size();
    }

    template <class D>
    inline auto pycontainer<D>::dimension() const -> size_type
    {
        return shape().size();
    }

    template <class D>
    inline auto pycontainer<D>::shape() const -> const shape_type&
    {
        return static_cast<const derived_type*>(this)-> shape_impl();
    }

    template <class D>
    inline auto pycontainer<D>::strides() const -> const strides_type&
    {
        return static_cast<const derived_type*>(this)->strides_impl();
    }

    template <class D>
    inline auto pycontainer<D>::backstrides() const -> const backstrides_type&
    {
        return static_cast<const derived_type*>(this)->backstrides_impl();
    }

    template <class D>
    template <class... Args>
    inline auto pycontainer<D>::operator()(Args... args) -> reference
    {
        size_type index = data_offset(strides(), static_cast<size_type>(args)...);
        return data()[index];
    }

    template <class D>
    template <class... Args>
    inline auto pycontainer<D>::operator()(Args... args) const -> const_reference
    {
        size_type index = data_offset(strides(), static_cast<size_type>(args)...);
        return data()[index];
    }

    template <class D>
    inline auto pycontainer<D>::operator[](const xindex& index) -> reference
    {
        return element(index.cbegin(), index.cend());
    }

    template <class D>
    inline auto pycontainer<D>::operator[](const xindex& index) const -> const_reference
    {
        return element(index.cbegin(), index.cend());
    }

    template <class D>
    template <class It>
    inline auto pycontainer<D>::element(It first, It last) -> reference
    {
        return data()[element_offset(strides(), first, last)];
    }

    template <class D>
    template <class It>
    inline auto pycontainer<D>::element(It first, It last) const -> const_reference
    {
        return data()[element_offset(strides(), first, last)];
    }

    template <class D>
    inline auto pycontainer<D>::data() -> container_type&
    {
        return static_cast<derived_type*>(this)->data_impl();
    }

    template <class D>
    inline auto pycontainer<D>::data() const -> const container_type&
    {
        return static_cast<const derived_type*>(this)->data_impl();
    }

    template <class D>
    template <class S>
    inline bool pycontainer<D>::broadcast_shape(S& shape) const
    {
        return xt::broadcast_shape(shape(), shape);
    }

    template <class D>
    template <class S>
    inline bool pycontainer<D>::is_trivial_broadcast(const S& strides) const
    {
        const strides_type& str = strides();
        return str.size() == strides.size() &&
            std::equal(str.cbegin(), str.cend(), strides.begin());
    }

    template <class D>
    inline auto pycontainer<D>::begin() -> iterator
    {
        return data().begin();
    }

    template <class D>
    inline auto pycontainer<D>::end() -> iterator
    {
        return data().end();
    }

    template <class D>
    inline auto pycontainer<D>::begin() const -> const_iterator
    {
        return data().cbegin();
    }

    template <class D>
    inline auto pycontainer<D>::end() const -> const_iterator
    {
        return data().cend();
    }

    template <class D>
    inline auto pycontainer<D>::cbegin() const -> const_iterator
    {
        return begin();
    }

    template <class D>
    inline auto pycontainer<D>::cend() const -> const_iterator
    {
        return end();
    }

    template <class D>
    inline auto pycontainer<D>::xbegin() -> broadcast_iterator
    {
        const shape_type& shape = shape();
        return broadcast_iterator(stepper_begin(shape), shape);
    }

    template <class D>
    inline auto pycontainer<D>::xend() -> broadcast_iterator
    {
        const shape_type& shape = shape();
        return broadcast_iterator(stepper_end(shape), shape);
    }

    template <class D>
    inline auto pycontainer<D>::xbegin() const -> const_broadcast_iterator
    {
        const shape_type& shape = shape();
        return const_broadcast_iterator(stepper_begin(shape), shape);
    }

    template <class D>
    inline auto pycontainer<D>::xend() const -> const_broadcast_iterator
    {
        const shape_type& shape = shape();
        return const_broadcast_iterator(stepper_end(shape), shape);
    }

    template <class D>
    inline auto pycontainer<D>::cxbegin() const -> const_broadcast_iterator
    {
        return xbegin();
    }

    template <class D>
    inline auto pycontainer<D>::cxend() const -> const_broadcast_iterator
    {
        return xend();
    }

    template <class D>
    template <class S>
    inline auto pycontainer<D>::xbegin(const S& shape) -> xiterator<stepper, S>
    {
        return xiterator<stepper, S>(stepper_begin(shape), shape);
    }

    template <class D>
    template <class S>
    inline auto pycontainer<D>::xend(const S& shape) -> xiterator<stepper, S>
    {
        return xiterator<stepper, S>(stepper_end(shape), shape);
    }

    template <class D>
    template <class S>
    inline auto pycontainer<D>::xbegin(const S& shape) const -> xiterator<const_stepper, S>
    {
        return xiterator<const_stepper, S>(stepper_begin(shape), shape);
    }

    template <class D>
    template <class S>
    inline auto pycontainer<D>::xend(const S& shape) const -> xiterator<const_stepper, S>
    {
        return xiterator<const_stepper, S>(stepper_end(shape), shape);
    }

    template <class D>
    template <class S>
    inline auto pycontainer<D>::cxbegin(const S& shape) const -> xiterator<const_stepper, S>
    {
        return xbegin(shape);
    }

    template <class D>
    template <class S>
    inline auto pycontainer<D>::cxend(const S& shape) const -> xiterator<const_stepper, S>
    {
        return xend(shape);
    }

    template <class D>
    template <class S>
    inline auto pycontainer<D>::stepper_begin(const S& shape) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(static_cast<derived_type*>(this), data().begin(), offset);
    }

    template <class D>
    template <class S>
    inline auto pycontainer<D>::stepper_end(const S& shape) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(static_cast<derived_type*>(this), data().end(), offset);
    }

    template <class D>
    template <class S>
    inline auto pycontainer<D>::stepper_begin(const S& shape) const -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(static_cast<const derived_type*>(this), data().begin(), offset);
    }

    template <class D>
    template <class S>
    inline auto pycontainer<D>::stepper_end(const S& shape) const -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(static_cast<const derived_type*>(this), data().end(), offset);
    }

}

#endif

