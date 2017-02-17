/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef PYBUFFER_ADAPTOR_HPP
#define PYBUFFER_ADAPTOR_HPP

#include <cstddef>

namespace xt
{

    template <class T>
    class pybuffer_adaptor
    {

    public:

        using value_type = T;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        using iterator = pointer;
        using const_iterator = const_pointer;

        pybuffer_adaptor() = default;
        pybuffer_adaptor(pointer data, size_type size);
        
        bool empty() const noexcept;
        size_type size() const noexcept;
        
        reference operator[](size_type i);
        const_reference operator[](size_type i) const;

        reference front();
        const_reference front() const;

        reference back();
        const_reference back() const;

        iterator begin();
        iterator end();

        const_iterator begin() const;
        const_iterator end() const;
        const_iterator cbegin() const;
        const_iterator cend() const;

    private:

        pointer p_data;
        size_type m_size;
    };

    /***********************************
     * pybuffer_adaptor implementation *
     ***********************************/

    template <class T>
    inline pybuffer_adaptor<T>::pybuffer_adaptor(pointer data, size_type size)
        : p_data(data), m_size(size)
    {
    }

    template <class T>
    inline bool pybuffer_adaptor<T>::empty() const noexcept
    {
        return m_size == 0;
    }

    template <class T>
    inline auto pybuffer_adaptor<T>::size() const noexcept -> size_type
    {
        return m_size;
    }

    template <class T>
    inline auto pybuffer_adaptor<T>::operator[](size_type i) -> reference
    {
        return p_data[i];
    }

    template <class T>
    inline auto pybuffer_adaptor<T>::operator[](size_type i) const -> const_reference
    {
        return p_data[i];
    }

    template <class T>
    inline auto pybuffer_adaptor<T>::front() -> reference
    {
        return p_data[0];
    }

    template <class T>
    inline auto pybuffer_adaptor<T>::front() const -> const_reference
    {
        return p_data[0];
    }

    template <class T>
    inline auto pybuffer_adaptor<T>::back() -> reference
    {
        return p_data[m_size - 1];
    }

    template <class T>
    inline auto pybuffer_adaptor<T>::back() const -> const_reference
    {
        return p_data[m_size - 1];
    }

    template <class T>
    inline auto pybuffer_adaptor<T>::begin() -> iterator
    {
        return p_data;
    }

    template <class T>
    inline auto pybuffer_adaptor<T>::end() -> iterator
    {
        return p_data + m_size;
    }

    template <class T>
    inline auto pybuffer_adaptor<T>::begin() const -> const_iterator
    {
        return const_iterator(p_data);
    }

    template <class T>
    inline auto pybuffer_adaptor<T>::end() const -> const_iterator
    {
        return const_iterator(p_data + m_size);
    }

    template <class T>
    inline auto pybuffer_adaptor<T>::cbegin() const -> const_iterator
    {
        return begin();
    }

    template <class T>
    inline auto pybuffer_adaptor<T>::cend() const -> const_iterator
    {
        return end();
    }
}

#endif

