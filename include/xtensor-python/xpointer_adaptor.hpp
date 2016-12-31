/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XPOINTER_ADAPTOR_HPP
#define XPOINTER_ADAPTOR_HPP

#include <iterator>
#include <algorithm>
#include <memory>

namespace xt
{

    template <class T, class A = std::allocator<T>>
    class xpointer_adaptor
    {

    public:

        using allocator_type = A;
        using value_type = typename allocator_type::value_type;
        using reference = typename allocator_type::reference;
        using const_reference = typename allocator_type::const_reference;
        using pointer = typename allocator_type::pointer;
        using const_pointer = typename allocator_type::const_pointer;
        using size_type = typename allocator_type::size_type;
        using difference_type = typename allocator_type::difference_type;

        using iterator = pointer;
        using const_iterator = const_pointer;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        xpointer_adaptor(pointer& data, size_type size,
                         const allocator_type& alloc = allocator_type());
        ~xpointer_adaptor() = default;

        xpointer_adaptor(const xpointer_adaptor&) = default;
        xpointer_adaptor& operator=(const xpointer_adaptor&);

        xpointer_adaptor(xpointer_adaptor&&) = default;
        xpointer_adaptor& operator=(xpointer_adaptor&&);

        bool empty() const noexcept;
        size_type size() const noexcept;

        void resize(size_type count);
        void resize(size_type count, const_reference value);

        reference operator[](size_type i);
        const_reference operator[](size_type i) const;

        reference front();
        const_reference front() const;

        reference back();
        const_reference back() const;

        pointer data() noexcept;
        const_pointer data() const noexcept;

        iterator begin() noexcept;
        iterator end() noexcept;

        const_iterator begin() const noexcept;
        const_iterator end() const noexcept;

        const_iterator cbegin() const noexcept;
        const_iterator cend() const noexcept;

        reverse_iterator rbegin() noexcept;
        reverse_iterator rend() noexcept;

        const_reverse_iterator rbegin() const noexcept;
        const_reverse_iterator rend() const noexcept;

        const_reverse_iterator crbegin() const noexcept;
        const_reverse_iterator crend() const noexcept;

    private:

        void resize_impl(size_type count);
        void swap_and_destroy(pointer& old_data, size_type& old_size,
                              pointer& new_data, size_type& new_size);

        pointer& p_data;
        size_type m_size;
        allocator_type m_allocator;
    };

    template <class T, class A>
    inline xpointer_adaptor<T, A>::xpointer_adaptor(pointer& data, size_type size,
                                                    const allocator_type& alloc)
        : p_data(data), m_size(size), m_allocator(alloc)
    {
    }

    template <class T, class A>
    inline xpointer_adaptor<T, A>&
    xpointer_adaptor<T, A>::operator=(const xpointer_adaptor& rhs)
    {
        size_type new_size = rhs.size();
        pointer new_data = m_allocator.allocate(new_size);
        std::copy(rhs.begin(), rhs.end(), new_data);
        swap_and_destroy(p_data, m_size, new_data, new_size);
        return *this;
    }

    template <class T, class A>
    inline xpointer_adaptor<T, A>&
    xpointer_adaptor<T, A>::operator=(xpointer_adaptor&& rhs)
    {
        std::swap(p_data, rhs.p_data);
        std::swap(m_size, rhs.m_size);
        return *this;
    }

    template <class T, class A>
    inline bool xpointer_adaptor<T, A>::empty() const noexcept
    {
        return size() == 0;
    }

    template <class T, class A>
    inline auto xpointer_adaptor<T, A>::size() const noexcept -> size_type
    {
        return m_size;
    }

    template <class T, class A>
    inline void xpointer_adaptor<T, A>::resize(size_type count)
    {
        resize_impl(count);
    }

    template <class T, class A>
    inline void xpointer_adaptor<T, A>::resize(size_type count, const_reference value)
    {
        size_type old_size = size();
        resize_impl(count);
        if (count > old_size)
        {
            std::fill(begin() + old_size, end(), value);
        }
    }

    template <class T, class A>
    inline auto xpointer_adaptor<T, A>::operator[](size_type i) -> reference
    {
        return p_data[i];
    }

    template <class T, class A>
    inline auto xpointer_adaptor<T, A>::operator[](size_type i) const -> const_reference
    {
        return p_data[i];
    }

    template <class T, class A>
    inline auto xpointer_adaptor<T, A>::front() -> reference
    {
        return p_data[0];
    }

    template <class T, class A>
    inline auto xpointer_adaptor<T, A>::front() const -> const_reference
    {
        return p_data[0];
    }

    template <class T, class A>
    inline auto xpointer_adaptor<T, A>::back() -> reference
    {
        return p_data[size() - 1];
    }

    template <class T, class A>
    inline auto xpointer_adaptor<T, A>::back() const -> const_reference
    {
        return p_data[size() - 1];
    }

    template <class T, class A>
    inline auto xpointer_adaptor<T, A>::data() noexcept -> pointer
    {
        return p_data;
    }

    template <class T, class A>
    inline auto xpointer_adaptor<T, A>::data() const noexcept -> const_pointer
    {
        return p_data;
    }

    template <class T, class A>
    inline auto xpointer_adaptor<T, A>::begin() noexcept -> iterator
    {
        return data();
    }

    template <class T, class A>
    inline auto xpointer_adaptor<T, A>::end() noexcept-> iterator
    {
        return data() + size();
    }

    template <class T, class A>
    inline auto xpointer_adaptor<T, A>::begin() const noexcept -> const_iterator
    {
        return data();
    }

    template <class T, class A>
    inline auto xpointer_adaptor<T, A>::end() const noexcept -> const_iterator
    {
        return data() + size();
    }

    template <class T, class A>
    inline auto xpointer_adaptor<T, A>::cbegin() const noexcept -> const_iterator
    {
        return begin();
    }

    template <class T, class A>
    inline auto xpointer_adaptor<T, A>::cend() const noexcept -> const_iterator
    {
        return end();
    }

    template <class T, class A>
    inline auto xpointer_adaptor<T, A>::rbegin() noexcept -> reverse_iterator
    {
        return reverse_iterator(end());
    }

    template <class T, class A>
    inline auto xpointer_adaptor<T, A>::rend() noexcept -> reverse_iterator
    {
        return reverse_iterator(begin());
    }

    template <class T, class A>
    inline auto xpointer_adaptor<T, A>::rbegin() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator(end());
    }

    template <class T, class A>
    inline auto xpointer_adaptor<T, A>::rend() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator(begin());
    }

    template <class T, class A>
    inline auto xpointer_adaptor<T, A>::crbegin() const noexcept -> const_reverse_iterator
    {
        return rbegin();
    }

    template <class T, class A>
    inline auto xpointer_adaptor<T, A>::crend() const noexcept -> const_reverse_iterator
    {
        return rend();
    }

    template <class T, class A>
    inline void xpointer_adaptor<T, A>::resize_impl(size_type count)
    {
        pointer new_data = m_allocator.allocate(count);
        size_type end = std::min(size(), count);
        std::move(begin(), begin() + end, new_data);
        swap_and_destroy(p_data, m_size, new_data, count);
    }

    template <class T, class A>
    inline void xpointer_adaptor<T, A>::swap_and_destroy(pointer& old_data, size_type& old_size,
                                                         pointer& new_data, size_type& new_size)
    {
        std::swap(old_data, new_data);
        std::swap(old_size, new_size);
        std::for_each(new_data, new_data + new_size,
            [this](T& value) { m_allocator.destroy(&value); });
        m_allocator.deallocate(new_data, new_size);
    }
}

#endif
