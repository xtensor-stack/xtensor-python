#ifndef PYALLOCATOR_HPP
#define PYALLOCATOR_HPP

#include <memory>

#include "Python.h"

namespace xt
{
    template<typename T>
    struct pyallocator
    {
        using value_type = T;

        static constexpr bool is_always_equal = true;

        pyallocator() = default;

        template<typename U>
        constexpr pyallocator(const pyallocator<U>&) {}

        T* allocate(std::size_t n) const {
            T* buffer = static_cast<T*>(PyMem_Malloc(n * sizeof(T)));
            if (!buffer) {
                throw std::bad_alloc{};
            }
            return buffer;
        }

        void deallocate(T* p, std::size_t) {
            PyMem_Free(p);
        }

        constexpr inline bool operator==(const pyallocator&) {
            return true;
        }

        constexpr inline bool operator!=(const pyallocator&) {
            return false;
        }

        template<typename U>
        struct rebind {
            using other = pyallocator<U>;
        };
    };

}

#endif
