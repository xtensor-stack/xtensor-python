/*
    Backport from pybind11/numpy.h v2.0 prerelease.

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/
#ifndef PYBIND_BACKPORT_HPP
#define PYBIND_BACKPORT_HPP

#include <cstddef>
#include <algorithm>

#include "pybind11/numpy.h"

namespace pybind11
{
    namespace backport
    {
        struct PyArray_Proxy
        {
            PyObject_HEAD
            char* data;
            int nd;
            ssize_t* dimensions;
            ssize_t* strides;
            PyObject* base;
            PyObject* descr;
            int flags;
        };
        
        struct PyArrayDescr_Proxy
        {
            PyObject_HEAD
            PyObject* typeobj;
            char kind;
            char type;
            char byteorder;
            char flags;
            int type_num;
            int elsize;
            int alignment;
            char* subarray;
            PyObject* fields;
            PyObject* names;
        };

        inline PyArray_Proxy* array_proxy(void* ptr) {
            return reinterpret_cast<PyArray_Proxy*>(ptr);
        }

        inline const PyArray_Proxy* array_proxy(const void* ptr) {
            return reinterpret_cast<const PyArray_Proxy*>(ptr);
        }

        inline PyArrayDescr_Proxy* array_descriptor_proxy(PyObject* ptr) {
            return reinterpret_cast<PyArrayDescr_Proxy*>(ptr);
        }

        inline const PyArrayDescr_Proxy* array_descriptor_proxy(const PyObject* ptr) {
            return reinterpret_cast<const PyArrayDescr_Proxy*>(ptr);
        }

        inline bool check_flags(const void* ptr, int flag) {
            return (flag == (array_proxy(ptr)->flags & flag));
        }

        class array : public pybind11::array
        {
        public:
            using size_type = std::size_t;

            using pybind11::array::array;

            array() = default;
            
            template<typename T>
            array(const std::vector<size_type>& shape,
                  const std::vector<size_type>& strides,
                  const T* ptr, handle base = {})
                : array(buffer_info(const_cast<T*>(ptr),
                                    sizeof(T),
                                    format_descriptor<T>::value,
                                    shape.size(), shape, strides))
            {
                if (base)
                {
                    throw std::runtime_error("array base is not supported yet");
                }
            }

            template<typename T>
            array(const std::vector<size_type> &shape, 
                  const T* ptr, handle base = {})
                : array(shape, default_strides(shape, sizeof(T)), ptr, base)
            {
            }

            template<typename T>
            array(size_type size, const T *ptr, handle base) 
                : array(std::vector<size_type>{size}, ptr)
            {
            }

            size_type size() const
            {
                return std::accumulate(shape(), shape() + ndim(), size_type(1), std::multiplies<size_type>());
            }

            size_type itemsize() const
            {
                return static_cast<size_type>(array_descriptor_proxy(array_proxy(m_ptr)->descr)->elsize);
            }

            size_type ndim() const
            {
                return static_cast<size_type>(array_proxy(m_ptr)->nd);
            }

            const size_type* shape() const
            {
                return reinterpret_cast<const size_type*>(array_proxy(m_ptr)->dimensions);
            }

            const size_type* strides() const
            {
                return reinterpret_cast<const size_type*>(array_proxy(m_ptr)->strides);
            }

            void* data()
            {
                return static_cast<void*>(array_proxy(m_ptr)->data);
            }

            void* mutable_data()
            {
                return static_cast<void *>(array_proxy(m_ptr)->data);
            }

        protected:

            template<size_t dim = 0>
            inline size_type byte_offset() const
            {
                return 0;
            }

            template <size_t dim = 0, class... Args>
            inline size_type byte_offset(size_type i, Args... args) const
            {
                return i * strides()[dim] + byte_offset<dim + 1>(args...);
            }

            static std::vector<size_type>
            default_strides(const std::vector<size_type>& shape, size_type itemsize)
            {
                auto ndim = shape.size();
                std::vector<size_type> strides(ndim);
                if (ndim)
                {
                    std::fill(strides.begin(), strides.end(), itemsize);
                    for (size_type i = 0; i < ndim - 1; ++i)
                    {
                        for (size_type j = 0; j < ndim - 1 - i; ++j)
                        {
                            strides[j] *= shape[ndim - 1 - i];
                        }
                    }
                }
                return strides;
            }
        };
    }
}

#endif
