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

#ifndef PyArray_GET_
#define PyArray_GET_(ptr, attr) \
        (reinterpret_cast<::pybind11::backport::PyArray_Proxy*>(ptr)->attr)
#endif
#ifndef PyArrayDescr_GET_
#define PyArrayDescr_GET_(ptr, attr) \
        (reinterpret_cast<::pybind11::backport::PyArrayDescr_Proxy*>(ptr)->attr)
#endif

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
                if (base) throw std::runtime_error("array base is not supported yet");
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
                return std::accumulate(shape(), shape() + ndim(), size_type{1}, std::multiplies<size_type>());
            }

            size_type itemsize() const
            {
                return static_cast<size_type>(PyArrayDescr_GET_(PyArray_GET_(m_ptr, descr), elsize));
            }

            size_type ndim() const
            {
                return static_cast<size_type>(PyArray_GET_(m_ptr, nd));
            }

            const size_type* shape() const
            {
                return reinterpret_cast<const size_type*>(PyArray_GET_(m_ptr, dimensions));
            }

            const size_type* strides() const
            {
                return reinterpret_cast<const size_type*>(PyArray_GET_(m_ptr, strides));
            }

            template<typename... Ix>
            void* data()
            {
                return static_cast<void*>(PyArray_GET_(m_ptr, data));
            }

            template<typename... Ix>
            void* mutable_data()
            {
                // check_writeable();
                return static_cast<void *>(PyArray_GET_(m_ptr, data));
            }

            template<typename... Ix>
            size_type offset_at(Ix... index) const
            {
                if (sizeof...(index) > ndim())
                {
                    fail_dim_check(sizeof...(index), "too many indices for an array");
                }
                return get_byte_offset(index...);
            }

            size_type offset_at() const
            {
                return 0;
            }

        protected:

            void fail_dim_check(size_type dim, const std::string& msg) const
            {
                throw index_error(msg + ": " + std::to_string(dim) +
                                  " (ndim = " + std::to_string(ndim()) + ")");
            }

            template<typename... Ix>
            size_type get_byte_offset(Ix... index) const
            {
                const size_type idx[] = { static_cast<size_type>(index)... };
                if (!std::equal(idx + 0, idx + sizeof...(index), shape(), std::less<size_type>{}))
                {
                    auto mismatch = std::mismatch(idx + 0, idx + sizeof...(index), shape(), std::less<size_type>{});
                    throw index_error(std::string("index ") + std::to_string(*mismatch.first) +
                                      " is out of bounds for axis " + std::to_string(mismatch.first - idx) +
                                      " with size " + std::to_string(*mismatch.second));
                }
                return std::inner_product(idx + 0, idx + sizeof...(index), strides(), size_type{0});
            }
            
            size_type get_byte_offset() const
            {
                return 0;
            }
            
            static std::vector<size_type>
            default_strides(const std::vector<size_type>& shape, size_type itemsize)
            {
                auto ndim = shape.size();
                std::vector<size_type> strides(ndim);
                if (ndim)
                {
                    std::fill(strides.begin(), strides.end(), itemsize);
                    for (size_type i = 0; i < ndim - 1; i++)
                        for (size_type j = 0; j < ndim - 1 - i; j++)
                            strides[j] *= shape[ndim - 1 - i];
                }
                return strides;
            }
        };
    }
}

#endif
