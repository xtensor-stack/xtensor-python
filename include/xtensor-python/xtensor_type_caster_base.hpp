/***************************************************************************
* Copyright (c) Wolf Vollprecht, Johan Mabille and Sylvain Corlay          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_TYPE_CASTER_HPP
#define XTENSOR_TYPE_CASTER_HPP

#include <cstddef>
#include <algorithm>
#include <vector>

#include "xtensor/xtensor.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace pybind11
{
    namespace detail
    {
        // Casts an xtensor (or xarray) type to numpy array.If given a base,
        // the numpy array references the src data, otherwise it'll make a copy.
        // The writeable attributes lets you specify writeable flag for the array.
        template <typename Type>
        handle xtensor_array_cast(const Type& src, handle base = handle(), bool writeable = true)
        {
            // TODO: make use of xt::pyarray instead of array.
            std::vector<std::size_t> python_strides(src.strides().size());
            std::transform(src.strides().begin(), src.strides().end(),
                           python_strides.begin(), [](auto v) {
                return sizeof(typename Type::value_type) * v;
            });

            std::vector<std::size_t> python_shape(src.shape().size());
            std::copy(src.shape().begin(), src.shape().end(), python_shape.begin());

            array a(python_shape, python_strides, src.begin(), base);

            if (!writeable)
            {
                array_proxy(a.ptr())->flags &= ~detail::npy_api::NPY_ARRAY_WRITEABLE_;
            }

            return a.release();
        }

        // Takes an lvalue ref to some xtensor (or xarray) type and a (python) base object, creating a numpy array that
        // reference the xtensor object's data with `base` as the python-registered base class (if omitted,
        // the base will be set to None, and lifetime management is up to the caller).  The numpy array is
        // non-writeable if the given type is const.
        template <typename Type, typename CType>
        handle xtensor_ref_array(CType& src, handle parent = none())
        {
            return xtensor_array_cast<Type>(src, parent, !std::is_const<CType>::value);
        }

        // Takes a pointer to xtensor (or xarray), builds a capsule around it, then returns a numpy
        // array that references the encapsulated data with a python-side reference to the capsule to tie
        // its destruction to that of any dependent python objects.  Const-ness is determined by whether or
        // not the CType of the pointer given is const.
        template <typename Type, typename CType>
        handle xtensor_encapsulate(CType* src)
        {
            capsule base(src, [](void* o) { delete static_cast<CType*>(o); });
            return xtensor_ref_array<Type>(*src, base);
        }

        // Base class of type_caster for xtensor and xarray
        template <class Type>
        struct xtensor_type_caster_base
        {
            bool load(handle src, bool)
            {
                return false;
            }

        private:

            // Cast implementation
            template <typename CType>
            static handle cast_impl(CType* src, return_value_policy policy, handle parent)
            {
                switch (policy)
                {
                case return_value_policy::take_ownership:
                case return_value_policy::automatic:
                    return xtensor_encapsulate<Type>(src);
                case return_value_policy::move:
                    return xtensor_encapsulate<Type>(new CType(std::move(*src)));
                case return_value_policy::copy:
                    return xtensor_array_cast<Type>(*src);
                case return_value_policy::reference:
                case return_value_policy::automatic_reference:
                    return xtensor_ref_array<Type>(*src);
                case return_value_policy::reference_internal:
                    return xtensor_ref_array<Type>(*src, parent);
                default:
                    throw cast_error("unhandled return_value_policy: should not happen!");
                };
            }

        public:

            // Normal returned non-reference, non-const value:
            static handle cast(Type&& src, return_value_policy /* policy */, handle parent)
            {
                return cast_impl(&src, return_value_policy::move, parent);
            }

            // If you return a non-reference const, we mark the numpy array readonly:
            static handle cast(const Type&& src, return_value_policy /* policy */, handle parent)
            {
                return cast_impl(&src, return_value_policy::move, parent);
            }

            // lvalue reference return; default (automatic) becomes copy
            static handle cast(Type& src, return_value_policy policy, handle parent)
            {
                if (policy == return_value_policy::automatic || policy == return_value_policy::automatic_reference)
                {
                    policy = return_value_policy::copy;
                }

                return cast_impl(&src, policy, parent);
            }

            // const lvalue reference return; default (automatic) becomes copy
            static handle cast(const Type& src, return_value_policy policy, handle parent)
            {
                if (policy == return_value_policy::automatic || policy == return_value_policy::automatic_reference)
                {
                    policy = return_value_policy::copy;
                }

                return cast(&src, policy, parent);
            }

            // non-const pointer return
            static handle cast(Type* src, return_value_policy policy, handle parent)
            {
                return cast_impl(src, policy, parent);
            }

            // const pointer return
            static handle cast(const Type* src, return_value_policy policy, handle parent)
            {
                return cast_impl(src, policy, parent);
            }

#ifdef PYBIND11_DESCR // The macro is removed from pybind11 since 2.3
            static PYBIND11_DESCR name()
            {
                return _("xt::xtensor");
            }
#else
            static constexpr auto name = _("xt::xtensor");
#endif

            template <typename T>
            using cast_op_type = cast_op_type<T>;
        };
    }
}

#endif
