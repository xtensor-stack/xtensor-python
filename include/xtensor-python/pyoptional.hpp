/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef PY_OPTIONAL_HPP
#define PY_OPTIONAL_HPP

#include "xtensor/xbuffer_adaptor.hpp"
#include "xtensor/xoptional_assembly.hpp"
#include "xtensor/xiterator.hpp"
#include "xtensor/xsemantic.hpp"

#include "pycontainer.hpp"
#include "pystrides_adaptor.hpp"
#include "xtensor_type_caster_base.hpp"

namespace xt
{
    template <class T>
    class pyoptional_array;
}

namespace pybind11
{
    namespace detail
    {
        template <class T>
        struct handle_type_name<xt::pyoptional_array<T>>
        {
            static PYBIND11_DESCR name()
            {
                return _("numpy.ma.MaskedArray[") + make_caster<T>::name() + _("]");
            }
        };

        template <class T>
        struct pyobject_caster<xt::pyoptional_array<T>>
        {
            using type = xt::pyoptional_array<T>;

            bool load(handle src, bool convert)
            {
                value = reinterpret_borrow<type>(src);
                return static_cast<bool>(value);
            }

            static handle cast(const handle& src, return_value_policy, handle)
            {
                return src.inc_ref();
            }

            PYBIND11_TYPE_CASTER(type, handle_type_name<type>::name());
        };
    }
}

namespace xt
{
    template <class T>
    using pyoptional_array_data_type = pyarray<T>;

    using pyoptional_array_mask_type = pyarray<bool>;

    template <class T>
    class pyoptional_array : public pybind11::object,
                             public xoptional_assembly<pyoptional_array_data_type<T>,
                                                       pyoptional_array_mask_type, false>
    {
    public:

        using self_type = pyoptional_array<T>;
        using base_type = xoptional_assembly<pyoptional_array_data_type<T>,
                                             pyoptional_array_mask_type, false>;

        using pydata_type = xt::pyoptional_array_data_type<T>;
        using pymask_type = xt::pyoptional_array_mask_type;

        using shape_type = typename pydata_type::shape_type;
        using inner_shape_type = typename pydata_type::inner_shape_type;

        pyoptional_array();
        pyoptional_array(pybind11::handle h, pybind11::object::borrowed_t b);

        self_type& operator=(const self_type& rhs) = default;
        self_type& operator=(self_type&& rhs) = default;
        pyoptional_array(const self_type& rhs) = default;

        template <class ST = shape_type>
        void resize(const ST& shape, layout_type l = DEFAULT_LAYOUT);

        template <class ST = shape_type>
        void reshape(const ST &shape);

        using base_type::base_type;
        using base_type::operator();
        using base_type::operator[];
        using base_type::begin;
        using base_type::end;
    };

    template <class T>
    pyoptional_array<T>::pyoptional_array()
        : pybind11::object(pybind11::module::import("numpy.ma").attr("MaskedArray")(pybind11::list(), pybind11::list()), pybind11::object::borrowed_t()),
          base_type(pydata_type(attr("_data"), pybind11::object::borrowed_t()),
                    pymask_type(attr("_mask"), pybind11::object::borrowed_t()))
    {
    }

    template <class T>
    pyoptional_array<T>::pyoptional_array(pybind11::handle h, pybind11::object::borrowed_t b)
        : pybind11::object(h, b),
          base_type(pydata_type(attr("_data"), b),
                    pymask_type(attr("_mask"), b))
    {
    }

    template <class T>
    template <class ST>
    void pyoptional_array<T>::resize(const ST& shape, layout_type l)
    {
        base_type::value().resize(shape, l);
        base_type::flag().resize(shape, l);
        std::fill(base_type::flag().storage_begin(), base_type::flag().storage_end(), 0);

        pybind11::object tmp(pybind11::module::import("numpy.ma").attr("MaskedArray")(base_type::value(), base_type::flag()));
        *static_cast<pybind11::object *>(this) = std::move(tmp);
    }

    template <class T>
    template <class ST>
    void pyoptional_array<T>::reshape(const ST& shape)
    {
        base_type::value().reshape(shape);
        base_type::flag().reshape(shape);
    }
}

#endif