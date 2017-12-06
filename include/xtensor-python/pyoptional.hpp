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
    using pyoptional_array_data_type = pyarray<T>;

    using pyoptional_array_mask_type = pyarray<bool>;

    template <class T>
    using pyoptional_array = xoptional_assembly<pyoptional_array_data_type<T>,
                                                pyoptional_array_mask_type>;
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
        struct type_caster<xt::pyoptional_array<T>> {
        public:

            bool load(const handle& src, bool)
            {
                using optional_type = xt::pyoptional_array<T>;
                using pymask_type = xt::pyoptional_array_mask_type;
                using pydata_type = xt::pyoptional_array_data_type<T>;
                value = optional_type(pydata_type(src.attr("_data")), pymask_type(src.attr("_mask")));
                return true;
            }

            static handle cast(xt::pyoptional_array<T> src, return_value_policy /* policy */, handle /* parent */) {
                // src.inc_ref();
            }

            PYBIND11_TYPE_CASTER(xt::pyoptional_array<T>, handle_type_name<T>::name());
        };
    }
}

#endif