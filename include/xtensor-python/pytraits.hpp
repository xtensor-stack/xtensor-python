#ifndef PY_TRAITS_HPP
#define PY_TRAITS_HPP

#include <pybind11/pybind11.h>

namespace pybind11
{
  namespace detail
  {
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

      static constexpr int type_num = value_list[pybind11::detail::is_fmt_numeric<value_type>::index];
    };

  }
}

#endif
