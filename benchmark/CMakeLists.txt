############################################################################
# Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

message(STATUS "Forcing tests build type to Release")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

include(${CMAKE_CURRENT_LIST_DIR}/test/set_compiler_flag.cmake)
if(CPP17)
  # User requested C++17, but compiler might not oblige.
  set_compiler_flag(
    _cxx_std_flag CXX
    "-std=c++17"  # this should work with GNU, Intel, PGI
    "/std:c++17"  # this should work with MSVC
  )
  if(_cxx_std_flag)
    message(STATUS "Building with C++17")
  endif()
else()
  set_compiler_flag(
    _cxx_std_flag CXX REQUIRED
    "-std=c++14"  # this should work with GNU, Intel, PGI
    "/std:c++14"  # this should work with MSVC
  )
  message(STATUS "Building with C++14")
endif()

if(NOT _cxx_std_flag)
  message(FATAL_ERROR "xtensor-python needs a C++14-compliant compiler.")
endif()

# Check for Link Time Optimization support
set_compiler_flag(
  _lto_flag CXX
  "-flto"  # this works with GNU and Clang
  "-ipo"  # this works with Intel
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR (CMAKE_CXX_COMPILER_ID MATCHES "Intel" AND NOT WIN32))
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_cxx_std_flag} -march=native -Wunused-parameter -Wextra -Wreorder -Wconversion")

    # Enable link time optimization and set the default symbol
    # visibility to hidden (very important to obtain small binaries)
    if(NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
        # Default symbol visibility
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
        # Check for Link Time Optimization support
        if(_lto_flag)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_lto_flag}")
        endif()
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MP /bigobj")
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
    foreach(flag_var
            CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
            CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        string(REPLACE "/MD" "-MT" ${flag_var} "${${flag_var}}")
    endforeach()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if(NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_cxx_std_flag} -march=native -Wunused-parameter -Wextra -Wreorder -Wconversion")
    if(NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
        # Default symbol visibility
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
        # Check for Link Time Optimization support
        if(_lto_flag)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_lto_flag}")
        endif()
    endif()
  else() # We are using clang-cl
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_cxx_std_flag} /EHsc /MP /bigobj")
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
    foreach(flag_var
            CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
            CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        string(REPLACE "/MD" "-MT" ${flag_var} "${${flag_var}}")
    endforeach()
  endif()
else()
  message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

set(XTENSOR_PYTHON_BENCHMARK
    main.cpp
)

set(XTENSOR_PYTHON_BENCHMARK_TARGET benchmark_xtensor_python)
add_library(${XTENSOR_PYTHON_BENCHMARK_TARGET} MODULE EXCLUDE_FROM_ALL
            ${XTENSOR_PYTHON_BENCHMARK} ${XTENSOR_PYTHON_HEADERS})

set_target_properties(${XTENSOR_PYTHON_BENCHMARK_TARGET} PROPERTIES PREFIX "")
set_target_properties(${XTENSOR_PYTHON_BENCHMARK_TARGET} PROPERTIES SUFFIX "${PYTHON_MODULE_EXTENSION}")

if (APPLE)
    target_link_libraries(${XTENSOR_PYTHON_BENCHMARK_TARGET} PRIVATE "-undefined dynamic_lookup")
elseif (MSVC)
    target_link_libraries(${XTENSOR_PYTHON_BENCHMARK_TARGET} ${PYTHON_LIBRARIES})
else ()
    target_link_libraries(${XTENSOR_PYTHON_BENCHMARK_TARGET} "-shared")
endif()

configure_file(benchmark_pyarray.py benchmark_pyarray.py COPYONLY)
configure_file(benchmark_pytensor.py benchmark_pytensor.py COPYONLY)
configure_file(benchmark_pybind_array.py benchmark_pybind_array.py COPYONLY)
configure_file(benchmark_pyvectorize.py benchmark_pyvectorize.py COPYONLY)
configure_file(benchmark_pybind_vectorize.py benchmark_pybind_vectorize.py COPYONLY)

add_custom_target(xbenchmark DEPENDS ${XTENSOR_PYTHON_BENCHMARK_TARGET})

