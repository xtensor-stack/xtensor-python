cmake_minimum_required(VERSION 3.1)

project(foobar)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
find_package(pybind11 REQUIRED)
find_package(xtensor-python REQUIRED)

add_library(foobar src/python.cpp)

if(MSVC)
    target_compile_options(foobar PRIVATE /EHsc /MP /bigobj)
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR
    CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR
    (CMAKE_CXX_COMPILER_ID MATCHES "Intel" AND NOT WIN32))
    target_compile_options(foobar PRIVATE -march=native -std=c++14)
endif()

target_link_libraries(foobar
    PRIVATE
    pybind11::pybind11
    xtensor
    xtensor-python)

set_target_properties(foobar
    PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}")
