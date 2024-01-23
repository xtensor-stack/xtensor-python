cmake_minimum_required(VERSION 3.18..3.21)

project(xt)

# The C++ functions are build to a library with name "_${PROJECT_NAME}"
# The Python library simply loads all functions
set(PYPROJECT_NAME "_${PROJECT_NAME}")

option(USE_WARNINGS "${PROJECT_NAME}: Build with runtime warnings" ON)
option(USE_XSIMD "${PROJECT_NAME}: Build with hardware optimization" OFF)

if (DEFINED ENV{SETUPTOOLS_SCM_PRETEND_VERSION})
    set(PROJECT_VERSION $ENV{SETUPTOOLS_SCM_PRETEND_VERSION})
    message(STATUS "Building ${PROJECT_NAME} ${PROJECT_VERSION} (read from SETUPTOOLS_SCM_PRETEND_VERSION)")
else()
    execute_process(
        COMMAND python -c "from setuptools_scm import get_version; print(get_version(root='..'))"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE PROJECT_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    message(STATUS "Building ${PROJECT_NAME} ${PROJECT_VERSION}")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(xtensor REQUIRED)
find_package(pybind11 REQUIRED CONFIG)

if (SKBUILD)
    find_package(NumPy REQUIRED)
else()
    find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
endif()

pybind11_add_module(${PYPROJECT_NAME} module/main.cpp)

target_compile_definitions(${PYPROJECT_NAME} PUBLIC VERSION_INFO=${PROJECT_VERSION})
target_link_libraries(${PYPROJECT_NAME} PUBLIC xtensor)
target_include_directories(${PYPROJECT_NAME} PUBLIC "../include")

if (SKBUILD)
    target_include_directories(${PYPROJECT_NAME} PUBLIC ${NumPy_INCLUDE_DIRS})
else()
    target_link_libraries(${PYPROJECT_NAME} PUBLIC pybind11::module Python::NumPy)
endif()

if (USE_SIMD)
    find_package(xsimd REQUIRED)
    target_link_libraries(${PYPROJECT_NAME} PUBLIC xtensor::optimize xtensor::use_xsimd)
    message(STATUS "Compiling ${PROJECT_NAME}-Python with hardware optimization")
endif()

if (SKBUILD)
    if(APPLE)
        set_target_properties(${PYPROJECT_NAME} PROPERTIES INSTALL_RPATH "@loader_path/${CMAKE_INSTALL_LIBDIR}")
    else()
        set_target_properties(${PYPROJECT_NAME} PROPERTIES INSTALL_RPATH "$ORIGIN/${CMAKE_INSTALL_LIBDIR}")
    endif()
    install(TARGETS ${PYPROJECT_NAME} DESTINATION .)
endif()
