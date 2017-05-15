############################################################################
# Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 3.1)
project(xtensor-python)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
set(XTENSOR_PYTHON_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Versionning
# ===========

set(XTENSOR_PYTHON_CONFIG_FILE
    "${XTENSOR_PYTHON_INCLUDE_DIR}/xtensor-python/xtensor_python_config.hpp")
file(STRINGS ${XTENSOR_PYTHON_CONFIG_FILE} xtensor_python_version_defines
    REGEX "#define XTENSOR_PYTHON_VERSION_(MAJOR|MINOR|PATCH)")
foreach(ver ${xtensor_python_version_defines})
    if(ver MATCHES "#define XTENSOR_PYTHON_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(XTENSOR_PYTHON_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()
set(${PROJECT_NAME}_VERSION 
    ${XTENSOR_PYTHON_VERSION_MAJOR}.${XTENSOR_PYTHON_VERSION_MINOR}.${XTENSOR_PYTHON_VERSION_PATCH})
message(STATUS "xtensor-python v${${PROJECT_NAME}_VERSION}")

# Build
# =====

set(XTENSOR_PYTHON_HEADERS
    ${XTENSOR_PYTHON_INCLUDE_DIR}/xtensor-python/pyarray.hpp
    ${XTENSOR_PYTHON_INCLUDE_DIR}/xtensor-python/pycontainer.hpp
    ${XTENSOR_PYTHON_INCLUDE_DIR}/xtensor-python/pystrides_adaptor.hpp
    ${XTENSOR_PYTHON_INCLUDE_DIR}/xtensor-python/pytensor.hpp
    ${XTENSOR_PYTHON_INCLUDE_DIR}/xtensor-python/pyvectorize.hpp
    ${XTENSOR_PYTHON_INCLUDE_DIR}/xtensor-python/xtensor_python_config.hpp
)

OPTION(BUILD_TESTS "xtensor test suite" OFF)
OPTION(DOWNLOAD_GTEST "build gtest from downloaded sources" OFF)

if(DOWNLOAD_GTEST OR GTEST_SRC_DIR)
    set(BUILD_TESTS ON)
endif()

if(BUILD_TESTS)
    include_directories(${XTENSOR_PYTHON_INCLUDE_DIR})
    find_package(xtensor REQUIRED)
    include_directories(${xtensor_INCLUDE_DIRS})
    find_package(NumPy REQUIRED)
    include_directories(${NUMPY_INCLUDE_DIRS})

    find_package(pybind11 REQUIRED)
    include_directories(${pybind11_INCLUDE_DIRS})

    if(MSVC)
        set(PYTHON_MODULE_EXTENSION ".pyd")
    else()
        set(PYTHON_MODULE_EXTENSION ".so")
    endif()

    add_subdirectory(test)
    add_subdirectory(benchmark)
endif()

# Installation
# ============

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(FILES ${XTENSOR_PYTHON_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/xtensor-python)

# GNUInstallDirs "DATADIR" wrong here; CMake search path wants "share".
set(XTENSOR_PYTHON_CMAKECONFIG_INSTALL_DIR "share/cmake/${PROJECT_NAME}" CACHE STRING "install path for xtensor-pythonConfig.cmake")

configure_package_config_file(${PROJECT_NAME}Config.cmake.in
                              "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                              INSTALL_DESTINATION ${XTENSOR_PYTHON_CMAKECONFIG_INSTALL_DIR})

# xtensor-python is header-only and does not depend on the architecture.
# Remove CMAKE_SIZEOF_VOID_P from xtensor-pythonConfigVersion.cmake so that an xtensor-pythonConfig.cmake
# generated for a 64 bit target can be used for 32 bit targets and vice versa.
set(_XTENSOR_CMAKE_SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})
unset(CMAKE_SIZEOF_VOID_P)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
                                 VERSION ${${PROJECT_NAME}_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
set(CMAKE_SIZEOF_VOID_P ${_XTENSOR_CMAKE_SIZEOF_VOID_P})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION ${XTENSOR_PYTHON_CMAKECONFIG_INSTALL_DIR})

