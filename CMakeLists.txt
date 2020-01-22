############################################################################
# Copyright (c) Wolf Vollprecht, Johan Mabille and Sylvain Corlay          #
# Copyright (c) QuantStack                                                 #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 3.1)
project(xtensor-python)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
set(XTENSOR_PYTHON_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Versioning
# ==========

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

# Dependencies
# ============

set(xtensor_REQUIRED_VERSION 0.21.2)
if(TARGET xtensor)
    set(xtensor_VERSION ${XTENSOR_VERSION_MAJOR}.${XTENSOR_VERSION_MINOR}.${XTENSOR_VERSION_PATCH})
    # Note: This is not SEMVER compatible comparison
    if( NOT ${xtensor_VERSION} VERSION_GREATER_EQUAL ${xtensor_REQUIRED_VERSION})
        message(ERROR "Mismatch xtensor versions. Found '${xtensor_VERSION}' but requires: '${xtensor_REQUIRED_VERSION}'")
    else()
        message(STATUS "Found xtensor v${xtensor_VERSION}")
    endif()
else()
    find_package(xtensor ${xtensor_REQUIRED_VERSION} REQUIRED)
    message(STATUS "Found xtensor: ${xtensor_INCLUDE_DIRS}/xtensor")
endif()
    
# Currently no required version for pybind11
if(TARGET pybind11)
    # pybind11 has a variable that indicates its version already, so use that
    message(STATUS "Found pybind11 v${pybind11_VERSION}")
else()
    find_package(pybind11 REQUIRED)
    message(STATUS "Found pybind11: ${pybind11_INCLUDE_DIRS}/pybind11")
endif()

find_package(NumPy REQUIRED)
message(STATUS "Found numpy: ${NUMPY_INCLUDE_DIRS}")

# Build
# =====

set(XTENSOR_PYTHON_HEADERS
    ${XTENSOR_PYTHON_INCLUDE_DIR}/xtensor-python/pyarray.hpp
    ${XTENSOR_PYTHON_INCLUDE_DIR}/xtensor-python/pyarray_backstrides.hpp
    ${XTENSOR_PYTHON_INCLUDE_DIR}/xtensor-python/pycontainer.hpp
    ${XTENSOR_PYTHON_INCLUDE_DIR}/xtensor-python/pystrides_adaptor.hpp
    ${XTENSOR_PYTHON_INCLUDE_DIR}/xtensor-python/pytensor.hpp
    ${XTENSOR_PYTHON_INCLUDE_DIR}/xtensor-python/pyvectorize.hpp
    ${XTENSOR_PYTHON_INCLUDE_DIR}/xtensor-python/xtensor_python_config.hpp
    ${XTENSOR_PYTHON_INCLUDE_DIR}/xtensor-python/xtensor_type_caster_base.hpp
)

add_library(xtensor-python INTERFACE)
target_include_directories(xtensor-python INTERFACE
    "$<BUILD_INTERFACE:${XTENSOR_PYTHON_INCLUDE_DIR};${pybind11_INCLUDE_DIRS};${NUMPY_INCLUDE_DIRS}>"
    $<INSTALL_INTERFACE:include>)
target_link_libraries(xtensor-python INTERFACE xtensor)
get_target_property(inc_dir xtensor-python INTERFACE_INCLUDE_DIRECTORIES)

OPTION(BUILD_TESTS "xtensor test suite" OFF)
OPTION(DOWNLOAD_GTEST "build gtest from downloaded sources" OFF)

if(DOWNLOAD_GTEST OR GTEST_SRC_DIR)
    set(BUILD_TESTS ON)
endif()

if(BUILD_TESTS)
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

install(TARGETS xtensor-python
        EXPORT ${PROJECT_NAME}-targets)

# Makes the project importable from the build directory
export(EXPORT ${PROJECT_NAME}-targets
       FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")

install(FILES ${XTENSOR_PYTHON_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/xtensor-python)

set(XTENSOR_PYTHON_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}" CACHE
    STRING "install path for xtensor-pythonConfig.cmake")

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
install(EXPORT ${PROJECT_NAME}-targets
        FILE ${PROJECT_NAME}Targets.cmake
        DESTINATION ${XTENSOR_PYTHON_CMAKECONFIG_INSTALL_DIR})

