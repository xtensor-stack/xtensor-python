cmake_minimum_required(VERSION 3.29)

project(mymodule)

find_package(pybind11 CONFIG REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-python REQUIRED)
find_package(Python3 REQUIRED COMPONENTS NumPy)

pybind11_add_module(mymodule main.cpp)
target_link_libraries(mymodule PUBLIC pybind11::module xtensor-python Python3::NumPy)

target_compile_definitions(mymodule PRIVATE VERSION_INFO=0.1.0)
