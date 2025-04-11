cmake_minimum_required(VERSION 3.29)

project(mymodule)

find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(pybind11 REQUIRED CONFIG)
find_package(xtensor REQUIRED)
find_package(xtensor-python REQUIRED)

pybind11_add_module(mymodule main.cpp)
target_link_libraries(mymodule PUBLIC pybind11::module xtensor-python Python::NumPy)

target_compile_definitions(mymodule PRIVATE VERSION_INFO=0.1.0)
