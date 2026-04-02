cmake_minimum_required(VERSION 3.14)
project(execution_core CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Ensure Pybind11 is located (can be injected via pip install pybind11)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

pybind11_add_module(hft_execution 
    src/order_manager.cpp 
    src/bindings.cpp
)

target_include_directories(hft_execution PRIVATE include)

# Apply aggresive compiler optimizations for HFT
target_compile_options(hft_execution PRIVATE -O3 -march=native -funroll-loops)
