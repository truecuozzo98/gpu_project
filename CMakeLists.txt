cmake_minimum_required(VERSION 3.27)
project(cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(cuda main.cu)

add_library(
        library
        random_graph_generator.cpp
        random_graph_generator.h
        sequential_mst.cpp
        sequential_mst.h
)

target_link_libraries(cuda PRIVATE library)


set_target_properties(cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_property(TARGET cuda PROPERTY CUDA_STANDARD 17)
set_property(TARGET library PROPERTY CXX_STANDARD 17)