cmake_minimum_required(VERSION 3.18)
project(CUDA_Test LANGUAGES CUDA CXX)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# SPECIFY YOUR OWN ARCHITECTURES AT HERE
set(CMAKE_CUDA_ARCHITECTURES 89)

add_executable(add_cpu cpu/add.cpp)
add_executable(hello gpu/hello/hello.cu)
add_executable(add_gpu_1 gpu/add/add_gpu_1.cu)
add_executable(add_gpu_2 gpu/add/add_gpu_2.cu)
add_executable(add_gpu_3 gpu/add/add_gpu_3.cu)
add_executable(add_gpu_4 gpu/add/add_gpu_4.cu)

# matrix_tranpose
add_executable(matrix_transpose gpu/matrix_transpose/matrix_transpose_2.cu)
target_include_directories(matrix_transpose PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(matrix_transpose PROPERTIES
    CUDA_ARCHITECTURES 75
    CUDA_STANDARD 17
)