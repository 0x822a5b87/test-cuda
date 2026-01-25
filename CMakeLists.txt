cmake_minimum_required(VERSION 3.18)
project(CUDA_Test LANGUAGES CUDA CXX)

# SPECIFY YOUR OWN ARCHITECTURES AT HERE
set(CMAKE_CUDA_ARCHITECTURES 89)

add_executable(add_cpu cpu/add.cpp)
add_executable(hello gpu/hello.cu)
add_executable(add_gpu_1 gpu/add_gpu_1.cu)
add_executable(add_gpu_2 gpu/add_gpu_2.cu)
add_executable(add_gpu_3 gpu/add_gpu_3.cu)
add_executable(add_gpu_4 gpu/add_gpu_4.cu)