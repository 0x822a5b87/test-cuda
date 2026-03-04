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

# matrix_transpose
add_executable(matrix_transpose gpu/matrix_transpose/matrix_transpose_2.cu)
target_include_directories(matrix_transpose PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# matrix_transpose_shared_memory
add_executable(matrix_transpose_shared_memory gpu/matrix_transpose/matrix_transpose_shared_memory.cu)
target_include_directories(matrix_transpose_shared_memory PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})


# sgemm
add_executable(naive_sgemm 
    gpu/sgemm/sgemm.cu 
    gpu/sgemm/sgemm_util.cu
    util.cu 
)
target_include_directories(naive_sgemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/gpu/sgemm)

add_executable(shared_sgemm 
    gpu/sgemm/sgemm_shared.cu 
    gpu/sgemm/sgemm_util.cu
    util.cu 
)
target_include_directories(shared_sgemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/gpu/sgemm)

add_executable(shared_sgemm_thread
        util.cu
        gpu/sgemm/sgemm_util.cu
        gpu/sgemm/sgemm_shared_thread.cu
)
target_include_directories(shared_sgemm_thread PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/gpu/sgemm)

add_executable(naive_reduce util.cu gpu/reduce/naive_reduce.cu)
target_include_directories(naive_reduce PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/gpu/reduce)

add_executable(naive_reduce_2 util.cu gpu/reduce/naive_reduce_2.cu)
target_include_directories(naive_reduce_2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/gpu/reduce)

add_executable(reduce_eliminate_divergence util.cu gpu/reduce/reduce_eliminate_divergence.cu)
target_include_directories(reduce_eliminate_divergence PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/gpu/reduce)

add_executable(reduce_vectorized util.cu gpu/reduce/reduce_vectorized.cu)
target_include_directories(reduce_vectorized PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/gpu/reduce)

add_executable(reduce_vectorized_2 util.cu gpu/reduce/reduce_vectorized_2.cu)
target_include_directories(reduce_vectorized_2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/gpu/reduce)

add_executable(reduce_shuffle util.cu gpu/reduce/reduce_shuffle.cu)
target_include_directories(reduce_shuffle PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/gpu/reduce)

add_executable(naive_conv gpu/conv/naive_conv.cu gpu/conv/conv_util.cu)
target_include_directories(naive_conv PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/gpu/conv)

add_executable(vector_addition leetgpu/vector_addition.cu)
add_executable(matrix_multiplication_v1 leetgpu/matrix_multiplication_v1.cu)
add_executable(matrix_multiplication_v2 leetgpu/matrix_multiplication_v2.cu)
add_executable(color_inversion leetgpu/color_inversion_v1.cu)
add_executable(matrix_addition leetgpu/matrix_addition.cu)

add_library(convolution_1d leetgpu/convolution_1d.cu)
add_library(reverse_array_v1 leetgpu/reverse_array_v1.cu)
add_library(relu leetgpu/relu.cu)
add_library(leaky_relu leetgpu/leaky_relu.cu)
add_library(matrix_copy leetgpu/matrix_copy.cu)
add_library(count_array_element leetgpu/count_array_element.cu)
add_library(count_2d_array_element leetgpu/count_2d_array_element.cu)
add_library(silu leetgpu/silu.cu)
add_library(SWiGLU leetgpu/SWiGLU.cu)
add_library(value_clipping leetgpu/value_clipping.cu)
add_library(interleave_arrays leetgpu/interleave_arrays.cu)
add_library(geglu leetgpu/geglu.cu)
add_library(rgb_to_grayscale leetgpu/rgb_to_grayscale.cu)

add_library(add leetgpu/add.cu)
add_library(add_v2 leetgpu/add_v2.cu)
add_library(dot_product leetgpu/dot_product.cu)
add_library(mse leetgpu/mse.cu)
add_library(mse_v2 leetgpu/mse_v2.cu)
add_library(softmax leetgpu/softmax.cu)
add_library(subarray_sum leetgpu/subarray_sum.cu)
add_library(subarray_sum_v2 leetgpu/subarray_sum_v2.cu)
add_library(subarray_sum_v3 leetgpu/subarray_sum_v3.cu)
add_library(subarray_sum_v4 leetgpu/subarray_sum_v4.cu)