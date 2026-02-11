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
