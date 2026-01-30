#include <util.cuh>
#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>

float *allocateMatricOnHost(const int rows, const int cols) {
    const auto h_matrix = static_cast<float *>(malloc(rows * cols * sizeof(float)));
    if (h_matrix == nullptr) {
        std::cerr << "Failed to allocate host matrix" << std::endl;
        exit(EXIT_FAILURE);
    }
    return h_matrix;
}

float *allocateMatrixOnDevice(const int rows, const int cols) {
    float *d_matrix;
    size_t size = rows * cols * sizeof(float);
    CHECK(cudaMalloc(&d_matrix, size));
    return d_matrix;
}
