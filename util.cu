#include <util.cuh>
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

float *allocateMatricOnHost(int rows, int cols)
{
    float *h_matrix = (float *)malloc(rows * cols * sizeof(float));
    if (h_matrix == nullptr)
    {
        std::cerr << "Failed to allocate host matrix" << std::endl;
        exit(EXIT_FAILURE);
    }
    return h_matrix;
}

float *allocateMatrixOnDevice(int rows, int cols)
{
    float *d_matrix;
    size_t size = rows * cols * sizeof(float);
    CHECK(cudaMalloc(&d_matrix, size));
    return d_matrix;
}