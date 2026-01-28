#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define ANSI_COLOR_RED "\x1b[31m"
#define ANSI_COLOR_RESET "\x1b[0m"

#define CHECK(call)                                                              \
    {                                                                            \
        const cudaError_t error = call;                                          \
        if (error != cudaSuccess)                                                \
        {                                                                        \
            fprintf(stderr, ANSI_COLOR_RED "CUDA Error:" ANSI_COLOR_RESET "\n"); \
            fprintf(stderr, "    File:    %s\n", __FILE__);                      \
            fprintf(stderr, "    Line:    %d\n", __LINE__);                      \
            fprintf(stderr, "    Code:    %d\n", error);                         \
            fprintf(stderr, "    Reason:  %s\n", cudaGetErrorString(error));     \
            exit(1);                                                             \
        }                                                                        \
    }

#define CHECK_LAST_KERNEL_ERROR()       \
    {                                   \
        CHECK(cudaGetLastError());      \
        CHECK(cudaDeviceSynchronize()); \
    }

#define CHECK_MATRIX_TRANSPOSE(h_in, h_out, nx, ny, errorCount) \
    do                                                          \
    {                                                           \
        for (int _r = 0; _r < (ny); _r++)                       \
        {                                                       \
            for (int _c = 0; _c < (nx); _c++)                   \
            {                                                   \
                int _idx_in = _r * (nx) + _c;                   \
                int _idx_out = _c * (ny) + _r;                  \
                if (h_in[_idx_in] != h_out[_idx_out])           \
                {                                               \
                    (errorCount)++;                             \
                }                                               \
            }                                                   \
        }                                                       \
    } while (0) 


float *allocateMatricOnHost(int rows, int cols);
float *allocateMatrixOnDevice(int rows, int cols);
