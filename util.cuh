#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#define CHECK(call)                                                              \
{                                                                                \
    const cudaError_t error = call;                                              \
    if (error != cudaSuccess)                                                    \
    {                                                                            \
        fprintf(stderr, ANSI_COLOR_RED "CUDA Error:" ANSI_COLOR_RESET "\n");     \
        fprintf(stderr, "    File:    %s\n", __FILE__);                          \
        fprintf(stderr, "    Line:    %d\n", __LINE__);                          \
        fprintf(stderr, "    Code:    %d\n", error);                             \
        fprintf(stderr, "    Reason:  %s\n", cudaGetErrorString(error));         \
        exit(1);                                                                 \
    }                                                                            \
}

#define CHECK_LAST_KERNEL_ERROR()                                                \
{                                                                                \
    CHECK(cudaGetLastError());                                                   \
    CHECK(cudaDeviceSynchronize());                                              \
}