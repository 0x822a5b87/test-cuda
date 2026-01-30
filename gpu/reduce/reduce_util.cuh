#pragma once

#ifndef REDUCE_UTIL_CUH
#define REDUCE_UTIL_CUH

#include <util.cuh>

inline int *allocateIntArrOnHost(size_t len) {
    int *h_arr = allocateArrOnHost<int>(len);
    for (int i = 0; i < len / sizeof(int); i++) {
        h_arr[i] = i % 10;
    }
    return h_arr;
}

inline void checkReduceResult(int gpu_sum, const int *h_arr, size_t len) {
    int cpu_sum = 0;
    for (int i = 0; i < len; i++) {
        cpu_sum += h_arr[i];
    }

    printf("GPU Result: %d\n", gpu_sum);
    printf("CPU Expected: %d\n", cpu_sum);
}

#endif // REDUCE_UTIL_CUH
