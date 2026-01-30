#pragma once

#ifndef REDUCE_UTIL_CUH
#define REDUCE_UTIL_CUH

#include <util.cuh>

inline int *allocateIntArrOnHost(size_t len) {
    int *h_arr = allocateArrOnHost<int>(sizeof(int) * len);
    for (int i = 0; i < len; i++) {
        h_arr[i] = i;
    }
    return h_arr;
}

#endif // REDUCE_UTIL_CUH
