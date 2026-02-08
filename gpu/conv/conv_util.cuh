#pragma once

#ifndef CONV_UTIL_CUH
#define CONV_UTIL_CUH

#include <util.cuh>

struct Conv2dDims {
    int n, c, h, w;
    int k, r, s;
    int out_h, out_w;

    __host__ __device__ inline int in_offset(const int idx_n, const int idx_c, const int idx_h, const int idx_w) const {
        return (idx_n * c * h * w) + (idx_c * h * w) + (idx_h * w) + idx_w;
    }

    __host__ __device__ inline int weight_offset(const int idx_k, const int idx_c, const int idx_h, const int idx_w) const {
        return (idx_k * c * r * s) + (idx_c * r * s) + (idx_h * s) + idx_w;
    }

    __host__ __device__ inline int out_offset(const int idx_n, const int idx_k, const int idx_out_h, const int idx_out_w) const {
        return (idx_n * k * out_h * out_w) + (idx_k * out_h * out_w) + (idx_out_h * out_w) + idx_out_w;
    }
};

struct Conv2dAttrs {
    int u, v;
    int p, q;
};

inline int conv2d_offset_c(const Conv2dDims &dims, const int c) {
    return c * dims.h * dims.w;
}

bool check_result(const float* cpu_res, const float* gpu_res, int size);

void conv2d_cpu(const float *in, const float *weight, float *out,
                const Conv2dDims &dims, const Conv2dAttrs &attrs);

#endif // CONV_UTIL_CUH
