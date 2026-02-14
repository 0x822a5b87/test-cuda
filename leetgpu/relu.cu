#include <cuda_runtime.h>

constexpr unsigned ThreadPerBlock = 256;
constexpr unsigned ValuePerThread = 3;

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

__global__ void relu_kernel_float4(const float4* input, float4* output, int N) {
    unsigned gx = blockDim.x * blockIdx.x * ValuePerThread + threadIdx.x;
    for (int i = 0; i < ValuePerThread; i++) {
        gx += ThreadPerBlock;
        if (gx < N) {
            float4 v = input[gx];
            v.x = fmaxf(0.0f, v.x);
            v.y = fmaxf(0.0f, v.y);
            v.z = fmaxf(0.0f, v.z);
            v.w = fmaxf(0.0f, v.w);
            output[gx] = v;
        } 
    }
}

__global__ void relu_kernel(const float* input, float* output, int N) {
    unsigned gx = blockDim.x * blockIdx.x + threadIdx.x;
    if (gx < N) {
        output[gx] = fmaxf(0.0f, input[gx]);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    size_t vec_num = N / 4;
    size_t remained_num = N % 4;

    if (vec_num > 0) {
        size_t valuePerBlock = ValuePerThread * ThreadPerBlock;
        size_t block_num = CEIL_DIV(vec_num, valuePerBlock);
        relu_kernel_float4<<<block_num, ThreadPerBlock>>>(
            reinterpret_cast<const float4*>(input),
            reinterpret_cast<float4*>(output),
            vec_num
        );
    }

    if (remained_num > 0) {
        relu_kernel<<<1, remained_num>>>(input + vec_num * 4, output + vec_num * 4, remained_num);
    }
}