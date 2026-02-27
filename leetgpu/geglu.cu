#include <cuda_runtime.h>
#include <math.h>

#define COUNT_PER_THREAD 4

__device__ inline float gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.707106781f));
}

__global__ void geglu_kernel(const float* __restrict__ input, float* __restrict__ output, int half_N) {
    const unsigned int grid_stride = gridDim.x * blockDim.x * COUNT_PER_THREAD;
    unsigned int base_idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int pos = base_idx; pos < half_N; pos += grid_stride) {
        #pragma unroll
        for (int i = 0; i < COUNT_PER_THREAD; ++i) {
            unsigned int idx = pos + i * (gridDim.x * blockDim.x);
            if (idx < half_N) {
                float x1 = __ldg(&input[idx]);
                float x2 = __ldg(&input[idx + half_N]);
                output[idx] = x1 * gelu_exact(x2);
            }
        }
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int half_N = N / 2;
    if (half_N <= 0) return;

    int threads = 256;
    int blocks = 428; 

    geglu_kernel<<<blocks, threads>>>(input, output, half_N);
}