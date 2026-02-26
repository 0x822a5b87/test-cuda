#include <cuda_runtime.h>

__device__ __forceinline__ float sigmoid(float x) {
    if (x > 18.0f) return 1.0f;
    if (x < -18.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float silu_scalar(float x) {
    return x * sigmoid(x);
}

__global__ void silu_kernel_optimized(const float* input, float* output, int N) {
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned grid_stride = gridDim.x * blockDim.x * 4;
    for (int pos = tid * 4; pos < N; pos += grid_stride) {
        if (pos + 3 < N) {
            float4 v = __ldg(reinterpret_cast<const float4*>(&input[pos]));
            v.x = silu_scalar(v.x);
            v.y = silu_scalar(v.y);
            v.z = silu_scalar(v.z);
            v.w = silu_scalar(v.w);
            reinterpret_cast<float4*>(&output[pos])[0] = v;
        } else {
            if (pos > N) {
                break;
            }
            for (unsigned i = pos; i < N; i++) {
                output[i] = silu_scalar(__ldg(&input[i]));
            }
        }
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    silu_kernel_optimized<<<412, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
