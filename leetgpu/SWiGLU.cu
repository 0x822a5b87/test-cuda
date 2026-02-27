#include <cuda_runtime.h>

const unsigned BLOCKS_PER_GRID_FOR_H100 = 428;
const unsigned COUNT_PER_THREAD = 4;

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float silu_scalar(float x) {
    return x * sigmoid(x);
}

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    // 每个 block tile 的宽度
    const unsigned block_tile = blockDim.x * 4 * COUNT_PER_THREAD;
    // 每个 grid 的宽度
    const unsigned grid_stride = gridDim.x * block_tile;
    // 线程每次循环跳转的float
    const unsigned thread_stride = blockDim.x * 4;
    // x1 的起始位置
    unsigned x1_pos_base = blockIdx.x * block_tile + threadIdx.x * 4;

    // 去执行下一个 grid
    for (unsigned x1_pos = x1_pos_base; x1_pos < halfN; x1_pos += grid_stride) {
        for (unsigned ti = 0; ti < COUNT_PER_THREAD; ti++) {
            unsigned idx = x1_pos + ti * thread_stride;
            if (idx + 3 < halfN) {
                float4 x = __ldg(reinterpret_cast<const float4*>(&input[idx]));
                float4 g = __ldg(reinterpret_cast<const float4*>(&input[idx + halfN]));
                float4 res;
                res.x = silu_scalar(x.x) * g.x;
                res.y = silu_scalar(x.y) * g.y;
                res.z = silu_scalar(x.z) * g.z;
                res.w = silu_scalar(x.w) * g.w;
                reinterpret_cast<float4*>(&output[idx])[0] = res;
            }
            else if (idx < halfN) {
                #pragma unroll
                for (unsigned i = idx; i < halfN; ++i) { // 使用局部变量 i
                    output[i] = silu_scalar(input[i]) * input[i + halfN];
                }
            }
        }
    }
}

__global__ void swiglu_scalar_kernel(const float* __restrict__ input, float* __restrict__ output, int halfN) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned i = tid; i < halfN; i += gridDim.x * blockDim.x) {
        output[i] = silu_scalar(input[i]) * input[i + halfN];
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    if (halfN % 4 == 0) {
        swiglu_kernel<<<BLOCKS_PER_GRID_FOR_H100, threadsPerBlock>>>(input, output, halfN);
    } else {
        swiglu_scalar_kernel<<<BLOCKS_PER_GRID_FOR_H100, threadsPerBlock>>>(input, output, halfN);
    }
    cudaDeviceSynchronize();
}
