#include <cuda_runtime.h>

#define VECTORIZED(c) ((c) * (COUNT_VECTORIZED))

constexpr unsigned BLOCKS_PER_GRID_FOR_H100 = 428;
constexpr unsigned COUNT_VECTORIZED = 4;
constexpr unsigned VECTORIZED_CHECKER = COUNT_VECTORIZED - 1;
// discriminate this value from vectorized number
constexpr unsigned COUNT_PER_THREAD = 8;

__device__ __forceinline__ float4 clip_vector(float4 val, float min_val, float max_val) {
    float4 res;
    res.x = fminf(fmaxf(val.x, min_val), max_val);
    res.y = fminf(fmaxf(val.y, min_val), max_val);
    res.z = fminf(fmaxf(val.z, min_val), max_val);
    res.w = fminf(fmaxf(val.w, min_val), max_val);
    return res;
}

__device__ __forceinline__ float clip_scalar(float val, float min_val, float max_val) {
    return fminf(fmaxf(val, min_val), max_val);
}

__global__ void clip_kernel(const float* input, float* output, float lo, float hi, int N) {
    const unsigned block_tile = VECTORIZED(blockDim.x) * COUNT_PER_THREAD;
    const unsigned grid_stride = gridDim.x * block_tile;
    const unsigned thread_stride = VECTORIZED(blockDim.x);

    const unsigned base = blockIdx.x * block_tile + VECTORIZED(threadIdx.x);
    for (unsigned pos = base; pos < N; pos += grid_stride) {
        for (unsigned ti = 0; ti < COUNT_PER_THREAD; ti++) {
            const unsigned idx = pos + ti * thread_stride;
            if (idx + VECTORIZED_CHECKER < N) {
                float4 val = __ldg(reinterpret_cast<const float4*>(&input[idx]));
                float4 res = clip_vector(val, lo, hi);
                reinterpret_cast<float4 *>(&output[idx])[0] = res;
            }
            else if (idx < N) {
                // 必须使用for循环处理，因为我们的 thread_stride 不是 +1，而是 +4
                for (unsigned i = idx; i < N; i++) {
                    output[i] = clip_scalar(__ldg(&input[i]), lo, hi);
                }
            }
        }
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, float lo, float hi, int N) {
    int threadsPerBlock = 256;
    clip_kernel<<<BLOCKS_PER_GRID_FOR_H100, threadsPerBlock>>>(input, output, lo, hi, N);
    cudaDeviceSynchronize();
}
