#include <cuda_runtime.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

constexpr unsigned FULl_MASK = 0xffffffff;
constexpr unsigned THREAD_PER_BLOCK = 256;
constexpr unsigned WARP_PER_BLOCK = CEIL_DIV(THREAD_PER_BLOCK, 32);

__device__ __forceinline__ float warp_reduce(float warp_sum)
{
    warp_sum += __shfl_down_sync(FULl_MASK, warp_sum, 16);
    warp_sum += __shfl_down_sync(FULl_MASK, warp_sum, 8);
    warp_sum += __shfl_down_sync(FULl_MASK, warp_sum, 4);
    warp_sum += __shfl_down_sync(FULl_MASK, warp_sum, 2);
    warp_sum += __shfl_down_sync(FULl_MASK, warp_sum, 1);
    return warp_sum;
}

__global__ void final_optimized_reduce(const float *input, float *output, int N)
{
    float local_sum = 0;
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned block_tile = gridDim.x * blockDim.x;
    for (unsigned idx = tid; idx < N; idx += block_tile)
    {
        local_sum += input[idx];
    }

    local_sum = warp_reduce(local_sum);

    __shared__ float ssm[32];
    unsigned lane = threadIdx.x % 32;
    unsigned warp_id = threadIdx.x / 32;
    if (lane == 0)
    {
        ssm[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        float warp_sum = (lane < WARP_PER_BLOCK) ? ssm[lane] : 0.0f;
        warp_sum = warp_reduce(warp_sum);
        if (lane == 0)
        {
            atomicAdd(output, warp_sum);
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *input, float *output, int N)
{
    final_optimized_reduce<<<432, THREAD_PER_BLOCK>>>(input, output, N);
    cudaDeviceSynchronize();
}
