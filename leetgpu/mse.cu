#include <cuda_runtime.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

constexpr unsigned FULl_MASK = 0xffffffff;
constexpr unsigned THREAD_PER_BLOCK = 256;
constexpr unsigned WARP_PER_BLOCK = CEIL_DIV(THREAD_PER_BLOCK, 32);

__device__ __forceinline__ float add_reduce(float val)
{
    val += __shfl_down_sync(FULl_MASK, val, 16);
    val += __shfl_down_sync(FULl_MASK, val, 8);
    val += __shfl_down_sync(FULl_MASK, val, 4);
    val += __shfl_down_sync(FULl_MASK, val, 2);
    val += __shfl_down_sync(FULl_MASK, val, 1);
    return val;
}

__global__ void mse_kernel(const float *predictions, const float *targets, float *mse, int N)
{
    float local_val = 0.0f;
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned block_tile = gridDim.x * blockDim.x;
    for (unsigned idx = tid; idx < N; idx += block_tile)
    {
        float delta = predictions[idx] - targets[idx];
        local_val += delta * delta;
    }

    local_val = add_reduce(local_val);
    __shared__ float ssm_data[32];
    unsigned lane = threadIdx.x % 32;
    unsigned warp_id = threadIdx.x / 32;
    if (lane == 0)
    {
        ssm_data[warp_id] = local_val;
    }
    __syncthreads();
    if (warp_id == 0)
    {
        float warp_val = lane < WARP_PER_BLOCK ? ssm_data[lane] : 0.0f;
        warp_val = add_reduce(warp_val);
        if (lane == 0)
        {
            atomicAdd(mse, warp_val / N);
        }
    }
}

// predictions, targets, mse are device pointers
extern "C" void solve(const float *predictions, const float *targets, float *mse, int N)
{
    mse_kernel<<<120, THREAD_PER_BLOCK>>>(predictions, targets, mse, N);
    cudaDeviceSynchronize();
}
