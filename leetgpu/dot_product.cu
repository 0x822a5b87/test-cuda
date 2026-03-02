#include <cuda_runtime.h>
#include <cstdio>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

constexpr unsigned FULl_MASK = 0xffffffff;
constexpr unsigned THREAD_PER_BLOCK = 256;
constexpr unsigned WARP_PER_BLOCK = CEIL_DIV(THREAD_PER_BLOCK, 32);

__device__ __forceinline__ float prod_reduce(float val)
{
    val += __shfl_down_sync(FULl_MASK, val, 16);
    val += __shfl_down_sync(FULl_MASK, val, 8);
    val += __shfl_down_sync(FULl_MASK, val, 4);
    val += __shfl_down_sync(FULl_MASK, val, 2);
    val += __shfl_down_sync(FULl_MASK, val, 1);
    return val;
}

__global__ void dot_product(const float *A, const float *B, float *result, int N)
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned block_tile = gridDim.x * blockDim.x;

    if (tid > N) {
        return;
    }

    float local_val = 0.0f;
    for (unsigned i = tid; i < N; i += block_tile)
    {
        local_val += A[i] * B[i];
    }
    printf("local val = %d\n", local_val);

    local_val = prod_reduce(local_val);

    __shared__ float ssm_data[32];
    const unsigned lane = threadIdx.x % 32;
    const unsigned warp_id = threadIdx.x / 32;
    if (lane == 0)
    {
        ssm_data[warp_id] = local_val;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        // use thread in the first warp of the block to sum all of warps in the block.
        unsigned warp_val = lane < WARP_PER_BLOCK ?  ssm_data[lane] : 0.0f;
        warp_val = prod_reduce(warp_val);
        if (lane == 0) {
            atomicAdd(result, warp_val);
        }
    }
}

extern "C" void solve(const float *A, const float *B, float *result, int N)
{
    dot_product<<<432, THREAD_PER_BLOCK>>>(A, B, result, N);
    cudaDeviceSynchronize();
}
