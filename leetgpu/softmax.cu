#include <cuda_runtime.h>
#include <cstdio>
#include <float.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

constexpr unsigned BLOCKS_FOR_TESLA = 120;
constexpr unsigned FULl_MASK = 0xffffffff;
constexpr unsigned THREAD_PER_BLOCK = 256;
constexpr unsigned WARP_PER_BLOCK = CEIL_DIV(THREAD_PER_BLOCK, 32);

__global__ void max_kernel(const float *input, float *max, int N)
{
    float local_max = -FLT_MAX;
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned block_tile = gridDim.x * blockDim.x;
    for (unsigned idx = tid; idx < N; idx += block_tile)
    {
        local_max = fmaxf(local_max, input[idx]);
    }

    local_max = fmaxf(local_max, __shfl_down_sync(FULl_MASK, local_max, 16));
    local_max = fmaxf(local_max, __shfl_down_sync(FULl_MASK, local_max, 8));
    local_max = fmaxf(local_max, __shfl_down_sync(FULl_MASK, local_max, 4));
    local_max = fmaxf(local_max, __shfl_down_sync(FULl_MASK, local_max, 2));
    local_max = fmaxf(local_max, __shfl_down_sync(FULl_MASK, local_max, 1));

    __shared__ float ssm_max[32];
    unsigned lane = threadIdx.x % 32;
    unsigned warp_id = threadIdx.x / 32;
    if (lane == 0)
    {
        ssm_max[warp_id] = local_max;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        float block_max = (lane < WARP_PER_BLOCK) ? ssm_max[lane] : -FLT_MAX;
        block_max = fmaxf(block_max, __shfl_down_sync(FULl_MASK, block_max, 16));
        block_max = fmaxf(block_max, __shfl_down_sync(FULl_MASK, block_max, 8));
        block_max = fmaxf(block_max, __shfl_down_sync(FULl_MASK, block_max, 4));
        block_max = fmaxf(block_max, __shfl_down_sync(FULl_MASK, block_max, 2));
        block_max = fmaxf(block_max, __shfl_down_sync(FULl_MASK, block_max, 1));
        if (lane == 0)
        {
            atomicMax((int *)max, __float_as_int(block_max));
        }
    }
}

__global__ void reduce_exp_sum_kernel(const float *input, float *max, float *sum, int N)
{
    float local_sum = 0;
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned block_tile = gridDim.x * blockDim.x;
    for (unsigned idx = tid; idx < N; idx += block_tile)
    {
        local_sum += expf(input[idx] - max[0]);
    }

    local_sum += __shfl_down_sync(FULl_MASK, local_sum, 16);
    local_sum += __shfl_down_sync(FULl_MASK, local_sum, 8);
    local_sum += __shfl_down_sync(FULl_MASK, local_sum, 4);
    local_sum += __shfl_down_sync(FULl_MASK, local_sum, 2);
    local_sum += __shfl_down_sync(FULl_MASK, local_sum, 1);

    __shared__ float ssm_sum[32];
    unsigned lane = threadIdx.x % 32;
    unsigned warp_id = threadIdx.x / 32;
    if (lane == 0)
    {
        ssm_sum[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        float block_sum = (lane < WARP_PER_BLOCK) ? ssm_sum[lane] : 0;
        block_sum += __shfl_down_sync(FULl_MASK, block_sum, 16);
        block_sum += __shfl_down_sync(FULl_MASK, block_sum, 8);
        block_sum += __shfl_down_sync(FULl_MASK, block_sum, 4);
        block_sum += __shfl_down_sync(FULl_MASK, block_sum, 2);
        block_sum += __shfl_down_sync(FULl_MASK, block_sum, 1);
        if (lane == 0)
        {
            atomicAdd(sum, block_sum);
        }
    }
}

__global__ void softmax_kernel(const float *input, float* output, float *max, float *sum, int N)
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned block_tile = gridDim.x * blockDim.x;
    for (unsigned idx = tid; idx < N; idx += block_tile)
    {
        float exp = input[idx] - *max;
        output[idx] = expf(exp) / *sum;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *input, float *output, int N)
{
    float *d_max;
    float *d_sum;
    cudaMallocAsync(&d_max, sizeof(float), cudaStreamDefault);
    cudaMallocAsync(&d_sum, sizeof(float), cudaStreamDefault);
    max_kernel<<<BLOCKS_FOR_TESLA, THREAD_PER_BLOCK>>>(input, d_max, N);
    reduce_exp_sum_kernel<<<BLOCKS_FOR_TESLA, THREAD_PER_BLOCK>>>(input, d_max, d_sum, N);
    softmax_kernel<<<BLOCKS_FOR_TESLA, THREAD_PER_BLOCK>>>(input, output, d_max, d_sum, N);

    {
        float h_max = 0.0f;
        float h_sum = 0.0f;
        cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
        printf("max = %f, sum = %f\n", h_max, h_sum);
    }

    cudaDeviceSynchronize();
}
