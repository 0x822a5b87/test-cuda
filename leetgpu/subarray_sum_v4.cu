#include <cuda_runtime.h>
#include <stdint.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define CLEAR_LOWER_16(addr) ((addr) & (~0xF))

constexpr unsigned THREAD_PER_BLOCK = 256;
constexpr unsigned WARP_PER_BLOCK = CEIL_DIV(THREAD_PER_BLOCK, 32);

__global__ void sum_kernel_masked(const int *base_ptr, int *output, int S, int E)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned stride = gridDim.x * blockDim.x;

    uintptr_t s_addr = reinterpret_cast<uintptr_t>(base_ptr + S);
    uintptr_t aligned_s_addr = CLEAR_LOWER_16(s_addr);
    unsigned vec_offset = (s_addr - aligned_s_addr) / sizeof(int);

    uintptr_t e_addr = reinterpret_cast<uintptr_t>(base_ptr + E + 1);
    unsigned vec_cnt = CEIL_DIV(e_addr - aligned_s_addr, 16);

    int local_sum = 0;
    int4 *data = reinterpret_cast<int4 *>(aligned_s_addr);
    // idx是在包含了prolog和epilog的int4数组中的索引
    for (unsigned idx = tid; idx < vec_cnt; idx += stride)
    {
        int4 val = data[idx];
        // sub_idx是在数组 [base_ptr + S, base_ptr + E + 1) 中的索引
        int idx_of_subarray = idx * 4 - vec_offset;
        int max_idx = E - S;

        local_sum += (0 <= idx_of_subarray && idx_of_subarray <= max_idx) ? val.x : 0;
        local_sum += (0 <= idx_of_subarray + 1 && idx_of_subarray + 1 <= max_idx) ? val.y : 0;
        local_sum += (0 <= idx_of_subarray + 2 && idx_of_subarray + 2 <= max_idx) ? val.z : 0;
        local_sum += (0 <= idx_of_subarray + 3 && idx_of_subarray + 3 <= max_idx) ? val.w : 0;
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }

    __shared__ int ssm_sum[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    if (lane == 0)
    {
        ssm_sum[wid] = local_sum;
    }
    __syncthreads();

    if (wid == 0)
    {
        int block_sum = (lane < WARP_PER_BLOCK) ? ssm_sum[lane] : 0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            block_sum += __shfl_down_sync(0xFFFFFFFF, block_sum, offset);
        }
        if (lane == 0)
        {
            atomicAdd(output, block_sum);
        }
    }
}

extern "C" void solve(const int *input, int *output, int N, int S, int E)
{
    cudaMemset(output, 0, sizeof(int));
    if (E < S)
        return;

    sum_kernel_masked<<<120, THREAD_PER_BLOCK>>>(input, output, S, E);
    cudaDeviceSynchronize();
}