#include <cuda_runtime.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#define WARP_SIZE 32
#define BLOCK_SIZE 32
#define COARSE_FACTOR 4

__global__ void subarr_2d_kernel(const int *input, int *output, int N, int M, int S_ROW, int E_ROW, int S_COL, int E_COL)
{
    int sum = 0;

    int base_col = blockIdx.x * blockDim.x * COARSE_FACTOR + threadIdx.x;
    int base_row = blockIdx.y * blockDim.y * COARSE_FACTOR + threadIdx.y;
#pragma unroll
    for (int cfr = 0; cfr < COARSE_FACTOR; cfr++)
    {
        // col = block_offset + thread_offset + thread_base_offset
        // block_offset = blockDim.x * COARSE_FACTOR * blockIdx.x
        // thread_offset = blockDim.x * cfc
        // thread_base_offset = threadIdx.x
        int row = base_row + cfr * blockDim.y;
#pragma unroll
        for (int cfc = 0; cfc < COARSE_FACTOR; cfc++)
        {
            int col = base_col + cfc * blockDim.x;
            if (row >= S_ROW && row <= E_ROW && col >= S_COL && col <= E_COL)
            {
                sum += input[col + M * row];
            }
        }
    }

#pragma unroll
    for (int stride = (WARP_SIZE >> 1); stride > 0; stride >>= 1)
    {
        sum += __shfl_down_sync(0xffffffff, sum, stride);
    }

    if ((threadIdx.x & 31) == 0)
    {
        atomicAdd(output, sum);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int *input, int *output, int N, int M, int S_ROW, int E_ROW, int S_COL,
                      int E_COL)
{
    // 每个block处理的数据量
    int stride = BLOCK_SIZE * COARSE_FACTOR;
    // 总共需要的block数量
    dim3 gridDim(CEIL_DIV(M, stride), CEIL_DIV(N, stride));
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    cudaMemset(output, 0, sizeof(int));
    subarr_2d_kernel<<<gridDim, blockDim>>>(input, output, N, M, S_ROW, E_ROW, S_COL, E_COL);
}