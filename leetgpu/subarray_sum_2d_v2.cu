#include <cuda_runtime.h>
#include <stdint.h>

#define WARP_SIZE 32

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define CLEAR_LOWER_16(addr) ((addr) & (~0xFULL))

// Assuming that we cover the 2D space by a 1D grid
__global__ void sum_kernel_2d_masked(const int *base_ptr, int *output,
                                     int M, int S_ROW, int E_ROW, int S_COL, int E_COL)
{
    int local_sum = 0;
    // 计算全局的tid，我们需要使用tid去计算当前线程所归属的warp
    // 这个warp决定了我们处理哪一行
    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    int global_wid = global_tid / WARP_SIZE;
    int lane = threadIdx.x % 32;
    // 总的warp数量，决定了我们外层循环时的时间
    int total_warps = (gridDim.x * blockDim.x) / WARP_SIZE;

    int sub_w = E_COL - S_COL + 1;
    int sub_h = E_ROW - S_ROW + 1;

    // 总共有 sub_h 行，我们每个warp负责其中的一行
    for (int idx = global_wid; idx < sub_h; idx += total_warps)
    {
        // 通过基准行和偏移量找到目标行的入口地址
        int row = S_ROW + idx;

        const int *row_ptr = base_ptr + row * M + S_COL;

        // 通过入口地址计算一个可以16字节对齐的指针
        uintptr_t s_addr = reinterpret_cast<uintptr_t>(row_ptr);
        uintptr_t aligned_row_ptr = CLEAR_LOWER_16(s_addr);
        int vec_offset = static_cast<int>(s_addr - aligned_row_ptr) / sizeof(int);

        uintptr_t e_addr = reinterpret_cast<uintptr_t>(row_ptr + sub_w);
        int vec_num = CEIL_DIV((e_addr - aligned_row_ptr), 16);

        const int4 *data = reinterpret_cast<const int4 *>(aligned_row_ptr);
        for (int i = lane; i < vec_num; i += WARP_SIZE)
        {
            int4 val = data[i];
            int idx_of_sub = i * 4 - vec_offset;

            local_sum += (0 <= idx_of_sub && idx_of_sub < sub_w) ? val.x : 0;
            local_sum += (0 <= idx_of_sub + 1 && idx_of_sub + 1 < sub_w) ? val.y : 0;
            local_sum += (0 <= idx_of_sub + 2 && idx_of_sub + 2 < sub_w) ? val.z : 0;
            local_sum += (0 <= idx_of_sub + 3 && idx_of_sub + 3 < sub_w) ? val.w : 0;
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }
    if (lane == 0)
    {
        atomicAdd(output, local_sum);
    }
}

extern "C" void solve(const int *input, int *output, int N, int M,
                      int S_ROW, int E_ROW, int S_COL, int E_COL)
{
    cudaMemset(output, 0, sizeof(int));
    if (E_ROW < S_ROW || E_COL < S_COL)
    {
        return;
    }
    // 启动 120 个 Block，每个 Block 256 线程，占满典型 GPU
    sum_kernel_2d_masked<<<120, 256>>>(input, output, M, S_ROW, E_ROW, S_COL, E_COL);
}