#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdio>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define CLEAR_LOWER_16(addr) ((addr) & (~0xFULL))

__global__ void sum_kernel_3d_masked(const int *base_ptr, int *output,
                                     int W, int H,
                                     int S_D, int E_D, int S_H, int E_H, int S_W, int E_W)
{
    // init parameters
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int global_wid = global_tid / warpSize;
    int lane = threadIdx.x % warpSize;
    int total_warps = (gridDim.x * blockDim.x) / 32;

    // init target info
    int sub_w = E_W - S_W + 1;
    int sub_h = E_H - S_H + 1;
    int sub_d = E_D - S_D + 1;
    int total_lines = sub_d * sub_h;

    // init start parameters
    int local_sum = 0;
    for (int idx = global_wid; idx < total_lines; idx += total_warps)
    {
        int deep = idx / sub_h;
        int row_in_deep = idx % sub_h;

        int curr_d = S_D + deep;
        int curr_h = S_H + row_in_deep;

        const int *row_ptr = base_ptr + (size_t)curr_d * W * H + (size_t)curr_h * W + S_W;
        uintptr_t s_addr = reinterpret_cast<uintptr_t>(row_ptr);
        uintptr_t s_aligned_addr = CLEAR_LOWER_16(s_addr);
        int vec_offset = (s_addr - s_aligned_addr) / sizeof(int);
        // printf("s_addr = %p, s_aligned_addr = %p, vec_offset = %d\n", s_addr, s_aligned_addr, vec_offset);

        uintptr_t e_addr = reinterpret_cast<uintptr_t>(row_ptr + sub_w);
        int vec_num = CEIL_DIV(e_addr - s_aligned_addr, 16);
        const int4 *data_line = reinterpret_cast<const int4 *>(s_aligned_addr);
        for (int line_idx = lane; line_idx < vec_num; line_idx += warpSize)
        {
            int4 val = data_line[line_idx];
            int s_ptr_val = line_idx * 4 - vec_offset;
            local_sum += (0 <= s_ptr_val + 0 && s_ptr_val + 0 < sub_w) ? val.x : 0;
            local_sum += (0 <= s_ptr_val + 1 && s_ptr_val + 1 < sub_w) ? val.y : 0;
            local_sum += (0 <= s_ptr_val + 2 && s_ptr_val + 2 < sub_w) ? val.z : 0;
            local_sum += (0 <= s_ptr_val + 3 && s_ptr_val + 3 < sub_w) ? val.w : 0;
        }
    }

#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
    {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, delta);
    }

    if (lane == 0)
    {
        atomicAdd(output, local_sum);
    }
}

extern "C" void solve(const int *input, int *output,
                      int D, int H, int W,
                      int S_D, int E_D, int S_H, int E_H, int S_W, int E_W)
{
    cudaMemset(output, 0, sizeof(int));
    if (E_D < S_D || E_H < S_H || E_W < S_W)
    {
        return;
    }
    sum_kernel_3d_masked<<<120, 256>>>(
        input, output,
        W, H,
        S_D, E_D, S_H, E_H, S_W, E_W);
}