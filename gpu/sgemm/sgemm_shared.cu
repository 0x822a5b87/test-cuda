#include <iostream>
#include <cuda_runtime.h>
#include <sgemm_util.cuh>

#define TILE_WIDTH 32

__global__ void shared_sgemm_kernel(const float *A, const float *B, float *C, int M, int N, int K)
{
    // sA 和 sB 都只能以 y 作为一级索引，x 作为二级索引，否则无法合并访问
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    const unsigned int ty = threadIdx.y;
    const unsigned int tx = threadIdx.x;

    // row 是 TILE_WIDTH 的行偏移， col 是 TILE_WIDTH 的列偏移
    const unsigned int row = blockIdx.y * blockDim.y + ty;
    const unsigned int col = blockIdx.x * blockDim.x + tx;

    float value = 0.0f;
    for (unsigned int m = 0; m < (K - 1 + TILE_WIDTH) / TILE_WIDTH; m++)
    {
        // 注意，我们的 A 和 B 有一个最大的区别是：
        // 1. A 的索引是从 (row * K) 开始的，因为它的行在 `block` 启动时就已经确定了，所以它的行偏移量也确定了；而它的列是会从左往右移动的；
        // 2. B 的索引是从 (tile_row * N) 开始的，因为它的行是会从上往下移动的；而它的列 col 是在启动时就确定的。

        // 搬运 sA，sA 是在矩阵 A 中一个从左往右移动的矩阵
        // 1. row 是我们的 `block` 中当前线程的起始行，也就是说 C(row, col) 就是我们要计算的这个点；
        // 2. m * TILE_WIDTH + tx 是我们在循环中的列。
        // 这里需要满足行和列都在矩阵的范围内
        unsigned int tile_col = m * TILE_WIDTH + tx;
        if (row < M && tile_col < K)
        {
            // row * K 是行偏移，(m * TILE_WIDTH + tx) 是列偏移
            sA[ty][tx] = A[row * K + tile_col];
        }
        else
        {
            sA[ty][tx] = 0.0f;
        }

        // 1. (m * TILE_WIDTH + ty) 是我们 TILE 的行；
        // 2. col 是列，这个在 `block` 启动时就已经确定了。
        int tile_row = m * TILE_WIDTH + ty;
        if (tile_row < K && col < N)
        {
            sB[ty][tx] = B[tile_row * N + col];
        }
        else
        {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++)
        {
            value += sA[ty][i] * sB[i][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

void launch_sgemm_kernel(const device_sgemm_t *d_sgemm)
{
    dim3 blockDim(32, 32);
    dim3 gridDim((d_sgemm->N + blockDim.x - 1) / blockDim.x,
                 (d_sgemm->M + blockDim.y - 1) / blockDim.y);

    shared_sgemm_kernel<<<gridDim, blockDim>>>(d_sgemm->A, d_sgemm->B, d_sgemm->C, d_sgemm->M, d_sgemm->N, d_sgemm->K);
    cudaDeviceSynchronize();
}

int main(int argc, char const *argv[])
{
    const int M = 1024 * 16, N = 512 * 16, K = 2048;
    run(M, N, K);
    return 0;
}
