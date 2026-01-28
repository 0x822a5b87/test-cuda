#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <sgemm_util.cuh>

#define TILE_WIDTH 32

// __global__ void shared_sgemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {

//     // 不论是 sA 还是 sB，都必须以 ty 作为一级索引，如果以 tx 作为一级索引的话，
//     // 在后续的计算阶段，线程在访问共享内存时会出现不连续访问，导致性能下降
//     __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
//     __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

//     int tx = threadIdx.x;
//     int ty = threadIdx.y;

//     int row = blockIdx.y * TILE_WIDTH + ty;
//     int col = blockIdx.x * TILE_WIDTH + tx;

//     float value = 0.0f;

//     // 因为我们的 K 维度可能大于 TILE_WIDTH，所以我们必须将 K 维度划分成多个 Tile 来处理
//     // 这个和我们之前的思路中，将矩阵划分成多个 block 来处理是类似的
//     for (int m = 0; m < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
//         // 每一次进入到一个新的 tile 时，我们利用线程将 A 和 B 的数据搬运到共享内存中

//         // row 是 TILE_WIDTH 的行偏移， col 是 TILE_WIDTH 的列偏移

//         // --- 搬运 A ---
//         // A 的行是固定的 row，列在随着 m 步进
//         // 我们的目标是，要将 A 矩阵的 (row, m * TILE_WIDTH + tx) 元素搬运到 sA[ty][tx]
//         // 这样我们在后续的循环中就可以直接从shared memory 中读取数据了
//         // row * K 是行偏移，(m * TILE_WIDTH + tx) 是列偏移
//         if (row < M && (m * TILE_WIDTH + tx) < K) {
//             sA[ty][tx] = A[row * K + (m * TILE_WIDTH + tx)];
//         } else {
//             sA[ty][tx] = 0.0f;
//         }

//         // --- 搬运 B ---
//         // B 的列是固定的 col，行在随着 m 步进
//         // 我们用 ty 映射到 Tile 的行偏移
//         if (col < N && (m * TILE_WIDTH + ty) < K) {
//             // 注意：这里的索引是 (行 * N + 列)
//             // 行是 (m * TILE_WIDTH + ty)，列是 col
//             sB[ty][tx] = B[(m * TILE_WIDTH + ty) * N + col];
//         } else {
//             sB[ty][tx] = 0.0f;
//         }

//         __syncthreads();

//         // --- 计算 ---
//         // 这里的 k 必须在 sA 走列，在 sB 走行
//         for (int k = 0; k < TILE_WIDTH; ++k) {
//             value += sA[ty][k] * sB[k][tx];
//         }

//         __syncthreads();
//     }

//     if (row < M && col < N) {
//         C[row * N + col] = value;
//     }
// }

__global__ void shared_sgemm_kernel(const float *A, const float *B, float *C, int M, int N, int K)
{
    // sA 和 sB 都只能以 y 作为一级索引，x 作为二级索引，否则无法合并访问
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // row 是 TILE_WIDTH 的行偏移， col 是 TILE_WIDTH 的列偏移
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    float value = 0.0f;
    for (int m = 0; m < (K - 1 + TILE_WIDTH) / TILE_WIDTH; m++)
    {
        // 注意，我们的 A 和 B 有一个最大的区别是：
        // 1. A 的索引是从 (row * K) 开始的，因为它的行在 `block` 启动时就已经确定了，所以它的行偏移量也确定了；而它的列是会从左往右移动的；
        // 2. B 的索引是从 (tile_row * N) 开始的，因为它的行是会从上往下移动的；而它的列 col 是在启动时就确定的。

        // 搬运 sA，sA 是在矩阵 A 中一个从左往右移动的矩阵
        // 1. row 是我们的 `block` 中当前线程的起始行，也就是说 C(row, col) 就是我们要计算的这个点；
        // 2. m * TILE_WIDTH + tx 是我们在循环中的列。
        // 这里需要满足行和列都在矩阵的范围内
        int tile_col = m * TILE_WIDTH + tx;
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
