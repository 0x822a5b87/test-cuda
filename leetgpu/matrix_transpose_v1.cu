#include <cuda_runtime.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

constexpr int TX = 8;
constexpr int TY = 16;

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float ssm[TY][TX];

    int local_row = threadIdx.y;
    int local_col = threadIdx.x;

    int block_row = blockIdx.y * TY;
    int block_col = blockIdx.x * TX;

    int global_row = block_row + local_row;
    int global_col = block_col + local_col;
    if (global_col < cols && global_row < rows) {
        // 转置矩阵到sm
        ssm[local_row][local_col] = input[global_row * cols + global_col];
    }
    __syncthreads();

    int gx_out = block_row + local_col;
    int gy_out = block_col + local_row;
    if (gx_out < rows && gy_out < cols) {
        output[gy_out * rows + gx_out] = ssm[local_col][local_row];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(TX, TY);
    dim3 blocksPerGrid(CEIL_DIV(cols, TX), CEIL_DIV(rows, TY));

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
