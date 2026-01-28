#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <sgemm_util.cuh>

__global__ void naive_sgemm_kernel(const device_sgemm_t d_sgemm)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float value = 0.0;
    if (row < d_sgemm.M && col < d_sgemm.N)
    {
        for (int k = 0; k < d_sgemm.K; ++k)
        {
            int index_a = row * d_sgemm.K + k;
            int index_b = k * d_sgemm.N + col;
            value += d_sgemm.A[index_a] * d_sgemm.B[index_b];
        }
        d_sgemm.C[row * d_sgemm.N + col] = value;
    }
}

void launch_sgemm_kernel(const device_sgemm_t *d_sgemm)
{
    dim3 blockDim(32, 8);
    dim3 gridDim((d_sgemm->N + blockDim.x - 1) / blockDim.x,
                 (d_sgemm->M + blockDim.y - 1) / blockDim.y);

    naive_sgemm_kernel<<<gridDim, blockDim>>>(*d_sgemm);
    cudaDeviceSynchronize();
}

int main(int argc, char const *argv[])
{
    const int M = 1024 * 16, N = 512 * 16, K = 2048;
    run(M, N, K);
    return 0;
}
