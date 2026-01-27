#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <util.cuh>

__global__ void transpose_tiled_kernel(float *out, const float *in, int width, int height)
{
    __shared__ float tile[32][32];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }
    __syncthreads();
    // Assuming that we have a 4 * 4 matrix, the matrix is:
    // 0 1 2 3
    // 4 5 6 7
    // 8 9 a b
    // c d e f
    // Assuming that we're running block 0, so the tile will be :
    // 0 4
    // 1 5
    // Now the submatrix is updated in shared memory, what we have to do now is to update the value in current block to the new block.
    int new_x = blockIdx.y * blockDim.y+ threadIdx.x;
    int new_y = blockIdx.x * blockDim.x + threadIdx.y;
    out[new_y * height + new_x] = tile[threadIdx.x][threadIdx.y];
}

int main(int argc, char const *argv[])
{
    int nx = 1024 * 32;
    int ny = 512 * 32;
    size_t total_bytes = nx * ny * sizeof(float);
    float *h_in, *h_out;
    CHECK(cudaHostAlloc(&h_in, total_bytes, cudaHostAllocDefault));
    CHECK(cudaHostAlloc(&h_out, total_bytes, cudaHostAllocDefault));
    for (int i = 0; i < nx * ny; i++)
    {
        h_in[i] = i;
        h_out[i] = 0;
    }

    dim3 block(32, 32);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    float *d_in;
    float *d_out;
    CHECK(cudaMalloc(&d_in, total_bytes));
    CHECK(cudaMalloc(&d_out, total_bytes));
    CHECK(cudaMemcpy(d_in, h_in, total_bytes, cudaMemcpyHostToDevice));
    transpose_tiled_kernel<<<grid, block>>>(d_out, d_in, nx, ny);
    CHECK(cudaMemcpy(h_out, d_out, total_bytes, cudaMemcpyDeviceToHost));
    int errorCount = 0;
    CHECK_MATRIX_TRANSPOSE(h_in, h_out, nx, ny, errorCount);
    if (errorCount > 0)
    {
        std::cout << "error count: " << errorCount << std::endl;
    }

    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_in));
    CHECK(cudaFreeHost(h_out));
    CHECK(cudaFreeHost(h_in));

    return 0;
}
