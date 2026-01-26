#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <util.cuh>

__global__ void transpose_kernel(float *out, const float *in, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        out[x * height + y] = in[y * width + x];
    }
}

int main(int argc, char const *argv[])
{
    int nx = 1024;
    int ny = 512;
    size_t total_bytes = nx * ny * sizeof(float);
    float *h_in, *h_out;
    CHECK(cudaHostAlloc(&h_in, total_bytes, cudaHostAllocDefault));
    CHECK(cudaHostAlloc(&h_out, total_bytes, cudaHostAllocDefault));
    for (int i = 0; i < nx * ny; i++) {
        h_in[i] = i;
        h_out[i] = 0;
    }

    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    float *d_in;
    float *d_out;
    CHECK(cudaMalloc(&d_in, total_bytes));
    CHECK(cudaMalloc(&d_out, total_bytes));
    CHECK(cudaMemcpy(d_in, h_in, total_bytes, cudaMemcpyHostToDevice));
    transpose_kernel<<<grid, block>>>(d_out, d_in, nx, ny);
    CHECK(cudaMemcpy(h_out, d_out, total_bytes, cudaMemcpyDeviceToHost));
    int errorCount = 0;
    for (int r = 0; r < ny; r++) {
        for (int c = 0; c < nx; c++) {
            int index_in = r * nx + c;
            int index_out = c * ny + r;
            if (h_in[index_in] != h_out[index_out]) {
                errorCount++;
            }
        }
    }
    if (errorCount > 0) {
        std::cout << "error count: " << errorCount << std::endl;
    }

    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_in));
    CHECK(cudaFreeHost(h_out));
    CHECK(cudaFreeHost(h_in));

    return 0;
}
