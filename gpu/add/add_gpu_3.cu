#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda_runtime.h>

__global__ void add_parallel(float* x, float* y , float* r, int n) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        *(r + i) += *(x + i) + *(y + i);
    }
}

void call_add() {
    int n = 100000000;
    int mem_size = sizeof(float) * n;

    float *x, *y, *r;
    cudaMallocHost((void**)&x, mem_size);
    cudaMallocHost((void**)&y, mem_size);
    cudaMallocHost((void**)&r, mem_size);

    for (int i = 0; i < n; ++i) {
        *(x + i) = 1;
        *(y + i) = 2;
        *(r + i) = 0;
    }

    float *d_x, *d_y, *d_r;
    cudaMalloc((void**)&d_x, mem_size);
    cudaMalloc((void**)&d_y, mem_size);
    cudaMalloc((void**)&d_r, mem_size);

    cudaMemcpy(d_x, x, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, mem_size, cudaMemcpyHostToDevice);

    add_parallel<<<136, 256>>>(d_x, d_y, d_r, n);
    cudaDeviceSynchronize();
    cudaMemcpy(r, d_r, mem_size, cudaMemcpyDeviceToHost);

    int count = 0;
    for (int i = 0; i < n; ++i) {
        if (*(r + i) != 3) {
            count++;
        }
    }

    if (count != 0) {
        printf("Total errors: %d\n", count);
    }

    cudaFreeHost(r);
    cudaFreeHost(y);
    cudaFreeHost(x);
    cudaFree(d_r);
    cudaFree(d_y);
    cudaFree(d_x);
}

int main(int argc, char const *argv[])
{
    auto start = std::chrono::high_resolution_clock::now();
    call_add();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    printf("Execution time: %f ms\n", elapsed.count());
    return 0;
}
