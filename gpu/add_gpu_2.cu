#include <stdio.h>
#include <stdlib.h>
#include <chrono>

__global__ void add_parallel(float* x, float* y , float* r, int n) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        *(r + i) += *(x + i) + *(y + i);
    }
}

__global__ void add_stride(float* x, float* y , float* r, int n) {
    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int i = index; i < n; i += stride) {
        *(r + i) += *(x + i) + *(y + i);
    }
}

__global__ void add(float* x, float* y , float* r, int n) {
    for (int i = 0; i < n; ++i) {
        *(r + i) += *(x + i) + *(y + i);
    }
}

__global__ void atomic_add(float* x, float* y , float* r, int n) {
    for (int i = 0; i < n; ++i) {
        float val = *(x + i) + *(y + i);
        atomicAdd(r + i, val); // 原子加法
    }
}

void call_add() {
    int n = 100000000;
    int mem_size = sizeof(float) * n;

    float *x, *y, *r;
    x = static_cast<float*>(malloc(mem_size));
    y = static_cast<float*>(malloc(mem_size));
    r = static_cast<float*>(malloc(mem_size));
    for (int i = 0; i < n; ++i) {
        *(x + i) = 1;
        *(y + i) = 2;
    }

    float *cuda_x, *cuda_y, *cuda_r;
    auto e = cudaMalloc(&cuda_x, mem_size);
    if (e != cudaSuccess) {
        printf("Error code: %d\n", e);
    }
    e = cudaMemcpy(cuda_x, x, mem_size, cudaMemcpyKind::cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        printf("Error code: %d\n", e);
    }

    cudaMalloc(&cuda_y, mem_size);
    cudaMemcpy(cuda_y, y, mem_size, cudaMemcpyKind::cudaMemcpyHostToDevice);

    cudaMalloc(&cuda_r, mem_size);
    cudaMemcpy(cuda_r, r, mem_size, cudaMemcpyKind::cudaMemcpyHostToDevice);

    add_parallel<<<136, 256>>>(cuda_x, cuda_y, cuda_r, n);

    cudaMemcpy(r, cuda_r, mem_size, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) {
        if (*(r + i) != 3) {
            printf("Error at index %d: %f\n", i, *(r + i));
            break;
        }
    }
 
    cudaFree(cuda_r);
    cudaFree(cuda_y);
    cudaFree(cuda_x);
    free(r);
    free(y);
    free(x);
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
