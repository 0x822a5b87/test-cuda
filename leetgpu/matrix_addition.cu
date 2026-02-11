#include <cstdio>
#include <cuda_runtime.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

constexpr int BX = 256;

__global__ void matrix_add_vectorized(const float4* A, const float4* B, float4* C, int N) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tx < N) {
        float4 a = A[tx];
        float4 b = B[tx];
        C[tx] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
}

__global__ void matrix_add_scalar(const float* A, const float* B, float* C, int N) {
    int bx = blockIdx.x * blockDim.x;
    int tx = bx + threadIdx.x;
    if (tx < N) {
        C[tx] = A[tx] + B[tx];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = BX;

    int total_elements = N * N;
    if (reinterpret_cast<size_t>(A) % 16 == 0
        && reinterpret_cast<size_t>(B) % 16 == 0
        && reinterpret_cast<size_t>(C) % 16 == 0
        && (N * N) % 4 == 0) {

        total_elements /= 4;
        dim3 blocksPerGrid(CEIL_DIV(total_elements, threadsPerBlock));
        const float4 * vA = reinterpret_cast<const float4*>(A);
        const float4 * vB = reinterpret_cast<const float4*>(B);
        float4 * vC = reinterpret_cast<float4*>(C);
        matrix_add_vectorized<<<blocksPerGrid, threadsPerBlock>>>(vA, vB, vC, total_elements);
    } else {
        dim3 blocksPerGrid(CEIL_DIV(total_elements, threadsPerBlock));
        matrix_add_scalar<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, total_elements);
    }

    cudaDeviceSynchronize();
}
