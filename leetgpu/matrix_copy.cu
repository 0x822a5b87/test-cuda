#include <cuda_runtime.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

const unsigned COUNT_PER_THREAD = 4;
const unsigned THREAD_PER_BLOCK = 64;

__global__ void copy_matrix_kernel_vectorized_grid_stide(const float4* A, float4* B, int N) {
    unsigned gx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned stride = gridDim.x * blockDim.x; 

    for (unsigned t = 0; t * stride < N; ++t) {
    #pragma unroll
    for (unsigned i = 0; i < COUNT_PER_THREAD; i++) {
        unsigned cx = gx + i * stride; 
        if (cx < N) {
            B[cx] = A[cx];
        }
    }
    }
}

__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    unsigned x = threadIdx.x;
    if (x < N) {
        B[x] = A[x];
    }
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, float* B, int N) {
    int total = N * N;
    int num_of_float4 = total / 4;
    int num_of_remained = total % 4;

    if (num_of_float4 > 0) {
        int blocks = CEIL_DIV(num_of_float4, THREAD_PER_BLOCK * COUNT_PER_THREAD);
        copy_matrix_kernel_vectorized_grid_stide<<<blocks, THREAD_PER_BLOCK>>>(
            reinterpret_cast<const float4*>(A),
            reinterpret_cast<float4*>(B),
            num_of_float4
        );
    }

    if (num_of_remained > 0) {
        copy_matrix_kernel<<<num_of_remained, 1>>>(
            A + num_of_float4 * 4,
            B + num_of_float4 * 4,
            num_of_remained
        );
    }

    cudaDeviceSynchronize();
}
