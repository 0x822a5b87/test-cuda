#include <cuda_runtime.h>

// Use float4 and grid-stride looping to optimize performance.
__global__ void vector_add_for_4(const float *A, const float *B, float *C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N / 4) {
        const auto *A4 = reinterpret_cast<const float4 *>(A);
        const auto* B4 = reinterpret_cast<const float4 *>(B);
        auto C4 = reinterpret_cast<float4 *>(C);
        const auto a = A4[tid];
        const auto b = B4[tid];
        float4 res;
        res.x = a.x + b.x;
        res.y = a.y + b.y;
        res.z = a.z + b.z;
        res.w = a.w + b.w;
        C4[tid] = res;
    }
}


__global__ void vector_add(const float *A, const float *B, float *C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}


// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    blocksPerGrid = (blocksPerGrid + 3) / 4;

    if (N % 4 == 0) {
        vector_add_for_4<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    } else {
        vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    }
    cudaDeviceSynchronize();
}

int main(int argc, char *argv[]) {

}
