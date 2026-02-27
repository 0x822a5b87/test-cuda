#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoid_kernel(const float *X, float *Y, int N)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N)
    {
        Y[idx] = __frcp_rn(1.0f + __expf(-X[idx]));
    }
}

// X, Y are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *X, float *Y, int N)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
    cudaDeviceSynchronize();
}
