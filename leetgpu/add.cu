#include <cuda_runtime.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

constexpr unsigned THREAD_PER_BLOCK = 256;

__global__ void add_kernel(const float *input, float *output, int N)
{
    __shared__ float ssm[THREAD_PER_BLOCK];

    const unsigned gx = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned tx = threadIdx.x;
    // 注意，这里对于超出范围的我们初始化为 0.0f，这非常重要，可以有效的减少我们后续的边界条件判定
    ssm[tx] = (gx < N) ? input[gx] : 0.0f;
    __syncthreads();

    for (unsigned stride = THREAD_PER_BLOCK / 2; stride >= 32; stride >>= 1)
    {
        if (tx < stride && tx < THREAD_PER_BLOCK)
        {
            ssm[tx] += ssm[tx + stride];
        }
        __syncthreads();
    }

    if (tx < 32) {
        float val = ssm[tx];

        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);

        if (tx == 0) {
            atomicAdd(output, ssm[0]);
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *input, float *output, int N)
{
    int blocksPerGrid = CEIL_DIV(N, THREAD_PER_BLOCK);
    add_kernel<<<blocksPerGrid, THREAD_PER_BLOCK>>>(input, output, N);
    cudaDeviceSynchronize();
}
