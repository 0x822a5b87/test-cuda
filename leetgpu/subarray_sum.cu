#include <cuda_runtime.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

__global__ void reduction_kernal(const int *input, int *output, const int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    int val = 0;
    int vals[4];
    if (i * 4 + 3 < N)
    {
        // 处理float4对应的内容
        const auto idx = i * 4;
        vals[0] = input[idx];
        vals[1] = input[idx + 1];
        vals[2] = input[idx + 2];
        vals[3] = input[idx + 3];
        val = vals[0] + vals[1] + vals[2] + vals[3];
    }
    else if (i * 4 < N)
    {
        const auto idx = i * 4;
        vals[0] = input[idx];
        vals[1] = idx + 1 < N ? input[idx + 1] : 0;
        vals[2] = idx + 2 < N ? input[idx + 2] : 0;
        vals[3] = idx + 3 < N ? input[idx + 3] : 0;
        val = vals[0] + vals[1] + vals[2] + vals[3];
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        constexpr unsigned FULL_MASK = 0xffffffff;
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    if (i % 32 == 0)
    {
        atomicAdd(output, val);
    }
}
// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int *input, int *output, int N, int S, int E)
{
    // 数组起始索引
    input = input + S;
    // 数组长度
    N = E - S + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = CEIL_DIV(CEIL_DIV(N, 4), threadsPerBlock);

    reduction_kernal<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
