#include <cuda_runtime.h>

#define VECTORIZED(c) ((c) * (COUNT_VECTORIZED))

constexpr unsigned BLOCKS_PER_GRID_FOR_H100 = 428;
constexpr unsigned COUNT_VECTORIZED = 4;
constexpr unsigned VECTORIZED_CHECKER = COUNT_VECTORIZED - 1;
// discriminate this value from vectorized number
constexpr unsigned COUNT_PER_THREAD = 8;

__global__ void interleave_kernel(const float *A, const float *B, float *output, int N)
{
    float4 res1, res2;

    const unsigned block_tile = VECTORIZED(blockDim.x) * COUNT_PER_THREAD;
    const unsigned grid_stride = gridDim.x * block_tile;
    const unsigned thread_stride = VECTORIZED(blockDim.x);

    const unsigned base = blockIdx.x * block_tile + VECTORIZED(threadIdx.x);

    for (unsigned pos = base; pos < N; pos += grid_stride)
    {
        for (unsigned ti = 0; ti < COUNT_PER_THREAD; ti++)
        {
            const unsigned idx = pos + ti * thread_stride;
            if (idx + VECTORIZED_CHECKER < N)
            {
                float4 val_a = __ldg(reinterpret_cast<const float4 *>(&A[idx]));
                float4 val_b = __ldg(reinterpret_cast<const float4 *>(&B[idx]));
                res1.x = val_a.x;
                res1.y = val_b.x;
                res1.z = val_a.y;
                res1.w = val_b.y;

                res2.x = val_a.z;
                res2.y = val_b.z;
                res2.z = val_a.w;
                res2.w = val_b.w;

                // 这里需要注意，需要先取 output 的指针再转换为 float4*
                // 每一个线程处理了 A 和 B 的 4 个元素，产生了 8 个交错元素
                // 这 8 个元素正好填满 output[2*idx] 到 output[2*idx + 7]
                reinterpret_cast<float4 *>(&output[2 * idx])[0] = res1;
                reinterpret_cast<float4 *>(&output[2 * idx + 4])[0] = res2;
            }
            else if (idx < N)
            {
                for (unsigned i = idx; i < N; i++)
                {
                    // 这里已经到了收尾阶段，极少量的非合并写入对最终写入性能影响可以忽略
                    output[2 * i] = __ldg(&A[i]);
                    output[2 * i + 1] = __ldg(&B[i]);
                }
            }
        }
    }
}

// A, B, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *output, int N)
{
    int threadsPerBlock = 256;
    interleave_kernel<<<BLOCKS_PER_GRID_FOR_H100, threadsPerBlock>>>(A, B, output, N);
    cudaDeviceSynchronize();
}
