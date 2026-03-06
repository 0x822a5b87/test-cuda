#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

// -------------------------------------------------------------------------------------------
// Kernel: 2D Max Pooling for NCHW layout
// -------------------------------------------------------------------------------------------
__global__ void max_pool_nchw_optimized(const float *__restrict__ input,
                                        float *__restrict__ output,
                                        int N, int C, int H, int W,
                                        int OUT_H, int OUT_W,
                                        int K, int S, int P)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;

    int nc = blockIdx.z;
    int c = nc % C;
    int n = nc / C;

    if (ow < OUT_W && oh < OUT_H)
    {
        const float *line_ptr = input + (size_t)n * C * H * W + (size_t)c * H * W;

        float max_val = -FLT_MAX;

        int ih_start = oh * S - P;
        int iw_start = ow * S - P;

#pragma unroll
        for (int kh = 0; kh < K; ++kh)
        {
            int ih = ih_start + kh;
            if (ih >= 0 && ih < H)
            {
#pragma unroll
                for (int kw = 0; kw < K; ++kw)
                {
                    int iw = iw_start + kw;
                    if (iw >= 0 && iw < W)
                    {
                        // ih = (oh * S - P) * W + iw
                        // ih 分为两个部分
                        // 1. (oh * S - P) * W，该部分对于同一行的数据是完全一致的
                        // 2. (ow * S - P) + kw，该部分 kw 一样，但是 ow * S 会导致，在 stride 不为1时发生非合并访问
                        float val = line_ptr[ih * W + iw];
                        if (val > max_val)
                            max_val = val;
                    }
                }
            }
        }

        // index ：n * (C * OUT_H * OUT_W) + c * (OUT_H * OUT_W) + oh * OUT_W + ow
        size_t out_idx = (size_t)n * C * OUT_H * OUT_W + (size_t)c * OUT_H * OUT_W + (size_t)oh * OUT_W + ow;
        output[out_idx] = max_val;
    }
}

// -------------------------------------------------------------------------------------------
// Host side: solve function
// -------------------------------------------------------------------------------------------
extern "C" void solve(const float *input, float *output, int N, int C, int H, int W,
                      int kernel_size, int stride, int padding)
{
    int OUT_W = (W + 2 * padding - kernel_size) / stride + 1;
    int OUT_H = (H + 2 * padding - kernel_size) / stride + 1;

    if (OUT_W <= 0 || OUT_H <= 0)
        return;

    dim3 block(32, 8, 1);

    // x -> OUT_W
    // y -> OUT_H
    // z -> N * C
    dim3 grid(
        CEIL_DIV(OUT_W, 32),
        CEIL_DIV(OUT_H, 8),
        N * C);

    max_pool_nchw_optimized<<<grid, block>>>(
        input, output,
        N, C, H, W,
        OUT_H, OUT_W,
        kernel_size, stride, padding);
}