#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

__global__ void max_pool_nchw_optimized(const float *__restrict__ input, float *output,
                                        int N, int C, int H, int W,
                                        int OH, int OW,
                                        int kernel_size, int stride, int padding)
{
    int nc = blockIdx.z * blockDim.z + threadIdx.z;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;

    if (ow < OW && oh < OH)
    {
        int batch = nc / C;
        int channel = nc % C;
        // (ow, oh) 在输入的 batch， channel 下的相对偏移量
        int s_h_input = oh * stride - padding;
        int s_w_input = ow * stride - padding;

        const float *s_ptr = input + (size_t)batch * C * H * W + (size_t)channel * H * W;
        float max_val = -FLT_MAX;
#pragma unroll
        for (int kh = 0; kh < kernel_size; kh++)
        {
            // keep the condition so that it can skip the unnecessary loop
            int h_input = s_h_input + kh;
            if (h_input >= 0 && h_input < H)
            {
#pragma unroll
                for (int kw = 0; kw < kernel_size; kw++)
                {
                    // int w_input = s_w_input + kw;
                    // float v = (w_input >= 0 && w_input < W) ? s_ptr[h_input * W + w_input] : -FLT_MAX;
                    // max_val = fmaxf(max_val, v);

                    int w_input = s_w_input + kw;
                    if (w_input >= 0 && w_input < W)
                    {
                        // ih = (oh * S - P) * W + iw
                        // ih 分为两个部分
                        // 1. (oh * S - P) * W，该部分对于同一行的数据是完全一致的
                        // 2. (ow * S - P) + kw，该部分 kw 一样，但是 ow * S 会导致，在 stride 不为1时发生非合并访问
                        float val = s_ptr[h_input * W + w_input];
                        if (val > max_val)
                            max_val = val;
                    }
                }
            }
        }

        size_t idx = (size_t)batch * C * OW * OH + (size_t)channel * OW * OH + oh * OW + ow;
        output[idx] = max_val;
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