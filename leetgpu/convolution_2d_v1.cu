#include <cuda_runtime.h>

constexpr unsigned TX = 32;
constexpr unsigned TY = 16;

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

__constant__ float kernels[256];

__global__ void conv2d_kern(const float *input, float *output, int input_rows, int input_cols, int kernel_rows, int kernel_cols)
{
    extern __shared__ float sm[];
    unsigned bx = blockIdx.x * blockDim.x, by = blockIdx.y * blockDim.y;
    unsigned tx = threadIdx.x, ty = threadIdx.y;
    unsigned gx = bx + tx, gy = by + ty;

    unsigned tile_w = TX + kernel_cols - 1;
    unsigned tile_h = TY + kernel_rows - 1;

    for (unsigned y = ty; y < tile_h; y += TY)
    {
        for (unsigned x = tx; x < tile_w; x += TX)
        {
            int global_x = bx + x;
            int global_y = by + y;
            if (global_x < input_cols && global_y < input_rows)
            {
                sm[y * tile_w + x] = input[global_y * input_cols + global_x];
            }
            else
            {
                sm[y * tile_w + x] = 0.0f;
            }
        }
    }
    __syncthreads();

    unsigned output_cols = input_cols - kernel_cols + 1;
    unsigned output_rows = input_rows - kernel_rows + 1;
    if (gx < output_cols && gy < output_rows)
    {
        float val = 0.0f;
#pragma unroll 4
        for (unsigned kh = 0; kh < kernel_rows; kh++)
        {
#pragma unroll 4
            for (unsigned kw = 0; kw < kernel_cols; kw++)
            {
                unsigned sm_x = tx + kw;
                unsigned sm_y = ty + kh;
                val += sm[sm_y * tile_w + sm_x] * kernels[kh * kernel_cols + kw];
            }
        }
        output[gy * output_cols + gx] = val;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float *input, const float *kernel, float *output,
                      int input_rows, int input_cols,
                      int kernel_rows, int kernel_cols)
{
    size_t kernel_size = kernel_rows * kernel_cols * sizeof(float);
    cudaMemcpyToSymbol(kernels, kernel, kernel_size);

    dim3 block(TX, TY);
    dim3 grid(CEIL_DIV(input_cols, TX), CEIL_DIV(input_rows, TY));
    unsigned s_mem_size = (TX + kernel_cols - 1) * (TY + kernel_rows - 1) * sizeof(float);
    conv2d_kern<<<grid, block, s_mem_size>>>(input, output, input_rows, input_cols, kernel_rows, kernel_cols);
    cudaDeviceSynchronize();
}
