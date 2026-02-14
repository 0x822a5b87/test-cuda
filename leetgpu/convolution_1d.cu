#include <cuda_runtime.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

constexpr unsigned TN = 256;
constexpr unsigned TILE = 256;

__global__ void convolution_1d_kernel(const float* input, const float* __restrict__ kernel, float* output, int input_size, int kernel_size, int output_size) {
    __shared__ float ssm_data[TN + TILE];

    int bx = blockIdx.x * blockDim.x;
    int tx = threadIdx.x;
    int gx = bx + tx;

    float sum = 0.0f;
    for (int k = 0; k < kernel_size; k += TILE) {
        int current_tile_w = min(TILE, kernel_size - k);
        for (int i = tx; i < TILE + current_tile_w; i += TN) {
            // bx 为 block 偏移量
            // k 为TILE的偏移量
            // i 为TILE内偏移量
            int load_id = bx + k + i;
            if (load_id < input_size) {
                ssm_data[i] = input[load_id];
            } else {
                ssm_data[i] = 0.0;
            }
        }
        __syncthreads();
        if (gx < output_size) {
            #pragma unroll
            for (int i = 0; i < current_tile_w; i++) {
                sum += ssm_data[tx + i] * kernel[k + i];
            }
        }
        __syncthreads();
    }
    if (gx < output_size) {
        output[gx] = sum;
    }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = TN;
    int blocksPerGrid = CEIL_DIV(output_size, threadsPerBlock);
    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output,
        input_size, kernel_size, output_size);
    cudaDeviceSynchronize();
}
