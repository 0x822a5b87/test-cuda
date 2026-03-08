#include <cuda_runtime.h>

#define KERNEL_SIZE 2047
#define TILE_SIZE 256
#define ELEM_PER_THREAD 4

__constant__ float d_kernel[KERNEL_SIZE];

__global__ void convolution_1d_kernel(const float *input, const float *kernel, float *output, int input_size, int kernel_size)
{
    extern __shared__ float smem[];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int index = id * ELEM_PER_THREAD;
    int thread_id = threadIdx.x;

    for (int count = ((ELEM_PER_THREAD * TILE_SIZE + kernel_size - 1 + ELEM_PER_THREAD - 1) / ELEM_PER_THREAD) * ELEM_PER_THREAD, i = 0;
         count > 0;
         count = count - ELEM_PER_THREAD * TILE_SIZE, i++)
    {
        if (thread_id * 4 < count)
        {
            if (i * TILE_SIZE * ELEM_PER_THREAD + index + ELEM_PER_THREAD - 1 < input_size)
            {
                float4 temp_vector = reinterpret_cast<const float4 *>(&input[i * TILE_SIZE * ELEM_PER_THREAD + index])[0];
                reinterpret_cast<float4 *>(&smem[i * TILE_SIZE * ELEM_PER_THREAD + thread_id * 4])[0] = temp_vector;
            }
            else if (i * TILE_SIZE * ELEM_PER_THREAD + index < input_size)
            {
                float temp_vector[ELEM_PER_THREAD];
                for (int j = 0; j < ELEM_PER_THREAD; j++)
                {
                    if (i * TILE_SIZE * ELEM_PER_THREAD + index + j < input_size)
                    {
                        temp_vector[j] = input[i * TILE_SIZE * ELEM_PER_THREAD + index + j];
                    }
                }
                for (int j = 0; j < ELEM_PER_THREAD; j++)
                {
                    if (i * TILE_SIZE * ELEM_PER_THREAD + index + j < input_size)
                    {
                        smem[i * TILE_SIZE * ELEM_PER_THREAD + thread_id * 4 + j] = temp_vector[j];
                    }
                }
            }
        }
    }
    __syncthreads();
    float sum[ELEM_PER_THREAD] = {0.0f};
    int smem_base = thread_id * ELEM_PER_THREAD;
    float4 curr_quad = *reinterpret_cast<float4 *>(&smem[smem_base]);
    for (int k = 0; k < kernel_size; k += 4)
    {
        float4 next_quad = {0.0f, 0.0f, 0.0f, 0.0f};
        next_quad = *reinterpret_cast<float4 *>(&smem[smem_base + k + 4]);
        if (k < kernel_size)
        {
            float w = d_kernel[k];
            sum[0] += curr_quad.x * w;
            sum[1] += curr_quad.y * w;
            sum[2] += curr_quad.z * w;
            sum[3] += curr_quad.w * w;
        }
        if (k + 1 < kernel_size)
        {
            float w = d_kernel[k + 1];
            sum[0] += curr_quad.y * w;
            sum[1] += curr_quad.z * w;
            sum[2] += curr_quad.w * w;
            sum[3] += next_quad.x * w;
        }
        if (k + 2 < kernel_size)
        {
            float w = d_kernel[k + 2];
            sum[0] += curr_quad.z * w;
            sum[1] += curr_quad.w * w;
            sum[2] += next_quad.x * w;
            sum[3] += next_quad.y * w;
        }
        if (k + 3 < kernel_size)
        {
            float w = d_kernel[k + 3];
            sum[0] += curr_quad.w * w;
            sum[1] += next_quad.x * w;
            sum[2] += next_quad.y * w;
            sum[3] += next_quad.z * w;
        }
        curr_quad = next_quad;
    }
    int out_index = blockIdx.x * blockDim.x * ELEM_PER_THREAD + thread_id * ELEM_PER_THREAD;
    if (out_index < input_size - kernel_size + 1)
    {
        for (int j = 0; j < ELEM_PER_THREAD; j++)
        {
            if (out_index + j < input_size - kernel_size + 1)
            {
                output[out_index + j] = sum[j];
            }
        }
    }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *input, const float *kernel, float *output, int input_size, int kernel_size)
{
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = TILE_SIZE;
    int elemPerBlock = threadsPerBlock * ELEM_PER_THREAD;
    int blocksPerGrid = (output_size + elemPerBlock - 1) / elemPerBlock;
    cudaMemcpyToSymbolAsync(d_kernel, kernel, kernel_size * (sizeof(float)));
    //    int smem_size=((ELEM_PER_THREAD*TILE_SIZE+kernel_size-1+ELEM_PER_THREAD-1)/ELEM_PER_THREAD)*ELEM_PER_THREAD*sizeof(float);
    int max_smem_elements = ELEM_PER_THREAD * TILE_SIZE + kernel_size + 7;
    int smem_size = ((max_smem_elements + 3) / 4) * 4 * sizeof(float);
    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock, smem_size>>>(input, kernel, output, input_size, kernel_size);
}