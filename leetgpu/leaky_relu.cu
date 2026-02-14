#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

constexpr unsigned ThreadPerBlock = 256;
constexpr unsigned ValuePerThread = 4;
constexpr float Alpha = 0.01;

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

template <typename T>
__device__ void leaky_relu(T &v, T alpha)
{
    v = (v < static_cast<T>(0)) ? v * alpha : v;
}

__global__ void leaky_relu_kernel(const float *input, float *output, int N)
{
    unsigned tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tx < N)
    {
        float v = input[tx];
        leaky_relu(v, Alpha);
        output[tx] = v;
    }
}

__global__ void leaky_relu_kernel_float4(const float4 *input, float4 *output, int N)
{
    unsigned base_offset = blockIdx.x * blockDim.x * ValuePerThread + threadIdx.x;
    for (int i = 0; i < ValuePerThread; i++)
    {
        unsigned gx = base_offset + i * ThreadPerBlock;
        if (gx < N) {
            float4 v = input[gx];
            leaky_relu(v.x, Alpha);
            leaky_relu(v.y, Alpha);
            leaky_relu(v.z, Alpha);
            leaky_relu(v.w, Alpha);
            output[gx] = v;
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *input, float *output, int N)
{
    const unsigned vec_num = N / 4;
    const unsigned remained_num = N % 4;

    if (vec_num > 0)
    {
        int valuePerBlock = ThreadPerBlock * ValuePerThread;
        int blocksPerGrid = CEIL_DIV(vec_num, valuePerBlock);
        leaky_relu_kernel_float4<<<blocksPerGrid, ThreadPerBlock>>>(
            reinterpret_cast<const float4 *>(input),
            reinterpret_cast<float4 *>(output),
            vec_num);
    }

    if (remained_num > 0)
    {
        leaky_relu_kernel<<<1, remained_num>>>(input + vec_num * 4, output + vec_num * 4, remained_num);
    }

    cudaDeviceSynchronize();
}


// 辅助函数：检查 CUDA 错误
void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        exit(-1);
    }
}

// int main(int argc, char** argv) {
//     // 1. 配置输入长度：默认 1,000,007 (一个质数，方便测试尾部逻辑)
//     int N = 1000007;
//     if (argc > 1) {
//         N = std::stoi(argv[1]);
//     }

//     std::cout << "Testing ReLU with N = " << N << std::endl;

//     // 2. 在 Host 端准备数据
//     std::vector<float> h_input(N);
//     std::vector<float> h_output_gpu(N);
//     std::vector<float> h_output_cpu(N);

//     for (int i = 0; i < N; ++i) {
//         // 生成 [-10.0, 10.0] 之间的随机数
//         h_input[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 20.0f - 10.0f;
//     }

//     // 3. 在 Device (GPU) 端分配内存
//     float *d_input, *d_output;
//     checkCuda(cudaMalloc(&d_input, N * sizeof(float)));
//     checkCuda(cudaMalloc(&d_output, N * sizeof(float)));

//     // 4. 将数据拷贝到 GPU
//     checkCuda(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

//     // 5. 调用你的 solve 函数
//     solve(d_input, d_output, N);

//     // 6. 将结果拷贝回 Host
//     checkCuda(cudaMemcpy(h_output_gpu.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

//     // 7. CPU 黄金参考逻辑 (用于检查结果)
//     for (int i = 0; i < N; ++i) {
//         h_output_cpu[i] = (h_input[i] > 0.0f) ? h_input[i] : h_input[i] * Alpha;
//     }

//     // 8. 结果校验
//     int errors = 0;
//     double max_err = 0;
//     for (int i = 0; i < N; ++i) {
//         float diff = std::abs(h_output_gpu[i] - h_output_cpu[i]);
//         if (diff > 1e-6) {
//             if (errors < 10) { // 只打印前 10 个错误
//                 std::cout << "Error at index " << i << ": CPU=" << h_output_cpu[i] 
//                           << ", GPU=" << h_output_gpu[i] << std::endl;
//             }
//             errors++;
//         }
//         max_err = std::max(max_err, (double)diff);
//     }

//     if (errors == 0) {
//         std::cout << "VERIFICATION PASSED! (Max Error: " << max_err << ")" << std::endl;
//     } else {
//         std::cout << "VERIFICATION FAILED! Total errors: " << errors << std::endl;
//     }

//     // 9. 释放内存
//     checkCuda(cudaFree(d_input));
//     checkCuda(cudaFree(d_output));

//     return 0;
// }