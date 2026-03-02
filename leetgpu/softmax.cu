 #include <cuda_runtime.h>

 const unsigned BLOCKS = 426;

__global__ void max_kernel(const float* input, float* max, int N) {}

__global__ void softmax_kernel(const float* input, float* output, int N) {}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    float* d_max;
    float* d_sum;
    cudaMallocAsync(&d_max, sizeof(float), cudaStreamDefault);
    cudaMallocAsync(&d_sum, sizeof(float), cudaStreamDefault);
    max_kernel<<<BLOCKS, threadsPerBlock>>>(input, d_max, N);
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
