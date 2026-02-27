#include <cuda_runtime.h>

__device__ __forceinline__ float gray(float R, float G, float B) {
    return 0.299 * R + 0.587 * G + 0.114 * B;
}

__global__ void rgb_to_grayscale_kernel(const float* input, float* output, int N) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned idx = tid * 3;
    if (idx >= N) {
        return;
    }
    float R = input[idx];
    float G = input[idx + 1];
    float B = input[idx + 2];
    output[idx] = gray(R, G, B);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int width, int height) {
    int total_pixels = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, total_pixels);
    cudaDeviceSynchronize();
}
