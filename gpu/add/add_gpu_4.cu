#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void add_parallel(const float* x, const float* y, float* r, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        r[i] = x[i] + y[i];
    }
}

struct add_ctx_t {
    float *h_x;
    float *h_y;
    float *h_r;
    float *d_x;
    float *d_y; 
    float *d_r;
};

void destroy_add_ctx(add_ctx_t& ctx) {
    cudaFree(ctx.d_r);
    cudaFree(ctx.d_y);
    cudaFree(ctx.d_x);
    cudaFreeHost(ctx.h_r);
    cudaFreeHost(ctx.h_y);
    cudaFreeHost(ctx.h_x);
}

add_ctx_t init_add_ctx(const int N, const size_t total_bytes) {
    float *h_x, *h_y, *h_r;
    cudaHostAlloc(&h_x, total_bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_y, total_bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_r, total_bytes, cudaHostAllocDefault);

    for (int i = 0; i < N; i++) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }

    float *d_x, *d_y, *d_r;
    cudaMalloc(&d_x, total_bytes);
    cudaMalloc(&d_y, total_bytes);
    cudaMalloc(&d_r, total_bytes);

    add_ctx_t ctx = {h_x, h_y, h_r, d_x, d_y, d_r};
    return ctx;
}

int main() {
    const int N = 100000000;
    const size_t total_bytes = N * sizeof(float);

    add_ctx_t ctx = init_add_ctx(N, total_bytes);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    const int nStreams = 4;
    const int streamSize = N / nStreams;
    const size_t streamBytes = streamSize * sizeof(float);
    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    for (int i = 0; i < nStreams; i++) {
        int offset = i * streamSize;

        cudaMemcpyAsync(ctx.d_x + offset, ctx.h_x + offset, streamBytes, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(ctx.d_y + offset, ctx.h_y + offset, streamBytes, cudaMemcpyHostToDevice, streams[i]);

        int threadsPerBlock = 256;
        int blocksPerGrid = (streamSize + threadsPerBlock - 1) / threadsPerBlock;
        std::cout << "blocksPerGrid: " << blocksPerGrid << std::endl;
        add_parallel<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(ctx.d_x + offset, ctx.d_y + offset, ctx.d_r + offset, streamSize);

        cudaMemcpyAsync(ctx.h_r + offset, ctx.d_r + offset, streamBytes, cudaMemcpyDeviceToHost, streams[i]);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();

    int count = 0;
    for (int i = 0; i < N; ++i) {
        if (*(ctx.h_r + i) != 3) {
            count++;
        }
    }

    if (count != 0) {
        std::cout << "Error count : " << count << std::endl;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Pipeline Execution Time (GPU Hardware): " << milliseconds << " ms" << std::endl;

    for (int i = 0; i < nStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    destroy_add_ctx(ctx);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}