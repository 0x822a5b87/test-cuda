#include "cuda_runtime.h"
#include <util.cuh>
#include <reduce_util.cuh>

template<int THREAD_NUM>
__global__ void reduce_eliminate_divergence(const int *arr, int *out, const int len) {
    __shared__ int ssm_data[THREAD_NUM];
    const unsigned tx = threadIdx.x;
    const unsigned global_tx = GLOBAL_TX;
    if (global_tx < len) {
        ssm_data[tx] = arr[global_tx];
    }

    __syncthreads();
    for (size_t step = THREAD_NUM >> 1; step > 0; step >>= 1) {
        if (tx + step < THREAD_NUM) {
            ssm_data[tx] += ssm_data[tx + step];
        }
        __syncthreads();
    }

    if (tx == 0) {
        atomicAdd(out, ssm_data[0]);
    }
}

int main(int argc, char *argv[]) {
    constexpr size_t len = 100000000;
    int h_out = 0;

    int *h_arr = allocateIntArrOnHost(sizeof(int) * len);

    int *d_out;
    CHECK(cudaMalloc(&d_out, sizeof(int)))
    int *d_arr = allocateArrOnDevice<int>(sizeof(int) * len);
    CHECK(cudaMemcpy(d_arr, h_arr, sizeof(int) * len, cudaMemcpyHostToDevice));

    constexpr int thread_num = 8;
    constexpr int block_num = (len + thread_num - 1) / thread_num;
    reduce_eliminate_divergence<thread_num><<<block_num, thread_num>>>(d_arr, d_out, len);

    CHECK(cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    checkReduceResult(h_out, h_arr, len);

    CHECK(cudaFree(d_arr));
    CHECK(cudaFree(d_out));
    free(h_arr);
}