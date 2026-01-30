#include <cuda_runtime.h>
#include <cstdio>
#include <util.cuh>

template<int THREAD_NUM>
__global__ void reduce_optimized_v2_kernel(const int *arr, int *out, const int len) {
    __shared__ int ssm_data[THREAD_NUM / 2];

    const unsigned tx = threadIdx.x;
    const unsigned i = blockIdx.x * (THREAD_NUM * 2) + tx;

    int sum = (i < len) ? arr[i] : 0;
    if (i + THREAD_NUM < len) {
        sum += arr[i + THREAD_NUM];
    }

    if (tx < THREAD_NUM / 2) {
    }

    __shared__ int full_ssm[THREAD_NUM];
    full_ssm[tx] = sum;
    __syncthreads();

    if (tx < THREAD_NUM / 2) {
        ssm_data[tx] = full_ssm[tx] + full_ssm[tx + THREAD_NUM / 2];
    }
    __syncthreads();

    for (int step = (THREAD_NUM / 2) / 2; step > 0; step >>= 1) {
        if (tx < step) {
            ssm_data[tx] += ssm_data[tx + step];
        }
        __syncthreads();
    }

    if (tx == 0) {
        atomicAdd(out, ssm_data[0]);
    }
}

int main() {
    constexpr size_t len = 1000;
    constexpr int thread_num = 8;

    int *h_arr = (int*)malloc(sizeof(int) * len);
    int cpu_sum = 0;
    for (int i = 0; i < len; i++) {
        h_arr[i] = i;
        cpu_sum += i;
    }

    int *d_arr, *d_out;
    CHECK(cudaMalloc(&d_arr, sizeof(int) * len));
    CHECK(cudaMalloc(&d_out, sizeof(int)));
    CHECK(cudaMemset(d_out, 0, sizeof(int)));
    CHECK(cudaMemcpy(d_arr, h_arr, sizeof(int) * len, cudaMemcpyHostToDevice));

    int block_num = (len + (thread_num * 2) - 1) / (thread_num * 2);

    reduce_optimized_v2_kernel<thread_num><<<block_num, thread_num>>>(d_arr, d_out, len);

    int h_out = 0;
    CHECK(cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    printf("GPU Result: %d\n", h_out);
    printf("CPU Expected: %d\n", cpu_sum);

    CHECK(cudaFree(d_arr));
    CHECK(cudaFree(d_out));
    free(h_arr);
    return 0;
}