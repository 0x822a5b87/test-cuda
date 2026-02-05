#include "cuda_runtime.h"
#include <util.cuh>
#include <reduce_util.cuh>

template<int THREAD_NUM>
__global__ void reduce_vectorized(const int *arr, int *out, const int len) {
    __shared__ int ssm[THREAD_NUM * 4];
    const unsigned tx = threadIdx.x;
    const unsigned global_tx = (blockIdx.x * blockDim.x) + tx;
    if (global_tx * 4 < len) {
        const int4 data = reinterpret_cast<const int4*>(arr)[global_tx];
        reinterpret_cast<int4*>(ssm)[tx] = data;
    } else {
        reinterpret_cast<int4*>(ssm)[tx] = make_int4(0, 0, 0, 0);
    }

    __syncthreads();

    const size_t start = tx * 4;
    // 这里，我们将 [tx * 4, tx * 4 + 3] 这段数据计算后存储到 tx
    // 当当前block的所有线程执行完毕之后，所有的数据相当于被压缩到了 [0, THREAD_NUM) 这段 ssm 内
    // 后续的 ssm 数据已经不再使用
    const int int4_sum = ssm[start] + ssm[start + 1] + ssm[start + 2] + ssm[start + 3];
    __syncthreads();
    ssm[tx] = int4_sum;
    __syncthreads();

    for (size_t step = THREAD_NUM >> 1; step > 0; step >>= 1) {
        if (tx < step) {
            ssm[tx] += ssm[tx + step];
        }
        __syncthreads();
    }

    if (tx == 0) {
        atomicAdd(out, ssm[0]);
    }
}

int main(int argc, char *argv[]) {
    constexpr size_t len = 100000000;
    int h_out = 0;

    int *h_arr = allocateIntArrOnHost(sizeof(int) * len);
    printf("h_arr = %p\n", h_arr);

    int *d_out;
    CHECK(cudaMalloc(&d_out, sizeof(int)))
    int *d_arr = allocateArrOnDevice<int>(sizeof(int) * len);
    printf("d_arr = %p\n", d_arr);
    CHECK(cudaMemcpy(d_arr, h_arr, sizeof(int) * len, cudaMemcpyHostToDevice));

    constexpr int thread_num = 32;
    constexpr int block_num = (len + thread_num - 1) / thread_num;
    reduce_vectorized<thread_num><<<block_num, thread_num>>>(d_arr, d_out, len);

    CHECK(cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    checkReduceResult(h_out, h_arr, len);

    CHECK(cudaFree(d_arr));
    CHECK(cudaFree(d_out));
    free(h_arr);
}