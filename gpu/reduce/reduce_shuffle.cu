#include "cuda_runtime.h"
#include <util.cuh>
#include <reduce_util.cuh>

constexpr unsigned NUMBER_PER_THREAD = 4;

template<int THREAD_NUM>
__global__ void reduce_vectorized(const int *arr, int *out, const int len) {
    __shared__ int ssm[THREAD_NUM];
    const unsigned tx = threadIdx.x;
    const unsigned global_tx = (blockIdx.x * blockDim.x) + tx;
    if (global_tx * NUMBER_PER_THREAD < len) {
        const auto [x, y, z, w] = reinterpret_cast<const int4*>(arr)[global_tx];
        ssm[tx] = x + y + z + w;
    } else {
        ssm[tx] = 0;
    }

    __syncthreads();

    // 我们使用 for 循环执行前面的部分
    for (size_t step = THREAD_NUM >> 1; step > 32; step >>= 1) {
        if (tx < step) {
            ssm[tx] += ssm[tx + step];
        }
        __syncthreads();
    }

    // 当执行到该位置时，由于我们在 for 循环中调用了 __syncthreads()，所以所有的线程都已经执行到该位置
    // 这也就表明，所有的数据都已经被规约到 ssm[0] ~ ssm[31]
    // 此时我们只需要使用 __shfl_down_sync 来计算最后的结果即可。
    if (tx < 32) {
        int sum = ssm[tx];
        // 注意，最后一次计算时 step > 32，也就是我们的值现在规约到了 ssm[0] ~ ssm[63]
        if (THREAD_NUM > 32 && tx + 32 < THREAD_NUM) {
            sum += ssm[tx + 32];
        }
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);
        if (tx == 0) {
            atomicAdd(out, sum);
        }
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

    constexpr int thread_num = 128;
    constexpr int stride = thread_num * NUMBER_PER_THREAD;
    constexpr int block_num = (len + stride - 1) / stride;
    reduce_vectorized<thread_num><<<block_num, thread_num>>>(d_arr, d_out, len);

    CHECK(cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    checkReduceResult(h_out, h_arr, len);

    CHECK(cudaFree(d_arr));
    CHECK(cudaFree(d_out));
    free(h_arr);
}