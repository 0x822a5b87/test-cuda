#include <cuda_runtime.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

const unsigned COUNT_PER_THREAD = 4;
const unsigned THREAD_PER_BLOCK = 128;

__global__ void count_equal_kernel(const int* input, int* output, int N, int K) {
    const int gx = blockDim.x * blockIdx.x * COUNT_PER_THREAD + threadIdx.x;
    for (int i = 0; i < COUNT_PER_THREAD; i++) {
        const int cx = gx + i * COUNT_PER_THREAD;
        if (cx >= N) {
            break;
        }
        if (input[cx] == K) {
            atomicAdd(output, 1);
        }
    }
}

__global__ void count_equal_kernel_vectorized(const int4* input, int* output, int N, int K) {
    __shared__ int k_counter[THREAD_PER_BLOCK];
    unsigned tx = threadIdx.x;
    k_counter[tx] = 0;
    unsigned gx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned grid_size = gridDim.x * blockDim.x;
    unsigned stride = grid_size * COUNT_PER_THREAD;
    for (int start_x = gx; start_x < N; start_x += stride) {
        for (int j = 0; j < COUNT_PER_THREAD; j++) {
            int cx = start_x + j * grid_size;
            if (cx < N) {
                int4 val = input[cx];
                if (val.x == K) k_counter[tx]++;
                if (val.y == K) k_counter[tx]++;
                if (val.z == K) k_counter[tx]++;
                if (val.w == K) k_counter[tx]++;
            }
        }
    }

    // 只有这里才需要等待其他的线程完成
    __syncthreads();
    if (tx == 0) {
        for (int i = 0; i < THREAD_PER_BLOCK; i++) {
            atomicAdd(output, k_counter[i]);
        }
    }
}

__global__ void count_equal_h100_optimized(const int4* input, int* output, int N4, int K) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = gridDim.x * blockDim.x;
    int local_sum = 0;

    for (int i = gtid; i < N4; i += grid_size) {
        int4 v = input[i];
        if (v.x == K) local_sum++;
        if (v.y == K) local_sum++;
        if (v.z == K) local_sum++;
        if (v.w == K) local_sum++;
    }

    for (int mask = 16; mask > 0; mask >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
    }

    if ((threadIdx.x & 31) == 0 && local_sum > 0) {
        atomicAdd(output, local_sum);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int K) {
    const int num_of_int4 = N / 4;
    const int num_of_remained = N % 4;

    if (num_of_int4 > 0) {
        const int num_of_blocks = CEIL_DIV(num_of_int4, THREAD_PER_BLOCK);
        count_equal_h100_optimized<<<num_of_blocks, THREAD_PER_BLOCK>>>(
            reinterpret_cast<const int4*>(input),
            output,
            num_of_int4,
            K
        );
    }

    if (num_of_remained > 0) {
        const int num_of_blocks_remains = CEIL_DIV(num_of_remained, COUNT_PER_THREAD);
        count_equal_kernel<<<num_of_blocks_remains, COUNT_PER_THREAD>>>(
            input + num_of_int4 * 4,
            output,
            num_of_remained,
            K);
    }

    cudaDeviceSynchronize();
}
