#include <cuda_runtime.h>
#include <cstdio>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

constexpr int BX = 256;

// 整体的逻辑可以概括为：
// 1. 在启动阶段，通过 CEIL_DIV(N, threadsPerBlock * 2) 使得我们的任务只会走到一半；
// 2. 线程同时从读取从左往右的block，以及block的从后往前的镜像 block；
// 3. 为了能够合并访问，在读取时，镜像 block 的需要通过 blockDim.x 进行偏移；
// 4. 此时，从左往右读取数据到 ssm_left 和 ssm_right
// 5. block同时存在一个从左往右和一个从右往左的滑动窗口，需要注意的是，左右两边判断条件不一致；
// 6. 对ssm_left和ssm_right我们反转后写入到input即可，不用担心会被重复反转，因为我们数据是从ssm取的，所以重复反转只是多了一次额外的开销
//      所以最多也只会有255 * 2次额外开销，属于可以接受范围。
__global__ void reverse_array(float* input, int N) {
    __shared__ float ssm_left[BX];
    __shared__ float ssm_right[BX];
    
    int tx = threadIdx.x;
    int bx = blockIdx.x * blockDim.x;

    int idx_l = bx + tx;
    // 锚点 A（基准点）：N（数组尾部边界）
    // 锚点 B（块边界）：N - bx。
    // 锚点 C（寻址起点）：N - bx - blockDim.x
    // 平移：+ tx
    int idx_r_seq = N - bx - blockDim.x + tx;

    if (idx_l < N) {
        ssm_left[tx] = input[idx_l];
    } else {
        ssm_left[tx] = 0.0;
    }
    if (idx_r_seq >= 0 && idx_r_seq < N) {
        ssm_right[tx] = input[idx_r_seq];
    } else {
        ssm_right[tx] = 0;
    }
    __syncthreads();

    // idx_l     = [0, 255]
    // idx_r_sqe = [-251, 4]
    // ssm_left  = [1, 2, 3, 4, 0, ..., 0]
    // ssm_right = [0, 0, ..., 1, 2, 3, 4]
    int mirrow_tx = BX - 1 - tx;
    if (idx_l < N) {
        // mirrow_idx       = [3, 0]
        input[idx_l] = ssm_right[mirrow_tx];
    }
    if (idx_r_seq >= 0 && idx_r_seq < N) {
        // mirrow_idx       = [3, 0]
        input[idx_r_seq] = ssm_left[mirrow_tx];
    }
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = BX;
    int blocksPerGrid = CEIL_DIV(N, threadsPerBlock * 2);

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
