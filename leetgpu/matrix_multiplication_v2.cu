#include <cuda_runtime.h>

#define ceil(x, y) (((x) + (y) - 1) / (y))

constexpr int THREAD_TILE_X = 8;
constexpr int THREAD_TILE_Y = 8;

constexpr int BLOCK_SIZE_X = 8;
constexpr int BLOCK_SIZE_Y = 8;
constexpr int THREAD_COUNT = BLOCK_SIZE_X * BLOCK_SIZE_Y;

constexpr int BX = BLOCK_SIZE_X * THREAD_TILE_X;
constexpr int BY = BLOCK_SIZE_Y * THREAD_TILE_Y;
constexpr int BK = 32;

__global__ void matrix_multiplication_kernel(const float *A, const float *B, float *C,
                                             int M, int N, int K) {
    // 我们把输出张量划分为多个 block，每个 block 又继续划分多个 THREAD
    // 每个THREAD 继续划分为多个 TILE
    // 这里 ssm_a 和 ssm_b 需要提供一个能力：在同一个 block 中，
    // 线程每次 for 循环移动时，覆盖当前 block 的所有线程的所有TILE所需要的数据
    // 一次读取，多次使用。
    // 那么，结合上面的结论和矩阵乘法的要求（计算元素 (x,y) 需要 A 的第 x 行和 B 的第 y 列）
    // 我们可以做出如下推论：
    // ssm_a 需要的矩阵是高度是 THREAD_TILE_Y * BLOCK_SIZE_Y
    // ssm_b 需要的矩阵是宽度是 THREAD_TILE_X * BLOCK_SIZE_X
    // 而 ssm_a 的宽度和 ssm_b 的高度则没有限制，只需要满足 ssm_a.width == ssm_b.height
    // 因为 :
    // 1. ssm_a 需要整行，它可以看做一个从索引0开始，向右划到到N结束的滑动块；
    // 2. ssm_b 需要整列，它可以看做一个从索引0开始，向下滑动到N结束的滑动块；
    // 也就是我们说，我们可以如下声明我们的 ssm_a 和 ssm_b，这里的 BK 是任意值都可以实现逻辑，只是性能不同
    __shared__ float ssm_a[BY][BK];
    __shared__ float ssm_b[BK][BX];

    float accum[THREAD_TILE_Y][THREAD_TILE_X] = {};
    float reg_for_a[THREAD_TILE_Y];
    float reg_for_b[THREAD_TILE_X];

    const int tid = static_cast<int>(threadIdx.y * BLOCK_SIZE_X + threadIdx.x);
    const int local_row = static_cast<int>(threadIdx.y);
    const int local_col = static_cast<int>(threadIdx.x);

    for (int i_k = 0; i_k < ceil(N, BK); i_k++) {
        // 每次计算当前TILE之前，我们要把所有的重新读取数据到ssm
        // 这里需要注意的是，当我们在搬运数据的时候，我们并不考虑TILE的概念
        // 从A和B搬运数据到ssm是一个完全独立的过程，它是输入张量某个区域到ssm的一比一映射
        // 那么，我们所需要考虑的就是：到底把哪个区域映射到我们的 ssm

        // 1. 对于 ssm_a，它的 x 轴随着 i_k 移动，y轴随着 block 移动；
        // 2. 对于 ssm_b，它的 y 轴随着 i_k 移动，x 轴随着 block 移动。

        // 对于 ssm_a，假设我们当前 block 的第一个线程的坐标是：
        // ty = blockIdx.y * blockDim.y + threadIdx.y
        // 那么我们需要填充到 ssm_a 的信息就是：
        // [(i_k * BK, ty), ((i_k + 1) * BK, ty))
        // [(i_k * BK, ty + 1), ((i_k + 1) * BK, ty + 1))
        // ...
        // [(i_k * BK, ty + BY - 1), ((i_k + 1) * BK, ty + BY - 1))
        // 得到一个 BY * BK 的矩阵，我们可以把这个矩阵看做一个整体，那么我们可以通过如下代码来实现搬运
        // for (int i = 0; i < BY; i++) {
        //     for (int j = 0; j < BX; j++) {
        //         ssm_a[x][y] = value;
        //     }
        // }
        // 我们需要将这个逻辑转换为GPU上实现的逻辑
        // 每个线程搬运一个数据，那么总共需要 (BY * BK) / (BLOCK_SIZE_X * BLOCK_SIZE_Y) 次（向上取整）
        for (int i = 0; i < ceil(BY * BK, THREAD_COUNT); ++i) {
            const int load_id = i * THREAD_COUNT + tid;
            const int r = load_id >> 6;
            const int c = load_id & 63;
            const int global_r = static_cast<int>(blockIdx.y * BY + r);
            const int global_c = i_k * BK + c;
            if (global_r < M && global_c < N) {
                ssm_a[r][c] = A[global_r * N + global_c];
            } else {
                ssm_a[r][c] = 0.0f;
            }
        }
        // 对于 ssm_b，假设我们当前 block 的第一个线程的坐标是：
        // tx = blockIdx.x * blockDim.x + threadIdx.x
        // 那么我们需要填充到 ssm_b 的信息就是：
        // [(tx, i_k * BX), (tx, (i_k + 1) * BX))
        // [(tx + 1, i_k * BX), (tx + 1, (i_k + 1) * BX))
        // ...
        // [(tx + BK - 1, i_k * BX), (tx + BK - 1, (i_k + 1) * BX))
        // 得到一个 BK * BX 的矩阵
        for (int i = 0; i < ceil(BK * BX, THREAD_COUNT); i++) {
            const int load_id = i * THREAD_COUNT + tid;
            const int r = load_id >> 6;
            const int c = load_id & 63;
            const int global_r = i_k * BK + r;
            const int global_c = static_cast<int>(blockIdx.x * BX + c);
            if (global_r < N && global_c < K) {
                ssm_b[r][c] = B[global_r * K + global_c];
            } else {
                ssm_b[r][c] = 0.0;
            }
        }
        __syncthreads();

        // 我们使用外积的方式来计算结果，外积可以看做是 [M * 1] * [1 * N] 的矩阵乘法，得到一个 [M * N] 的矩阵
        // 每个线程负责一个TILE，注意，在之前我们的逻辑中
        // 我们特意使用了TILE和线程交错排布的模式，但是这里我们改用了
        // TILE紧密排布的模式(TILE0, TILE1, ...)，因为之前我们是访问显存，
        // 而现在访问的是寄存器和SharedMemory，不再需要考虑合并读取。
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            // 这里，进入我们的累加循环，我们的矩阵乘法是 [THREAD_TILE_Y * BK] * [BK * THREAD_TILE_X]
            // 这个位置非常容易误解成我们浪费了一些元素，因为我们把内积转换成了外积
            // 如果我们采用如下的方式：
            // for (int i = 0; i < BY; i++){}
            // for (int i = 0; i < BX; i++){}
            // 我们计算的就不是 TILE 的结果，而是整个block的结果，那样我们相当于多个线程重复计算了
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_Y; i++) {
                reg_for_a[i] = ssm_a[local_row * THREAD_TILE_Y + i][k];
            }
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_X; i++) {
                reg_for_b[i] = ssm_b[k][local_col * THREAD_TILE_X + i];
            }
            #pragma unroll
            for (int r = 0; r < THREAD_TILE_Y; r++) {
                #pragma unroll
                for(int c = 0; c < THREAD_TILE_X; c++) {
                    accum[r][c] += reg_for_a[r] * reg_for_b[c];
                }
            }
        }
        __syncthreads();
    }

    for (int r = 0; r < THREAD_TILE_Y; r++) {
        for (int c = 0; c < THREAD_TILE_X; c++) {
            int global_r = static_cast<int>(blockIdx.y * BY + threadIdx.y * THREAD_TILE_Y + r);
            int global_c = static_cast<int>(blockIdx.x * BX + threadIdx.x * THREAD_TILE_X + c);
            if (global_r < M && global_c < K) {
                C[global_r * K + global_c] = accum[r][c];
            }
        }
    }
}


// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *C, int M, int N, int K) {
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    const int stride_x = static_cast<int>(threadsPerBlock.x) * THREAD_TILE_X;
    const int stride_y = static_cast<int>(threadsPerBlock.y) * THREAD_TILE_Y;
    dim3 blocksPerGrid(ceil(K, stride_x), ceil(M, stride_y));

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

int main(int argc, char const *argv[])
{
    /* code */
    return 0;
}
