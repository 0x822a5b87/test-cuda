#include <cuda_runtime.h>

#define ceil(x, y) (((x) + (y) - 1) / (y))

constexpr int THREAD_TILE_X = 8;
constexpr int THREAD_TILE_Y = 8;

constexpr int BLOCK_SIZE_X = 16;
constexpr int BLOCK_SIZE_Y = 16;

constexpr int STRIDE_X = BLOCK_SIZE_X * THREAD_TILE_X;
constexpr int STRIDE_Y = BLOCK_SIZE_Y * THREAD_TILE_Y;

// v1 版本，我们引入TILE机制实现线程粗化逻辑，同时我们通过将内积转换为外积来优化性能。
__global__ void matrix_multiplication_kernel(const float *A, const float *B, float *C,
                                             int M, int N, int K) {
    // 初始化结果数组
    float accum[THREAD_TILE_Y][THREAD_TILE_X];
    // 防止编译器优化到数组到显存
#pragma unroll
    for (int i_row = 0; i_row < THREAD_TILE_Y; ++i_row) {
#pragma unroll
        for (int i_col = 0; i_col < THREAD_TILE_X; ++i_col) {
            accum[i_row][i_col] = 0;
        }
    }

    // 现在我们要开始从内存读取数据并累加到结果数组
    // 按照我们的交错分布，我们在 accum[i_row][i_col] 这一个元素，对应的输出的点应该是
    // row = blockIdx.y * STRIDE_Y + i_row * BLOCK_SIZE_Y + threadIdx.y
    // col = blockIdx.x * STRIDE_X + i_col * BLOCK_SIZE_X + threadIdx.x

    // row = block行偏移量 + TILE行偏移量 + 线程相对行偏移量
    // block 行偏移量是相同的，i_row 对于所有的线程相同，而在block中左右相邻的线程y是一样的
    // 也就是说 row 的完全一致的，可以通过广播实现访问

    // col = block列偏移量 + TILE列偏移量 + 线程相对列偏移量
    // 这里block列偏移量是固定的，i_col 对所有的线程都是相同的，所以相邻线程的TILE列偏移量相同，
    // 唯一不同的是 threadIdx.x，而这个值不同线程之间是连续的，最后他们在内存中的数据可以合并访问
    const int tile_row_offset = static_cast<int>(blockIdx.y * STRIDE_Y + threadIdx.y);
    const int tile_col_offset = static_cast<int>(blockIdx.x * STRIDE_X + threadIdx.x);
    float col_of_a[THREAD_TILE_Y];
    float row_of_b[THREAD_TILE_X];
    for (int i_factor = 0; i_factor < N; ++i_factor) {
        // 关于 TILE 的遍历，我们可以看到 TILE 不是连续的
        // 因为这里在A的行和B的列的移动，是通过最外层的for循环实现的
        // 这里的 col_of_a 和 row_of_b 其实是TILE负责的区域在变化
        // 这里的逻辑其实可以理解为：
#pragma unroll
        for (int i_row = 0; i_row < THREAD_TILE_Y; ++i_row) {
            const int y = tile_row_offset + i_row * BLOCK_SIZE_Y;
            // 这个位置，同一个 block 下存在左右相邻线程 y 相等，那么他们访问的是同一个地址，
            // 可以通过广播传输数据。
            col_of_a[i_row] = y < M ? A[y * N + i_factor] : 0.0f;
        }
#pragma unroll
        for (int i_col = 0; i_col < THREAD_TILE_X; ++i_col) {
            // 这个位置，同一个 block 内，线程地址连续，可以合并访问。
            const int x = tile_col_offset + i_col * BLOCK_SIZE_X;
            row_of_b[i_col] = x < K ? B[i_factor * K + x] : 0.0f;
        }

#pragma unroll
        for (int i_row = 0; i_row < THREAD_TILE_Y; ++i_row) {
#pragma unroll
            for (int i_col = 0; i_col < THREAD_TILE_X; ++i_col) {
                accum[i_row][i_col] += col_of_a[i_row] * row_of_b[i_col];
            }
        }
    }

#pragma unroll
    for (int i = 0; i < THREAD_TILE_Y; i++) {
#pragma unroll
        for (int j = 0; j < THREAD_TILE_X; j++) {
            const int row_of_c = tile_row_offset + i * BLOCK_SIZE_Y;
            const int col_of_c = tile_col_offset + j * BLOCK_SIZE_X;
            if (row_of_c < M && col_of_c < K) {
                C[row_of_c * K + col_of_c] = accum[i][j];
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

int main(int argc, char *argv[]) {
}
