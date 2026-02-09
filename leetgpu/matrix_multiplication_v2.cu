#include <cuda_runtime.h>

#define ceil(x, y) (((x) + (y) - 1) / (y))

constexpr int THREAD_TILE_X = 8;
constexpr int THREAD_TILE_Y = 8;

constexpr int BLOCK_SIZE_X = 16;
constexpr int BLOCK_SIZE_Y = 16;

constexpr int BLOCK_SSM_X = BLOCK_SIZE_X * THREAD_TILE_X;
constexpr int BLOCK_SSM_Y = BLOCK_SIZE_Y * THREAD_TILE_Y;

__global__ void matrix_multiplication_kernel(const float *A, const float *B, float *C,
                                             int M, int N, int K) {
    __shared__ float ssm_a[BLOCK_SSM_X][BLOCK_SSM_Y];
    __shared__ float ssm_b[BLOCK_SSM_Y][BLOCK_SSM_X];

    const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    for (int i_row = 0; i_row < THREAD_TILE_Y; i_row++) {
        for (int i_col = 0; i_col < THREAD_TILE_X; i_col++) {

        }
    }
}


// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *C, int M, int N, int K) {
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    const int stride_x = static_cast<int>(threadsPerBlock.x) * THREAD_TILE_X;
    const int stride_y = static_cast<int>(threadsPerBlock.y) * THREAD_TILE_Y;
    dim3 blocksPerGrid(ceil(K, stride_x), ceil(K, stride_y));

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

int main(int argc, char *argv[]) {
}
