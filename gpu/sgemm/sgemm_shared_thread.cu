#include <iostream>
#include <cuda_runtime.h>
#include <sgemm_util.cuh>

constexpr int POWER = 16;
constexpr int M = 1024 * POWER;
constexpr int N = 512 * POWER;
constexpr int K = 128 * POWER;

constexpr int BX = 64;
constexpr int BY = 64;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 4;

#define ceil(x, y) (((x) + (y) - 1) / (y))

__global__ void tiled_sgemm_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float sA[BX][BK];
    __shared__ float sB[BK][BY];

    // 线程私有的变量
    float res[TM][TN];
    for (auto &cols : res) {
        for (auto &v : cols) {
            v = 0.0f;
        }
    }

    const unsigned ty = threadIdx.y;
    const unsigned tx = threadIdx.x;
    const unsigned tid = ty * blockDim.x + tx;
    const unsigned thread_num = blockDim.x * blockDim.y;

    // LOOP FOR THE BLOCK
    for (size_t m = 0; m < ((K + BK - 1) / BK); m++) {
        // LOOP FOR sA
        for (size_t i = 0; i < ceil(BX * BK, thread_num); i++) {
            const unsigned load_id = tid + i * thread_num;
            if (load_id < BX * BK) {
                const unsigned r = load_id / BK;
                const unsigned c = load_id % BK;
                // 我们的矩阵在横向移动，所以它的行是随着 r 变化而变化，而列随着矩阵的移动也会变化（和 m 关联）
                // 此外，我们不能使用 blockIdx.y * blockDim.y 来计算行偏移量，因为我们的线程粗化了
                const unsigned global_row = blockIdx.y * BX + r;
                const unsigned global_col = m * BK + c;
                if (global_row < M && global_col < K) {
                    sA[r][c] = A[global_row * K + global_col];
                } else {
                    sA[r][c] = 0.0f;
                }
            }
        }

        // LOOP FOR sB
        for (size_t i = 0; i < ceil(BK * BY, thread_num); i++) {
            const unsigned load_id = tid + i * thread_num;
            if (load_id < BK * BY) {
                const unsigned r = load_id / BY;
                const unsigned c = load_id % BY;
                const unsigned global_row = m * BK + r;
                const unsigned global_col = blockIdx.x * BY + c;
                if (global_row < K && global_col < N) {
                    sB[r][c] = B[global_row * N + global_col];
                } else {
                    sB[r][c] = 0.0f;
                }
            }
        }

        __syncthreads();

        // 我们前面的算法中，使用的是内积，内积的实现通常是
        // 取出 A 的第 M 行和 B 的第 N 列，计算他们的积。
        // 内积更直观，缺点是每次计算一个点，我们就需要读取一整行和一整列
        // 而这个位置，使用外积效率更高，因为我们外积读取一行和一列，便可以计算出一个矩阵
        // 也就是，对于 `M * K` 和 `K * N` 的矩阵，我们只需要在 K 次循环中
        // 进行 M 次访问（读取A的行）和 N 次访问（读取 B 的列），
        // 在计算完成之后 A 的这一行和 B 的这一列使命就完成了
        // 总共需要 BK * (TM + TN) 次 SSM 访问和 BK * TM * TN * 2 次寄存器访问
        for (int k = 0; k < BK; k++) {
            float a_reg[TM];
            float b_reg[TN];

            for (int i = 0; i < TM; i++) {
                a_reg[i] = sA[ty * TM + i][k];
            }
            for (int j = 0; j < TN; j++) {
                b_reg[j] = sB[k][tx * TN + j];
            }

            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    res[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    // Now the calculation of the block is finish, output the result to C
    for (size_t i = 0; i < TM; i++) {
        for (size_t j = 0; j < TN; j++) {
            unsigned global_row = blockIdx.y * BX + ty * TM + i;
            unsigned global_col = blockIdx.x * BY + tx * TN + j;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = res[i][j];
            }
        }
    }
}

void launch_sgemm_kernel(const device_sgemm_t *d_sgemm) {
    dim3 block(BY / TN, BX / TM);
    dim3 grid(ceil(d_sgemm->N, BY), ceil(d_sgemm->M, BX));
    tiled_sgemm_kernel<<<grid, block>>>(d_sgemm->A, d_sgemm->B, d_sgemm->C, M, N, K);
}

int main(int argc, char const *argv[]) {
    run(M, N, K);
    return 0;
}
