#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define WARP_SIZE 32

__global__ void sum_kernel_optimized(const int* input, int* output, int N) {
    int sum = 0;
    // 1. 每个线程计算自己的网格跨步起点
    // 使用 int4 向量化读取，每个线程单次读取 4 个 int
    const int4* v_input = reinterpret_cast<const int4*>(input);
    int n_vec = N / 4;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // 2. 主循环：向量化读取 (L2 友好，指令并行度高)
    for (int i = tid; i < n_vec; i += stride) {
        int4 data = v_input[i]; // 对应 PTX 中的 ld.global.v4.u32
        sum += data.x + data.y + data.z + data.w;
    }

    // 3. 处理末尾无法凑齐 int4 的残数 (由第一个线程处理)
    if (tid == 0) {
        for (int i = n_vec * 4; i < N; ++i) {
            atomicAdd(output, input[i]);
        }
    }

    // 4. Warp 内规约 (Shuffle Down)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // 5. 将每个 Warp 的结果原子累加到全局
    if ((tid & 31) == 0) {
        atomicAdd(output, sum);
    }
}

extern "C" void solve(const int* input, int* output, int N, int S, int E) {
    cudaMemset(output, 0, sizeof(int));

    // 计算真实的起始地址和长度
    const int* start_ptr = input + S;
    int total_n = E - S + 1;

    // Prolog
    int alignment_offset = ((long long)start_ptr % 16) / sizeof(int);
    if (alignment_offset != 0) {
        alignment_offset = 4 - alignment_offset;
    }
    
    if (alignment_offset > total_n) alignment_offset = total_n;
    if (alignment_offset > 0) {
        int host_prolog[4];
        cudaMemcpy(host_prolog, start_ptr, alignment_offset * sizeof(int), cudaMemcpyDeviceToHost);
        int prolog_sum = 0;
        for(int i=0; i<alignment_offset; i++) prolog_sum += host_prolog[i];
        cudaMemcpy(output, &prolog_sum, sizeof(int), cudaMemcpyHostToDevice);
    }

    // 更新主循环的指针和长度
    const int* aligned_ptr = start_ptr + alignment_offset;
    int main_n = total_n - alignment_offset;

    if (main_n > 0) {
        int threadsPerBlock = 256;
        // 这里的 Block 数可以根据 GPU 的 SM 数量动态调整，通常 160-320 效果最好
        int blocksPerGrid = 160; 
        sum_kernel_optimized<<<blocksPerGrid, threadsPerBlock>>>(aligned_ptr, output, main_n);
    }
}