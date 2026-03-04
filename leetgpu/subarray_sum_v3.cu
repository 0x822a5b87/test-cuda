#include <cuda_runtime.h>
#include <stdint.h>

#define WARP_SIZE 32

__global__ void sum_kernel_adaptive(const int* input, int* output, int total_n, int alignment_offset) {
    int sum = 0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // 让最前面的几个线程处理非对齐的数据
    if (tid < alignment_offset) {
        atomicAdd(output, input[tid]);
    }

    // 指针平移到对齐点
    const int4* v_input = reinterpret_cast<const int4*>(input + alignment_offset);
    int main_n_vec = (total_n - alignment_offset) / 4;

    for (int i = tid; i < main_n_vec; i += stride) {
        // 强制 128-bit 指令，PTX 将生成 ld.global.v4.u32
        int4 data = v_input[i]; 
        sum += data.x + data.y + data.z + data.w;
    }

    // 处理末尾无法凑齐 int4 的部分 (Epilog)
    int epilog_start_idx = alignment_offset + main_n_vec * 4;
    int rem = total_n - epilog_start_idx;
    if (tid < rem) {
        atomicAdd(output, input[epilog_start_idx + tid]);
    }

    // --- 4. Warp 级规约 (Shuffle Down) ---
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // --- 5. 结果汇总 ---
    if ((tid & 31) == 0) {
        atomicAdd(output, sum);
    }
}

extern "C" void solve(const int* input, int* output, int N, int S, int E) {
    cudaMemset(output, 0, sizeof(int));

    const int* start_ptr = input + S;
    int total_n = E - S + 1;
    if (total_n <= 0) return;

    // 2. 在 CPU 计算对齐偏移 (4个int = 16字节)
    // 计算当前起始地址到下一个 16 字节对齐点需要补多少个 int
    uintptr_t addr = reinterpret_cast<uintptr_t>(start_ptr);
    int alignment_offset = (16 - (addr % 16)) % 16 / sizeof(int);

    // 如果偏移量超过了总数，主循环将不执行
    if (alignment_offset > total_n) alignment_offset = total_n;

    int threadsPerBlock = 256;
    int blocksPerGrid = 160; 

    sum_kernel_adaptive<<<blocksPerGrid, threadsPerBlock>>>(start_ptr, output, total_n, alignment_offset);
    
    // LeetGPU 通常需要同步以确保计时准确
    cudaDeviceSynchronize();
}