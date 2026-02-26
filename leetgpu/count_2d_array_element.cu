#include <cuda_runtime.h>
#include <cstdint>

__global__ void count_2d_equal_kernel(const int* input, int* output, int N, int M, int K) {
    const auto tid = threadIdx.x;
    auto g_tid = tid + blockDim.x * blockIdx.x;
    const auto t_elements = M * N;
    const auto warp_id = (tid >> 5);
    const auto lane_id = (tid & 31);

    const auto reg_K = K;

    std::uint32_t local_count{};
    while(g_tid < t_elements) {
        local_count += (input[g_tid] == reg_K);
        g_tid += (gridDim.x * blockDim.x);
    }

    auto red_K = local_count;
    for (std::uint32_t off = 16; off; off >>= 1){
        red_K += __shfl_down_sync(0xFFFFFFFF, red_K, off);
    }
    
    __shared__ std::int32_t shared_reduce[32];

    if (!lane_id) {
        shared_reduce[warp_id] = red_K;
    }
    __syncthreads();

    // do reduction again on shared memory with the first warp,
    // then atomic add to output
    if(!warp_id) {
        auto per_warp_count = shared_reduce[lane_id];
        for (std::uint32_t off = 16; off; off >>= 1){
            per_warp_count += __shfl_down_sync(0xFFFFFFFF, per_warp_count, off);
        }

        if (!lane_id) {
            atomicAdd(output, per_warp_count);
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K) {
    dim3 threadsPerBlock(1024);
    dim3 blocksPerGrid((M*N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    count_2d_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, M, K);
    cudaDeviceSynchronize();
}