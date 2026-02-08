#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <conv_util.cuh>

__global__ void naive_conv2d_c_oriented(const float *in, float *weight, float *out, const Conv2dDims dims, const Conv2dAttrs attrs) {
    // 我们将整个输出张量的一个通道中的所有算子压缩到x轴，所以这里x代表了在某个通道中的相对偏移量
    // idx_area = height * w + width;
    const int idx_area_out = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    // 这里代表了第 idx_k 个核函数，这里非常容易误解的是
    // 在通道优先中，y 表示的是通道的索引，实际上这里和通道没有任何关系
    const int idx_k = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    // 表示第几张图片
    const int idx_n = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);

    if (idx_area_out >= dims.out_h * dims.out_w || idx_k >= dims.k || idx_n >= dims.n) {
        return;
    }

    // 计算输出张量元素在矩阵中的相对坐标
    const int idx_h_in_area = idx_area_out / dims.out_w;
    const int idx_w_in_area = idx_area_out % dims.out_w;

    // 通过输出张量元素在矩阵中的相对坐标计算在输入张量中的起始位置
    const int pos_ori_h = idx_h_in_area * attrs.u - attrs.p;
    const int pos_ori_w = idx_w_in_area * attrs.v - attrs.q;

    float sum = 0.0f;
    for (int k_h_num = 0; k_h_num < dims.r; k_h_num++) {
        for (int k_w_num = 0; k_w_num < dims.s; k_w_num++) {
            for (int k_c_num = 0; k_c_num < dims.c; k_c_num++) {
                const int pos_cur_h = pos_ori_h + k_h_num;
                const int pos_cur_w = pos_ori_w + k_w_num;
                if (0 <= pos_cur_h && pos_cur_h < dims.h && 0 <= pos_cur_w && pos_cur_w < dims.w) {
                    // 通道优先的含义是：计算**同一个核函数**下，输入张量不同通道间具有相同相对索引的元素
                    // 所以这个位置他的相对索引和 idx_k 没有任何关系，只和输入张量的通道有关
                    const int in_offset = dims.in_offset(idx_n, k_c_num, pos_cur_h, pos_cur_w);
                    // 核函数的索引只和核函数以及 idx_k 有关
                    const int weight_offset = dims.weight_offset(idx_k, k_c_num, k_h_num, k_w_num);
                    sum += in[in_offset] * weight[weight_offset];
                }
            }
        }
    }
    const unsigned out_offset = dims.out_offset(idx_n, idx_k, idx_h_in_area, idx_w_in_area);
    out[out_offset] = sum;
}

int main() {
    // 定义输入数据和卷积核的尺寸
    constexpr int n = 2000; // batch size
    constexpr int c = 2; // 通道数
    constexpr int h = 10; // 数据高
    constexpr int w = 10; // 数据宽
    constexpr int k = 5; // 卷积核数量
    constexpr int r = 3; // 卷积核高
    constexpr int s = 3; // 卷积核宽
    constexpr int u = 1; // 卷积在高方向上的步长
    constexpr int v = 1; // 卷积在宽方向上的步长
    constexpr int p = 0; // 卷积在高方向上的补边
    constexpr int q = 0; // 卷积在宽方向上的补边
    constexpr int out_h = (h - r + 2 * p) / u + 1; // 输出高
    constexpr int out_w = (w - s + 2 * q) / v + 1; // 输出宽

    constexpr auto dims = Conv2dDims{
        n, c, h, w,
        k, r, s,
        out_h, out_w
    };

    constexpr auto attrs = Conv2dAttrs{
        u, v, p, q
    };

    // 分配内存并随机生成输入数据和卷积核
    float *in_device, *weight_device, *out_device;

    auto *in = static_cast<float *>(malloc(n * c * h * w * sizeof(float)));
    auto *weight = static_cast<float *>(malloc(k * c * r * s * sizeof(float)));
    auto *out = static_cast<float *>(malloc(n * k * out_h * out_w * sizeof(float)));

    CHECK(cudaMalloc(reinterpret_cast<void **>(&in_device), n * c * h * w * sizeof(float)));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&weight_device), k * c * r * s * sizeof(float)));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&out_device), n * k * out_h * out_w * sizeof(float)));

    // 随机生成输入数据和卷积核
    for (int i = 0; i < n * c * h * w; ++i) {
        in[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < k * c * r * s; ++i) {
        weight[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 将输入数据和卷积核拷贝到 GPU
    CHECK(cudaMemcpy(in_device, in, n * c * h * w * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(weight_device, weight, k * c * r * s * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(out_device, out, n * k * out_h * out_w * sizeof(float), cudaMemcpyHostToDevice));

    // 这里我们使用通道优先的策略，我们的 block 中包含图片的所有通道。
    constexpr int blockDim_y = k;
    // 将输入图片的一层压缩为一行，并且由于我们使用通道优先的策略，所以
    // 我们需要根据通道的数量来调整我们的矩阵的大小
    constexpr int max_threads = 1024;
    constexpr int ideal_x = (out_h * out_w + k - 1) / k;
    constexpr int blockDim_x = std::max(1, std::min(ideal_x, max_threads / k));
    dim3 blockDim(blockDim_x, blockDim_y);

    // 计算所需的全部 block，这里需要注意的是，grid 的移动仍然遵循：
    // 先移动x，当x轴的元素计算完成再移动 y；
    // 当y轴的元素也计算完成时，说明我们当前图片已经计算完成：因为我们是通道优先的
    // 此时当前图片的全部通道都计算完毕，开始计算下张图片。
    constexpr int gridDim_x = (out_h * out_w + blockDim_x - 1) / blockDim_x;
    constexpr int gridDim_y = (k + blockDim_y - 1) / blockDim_y;
    dim3 gridDim(gridDim_x, gridDim_y, n);

    naive_conv2d_c_oriented<<<gridDim, blockDim>>>(in_device, weight_device, out_device, dims, attrs);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(out, out_device, n * k * out_h * out_w * sizeof(float), cudaMemcpyDeviceToHost));

    const auto out_cpu = static_cast<float *>(malloc(n * k * out_h * out_w * sizeof(float)));
    conv2d_cpu(in, weight, out_cpu, dims, attrs);

    if (check_result(out_cpu, out, n * k * out_h * out_w)) {
        std::cout << "Verification passed!" << std::endl;

        constexpr int iter = 1;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, nullptr);
        for (int i = 0; i < iter; i++) {
            naive_conv2d_c_oriented<<<gridDim, blockDim>>>(in_device, weight_device, out_device, dims, attrs);
        }
        cudaEventRecord(stop, nullptr);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        std::cout << "GPU time: " << 1000 * elapsedTime / static_cast<float>(iter) << "us" << std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaFree(in_device);
    cudaFree(weight_device);
    cudaFree(out_device);
    free(in);
    free(weight);
    free(out);

    return 0;
}
