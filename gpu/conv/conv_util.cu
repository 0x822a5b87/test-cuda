#include <cstdio>
#include <conv_util.cuh>

void conv2d_cpu(const float *in, const float *weight, float *out, const Conv2dDims &dims, const Conv2dAttrs &attrs) {
    // 每张图片单独计算
    for (int n_num = 0; n_num < dims.n; ++n_num) {
        // 在计算完之后，我们的原始厚度 c（通道）将被转换为新的厚度 k
        for (int k_num = 0; k_num < dims.k; ++k_num) {
            // 这里 oh_num 和 ow_num 都是输出上的点，也就是说每个点都需要计算
            // 我们这里改变了从输入到输出的顺序，而是通过输出反推输入
            for (int oh_num = 0; oh_num < dims.out_h; oh_num++) {
                for (int ow_num = 0; ow_num < dims.out_w; ow_num++) {
                    // 我们通过 p 和 q，将我们输出张量中的节点映射到了输入张量的左上角节点而不是中心节点
                    // 这意味着我们的可以执行互相关而不是卷积
                    float sum = 0.0;
                    const int h_pos = oh_num * attrs.u - attrs.p;
                    const int w_pos = ow_num * attrs.v - attrs.q;
                    for (int c_num = 0; c_num < dims.c; ++c_num) {
                        for (int h_num = 0; h_num < dims.r; ++h_num) {
                            for (int w_num = 0; w_num < dims.s; ++w_num) {
                                const int in_w = w_pos + w_num;
                                const int in_h = h_pos + h_num;
                                if (0 <= in_w && in_w < dims.w && 0 <= in_h && in_h < dims.h) {
                                    const float in_val = in[dims.in_offset(n_num, c_num, in_h, in_w)];
                                    const float k_val = weight[dims.weight_offset(k_num, c_num, h_num, w_num)];
                                    sum += in_val * k_val;
                                }
                            }
                        }
                    }
                    // 最终生成的结果为：
                    // [
                    //      图片1 -> [
                    //          核函数1 -> 核函数1生成的张量,
                    //          核函数2 -> 核函数2生成的张量,
                    //          ...
                    //      ],
                    //      图片2 -> [],
                    //      ...
                    // ]
                    out[dims.out_offset(n_num, k_num, oh_num, ow_num)] = sum;
                }
            }
        }
    }
}

bool check_result(const float* cpu_res, const float* gpu_res, int size) {
    for (int i = 0; i < size; ++i) {
        const float diff = std::abs(cpu_res[i] - gpu_res[i]);
        if (diff > 1e-4) {
            printf("Check failed at index %d: CPU=%f, GPU=%f\n", i, cpu_res[i], gpu_res[i]);
            return false;
        }
    }
    return true;
}
