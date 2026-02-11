#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

constexpr int BX = 32;          // block中的x线程
constexpr int BY = 8;          // block中的y线程

constexpr int BK = 1;          // block内x轴步长，每次处理BK组RGB元素

constexpr int SX = BX * BK;     // block的x轴步长，每个RGB元素占据四个字节
constexpr int SY = BY;          // block的y轴步长，我们这里y轴为1

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    uchar4* pixel4 = reinterpret_cast<uchar4*>(image);
    // 注意，这里我们使用的是TILE和线程交错排布，所以相对偏移量不需要乘以BK
    const int gx = blockIdx.x * SX + threadIdx.x;
    const int gy = blockIdx.y * SY + threadIdx.y;

    for (int i = 0; i < BK; i++) {
        const int tx = gx + i * BX;
        if (tx < width && gy < height) {
            int idx = gy * width + (gx + i * BX);
            uchar4 pixel = pixel4[idx];
            pixel.x = ~pixel.x;
            pixel.y = ~pixel.y;
            pixel.z = ~pixel.z;
            pixel4[idx] = pixel;
        }
    }
}

// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    dim3 threadsPerBlock(BX, BY);
    dim3 blocksPerGrid(CEIL_DIV(width, SX), CEIL_DIV(height, SY));
    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}

int main() {
    // 1. 定义图像尺寸 (假设为 1024x1024 的 RGBA 图像)
    int width = 5120; 
    int height = 4096;
    size_t img_size = width * height * 4; // 每个像素 4 字节

    // 2. 在 Host 端分配并初始化数据
    std::vector<unsigned char> h_image(img_size);
    for (int i = 0; i < img_size; ++i) {
        h_image[i] = (unsigned char)(i % 256); // 填充一些模拟颜色数据
    }

    std::cout << "Original first pixel: R=" << (int)h_image[0] 
              << " G=" << (int)h_image[1] 
              << " B=" << (int)h_image[2]
              <<"  M=" << (int)h_image[3] << std::endl;

    // 3. 在 Device 端分配显存
    unsigned char* d_image;
    cudaError_t err = cudaMalloc(&d_image, img_size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Malloc failed!" << std::endl;
        return -1;
    }

    // 4. 将数据从 Host 拷贝到 Device
    cudaMemcpy(d_image, h_image.data(), img_size, cudaMemcpyHostToDevice);

    // 5. 调用你的 solve 函数
    // 注意：这里的 width 传的是像素宽度（正如我们之前讨论的像素视角）
    solve(d_image, width, height);

    // 6. 将处理后的数据拷贝回 Host
    cudaMemcpy(h_image.data(), d_image, img_size, cudaMemcpyDeviceToHost);

    // 7. 验证结果 (255 - x)
    std::cout << "Inverted first pixel: R=" << (int)h_image[0] 
              << " G=" << (int)h_image[1] 
              << " B=" << (int)h_image[2] 
              << " M=" << (int)h_image[3]<< std::endl;

    // 8. 释放显存
    cudaFree(d_image);

    std::cout << "Successfully completed image inversion test." << std::endl;
    return 0;
}