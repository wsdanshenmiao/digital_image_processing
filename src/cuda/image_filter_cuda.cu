#include "cuda/image_filter_cuda.h"

#include <cuda_runtime.h>
#include <cstdio>


__global__ void domain_average_filter2d_kernel(uchar3 * input, uchar3 * output, dim3 dim, int radius)
{
    uint globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint globalIdy = blockIdx.y * blockDim.y + threadIdx.y;
    uint global_index = globalIdy * dim.x + globalIdx;

    if(globalIdx < radius || globalIdy < radius ||
        globalIdx >= dim.x - radius || globalIdy >= dim.y - radius){
            output[global_index] = input[global_index];
            return;
    }

    uint3 sum = make_uint3(0, 0, 0);
    for(int i = -radius; i <= radius; ++i){
        for(int j = -radius; j <= radius; ++j){
            uchar3 pixel = input[(globalIdy + i) * dim.x + (globalIdx + j)];
            sum.x += pixel.x;
            sum.y += pixel.y;
            sum.z += pixel.z;
        }
    }

    uint area = (2 * radius + 1) * (2 * radius + 1);
    output[global_index] = make_uchar3(sum.x / area, sum.y / area, sum.z / area);
}

namespace dsm{
    std::vector<uint8_t> image_filter_cuda::domain_average_filter2d(const std::vector<uint8_t>& input, size_t width, size_t height, int radius)
    {
        auto size = std::size(input);
        if(radius <= 0 || width * height * 3 != size){
            return input;
        }

        // malloc device memory and copy input data
        uchar3* d_input = nullptr;
        uchar3* d_output = nullptr;
        cudaMalloc(&d_input, sizeof(uchar3) * (width * height));
        cudaMalloc(&d_output, sizeof(uchar3) * (width * height));
        cudaMemcpy(d_input, std::data(input), sizeof(uchar3) * (width * height), cudaMemcpyHostToDevice);

        // launch kernel
        dim3 blockDim = std::max(width, height) < 256 ? dim3(1,1,1) : dim3(16,16,1);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
        domain_average_filter2d_kernel<<<gridDim, blockDim>>>(d_input, d_output, dim3(width, height, 1), radius);

        cudaDeviceSynchronize();

        // copy back output data
        std::vector<uint8_t> output(size);
        cudaMemcpy(std::data(output), d_output, sizeof(uchar3) * (width * height), cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);

        return output;
    }
}

