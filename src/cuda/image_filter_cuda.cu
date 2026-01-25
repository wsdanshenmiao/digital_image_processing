#include "cuda/image_filter_cuda.h"

#include <cuda_runtime.h>
#include <cstdio>


__global__ void domain_average_filter2d_kernel(uchar3* input, uchar3* output, dim3 dim, int radius)
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

std::vector<uint8_t> dsm::image_filter_cuda::domain_average_filter2d(const std::vector<uint8_t>& input, size_t width, size_t height, int radius)
{
    auto size = std::size(input);
    if(radius <= 0 || width * height * 3 != size){
        return input;
    }

    uchar3* d_input = nullptr;
    uchar3* d_output = nullptr;

    auto check_cuda_error = [&d_input, &d_output](cudaError_t err, const char* msg){
        bool error = err != cudaSuccess;
        if(error){
            fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
            if(d_input != nullptr)
                cudaFree(d_input);
            if(d_output != nullptr)
                cudaFree(d_output);
        }
        return error;
    };

    // malloc device memory and copy input data
    cudaError_t err = cudaMalloc(&d_input, sizeof(uchar3) * (width * height));
    if(check_cuda_error(err, "cudaMalloc d_input"))
        return input;
    err = cudaMalloc(&d_output, sizeof(uchar3) * (width * height));
    if(check_cuda_error(err, "cudaMalloc d_output"))
        return input;
    err = cudaMemcpy(d_input, std::data(input), sizeof(uchar3) * (width * height), cudaMemcpyHostToDevice);
    if(check_cuda_error(err, "cudaMemcpy to d_input"))
        return input;

    // launch kernel
    dim3 blockDim = std::max(width, height) < 256 ? dim3(1,1,1) : dim3(16,16,1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    domain_average_filter2d_kernel<<<gridDim, blockDim>>>(d_input, d_output, dim3(width, height, 1), radius);

    // check for kernel launch errors
    if(check_cuda_error(cudaGetLastError(), "Kernel launch failed"))
        return input;

    err = cudaDeviceSynchronize();
    if(check_cuda_error(err, "cudaDeviceSynchronize failed"))
        return input;

    // copy back output data
    std::vector<uint8_t> output(size);
    err = cudaMemcpy(std::data(output), d_output, sizeof(uchar3) * (width * height), cudaMemcpyDeviceToHost);
    if(check_cuda_error(err, "cudaMemcpy to output"))
        return input;

    err = cudaFree(d_input);
    if(check_cuda_error(err, "cudaFree d_input"))
        return output;
    err = cudaFree(d_output);
    if(check_cuda_error(err, "cudaFree d_output"))
        return output;

    return output;
}


