#include "cuda/image_filter_cuda.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>


__global__ void domain_average_filter2d_kernel(
    uchar3* input, uchar3* output, 
    dim3 dim, int radius, 
    bool isHorizontal)
{
    uint dispatch_index_x = blockIdx.x * blockDim.x + threadIdx.x;
    uint dispatch_index_y = blockIdx.y * blockDim.y + threadIdx.y;
    uint dispatch_index = isHorizontal ? dispatch_index_y * dim.x + dispatch_index_x :
        dispatch_index_x * dim.x + dispatch_index_y;

    if(dispatch_index_x >= (isHorizontal ? dim.x : dim.y) || 
        dispatch_index_y >= (isHorizontal ? dim.y : dim.x) ||
        dispatch_index_x < 0 || dispatch_index_y < 0){
        return;
    }

    extern __shared__ uchar3 domain_average_shared_mem[];
    // load near pixel if needed
    if(threadIdx.x < radius){
        int input_index = isHorizontal ? (dispatch_index_y * dim.x + max(dispatch_index_x - radius, 0)) : 
            (max(dispatch_index_x - radius, 0) * dim.x + dispatch_index_y);
        domain_average_shared_mem[threadIdx.y * blockDim.x + threadIdx.x] = input[input_index];
    }
    if(threadIdx.x >= blockDim.x - radius){
        int input_index = isHorizontal ? (dispatch_index_y * dim.x + min(dispatch_index_x + radius, dim.x - 1)) : 
            (min(dispatch_index_x + radius, dim.y - 1) * dim.x + dispatch_index_y);
        domain_average_shared_mem[threadIdx.y * blockDim.x + threadIdx.x + 2 * radius] = input[input_index];
    }

    // current pixel value
    domain_average_shared_mem[threadIdx.y * blockDim.x + threadIdx.x + radius] = input[dispatch_index];
    __syncthreads();

    if(dispatch_index_x < radius || dispatch_index_y < radius ||
        dispatch_index_x >= (isHorizontal ? dim.x : dim.y) - radius || 
        dispatch_index_y >= (isHorizontal ? dim.y : dim.x) - radius){
        output[dispatch_index] = input[dispatch_index];
        return;
    }

    uint3 sum = make_uint3(0, 0, 0);
    for(int i = 0; i <= 2 * radius; ++i){
        uchar3 pixel = domain_average_shared_mem[threadIdx.y * blockDim.x + threadIdx.x + i];
        sum.x += pixel.x;
        sum.y += pixel.y;
        sum.z += pixel.z;
    }

    uint area = 2 * radius + 1;
    output[dispatch_index] = make_uchar3(sum.x / area, sum.y / area, sum.z / area);
}

template <typename Func>
__global__ void median_filter2d_kernel(uint8_t* input, uint8_t* output, dim3 dim, int radius, Func func)
{
    uint dispatch_index_x = blockIdx.x * blockDim.x + threadIdx.x;
    uint dispatch_index_y = blockIdx.y * blockDim.y + threadIdx.y;
    uint dispatch_index = dispatch_index_y * dim.x + dispatch_index_x;

    if( dispatch_index_x < 0 || dispatch_index_y < 0 ||
        dispatch_index_x >= dim.x || dispatch_index_y >= dim.y){
        return;
    }

    if(dispatch_index_y < radius || dispatch_index_x < radius ||
        dispatch_index_x >= dim.x - radius || dispatch_index_y >= dim.y - radius){
        output[dispatch_index] = input[dispatch_index];
        return;
    }

    // load data to shared memory
    extern __shared__ uint8_t median_shared_mem[];
    int diameter = 2 * radius + 1;
    uint shared_base_index = (threadIdx.y * blockDim.x + threadIdx.x) * diameter * diameter;
    for(int i = -radius; i <= radius; ++i){
        for(int j = -radius; j <= radius; ++j){
            median_shared_mem[shared_base_index + (i + radius) * diameter + (j + radius)] = 
                input[(dispatch_index_y + i) * dim.x + dispatch_index_x + j];
        }
    }

    uint8_t median = func(median_shared_mem + shared_base_index, diameter * diameter);
    output[dispatch_index] = median;
}

std::vector<uint8_t> dsm::image_filter_cuda::domain_average_filter2d(const std::vector<uint8_t>& input, size_t width, size_t height, int radius)
{
    auto size = std::size(input);
    if(radius <= 0){
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
    const int thread_width = 8;
    radius = std::min(thread_width, radius);
    size_t shared_mem_size = (thread_width + 2 * radius) * sizeof(uchar3);
    dim3 blockDim{thread_width, 1, 1};
    // launch horizontal pass
    dim3 gridDim{(width + blockDim.x - 1) / blockDim.x, height, 1};
    domain_average_filter2d_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        d_input, d_output, dim3(width, height, 1), radius, true);
    // launch vertical pass
    gridDim = dim3((height + blockDim.x - 1) / blockDim.x, width, 1);
    domain_average_filter2d_kernel<<<gridDim, blockDim, shared_mem_size>>>(
        d_output, d_input, dim3(width, height, 1), radius, false);

    // check for kernel launch errors
    if(check_cuda_error(cudaGetLastError(), "Kernel launch failed"))
        return input;

    // synchronize to ensure kernel completion
    err = cudaDeviceSynchronize();
    if(check_cuda_error(err, "cudaDeviceSynchronize failed"))
        return input;

    // copy back output data
    std::vector<uint8_t> output(size);
    err = cudaMemcpy(std::data(output), d_input, sizeof(uchar3) * (width * height), cudaMemcpyDeviceToHost);
    if(check_cuda_error(err, "cudaMemcpy to output"))
        return input;

    err = cudaFree(d_input);
    d_input = nullptr;
    if(check_cuda_error(err, "cudaFree d_input"))
        return output;
    err = cudaFree(d_output);
    d_output = nullptr;
    if(check_cuda_error(err, "cudaFree d_output"))
        return output;

    return output;
}

std::vector<uint8_t> dsm::image_filter_cuda::median_filter2d(const std::vector<uint8_t>& input, size_t width, size_t height, int radius)
{
    auto size = std::size(input);
    if(radius <= 0)
        return input;

    // malloc gpu memory
    uint8_t* d_input = nullptr;
    uint8_t* d_output = nullptr;
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

    cudaError_t err = cudaMalloc(&d_input, sizeof(uint8_t) * width * height);
    if(check_cuda_error(err, "cudaMalloc d_input"))
        return input;
    err = cudaMalloc(&d_output, sizeof(uint8_t) * width * height);
    if(check_cuda_error(err, "cudaMalloc d_output"))
        return input;
    // copy input data to gpu
    err = cudaMemcpy(d_input, std::data(input), sizeof(uint8_t) * width * height, cudaMemcpyHostToDevice);
    if(check_cuda_error(err, "cudaMemcpy to d_input"))
        return input;

    // launch kernel
    constexpr int thread_width = 16;
    size_t filter_size = (2 * radius + 1) * (2 * radius + 1);
    // each thread has a block of filter_size pixels in shared memory
    auto kernel_func = [] __device__ (uint8_t* pixels, int count) -> uint8_t {
        if(pixels == nullptr)
            return 0;
        
        // use intersection sort pixels in shared memory using insertion sort
        for(int i = 1; i < count; ++i){
            uint8_t curr_pixel = pixels[i];
            int j = i - 1;
            for(; j >= 0 && pixels[j] > curr_pixel; j--){
                pixels[j + 1] = pixels[j];
            }
            pixels[j + 1] = curr_pixel;
        }

        return pixels[count / 2];
    };
    size_t shared_mem_size = thread_width * thread_width * filter_size * sizeof(uint8_t);
    radius = std::min(thread_width, radius);
    dim3 blockDim{thread_width, thread_width, 1};
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);
    median_filter2d_kernel<<<gridDim, blockDim, shared_mem_size>>>(d_input, d_output, dim3(width, height, 1), radius, kernel_func);

    // check for kernel launch errors
    if(check_cuda_error(cudaGetLastError(), "Kernel launch failed"))
        return input;

    err = cudaDeviceSynchronize();
    if(check_cuda_error(err, "cudaDeviceSynchronize failed"))
        return input;

    // copy back output data
    std::vector<uint8_t> output(size);
    err = cudaMemcpy(std::data(output), d_output, sizeof(uint8_t) * (width * height), cudaMemcpyDeviceToHost);
    if(check_cuda_error(err, "cudaMemcpy to output"))
        return input;

    err = cudaFree(d_input);
    d_input = nullptr;
    if(check_cuda_error(err, "cudaFree d_input"))
        return output;
    err = cudaFree(d_output);
    d_output = nullptr;
    if(check_cuda_error(err, "cudaFree d_output"))
        return output;

    return output;
}



