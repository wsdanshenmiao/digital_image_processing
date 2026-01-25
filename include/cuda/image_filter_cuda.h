#pragma once
#ifndef __IMAGE_FILTER_CUDA_H__
#define __IMAGE_FILTER_CUDA_H__

#include <vector>
#include <cstdint>

namespace dsm{
    class image_filter_cuda
    {
    public:
        static std::vector<uint8_t> domain_average_filter2d(const std::vector<uint8_t>& input, size_t width, size_t height, int radius);
    };


}

#endif // __IMAGE_FILTER_CUDA_H__