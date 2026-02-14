#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "cuda/image_filter_cuda.h"

#include <stb_image.h>
#include <stb_image_write.h>

#include <cstdio>
#include <vector>
#include <algorithm>
#include <numbers>
#include <chrono>
#include <filesystem>

std::vector<uint8_t> test_image_filter_cuda(const std::vector<uint8_t>& img_data, int width, int height)
{
    std::vector<uint8_t> output_data{};
    // output_data = dsm::image_filter_cuda::domain_average_filter2d(img_data, width, height, 2);
    output_data = dsm::image_filter_cuda::median_filter2d(img_data, width, height, 2);
    return output_data;
}

void test_image(const std::string& filepath, int run_count)
{
    int width, height, components;
    int channels = 3;
    stbi_uc* img_data= stbi_load(filepath.c_str(), &width, &height, &components, channels);
    if (img_data == nullptr) {
        std::printf("Failed to load image: %s\n", filepath.c_str());
        return;
    }
    std::printf("Image loaded: %s (width: %d, height: %d, channels: %d)\n", filepath.c_str(), width, height, channels);
    
    uint32_t size = static_cast<uint32_t>(width * height * channels);
    std::vector<uint8_t> img_data_vec(size);
    std::copy(img_data, img_data + size, img_data_vec.begin());
    std::vector<uint8_t> luminance(size / 3);
    for(int i = 0; i < size; i += 3){
        luminance[i / 3] = (uint8_t)(0.299f * img_data_vec[i] + 0.587f * img_data_vec[i + 1] + 0.114f * img_data_vec[i + 2]);
    }

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<uint8_t> output_data{};
    for(int i = 0; i < run_count; ++i){
        output_data = test_image_filter_cuda(luminance, width, height);
        channels = 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::printf("run count: %d, total time cost: %lld ms, average time cost: %lld ms\n", run_count, duration.count(), duration.count() / run_count);

    std::filesystem::path path{filepath};
    auto output_filepath = path.parent_path().empty() ? 
        ("output/" + path.filename().string()) : 
        (path.parent_path().string() + "/output/" + path.filename().string());
    stbi_write_png(output_filepath.c_str(), width, output_data.size() / (width * channels), channels, output_data.data(), width * channels);

    if(img_data != nullptr) {
        stbi_image_free(img_data);
    }
    std::printf("Image processing completed successfully. output saved to %s\n", output_filepath.c_str());
}

int main()
{
    test_image("asset/madoka_homura.jpg", 1);

    return 0;
}