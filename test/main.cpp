#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "image_filter.h"
#include "image_coding/huffman_coding.h"
#include "image_coding/shannon_fano_coding.h"

#include <stb_image.h>
#include <stb_image_write.h>

#include <print>
#include <ranges>
#include <vector>
#include <algorithm>
#include <numbers>
#include <chrono>

struct Color
{
    float r, g, b;

    Color& operator+=(const Color& other) 
    {
        r = r + other.r;
        g = g + other.g;
        b = b + other.b;
        return *this;
    }
    Color& operator-=(const Color& other) 
    {
        r = r - other.r;
        g = g - other.g;
        b = b - other.b;
        return *this;
    }
    Color& operator/=(float divisor) 
    {
        if (divisor != 0) r = r / divisor;
        if (divisor != 0) g = g / divisor;
        if (divisor != 0) b = b / divisor;
        return *this;
    }
    Color& operator*=(float multiplier) 
    {
        r = r * multiplier;
        g = g * multiplier;
        b = b * multiplier;
        return *this;
    }

    Color operator+(const Color& other) const
    {
        Color result = *this;
        result += other;
        return result;
    }
    Color operator/(float divisor) const
    {
        Color result = *this;
        result /= divisor;
        return result;
    }
    Color operator-(const Color& other) const
    {
        Color result = *this;
        result -= other;
        return result;
    }
    Color operator*(float multiplier) const
    {
        Color result = *this;
        result *= multiplier;
        return result;
    }
    
    bool operator==(const Color& other) const = default;
};

Color abs(const Color& color) 
{
    return Color{std::abs(color.r), std::abs(color.g), std::abs(color.b)};
}

Color max(const Color& a, const Color& b)
{
    return Color{std::max(a.r, b.r), std::max(a.g, b.g), std::max(a.b, b.b)};
}


template <std::ranges::random_access_range Container>
auto test_image_filter(Container&& input, size_t width, size_t height, size_t channels)
{    
    std::vector<Color> img_vector = input
        | std::views::chunk(channels) 
        | std::views::transform([](auto chunk) {
            auto it = chunk.begin();
            float r = static_cast<float>(*it++) / 255.0f;
            float g = static_cast<float>(*it++) / 255.0f;
            float b = static_cast<float>(*it++) / 255.0f;
            return Color{r, g, b};}) 
        | std::ranges::to<std::vector>();

    auto luminance = img_vector 
        | std::views::transform([](const auto& col){
            auto lumi = col.r * 0.299f + col.g * 0.587f + col.b * 0.114f;
            return Color{lumi, lumi, lumi}; });

    auto cmp_func = [](const Color& a, const Color& b) {
        float luminance_a = 0.299f * a.r + 0.587f * a.g + 0.114f * a.b;
        float luminance_b = 0.299f * b.r + 0.587f * b.g + 0.114f * b.b;
        return luminance_a < luminance_b;
    };

    // auto result = dsm::image_filter::domain_average_filter1d(img_vector, 5);
    // auto result = dsm::image_filter::domain_average_filter2d(img_vector, width, height, 5);
    // auto result = dsm::image_filter::median_filter1d(img_vector, 5, cmp_func);
    // auto result = dsm::image_filter::median_filter2d(img_vector, width, height, 5, cmp_func);
    // auto result = dsm::image_filter::gradient_filter1d(img_vector);
    // auto result = dsm::image_filter::gradient_filter2d(luminance, width, height);
    // auto result = dsm::image_filter::robert_gradient_filter(luminance, width, height);
    // auto result = dsm::image_filter::laplacian_filter(img_vector, width, height);
    // auto result = dsm::image_filter::directional_filter(img_vector, width, height, std::numbers::pi_v<float> / 4.0f);
    // auto result = dsm::image_filter::sobel_filter(luminance, width, height, cmp_func);
    // auto result = dsm::image_filter::prewitt_filter(luminance, width, height, cmp_func);
    auto result = dsm::image_filter::kirsch_filter(luminance, width, height, cmp_func);
    
    std::vector<uint8_t> output_data = result
        | std::views::transform([](const Color& color) {
            uint8_t r = static_cast<uint8_t>(std::clamp(color.r * 255.0f, 0.0f, 255.0f));
            uint8_t g = static_cast<uint8_t>(std::clamp(color.g * 255.0f, 0.0f, 255.0f));
            uint8_t b = static_cast<uint8_t>(std::clamp(color.b * 255.0f, 0.0f, 255.0f));
            return std::array<uint8_t, 3>{r, g, b};})
        | std::views::join
        | std::ranges::to<std::vector>();

    return output_data;
}

template <std::ranges::range Container>
std::vector<uint8_t> test_image_coding(Container&& input)
{
    dsm::image_coding::shannon_fano_coder coder{};
    coder.split_mid_encode(std::forward<Container>(input));
    auto data = coder.decode();
    return data;
}


void test_image(int run_count)
{
    int width, height, components;
    int channels = 3;
    std::string mhfilename = "madoka_homura";
    std::string rtfilename = "raytracing";
    auto selected_filename = mhfilename;
    std::string suffixjpg = ".jpg";
    std::string suffixpng = ".png";
    auto filepath = "asset/" + selected_filename + suffixjpg;
    stbi_uc* img_data= stbi_load(filepath.c_str(), &width, &height, &components, channels);
    if (img_data == nullptr) 
        return;
    
    uint32_t size = width * height * channels;
    auto input_data = std::span{ img_data, size };
    std::vector<uint8_t> input_data{};
    
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<uint8_t> output_data;
    for(int i = 0; i < run_count; ++i){
        // auto output_data = test_image_filter(input_data, width, height, channels);
        output_data = test_image_coding(input_data);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::println("run count: {}, total time cost: {}, average time cost: {}", run_count, duration, duration / run_count);

    for (const auto& [index, data] : input_data | std::views::enumerate) {
        assert(data == output_data[index]);
    }

    auto output_filepath = "asset/output/" + selected_filename + ".png";
    stbi_write_png(output_filepath.c_str(), width, output_data.size() / (width * channels), channels, output_data.data(), width * channels);

    if(img_data != nullptr) {
        stbi_image_free(img_data);
    }
    std::println("Image processing completed successfully. output saved to {}", output_filepath);
}

int main(int argc, char* argv[]) 
{
    int run_count = argc <= 1 ? 1 : std::clamp((int)std::atof(argv[1]), 1, 10);
    test_image(run_count);
    return 0;
}