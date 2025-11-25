#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "image_filter.h"

#include <stb_image.h>
#include <stb_image_write.h>

#include <print>
#include <ranges>
#include <vector>
#include <algorithm>

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

int main() 
{
    int width, height, components;
    int channels = 3;
    std::string filename = "madoka_homura";
    std::string suffix = ".jpg";
    auto filepath = "asset/" + filename + suffix;
    stbi_uc* img_data= stbi_load(filepath.c_str(), &width, &height, &components, channels);
    if (img_data == nullptr) 
        return -1;
    
    uint32_t size = width * height * channels;
    std::vector<Color> img_vector = std::span(img_data, size)
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

    // auto result = dsm::image_filter::domain_average_filter1d(img_vector, 5);
    // auto result = dsm::image_filter::domain_average_filter2d(img_vector, width, height, 5);
    // auto result = dsm::image_filter::median_filter1d(img_vector, 5, [](const Color& a, const Color& b) {
    //     float luminance_a = 0.299f * a.r + 0.587f * a.g + 0.114f * a.b;
    //     float luminance_b = 0.299f * b.r + 0.587f * b.g + 0.114f * b.b;
    //     return luminance_a < luminance_b;
    // });
    // auto result = dsm::image_filter::median_filter2d(img_vector, width, height, 5, 
    //     [](const Color& a, const Color& b) {
    //     float luminance_a = 0.299f * a.r + 0.587f * a.g + 0.114f * a.b;
    //     float luminance_b = 0.299f * b.r + 0.587f * b.g + 0.114f * b.b;
    //     return luminance_a < luminance_b;
    // });
    // auto result = dsm::image_filter::gradient_filter1d(img_vector);
    // auto result = dsm::image_filter::gradient_filter2d(luminance, width, height);
    // auto result = dsm::image_filter::robert_gradient_filter(luminance, width, height);
    auto result = dsm::image_filter::laplacian_filter(img_vector, width, height);

    std::vector<uint8_t> output_data = result
        | std::views::transform([](const Color& color) {
            uint8_t r = static_cast<uint8_t>(std::clamp(color.r * 255.0f, 0.0f, 255.0f));
            uint8_t g = static_cast<uint8_t>(std::clamp(color.g * 255.0f, 0.0f, 255.0f));
            uint8_t b = static_cast<uint8_t>(std::clamp(color.b * 255.0f, 0.0f, 255.0f));
            return std::array<uint8_t, 3>{r, g, b};})
        | std::views::join
        | std::ranges::to<std::vector>();

    auto output_filepath = "asset/output/" + filename + ".png";
    stbi_write_png(output_filepath.c_str(), width, height, channels, output_data.data(), width * channels);

    if(img_data != nullptr) {
        stbi_image_free(img_data);
    }
    std::println("Image processing completed successfully.");
    return 0;
}