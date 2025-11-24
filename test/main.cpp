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

    Color& operator/=(float divisor) 
    {
        if (divisor != 0) r = r / divisor;
        if (divisor != 0) g = g / divisor;
        if (divisor != 0) b = b / divisor;
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
    
    bool operator==(const Color& other) const = default;
};

int main() 
{
    int width, height, components;
    int channels = 3;
    stbi_uc* img_data= stbi_load("asset/madoka_homura.jpg", &width, &height, &components, channels);
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

    // auto result = dsm::image_filter::domain_average1d(img_vector10);
    auto result = dsm::image_filter::domain_average2d(img_vector, width, height, 5);

    std::vector<uint8_t> output_data = result
        | std::views::transform([](const Color& color) {
            uint8_t r = static_cast<uint8_t>(std::clamp(color.r * 255.0f, 0.0f, 255.0f));
            uint8_t g = static_cast<uint8_t>(std::clamp(color.g * 255.0f, 0.0f, 255.0f));
            uint8_t b = static_cast<uint8_t>(std::clamp(color.b * 255.0f, 0.0f, 255.0f));
            return std::array<uint8_t, 3>{r, g, b};})
        | std::views::join
        | std::ranges::to<std::vector>();

    stbi_write_png("asset/output/output.png", width, height, channels, output_data.data(), width * channels);

    if(img_data != nullptr) {
        stbi_image_free(img_data);
    }
    std::println("Image processing completed successfully.");
    return 0;
}