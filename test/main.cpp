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
#include <filesystem>

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

enum FilterType
{
    DomainAverage,
    Median,
    Gradient,
    RobertGradient,
    Laplacian,
    Directional,
    Sobel,
    Prewitt,
    Kirsch
};

enum CodingType
{
    Huffman,
    ShannonFano,
    ShannonFanoSplitMid
};


template <std::ranges::random_access_range Container>
auto run_image_filter(FilterType filter_type, Container&& input, size_t width, size_t height, size_t channels)
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
            return Color{lumi, lumi, lumi}; 
        });

    auto cmp_func = [](const Color& a, const Color& b) {
        float luminance_a = 0.299f * a.r + 0.587f * a.g + 0.114f * a.b;
        float luminance_b = 0.299f * b.r + 0.587f * b.g + 0.114f * b.b;
        return luminance_a < luminance_b;
    };

    std::vector<Color> result{};
    switch (filter_type){
    case FilterType::DomainAverage:
        result = dsm::image_filter::domain_average_filter2d(img_vector, width, height, 2);
        break;
    case FilterType::Median:
        result = dsm::image_filter::median_filter2d(img_vector, width, height, 2, cmp_func);
        break;
    case FilterType::Gradient:
        result = dsm::image_filter::gradient_filter2d(luminance, width, height);
        break;
    case FilterType::RobertGradient:
        result = dsm::image_filter::robert_gradient_filter(luminance, width, height);
        break;
    case FilterType::Laplacian:
        result = dsm::image_filter::laplacian_filter(img_vector, width, height);
        break;
    case FilterType::Directional:
        result = dsm::image_filter::directional_filter(img_vector, width, height, std::numbers::pi_v<float> / 4.0f);
        break;
    case FilterType::Sobel:
        result = dsm::image_filter::sobel_filter(luminance, width, height, cmp_func);
        break;
    case FilterType::Prewitt:
        result = dsm::image_filter::prewitt_filter(luminance, width, height, cmp_func);
        break;
    case FilterType::Kirsch:
        result = dsm::image_filter::kirsch_filter(luminance, width, height, cmp_func);
    default:
        break;
    }
    
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
std::vector<uint8_t> run_image_coding(CodingType coding_type, Container&& input)
{
    switch (coding_type) {
    case CodingType::Huffman: {
        dsm::image_coding::huffman_coder coder{};
        coder.encode(input);
        return coder.decode();
    }
    case CodingType::ShannonFano: {
        dsm::image_coding::shannon_fano_coder coder{};
        coder.encode(input);
        return coder.decode();
    }
    case CodingType::ShannonFanoSplitMid: {
        dsm::image_coding::shannon_fano_coder coder{};
        coder.split_mid_encode(input);
        return coder.decode();
    }
    default:
        break;
    }
    return {};
}

template <typename ProcessType> requires std::is_same_v<ProcessType, FilterType> || std::is_same_v<ProcessType, CodingType>
void test_image(const std::string& filepath, ProcessType process_type, int run_count)
{
    int width, height, components;
    int channels = 3;
    stbi_uc* img_data= stbi_load(filepath.c_str(), &width, &height, &components, channels);
    if (img_data == nullptr) 
        return;
    
    uint32_t size = width * height * channels;
    auto input_data = std::span{ img_data, size };

    std::println("Image loaded: {} (width: {}, height: {}, channels: {})", filepath, width, height, channels);
    
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<uint8_t> output_data{};
    for(int i = 0; i < run_count; ++i){
        if constexpr(std::is_same_v<ProcessType, FilterType>) {
            output_data = run_image_filter(process_type, input_data, width, height, channels);
        }
        else if constexpr(std::is_same_v<ProcessType, CodingType>) {
            output_data = run_image_coding(process_type, input_data);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::println("run count: {}, total time cost: {}, average time cost: {}", run_count, duration, duration / run_count);

    std::filesystem::path path{filepath};
    auto output_filepath = path.parent_path().empty() ? 
        ("output/" + path.filename().string()) : 
        (path.parent_path().string() + "/output/" + path.filename().string());
    stbi_write_png(output_filepath.c_str(), width, output_data.size() / (width * channels), channels, output_data.data(), width * channels);

    if(img_data != nullptr) {
        stbi_image_free(img_data);
    }
    std::println("Image processing completed successfully. output saved to {}", output_filepath);
}

void print_usage(const char* program_name) 
{
    std::println("Usage: {} <image_path> <process_type> <run_count>", program_name);
    std::println("Example: {} test_image.png 5 1", program_name);
    std::println();
    std::println("Filter Types: \n1.DomainAverage\n2.Median\n3.Gradient\n4.RobertGradient\n5.Laplacian\n6.Directional\n7.Sobel\n8.Prewitt\n9.Kirsch");
    std::println();
    std::println("Coding Types: \n10.Huffman\n11.ShannonFano\n12.ShannonFanoSplitMid");
}

int main(int argc, char* argv[]) 
{
    if(argc <= 2) {
        print_usage(argv[0]);
    }
    else {
        std::string image_path = argv[1];
        int process_type = std::atof(argv[2]);
        process_type = std::clamp(process_type, 1, 12) == process_type ? process_type : 1;
        int run_count = argc <= 3 ? 1 : std::clamp((int)std::atof(argv[3]), 1, 10);
        if(9 < process_type && process_type <= 12){
            test_image(image_path, CodingType(process_type - 10), run_count);
        }
        else if(process_type <= 9){
            test_image(image_path, FilterType(process_type - 1), run_count);    
        }
    }
    return 0;
}