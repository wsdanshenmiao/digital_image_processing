#ifndef __IMAGE_FILTER_H__
#define __IMAGE_FILTER_H__

#include <ranges>
#include <vector>
#include <print>

namespace dsm{
    template <typename T>
    concept AllowAdditionAndDivision = requires(T a, float b) { a + a; a += a; a / b; a /= b; };

    class image_filter
    {
    private:

    public:
        template <std::ranges::random_access_range Container> requires 
            AllowAdditionAndDivision<std::ranges::range_value_t<Container>>
        static auto domain_average1d(Container&& input, int radius);
        
        template <std::ranges::random_access_range Container> requires 
            AllowAdditionAndDivision<std::ranges::range_value_t<Container>>
        static auto domain_average2d(Container&& input, size_t width, size_t height, int radius);

        template<std::ranges::random_access_range Container>
        static auto median_filter1d(Container&& input, int radius);
    };
    
    template <std::ranges::random_access_range Container> requires 
        AllowAdditionAndDivision<std::ranges::range_value_t<Container>>
    inline auto image_filter::domain_average1d(Container&& input, int radius)
    {
        using namespace std;
        using T = ranges::range_value_t<Container>;
        // Avoid uint8 calculation overflow
        using Type = std::conditional_t<std::is_integral_v<T> && sizeof(T) < sizeof(int), int, T>;

        std::vector<T> output{ranges::begin(input), ranges::end(input)};
        if(std::size(output) < static_cast<size_t>(2 * radius + 1) || radius < 1) {
            return output;
        }

        auto end_it = ranges::prev(ranges::end(output), radius);
        float inv_size = static_cast<float>(2 * radius + 1);
        for(auto it = ranges::next(ranges::begin(output), radius); it != end_it; ++it) {
            Type sum = Type{};
            auto end_prev_it = std::next(it, radius);
            for(auto prev_it = std::prev(it, radius); prev_it <= end_prev_it; ++prev_it) {
                sum += static_cast<Type>(*prev_it);
            }
            sum *= inv_size;
            *it = static_cast<T>(sum);
        }

        return output;
    }

    template <std::ranges::random_access_range Container> requires
        AllowAdditionAndDivision<std::ranges::range_value_t<Container>>
    inline auto image_filter::domain_average2d(Container &&input, size_t width, size_t height, int radius)
    {
        using namespace std;
        using T = ranges::range_value_t<Container>;
        using Type = std::conditional_t<std::is_integral_v<T> && sizeof(T) < sizeof(int), int, T>;

        std::vector<T> output{input.begin(), input.end()};
        if(std::size(output) < static_cast<size_t>((2 * radius + 1) * 2) || radius < 1){
            return output;
        }
        auto output_view = output | views::chunk(width);

        // process rows
        float inv_size = 1 / static_cast<float>((2 * radius + 1) * (2 * radius + 1));
        for(size_t i = radius; i < height - radius; ++i) {
            for(size_t j = radius; j < width - radius; ++j) {
                Type sum = Type{};
                for(int m = -radius; m <= radius; ++m) {
                    for(int n = -radius; n <= radius; ++n) {
                        sum += static_cast<Type>(output_view[i + m][j + n]);
                    }
                }
                sum *= inv_size;
                output_view[i][j] = static_cast<T>(sum);
            }
        }

        return output;
    }
    
    template <std::ranges::random_access_range Container>
    inline auto image_filter::median_filter1d(Container &&input, int radius)
    {
        using namespace std;
        using T = ranges::range_value_t<Container>;
        
        std::vector<T> output{input.begin(), input.end()};
        if(std::size(output) < static_cast<size>(2 * radius + 1)){
            return output;
        }

        auto end_it = ranges::prev(ranges::end(output), radius);
        for(auto it = ranges::next(ranges::begin(output), radius); it != end_it; ++it) {
            auto view = ranges::subrange(ranges::prev(it, radius), ranges::next(it, radius + 1));
            auto mid_it = std::ranges::nth_element(view, ranges::next(ranges::begin(view), radius));
            *it = *mid_it;
        }

        return output;
    }
}

#endif
