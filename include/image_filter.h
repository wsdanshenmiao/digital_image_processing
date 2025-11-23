#ifndef __IMAGE_PROCESSING_H__
#define __IMAGE_PROCESSING_H__

#include <ranges>
#include <vector>

namespace dsm{

    template <typename T>
    concept AllowAddition = requires(T a, T b) { a + b; };
    template <typename T>
    concept AllowSubtraction = requires(T a, T b) { a - b; };
    template <typename T>
    concept AllowMultiplication = requires(T a, T b) { a * b; };
    template <typename T>
    concept AllowDivision = requires(T a, T b) { a / b; };
    template <typename T>
    concept AllowArithmetic = AllowAddition<T> && AllowSubtraction<T> && AllowMultiplication<T> && AllowDivision<T>;

    class image_filter
    {
    public:
        template <std::ranges::range Container> requires AllowAddition<std::ranges::range_value_t<Container>>
        static std::vector<std::ranges::range_value_t<Container>> domain_average(Container&& input);
    };
    
    template <std::ranges::range Container> requires AllowAddition<std::ranges::range_value_t<Container>>
    inline std::vector<std::ranges::range_value_t<Container>> image_filter::domain_average(Container&& input)
    {
        using T = std::ranges::range_value_t<Container>;

        std::vector<T> output{std::ranges::begin(input), std::ranges::end(input)};
        return output;
    }
}

#endif
