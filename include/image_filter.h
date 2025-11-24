#ifndef __IMAGE_FILTER_H__
#define __IMAGE_FILTER_H__

#include <ranges>
#include <vector>
#include <algorithm>
#include <numeric>

namespace dsm{
    template <typename T>
    concept AllowAdditionAndDivision = requires(T a, float b) { a + a; a += a; a * b; a *= b; };

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

        template<std::ranges::random_access_range Container, typename Comparator = std::ranges::less>
        static auto median_filter1d(Container&& input, int radius, Comparator comp = Comparator{});

        template<std::ranges::random_access_range Container, typename Comparator = std::ranges::less>
        static auto median_filter2d(Container&& input, size_t width, size_t height, int radius, Comparator comp = Comparator{});
    };
    
    template <std::ranges::random_access_range Container> requires 
        AllowAdditionAndDivision<std::ranges::range_value_t<Container>>
    inline auto image_filter::domain_average1d(Container&& input, int radius)
    {
        using namespace std;
        using T = ranges::range_value_t<Container>;
        // Avoid uint8 calculation overflow
        using Type = std::conditional_t<std::is_integral_v<T> && sizeof(T) < sizeof(int), int, T>;

        auto input_begin = ranges::begin(input);

        std::vector<T> output{input_begin, ranges::end(input)};
        if(ranges::size(output) < static_cast<size_t>(2 * radius + 1) || radius < 1) {
            return output;
        }

        auto end_it = ranges::prev(ranges::end(input), radius);
        float inv_size = 1.f / static_cast<float>(2 * radius + 1);
        for(auto it = ranges::next(input_begin, radius); it != end_it; ++it) {
            Type sum = Type{};
            auto end_prev_it = ranges::next(it, radius);
            for(auto prev_it = ranges::prev(it, radius); prev_it <= end_prev_it; ++prev_it) {
                sum += static_cast<Type>(*prev_it);
            }
            sum *= inv_size;
            auto dist = ranges::distance(input_begin, it);
            auto output_it = ranges::next(ranges::begin(output), dist);
            *output_it = static_cast<T>(sum);
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

        auto input_begin = ranges::begin(input);
        std::vector<T> output{input_begin, ranges::end(input)};
        if(ranges::size(output) < static_cast<size_t>((2 * radius + 1) * 2) || radius < 1){
            return output;
        }
        auto input_view = input | views::chunk(width);
        auto input_view_begin = ranges::begin(input_view);

        // process rows
        float inv_size = 1.f / static_cast<float>((2 * radius + 1) * (2 * radius + 1));
        auto end_row_it = ranges::prev(ranges::end(input_view), radius);
        for(auto row_it = ranges::next(input_view_begin, radius); row_it != end_row_it; ++row_it) {
            // distance from the current line to the beginning of the input view
            auto dist_col = ranges::distance(input_view_begin, row_it);
            auto row_begin = ranges::begin(*row_it);
            for(auto it = ranges::next(row_begin, radius); 
                it != ranges::prev(ranges::end(*row_it), radius); ++it) {
                Type sum = Type{};
                // distance from the current element to the beginning of the line
                auto dist_row = ranges::distance(row_begin, it);
                // each line in kernel
                for(auto it_m = ranges::prev(row_it, radius); it_m <= ranges::next(row_it, radius); ++it_m) {
                    auto begin_it_m = ranges::begin(*it_m);
                    for(auto it_n = ranges::next(begin_it_m, dist_row - radius); 
                        it_n <= ranges::next(begin_it_m, dist_row + radius); ++it_n) {
                        sum += static_cast<Type>(*it_n);
                    }
                }
                sum *= inv_size;
                auto output_it = ranges::next(ranges::begin(output), dist_col * width + dist_row);
                *output_it = static_cast<T>(sum);
            }
        }

        return output;
    }
    
    template<std::ranges::random_access_range Container, typename Comparator>
    inline auto image_filter::median_filter1d(Container &&input, int radius, Comparator comp)
    {
        using namespace std;
        using T = ranges::range_value_t<Container>;
        
        std::vector<T> output{input.begin(), input.end()};
        if(std::size(output) < static_cast<size_t>(2 * radius + 1)){
            return output;
        }

        std::vector<size_t> indices(2 * radius + 1);
        auto end_it = ranges::prev(ranges::end(input), radius);
        for(auto it = ranges::next(ranges::begin(input), radius); it != end_it; ++it) {
            std::iota(std::begin(indices), std::end(indices), 0);
            auto view = ranges::subrange(ranges::prev(it, radius), ranges::next(it, radius + 1));
            auto mid_it = ranges::next(ranges::begin(indices), radius);
            ranges::nth_element(indices, mid_it, [&](const auto& a, const auto& b) {
                return comp(view[a], view[b]);
            });
            auto dist = ranges::distance(ranges::begin(input), it);
            auto output_it = ranges::next(ranges::begin(output), dist);
            *output_it = view[*mid_it];
        }

        return output;
    }
    
    template <std::ranges::random_access_range Container, typename Comparator>
    inline auto image_filter::median_filter2d(Container &&input, size_t width, size_t height, int radius, Comparator comp)
    {
        using namespace std;
        using T = ranges::range_value_t<Container>;

        std::vector<T> output{input.begin(), input.end()};
        if(ranges::size(input) < static_cast<size_t>((2 * radius + 1) * (2 * radius + 1)) || radius < 1){
            return output;
        }

        auto input_view = input | views::chunk(width);

        
        return output;
    }
}

#endif
