#ifndef __IMAGE_FILTER_H__
#define __IMAGE_FILTER_H__

#include <ranges>
#include <vector>
#include <algorithm>
#include <numeric>

namespace dsm{
    template <typename T>
    concept Addible = requires(T a, float b) { a + a; a += a; };
    template <typename T>
    concept Subtractable = requires(T a, float b) { a - a; a -= a; };
    template <typename T>
    concept Multiplicable = requires(T a, float b) { a * b; a *= b; };
    template <typename T>
    concept Divisible = requires(T a, float b) { a / b; a /= b; };

    class image_filter
    {
    private:

    public:
        template <std::ranges::random_access_range Container> requires 
            Addible<std::ranges::range_value_t<Container>> && 
            Multiplicable<std::ranges::range_value_t<Container>>
        static auto domain_average_filter1d(Container&& input, int radius);
        
        template <std::ranges::random_access_range Container> requires 
            Addible<std::ranges::range_value_t<Container>> && 
            Multiplicable<std::ranges::range_value_t<Container>>
        static auto domain_average_filter2d(Container&& input, size_t width, size_t height, int radius);

        template<std::ranges::random_access_range Container, typename Comparator = std::ranges::less>
        static auto median_filter1d(Container&& input, int radius, Comparator&& comp = Comparator{});

        template<std::ranges::random_access_range Container, typename Comparator = std::ranges::less>
        static auto median_filter2d(Container&& input, size_t width, size_t height, int radius, Comparator&& comp = Comparator{});
    
        template<std::ranges::random_access_range Container> requires
            Subtractable<std::ranges::range_value_t<Container>>
        static auto gradient_filter1d(Container&& input);

        template<std::ranges::random_access_range Container> requires
            Subtractable<std::ranges::range_value_t<Container>> &&
            Addible<std::ranges::range_value_t<Container>>
        static auto gradient_filter2d(Container&& input, size_t width, size_t height);
    };
    
    template <std::ranges::random_access_range Container> requires 
        Addible<std::ranges::range_value_t<Container>> && 
        Multiplicable<std::ranges::range_value_t<Container>>
    inline auto image_filter::domain_average_filter1d(Container&& input, int radius)
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
        Addible<std::ranges::range_value_t<Container>> && 
        Multiplicable<std::ranges::range_value_t<Container>>
    inline auto image_filter::domain_average_filter2d(Container &&input, size_t width, size_t height, int radius)
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

                std::vector<ranges::subrange<decltype(row_begin)>> kernel_lines{};
                kernel_lines.reserve(radius * 2 + 1);
                // collect each line in kernel
                for(auto it_m = ranges::prev(row_it, radius); it_m <= ranges::next(row_it, radius); ++it_m) {
                    auto begin_it_m = ranges::begin(*it_m);
                    kernel_lines.emplace_back(ranges::next(begin_it_m, dist_row - radius),
                        ranges::next(begin_it_m, dist_row + radius + 1));
                }
                auto kernel_line = kernel_lines | views::join;
                for(const auto& val : kernel_line) {
                    sum += static_cast<Type>(val);
                }
                sum *= inv_size;
                auto output_it = ranges::next(ranges::begin(output), dist_col * width + dist_row);
                *output_it = static_cast<T>(sum);
            }
        }

        return output;
    }
    
    template<std::ranges::random_access_range Container, typename Comparator>
    inline auto image_filter::median_filter1d(Container &&input, int radius, Comparator&& comp)
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
    inline auto image_filter::median_filter2d(Container &&input, size_t width, size_t height, int radius, Comparator&& comp)
    {
        using namespace std;
        using T = ranges::range_value_t<Container>;

        std::vector<T> output{input.begin(), input.end()};
        if(ranges::size(input) < static_cast<size_t>((2 * radius + 1) * (2 * radius + 1)) || radius < 1){
            return output;
        }

        auto input_view = input | views::chunk(width);
        auto end_row_it = ranges::prev(ranges::end(input_view), radius);
        for(auto row_it = ranges::next(ranges::begin(input_view), radius); row_it != end_row_it; ++row_it) {
            auto dist_col = ranges::distance(ranges::begin(input_view), row_it);
            auto row_begin = ranges::begin(*row_it);
            for(auto it = ranges::next(row_begin, radius); 
                it != ranges::prev(ranges::end(*row_it), radius); ++it) {
                auto dist_row = ranges::distance(row_begin, it);
               
                std::vector<ranges::subrange<decltype(row_begin)>> kernel_lines{};
                kernel_lines.reserve(2 * radius + 1);
                // collect each line in kernel
                for(auto it_m = ranges::prev(row_it, radius); it_m <= ranges::next(row_it, radius); ++it_m) {
                    auto begin_it_m = ranges::begin(*it_m);
                    kernel_lines.emplace_back(ranges::next(begin_it_m, dist_row - radius),
                        ranges::next(begin_it_m, dist_row + radius + 1));
                }

                auto kernel_values = kernel_lines | views::join;
                auto kernel_begin = ranges::begin(kernel_values);
                std::vector<size_t> indices(ranges::size(kernel_lines) * ranges::size(kernel_lines));
                std::iota(ranges::begin(indices), ranges::end(indices), 0);

                auto mid_it = ranges::next(ranges::begin(indices), ranges::size(indices) / 2);
                ranges::nth_element(indices, mid_it, [&](const auto& a, const auto& b) {
                    return comp(*ranges::next(kernel_begin, a), *ranges::next(kernel_begin, b));
                });

                auto output_it = ranges::next(ranges::begin(output), dist_col * width + dist_row);
                *output_it = *ranges::next(kernel_begin, *mid_it);
            }
        }

        return output;
    }
    
    template <std::ranges::random_access_range Container> requires
        Subtractable<std::ranges::range_value_t<Container>>
    inline auto image_filter::gradient_filter1d(Container &&input)
    {
        using namespace std;
        using T = ranges::range_value_t<Container>;

        std::vector<T> output(ranges::size(input));
        if(ranges::size(output) < 2){
            return output;
        }

        for(auto it = ranges::begin(input); it != ranges::prev(ranges::end(input)); ++it) {
            auto dist = ranges::distance(ranges::begin(input), it);
            auto output_it = ranges::next(ranges::begin(output), dist);
            *output_it = abs(*it - *ranges::next(it));
        }
        output.back() = *ranges::prev(ranges::end(input), 2);

        return output;
    }
    
    template <std::ranges::random_access_range Container> requires
        Subtractable<std::ranges::range_value_t<Container>> &&
        Addible<std::ranges::range_value_t<Container>>
    inline auto image_filter::gradient_filter2d(Container &&input, size_t width, size_t height)
    {
        using namespace std;
        using T = ranges::range_value_t<Container>;
        
        if(width < 2 || height < 2){
            return std::vector<T>{ranges::begin(input), ranges::end(input)};
        }

        auto input_view = input | views::chunk(width);
        std::vector<T> output(ranges::size(input));

        for(auto row_it = ranges::begin(input_view); row_it != ranges::end(input_view); ++row_it) {
            auto dist_col = ranges::distance(ranges::begin(input_view), row_it);
            auto next_row = ranges::next(row_it);
            if(next_row == ranges::end(input_view)) {
                auto output_it = ranges::next(ranges::begin(output), dist_col * width);
                ranges::copy(*ranges::prev(row_it), output_it);
                break;
            }

            auto begin_row_it = ranges::begin(*row_it);
            auto end_row_it = ranges::end(*row_it);
            for(auto it = begin_row_it; it != end_row_it; ++it) {
                auto dist_row = ranges::distance(begin_row_it, it);
                auto dist = dist_col * width + dist_row;
                auto output_it = ranges::next(ranges::begin(output), dist);
                if(ranges::next(it) == end_row_it) {
                    *output_it = *ranges::prev(it);
                }
                else {
                    T gx = abs(*it - *ranges::next(it));
                    T gy = abs(*it - *ranges::next(ranges::begin(*next_row), dist_row));
                    *output_it = gx + gy;
                }
            }
        }

        return output;
    }
}

#endif
