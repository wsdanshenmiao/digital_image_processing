#ifndef __IMAGE_FILTER_H__
#define __IMAGE_FILTER_H__

#include <ranges>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace dsm{
    template <typename T>
    concept Addible = requires(T a, float b) { a + a; a += a; };
    template <typename T>
    concept Subtractable = requires(T a, float b) { a - a; a -= a; };
    template <typename T>
    concept Multiplicable = requires(T a, float b) { a * b; a *= b; };
    template <typename T>
    concept Divisible = requires(T a, float b) { a / b; a /= b; };

    template <typename T>
    concept MultiFloat = requires(T a, float b) { a * b; a *= b; };

    class image_filter
    {
    private:

    public:
        template <std::ranges::random_access_range Container> requires 
            Addible<std::ranges::range_value_t<Container>> && 
            MultiFloat<std::ranges::range_value_t<Container>>
        static auto domain_average_filter1d(Container&& input, int radius);
        
        template <std::ranges::random_access_range Container> requires 
            Addible<std::ranges::range_value_t<Container>> && 
            MultiFloat<std::ranges::range_value_t<Container>>
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

        template<std::ranges::random_access_range Container> requires
            Subtractable<std::ranges::range_value_t<Container>> &&
            Addible<std::ranges::range_value_t<Container>>
        static auto robert_gradient_filter(Container&& input, size_t width, size_t height);

        template<std::ranges::random_access_range Container> requires
            Addible<std::ranges::range_value_t<Container>> &&
            MultiFloat<std::ranges::range_value_t<Container>>
        static auto laplacian_filter(Container&& input, size_t width, size_t height);

        template<std::ranges::random_access_range Container> requires
            Addible<std::ranges::range_value_t<Container>> &&
            MultiFloat<std::ranges::range_value_t<Container>>
        static auto directional_filter(Container&& input, size_t width, size_t height, float angle);

    private:
        template<std::ranges::random_access_range Container, typename Func>
        static auto filter_kernel2(Container&& input, size_t width, size_t height, Func&& func);
    
        template<std::ranges::random_access_range Container, typename Func>
        static auto filter_kernel(Container&& input, size_t width, size_t height, int radius, Func&& func);
    };
    
    template <std::ranges::random_access_range Container> requires 
        Addible<std::ranges::range_value_t<Container>> && 
        MultiFloat<std::ranges::range_value_t<Container>>
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
        MultiFloat<std::ranges::range_value_t<Container>>
    inline auto image_filter::domain_average_filter2d(Container &&input, size_t width, size_t height, int radius)
    {
        float inv_size = 1.f / static_cast<float>((2 * radius + 1) * (2 * radius + 1));
        return filter_kernel(input, width, height, radius, 
            [inv_size](const auto& kernel_values) {
                using Subrange = std::ranges::range_value_t<decltype(kernel_values)>;
                using T = std::ranges::range_value_t<Subrange>;
                using Type = std::conditional_t<std::is_integral_v<T> && sizeof(T) < sizeof(int), int, T>;
                Type sum = Type{};
                for(const auto& val : kernel_values | std::views::join) {
                    sum += static_cast<Type>(val);
                }
                sum *= inv_size;
                return static_cast<T>(sum);
            });
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
        size_t kernel_size = (2 * radius + 1) * (2 * radius + 1);
        return filter_kernel(input, width, height, radius, 
            [kernel_size, comp](const auto& kernel_values) {
                using namespace std;
                auto joined_kernel = kernel_values | views::join;
                auto kernel_begin = ranges::begin(joined_kernel);

                std::vector<size_t> indices(kernel_size);
                ranges::iota(indices, 0);
                auto mid_it = ranges::next(ranges::begin(indices), ranges::size(indices) / 2);
                ranges::nth_element(indices, mid_it, [&](const auto& a, const auto& b) {
                    return comp(*ranges::next(kernel_begin, a), *ranges::next(kernel_begin, b));
                });

                return *ranges::next(kernel_begin, *mid_it);
            });
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
        return filter_kernel2(std::forward<Container>(input), width, height,
            [](const auto& kernel_values) {
                using It = std::ranges::range_value_t<decltype(kernel_values)>;
                using T = std::iter_value_t<It>;
                T gx = abs(*kernel_values[0] - *kernel_values[1]);
                T gy = abs(*kernel_values[0] - *kernel_values[2]);
                return gx + gy;
            });
    }

    template<std::ranges::random_access_range Container> requires
        Subtractable<std::ranges::range_value_t<Container>> &&
        Addible<std::ranges::range_value_t<Container>>
    inline auto image_filter::robert_gradient_filter(Container&& input, size_t width, size_t height)
    {
        return filter_kernel2(std::forward<Container>(input), width, height,
            [](const auto& kernel_values) {
                static_assert(std::ranges::size(kernel_values) == 4, "Neighbors size must be 4.");
                using It = std::ranges::range_value_t<decltype(kernel_values)>;
                using T = std::iter_value_t<It>;
                T gx = abs(*kernel_values[0] - *kernel_values[3]);
                T gy = abs(*kernel_values[1] - *kernel_values[2]);
                return gx + gy;
            });
    }

    template <std::ranges::random_access_range Container> requires
        Addible<std::ranges::range_value_t<Container>> &&
        MultiFloat<std::ranges::range_value_t<Container>>
    inline auto image_filter::laplacian_filter(Container &&input, size_t width, size_t height)
    {
        return filter_kernel(std::forward<Container>(input), width, height, 1,
            [](const auto& kernel_values){
                using Subrange = std::ranges::range_value_t<decltype(kernel_values)>;
                using T = std::ranges::range_value_t<Subrange>;
                using Type = std::conditional_t<std::is_integral_v<T> && sizeof(T) < sizeof(int), int, T>;
                static std::array<float, 9> kernel = { 0, -1, 0, -1, 4, -1, 0, -1, 0 };
                Type sum{};
                for(const auto& [index, val] : kernel_values | std::views::join | std::views::enumerate) {
                    sum += val * kernel[index];
                }
                return static_cast<T>(sum);
            });
    }

    template<std::ranges::random_access_range Container> requires
        Addible<std::ranges::range_value_t<Container>> &&
        MultiFloat<std::ranges::range_value_t<Container>>
    inline auto image_filter::directional_filter(Container &&input, size_t width, size_t height, float angle)
    {
        float cos_angle = std::cos(angle);
        float cos_angle_sq = cos_angle * cos_angle;
        float sin_angle = std::sqrt(1 - cos_angle_sq);
        return filter_kernel(std::forward<Container>(input), width, height, 1,
            [=](const auto& kernel_values){
                using Subrange = std::ranges::range_value_t<decltype(kernel_values)>;
                using T = std::ranges::range_value_t<Subrange>;
                using Type = std::conditional_t<std::is_integral_v<T> && sizeof(T) < sizeof(int), int, T>;
                static std::array<float, 9> kernel_xx = {1, -2, 1, 2, -4, 2, 1, -2, 1};
                static std::array<float, 9> kernel_yy = {1, 2, 1, -2, -4, -2, 1, 2, 1};
                static std::array<float, 9> kernel_xy = {-1, 0, 1, 0, 0, 0, 1, 0, -1};

                Type sum{};
                for(const auto& [index, val] : kernel_values | std::views::join | std::views::enumerate) {
                    float kernel = cos_angle_sq * kernel_xx[index]
                        + sin_angle * sin_angle * kernel_yy[index]
                        + 2 * sin_angle * cos_angle * kernel_xy[index];
                    sum += val * kernel;
                }
                return static_cast<T>(sum);
            });
    }

    template<std::ranges::random_access_range Container, typename Func>
    inline auto image_filter::filter_kernel2(Container &&input, size_t width, size_t height, Func&& func)
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
                    std::array<decltype(it), 4> kernel_values{
                        it,
                        ranges::next(it),
                        ranges::next(ranges::begin(*next_row), dist_row),
                        ranges::next(ranges::begin(*next_row), dist_row + 1)
                    };
                    *output_it = func(kernel_values);
                }
            }
        }

        return output;
    }
    
    template <std::ranges::random_access_range Container, typename Func>
    inline auto image_filter::filter_kernel(Container &&input, size_t width, size_t height, int radius, Func &&func)
    {
        using namespace std;
        using T = ranges::range_value_t<Container>;

        auto input_begin = ranges::begin(input);
        std::vector<T> output{input_begin, ranges::end(input)};
        if(ranges::size(output) < static_cast<size_t>((2 * radius + 1) * 2) || radius < 1){
            return output;
        }
        auto input_view = input | views::chunk(width);
        auto input_view_begin = ranges::begin(input_view);

        // process rows
        auto end_row_it = ranges::prev(ranges::end(input_view), radius);
        for(auto row_it = ranges::next(input_view_begin, radius); row_it != end_row_it; ++row_it) {
            // distance from the current line to the beginning of the input view
            auto dist_col = ranges::distance(input_view_begin, row_it);
            auto row_begin = ranges::begin(*row_it);
            for(auto it = ranges::next(row_begin, radius); 
                it != ranges::prev(ranges::end(*row_it), radius); ++it) {
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
                auto output_it = ranges::next(ranges::begin(output), dist_col * width + dist_row);
                *output_it = func(kernel_lines);
            }
        }

        return output;
    }

}

#endif
