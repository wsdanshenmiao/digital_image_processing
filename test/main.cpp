#include "image_filter.h"

#include <print>
#include <ranges>
#include <vector>

int main() 
{
    auto vec = std::vector<int>{1, 2, 3, 4, 5};
    auto view = vec | std::views::transform([](int x) { return x * 2; });
    auto result = dsm::image_filter::domain_average(view);
    for(const auto& val : result) {
        std::println("{}", val);
    }
    return 0;
}