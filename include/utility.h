#pragma once
#ifndef __UTILITY_H__
#define __UTILITY_H__

#include <array>
#include <ranges>
#include <cassert>
#include <cstdint>
#include <utility>

namespace dsm::utility {

    // Calculate histogram of the input image data
    template<std::ranges::range Container>
    inline auto calculate_histogram(Container&& input)
    {
        static_assert(std::same_as<std::ranges::range_value_t<Container>, uint8_t>,
            "Input container's value type must be uint8_t.");
        
        std::array<size_t, std::numeric_limits<uint8_t>::max() + 1> histogram{};
        for(const auto& val : input){
            assert(std::in_range<uint8_t>(val));
            ++histogram[val];
        }

        return histogram;
    }
}


#endif