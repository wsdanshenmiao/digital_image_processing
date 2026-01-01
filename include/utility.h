#pragma once
#ifndef __UTILITY_H__
#define __UTILITY_H__

#include <array>
#include <ranges>
#include <cassert>
#include <cstdint>
#include <utility>

namespace dsm::utility {

    template <typename Container>
    concept uint8_range = std::ranges::range<Container> && std::same_as<std::ranges::range_value_t<Container>, uint8_t>;

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

    /// @brief Use mapping table encode data
    /// @tparam Container
    /// @param input a set of image data, the value type must be uint8_t
    /// @param mapping_table a mapping table that maps each uint8_t value to a pair of (code, length)
    ///                     the code must be stored in the lower bits of uint32_t
    /// @return encoded data
    template <uint8_range Container>
    inline std::vector<uint8_t> encode_data_width_mapping_table(Container&& input, const std::unordered_map<uint8_t, std::pair<uint32_t, uint8_t>>& mapping_table)
    {
        std::vector<uint8_t> output{};
        uint8_t current_byte = 0;
        uint8_t counter = 0;
        for(uint8_t byte : input){
            assert(mapping_table.contains(byte));
            auto [code, length] = mapping_table.at(byte);
            // write bits to output
            while (code > 0) {
                uint8_t remaining_len = 8 - counter;
                uint8_t need_len = std::min(remaining_len, length);
                current_byte <<= need_len;
                current_byte |= (code >> (length - need_len));
                counter += need_len;
                length -= need_len;
                code &= (1u << length) - 1;
                if(counter >= 8){
                    output.push_back(current_byte);
                    current_byte = 0;
                    counter = 0;
                }
            }
        }
        
        return output;
    }
}


#endif