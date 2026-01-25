#pragma once
#ifndef __SHANNON_FANO_CODING_H__
#define __SHANNON_FANO_CODING_H__


#include <vector>
#include <ranges>
#include <unordered_map>
#include <algorithm>
#include <cassert>
#include <print>
#include "utility.h"

namespace dsm::image_coding {

    class shannon_fano_coder
    {
    private:
        using mapping_table = std::array<std::pair<uint32_t, uint8_t>, 256>;

        /// @brief encoding tree node, left is 0, right is 1
        struct node
        {
            uint8_t value{};
            std::unique_ptr<node> left{};
            std::unique_ptr<node> right{};
        };

    public:
        template <utility::uint8_range Container>
        void encode(Container&& input)
        {
            auto mapping_table = generate_mapping_table_cdf_decimal(input);
            m_encoded_tree_root = build_encoding_tree(mapping_table);
            m_encoded_data = utility::encode_data_width_mapping_table(
                std::forward<Container>(input), mapping_table, m_encoded_tree_root->value);
        }

        std::vector<uint8_t> decode()
        {
            std::vector<uint8_t> output{};

            node* curr_node = m_encoded_tree_root.get();
            auto end_len = m_encoded_tree_root->value - 1;
            for(auto [index, data] : m_encoded_data | std::views::enumerate){
                int counter = (static_cast<size_t>(index) == m_encoded_data.size() - 1) ? end_len : 7;
                while (counter >= 0) {
                    assert(curr_node != nullptr);
                    curr_node = (data & (1 << counter)) ? curr_node->right.get() : curr_node->left.get();
                    --counter;
                    if (curr_node->left == nullptr && curr_node->right == nullptr) {
                        output.push_back(curr_node->value);
                        curr_node = m_encoded_tree_root.get();
                    }
                }
            }

            return output;
        }


        template <utility::uint8_range Container>
        void split_mid_encode(Container&& input)
        {
            auto mapping_table = generate_mapping_table_split_mid(input);
            m_encoded_tree_root = build_encoding_tree(mapping_table);
            m_encoded_data = utility::encode_data_width_mapping_table(
                std::forward<Container>(input), mapping_table, m_encoded_tree_root->value);
        }

        const std::vector<uint8_t>& get_encoded_data() const noexcept
        {
            return m_encoded_data;
        }



    private:
        template <utility::uint8_range Container>
        auto generate_frequency_table(Container&& input)
        {
            // calculate frequency
            std::array<std::pair<uint8_t, size_t>, 256> frequency_table{};
            auto size = std::ranges::size(input);
            if(size > 0){
                for(const auto& byte : input){
                    frequency_table[byte].first = byte;
                    ++frequency_table[byte].second;
                }
                std::ranges::sort(frequency_table, [](const auto& l, const auto& r){
                    return l.second > r.second;
                });
            }

            return frequency_table;
        }

        template <utility::uint8_range Container>
        mapping_table generate_mapping_table_cdf_decimal(Container&& input)
        {
            auto size = std::ranges::size(input);
            auto frequency_table = generate_frequency_table(std::forward<Container>(input));

            // calculate cdf and build mapping table
            std::array<float, 256> cdf{};
            float pre_prob = static_cast<float>(frequency_table[0].second) / size;
            mapping_table table{};
            for(auto i = 1; i < std::size(frequency_table); ++i){
                if(frequency_table[i].second == 0){
                    break;
                }
                uint8_t symbol = frequency_table[i].first;
                float prob = static_cast<float>(frequency_table[i].second) / size;
                cdf[i] = cdf[i - 1] + pre_prob;
                pre_prob = prob;
                uint8_t word_length = static_cast<uint8_t>(std::max(std::ceil(-std::log2(prob)), 1.f));
                table[symbol] = std::make_pair(decimal_to_binary(cdf[i], word_length), word_length);
            }
            auto first_world_length = static_cast<uint8_t>(std::max(std::ceil(-std::log2(static_cast<float>(frequency_table[0].second) / size)), 1.f));
            table[frequency_table[0].first] = {0, first_world_length};

            return table;
        }

        void assign_codes(
            std::span<std::pair<uint32_t, uint8_t>> freq_table, 
            std::span<std::pair<uint8_t, size_t>> prefix_sum, 
            size_t start_freq,
            uint32_t curr_code,
            uint8_t curr_length)
        {
            // end recursion
            if(freq_table.size() <= 1){
                // write value into table
                if(!freq_table.empty()){
                    freq_table[0] = {curr_code, curr_length};
                }
                return;
            }

            // search for the position that is greater or equal the median
            ptrdiff_t mid_val = start_freq + (prefix_sum.back().second - start_freq) / 2;
            auto mid_it = std::ranges::lower_bound(prefix_sum, std::pair{0, mid_val}, [](auto& l, auto& r) {
                return l.second < r.second;
            });
            // check which one is closer to the mid_val
            if(mid_it != prefix_sum.begin()){
                auto pre_it = std::prev(mid_it);
                if(std::abs(static_cast<ptrdiff_t>(mid_it->second) - mid_val) >= 
                   std::abs(static_cast<ptrdiff_t>(pre_it->second) - mid_val)) {
                    mid_it = pre_it;
                }
            }
            // verify each side is not empty
            if(mid_it == prefix_sum.end()){
                mid_it = std::prev(mid_it);
            }
            size_t mid_index = std::distance(prefix_sum.begin(), mid_it);

            assign_codes(freq_table.subspan(0, mid_index + 1), 
                prefix_sum.subspan(0, mid_index + 1), 
                start_freq,
                (curr_code << 1) | 1,
                curr_length + 1);
            assign_codes(freq_table.subspan(mid_index + 1), 
                prefix_sum.subspan(mid_index + 1), 
                prefix_sum[mid_index].second,
                (curr_code << 1) | 0,
                curr_length + 1);
        }

        template <utility::uint8_range Container>
        mapping_table generate_mapping_table_split_mid(Container&& input)
        {
            auto size = std::ranges::size(input);
            auto frequency_table = generate_frequency_table(std::forward<Container>(input));
            
            std::array<std::pair<uint8_t, size_t>, 256> prefix_sum_array{};
            auto end_index = prefix_sum_array.size() - 1;
            for(size_t index = 0; index < prefix_sum_array.size(); ++index){
                auto frequency = frequency_table[index].second;
                if(frequency == 0){
                    end_index = index == 0 ? index : index - 1;
                    break;
                }
                auto& [symbol, sum] = prefix_sum_array[index];
                symbol = frequency_table[index].first;
                sum = frequency + (index > 0 ? prefix_sum_array[index - 1].second : 0);
            }

            mapping_table ordered_table{};
            auto count = end_index + 1;
            assign_codes(std::span(ordered_table.begin(), count), std::span(prefix_sum_array.begin(), count), 0, 0, 0);
            
            mapping_table table{};
            for (size_t i = 0; i < ordered_table.size(); ++i) {
                table[frequency_table[i].first] = ordered_table[i];
            }

            return table;
        }

        std::unique_ptr<node> build_encoding_tree(const mapping_table& table)
        {
            auto root = std::make_unique<node>();
            for(const auto& [symbol, data] : table | std::views::enumerate){
                auto [code, len] = data;
                if(len <= 0){
                    continue;
                }
                node* curr_node = root.get();
                for(uint8_t i = 0; i < len; ++i){
                    auto& tmp_node = (code >> (len - i - 1)) & 1 ? curr_node->right : curr_node->left;
                    if(tmp_node == nullptr){
                        tmp_node = std::make_unique<node>();
                    }
                    curr_node = tmp_node.get();
                }
                curr_node->value = symbol;
            }

            return root;
        }

        uint32_t decimal_to_binary(float decimal, uint8_t length)
        {
            length = std::min(length, static_cast<uint8_t>(sizeof(uint32_t) * 8));
            uint32_t binary = 0;
            for(uint8_t i = 0; i < length; ++i){
                decimal *= 2;
                if(decimal >= 1.0f){
                    binary |= (1 << (length - i - 1));
                    decimal -= 1.0f;
                }
            }
            return binary;
        }

        

    private:
        std::vector<uint8_t> m_encoded_data{};
        std::unique_ptr<node> m_encoded_tree_root;
    };

}

#endif // __SHANNON_FANO_CODING_H__