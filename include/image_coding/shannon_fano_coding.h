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
            auto mapping_table = generate_mapping_table(input);
            m_encoded_tree_root = build_encoding_tree(mapping_table);
            m_encoded_data = utility::encode_data_width_mapping_table(std::forward<Container>(input), mapping_table, m_encoded_tree_root->value);
        }

        std::vector<uint8_t> decode()
        {
            std::vector<uint8_t> output{};

            node* curr_node = m_encoded_tree_root.get();
            auto end_len = m_encoded_tree_root->value - 1;
            for(auto [index, data] : m_encoded_data | std::views::enumerate){
                int counter = (index == m_encoded_data.size() - 1) ? end_len : 7;
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


    private:
        template <utility::uint8_range Container>
        auto generate_mapping_table(Container&& input)
        {
            mapping_table table{};
            auto size = std::ranges::size(input);
            if(size <= 0){
                return table;
            }

            // calculate frequency
            std::array<std::pair<uint8_t, size_t>, 256> frequency_table{};
            for(const auto& byte : input){
                frequency_table[byte].first = byte;
                ++frequency_table[byte].second;
            }
            std::ranges::sort(frequency_table, [](const auto& l, const auto& r){
                return l.second > r.second;
            });

            // calculate word lengths
            std::vector<std::pair<uint8_t, float>> pdf{};
            std::array<uint8_t, 256> word_length{};
            for(const auto& [symbol, freq] : frequency_table){
                if(freq == 0){
                    break;
                }
                float prob = static_cast<float>(freq) / size;
                pdf.emplace_back(symbol, prob);
                word_length[symbol] = static_cast<uint8_t>(std::max(std::ceil(-std::log2(prob)), 1.f));
            }

            // calculate cdf and build mapping table
            std::array<float, 256> cdf{};
            for(auto i = 1; i < std::size(pdf); ++i){
                auto [pre_symbol, pre_prob] = pdf[i - 1];
                auto [symbol, prob] = pdf[i];
                cdf[i] = cdf[i - 1] + pre_prob;
                table[symbol] = std::make_pair(
                    decimal_to_binary(cdf[i], word_length[symbol]),
                    word_length[symbol] );
            }
            table[pdf[0].first] = {0, word_length[pdf[0].first]};
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