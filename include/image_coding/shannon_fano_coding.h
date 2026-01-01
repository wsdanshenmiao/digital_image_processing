#pragma once
#ifndef __SHANNON_FANO_CODING_H__
#define __SHANNON_FANO_CODING_H__


#include <vector>
#include <ranges>
#include <unordered_map>
#include <algorithm>
#include <cassert>
#include "utility.h"

namespace dsm::image_coding {

    class shannon_fano_coder
    {
    public:
        template <utility::uint8_range Container>
        std::vector<uint8_t> encode(Container&& input)
        {
            m_mapping_table = generate_mapping_table(input);
            if(m_mapping_table.empty()){
                return {};
            }

            return utility::encode_data_width_mapping_table(std::forward<Container>(input), m_mapping_table);
        }

        std::vector<uint8_t> decode()
        {
            std::vector<uint8_t> output{};

            return output;
        }


    private:
        template <utility::uint8_range Container>
        auto generate_mapping_table(Container&& input)
        {
            std::unordered_map<uint8_t, std::pair<uint32_t, uint8_t>> mapping_table{};
            auto size = std::ranges::size(input);
            if(size <= 0){
                return mapping_table;
            }

            // calculate frequency
            std::unordered_map<uint8_t, size_t> frequency_table{};
            for(const auto& byte : input){
                if(auto it = frequency_table.find(byte); it != frequency_table.end()){
                    it->second++;
                }
                else{
                    frequency_table[byte] = 1;
                }
            }

            // calculate word lengths
            std::vector<std::pair<uint8_t, float>> pdf{};
            std::array<uint8_t, 256> word_length{};
            for(const auto& [symbol, freq] : frequency_table){
                float prob = static_cast<float>(freq) / size;
                pdf.emplace_back(symbol, prob);
                word_length[symbol] = static_cast<uint8_t>(std::ceil(-std::log2(prob)));
            }
            assert(pdf.size() > 0);

            std::ranges::sort(pdf, [](const auto& a, const auto& b){
                return a.second > b.second;
            });

            // calculate cdf and build mapping table
            std::array<float, 256> cdf{};
            for(auto i = 1; i < std::size(pdf); ++i){
                auto [pre_symbol, pre_prob] = pdf[i - 1];
                auto [symbol, prob] = pdf[i];
                cdf[i] = cdf[i - 1] + pre_prob;
                mapping_table[symbol] = std::make_pair(
                    decimal_to_binary(cdf[i], word_length[symbol]),
                    word_length[symbol] );
            }
            mapping_table[pdf[0].first] = {0, 1};
            return mapping_table;
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
        std::unordered_map<uint8_t, std::pair<uint32_t, uint8_t>> m_mapping_table{};
        std::vector<uint8_t> m_encoded_data{};
    };

}

#endif // __SHANNON_FANO_CODING_H__