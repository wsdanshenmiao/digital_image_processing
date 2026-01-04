#pragma once
#ifndef __HUFFMAN_CODING_H__
#define __HUFFMAN_CODING_H__

#include <ranges>
#include <vector>
#include <memory>
#include <queue>
#include <stack>
#include <algorithm>
#include <unordered_map>

#include "utility.h"

namespace dsm::image_coding {

    class huffman_coder
    {
    private:
        struct node
        {
            // symbol value, is not one symbol, root value is padding count of last byte
            uint8_t value{};
            size_t frequency{};
            std::unique_ptr<node> left{};
            std::unique_ptr<node> right{};

            node(uint8_t val, size_t freq) : value(val), frequency(freq), left(nullptr), right(nullptr) {}
            node(uint8_t val, size_t freq, std::unique_ptr<node> l, std::unique_ptr<node> r)
                : value(val), frequency(freq), left(std::move(l)), right(std::move(r)) {}
        };

    public:
        const std::vector<uint8_t>& get_encoded_data() const noexcept { return m_encoded_bits; }

        template <utility::uint8_range Container>
        void encode(Container&& input)
        {
            if(std::ranges::empty(input)) {
                return;
            }

            auto histogram = dsm::utility::calculate_histogram(input);
            m_huffman_tree_root = build_huffman_tree(histogram);
            auto mapping_table = generate_mapping_table(m_huffman_tree_root.get());

            int counter = 7;
            uint8_t current_byte = 0;
            m_encoded_bits.reserve(std::ranges::size(input));
            // encode input data
            for(const auto& data : input){
                const auto& bits = mapping_table.at(data);
                for(const auto& bit : bits){
                    current_byte |= (bit ? 1 : 0) << counter;
                    if (--counter < 0) {
                        m_encoded_bits.push_back(current_byte);
                        counter = 7;
                        current_byte = 0;
                    }
                }
            }
            if (counter != 7) {
                m_encoded_bits.push_back(current_byte);
                m_huffman_tree_root->value = static_cast<uint8_t>(counter);
            }
            else {
                m_huffman_tree_root->value = std::numeric_limits<uint8_t>::max();
            }
        }

        std::vector<uint8_t> decode()
        {
            if(m_huffman_tree_root == nullptr){
                return {};
            }

            std::vector<uint8_t> decoded_data;
            decoded_data.reserve(m_encoded_bits.size());
            const node* current_node = m_huffman_tree_root.get();

            ptrdiff_t end_index = m_encoded_bits.size() - 1;
            uint8_t root_val = m_huffman_tree_root->value;
            for(const auto& [index, byte] : m_encoded_bits | std::views::enumerate) {
                int counter = 7;
                uint8_t end_count = (index == end_index) ? (root_val + 1) : 0;
                while(counter >= end_count) {
                    bool bit = (byte >> counter--) & 1;
                    current_node = bit ? current_node->right.get() : current_node->left.get();

                    // reach leaf node
                    if(current_node->left == nullptr && current_node->right == nullptr){
                        // decode data
                        decoded_data.push_back(current_node->value);
                        // reset to root
                        current_node = m_huffman_tree_root.get();
                    }
                }
            }

            return decoded_data;
        }

    private:
        std::unique_ptr<node> build_huffman_tree(std::span<const size_t> histogram)
        {
            if(histogram.empty()) {
                return nullptr;
            }

            auto compare = [](const node* a, const node* b) {
                return a->frequency > b->frequency;
            };
            std::priority_queue<node*, std::vector<node*>, decltype(compare)> min_heap{compare};
            // Build leaf nodes and push into min-heap
            for(const auto& [index, freq] : histogram | std::views::enumerate) {
                if (freq > 0) {
                    min_heap.emplace(new node(index, freq));
                }
            }
            assert(!min_heap.empty());
            // only one unique symbol
            if (min_heap.size() == 1) {
                auto n = std::unique_ptr<node>(min_heap.top());
                min_heap.pop();
                return std::make_unique<node>(0, 0, nullptr, std::move(n));
            }

            while (min_heap.size() > 1) {
                auto left = std::unique_ptr<node>(min_heap.top());
                min_heap.pop();
                auto right = std::unique_ptr<node>(min_heap.top());
                min_heap.pop();
                // ensure left has smaller frequency, left edge is 0, right edge is 1
                if(compare(left.get(), right.get())) {
                    std::swap(left, right);
                }

                // sort by frequency
                auto sum_freq = left->frequency + right->frequency;
                auto merged_node = new node(0, sum_freq, std::move(left), std::move(right));
                min_heap.emplace(merged_node);
            }

            return std::unique_ptr<node>(min_heap.top());
        }

        auto generate_mapping_table(const node* root)
        {
            assert(root != nullptr);

            std::array<std::vector<bool>, sm_symbol_count> mapping_table;
            std::stack<std::pair<const node*, std::vector<bool>>> node_stack;

            bool is_unique_symbol = root->left == nullptr && root->right == nullptr;
            // push root's children to stack
            // if only one unique symbol, assign it a code of '0'
            node_stack.emplace(root, is_unique_symbol ? std::vector<bool>{false} : std::vector<bool>{});

            while (!node_stack.empty()) {
                auto [node, current_code] = std::move(node_stack.top());
                node_stack.pop();

                // node is leaf, write bits
                if(node->left == nullptr && node->right == nullptr) {
                    mapping_table[node->value] = current_code;
                }
                else{
                    if (node->left != nullptr) {
                        // left edge is 0
                        auto left_code = current_code;
                        left_code.push_back(false);
                        node_stack.emplace(node->left.get(), std::move(left_code));
                    }
                    if (node->right != nullptr) {
                        // right edge is 1
                        auto right_code = current_code;
                        right_code.push_back(true);
                        node_stack.emplace(node->right.get(), std::move(right_code));
                    }
                }
            }

            return mapping_table;
        }

    private:
        inline static constexpr size_t sm_symbol_count = std::numeric_limits<uint8_t>::max() + 1;

        std::vector<uint8_t> m_encoded_bits;
        std::unique_ptr<node> m_huffman_tree_root;
    };

} // namespace dsm::image_coding


#endif