#pragma once
#include "dsv41/weights.hpp"
#include <array>
#include <memory>
#include <span>
#include <vector>

namespace dsv41 {
struct EngramMetadata {
    std::string revision, identity;
    std::vector<std::uint64_t> token_map, layer_ids, num_embeddings;
    std::vector<std::array<std::uint64_t,4>> multipliers;
    std::vector<std::array<std::uint64_t,24>> primes, offsets;
    std::uint64_t pad_id = 0, compressed_vocab_size = 0;
    static std::shared_ptr<const EngramMetadata> load(const std::filesystem::path& path);
};
class EngramHashState {
public:
    explicit EngramHashState(std::shared_ptr<const EngramMetadata> metadata);
    // Batch size one. Masked tokens and all earlier lookbacks are replaced by pad.
    std::vector<std::uint64_t> append(std::span<const std::uint32_t> ids,
                                    std::span<const std::uint8_t> mask, std::uint64_t start_position);
    void reset();
    std::uint64_t position() const { return position_; }
private:
    std::shared_ptr<const EngramMetadata> metadata_;
    std::array<std::int64_t,3> previous_{};
    std::size_t history_size_ = 0;
    std::uint64_t position_ = 0;
};
enum class EngramReadMode { Mmap, Pread };
struct PackedEngramRows {
    std::vector<std::uint8_t> values; // [rows,256]
    std::vector<std::uint8_t> scales; // [rows,8]
};
class EngramStore {
public:
    EngramStore(WeightCatalog& catalog, std::uint64_t layer_id, std::uint64_t rows, EngramReadMode mode);
    PackedEngramRows gather(std::span<const std::uint64_t> row_ids) const;
    std::vector<std::uint64_t> pages(std::span<const std::uint64_t> row_ids) const;
    std::uint64_t backing_bytes() const { return weight_.size_bytes+scale_.size_bytes; }
private:
    TensorFile weight_, scale_;
    MappedTensor mapped_weight_, mapped_scale_;
    std::uint64_t rows_;
    EngramReadMode mode_;
};
std::uint16_t engram_bf16(std::uint8_t value, std::uint8_t scale);
std::vector<std::uint16_t> dequantize_engram(const PackedEngramRows& rows);
}
