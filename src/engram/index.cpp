#include "dsv41/engram.hpp"
#include "dsv41/checkpoint_atlas.hpp"
#include <limits>
#include <set>
#include <stdexcept>

namespace dsv41 {
namespace {
void check(bool ok, const char* msg) { if (!ok) throw std::runtime_error(msg); }
std::uint64_t number(const nlohmann::json& v) {
    check(v.is_number_unsigned(),"invalid Engram unsigned integer"); return v.get<std::uint64_t>();
}
}
std::shared_ptr<const EngramMetadata> EngramMetadata::load(const std::filesystem::path& path) {
    const auto j = read_json_file(path);
    check(j.at("schema_version") == 1 && j.at("head_dim") == 256 && j.at("n_heads") == 8 &&
          j.at("max_ngram_size") == 4,"unsupported Engram metadata layout");
    auto m = std::make_shared<EngramMetadata>();
    m->identity = sha256_text(j.dump()); m->revision = j.at("revision");
    for (const auto& n : j.at("token_map")) m->token_map.push_back(number(n));
    for (const auto& n : j.at("layer_ids")) m->layer_ids.push_back(number(n));
    for (const auto& n : j.at("num_embeddings")) m->num_embeddings.push_back(number(n));
    m->pad_id = number(j.at("pad_id")); m->compressed_vocab_size = number(j.at("compressed_vocab_size"));
    check(!m->token_map.empty() && m->compressed_vocab_size > 0 && m->pad_id < m->compressed_vocab_size,"invalid compressed vocabulary");
    for (auto id : m->token_map) check(id < m->compressed_vocab_size,"token map out of bounds");
    check(m->layer_ids.size() == 2 && m->num_embeddings.size() == 2 &&
          j.at("primes").size() == 2 && j.at("offsets").size() == 2 && j.at("multipliers").size() == 2,
          "Engram requires two layers");
    check(m->layer_ids[0] == 1 && m->layer_ids[1] == 14,"unexpected Engram source layers");
    std::set<std::uint64_t> seen_primes;
    for (std::size_t l = 0; l < 2; ++l) {
        check(j.at("multipliers")[l].size() == 4 && j.at("primes")[l].size() == 3 &&
              j.at("offsets")[l].size() == 24,"bad Engram hash dimensions");
        std::array<std::uint64_t,4> multipliers{};
        for (std::size_t i = 0; i < 4; ++i) {
            const auto n = number(j.at("multipliers")[l][i]);
            check(n%2 == 1 && n <= std::uint64_t(std::numeric_limits<std::int64_t>::max())/m->compressed_vocab_size,
                  "hash product could overflow signed int64");
            multipliers[i] = n;
        }
        std::array<std::uint64_t,24> primes{}, offsets{};
        std::uint64_t offset = 0;
        for (std::size_t i = 0; i < 24; ++i) {
            check(j.at("primes")[l][i/8].size() == 8,"bad prime dimensions");
            const auto prime = number(j.at("primes")[l][i/8][i%8]);
            check(prime > 1 && seen_primes.insert(prime).second,"invalid/duplicate hash bucket modulus");
            check(number(j.at("offsets")[l][i]) == offset && offset <= m->num_embeddings[l] &&
                  prime <= m->num_embeddings[l]-offset,"invalid Engram bucket range");
            primes[i] = prime; offsets[i] = offset; offset += prime;
        }
        check(offset == m->num_embeddings[l],"bucket ranges do not cover Engram table");
        m->multipliers.push_back(multipliers); m->primes.push_back(primes); m->offsets.push_back(offsets);
    }
    return m;
}
EngramHashState::EngramHashState(std::shared_ptr<const EngramMetadata> m): metadata_(std::move(m)) {
    check(bool(metadata_),"missing Engram metadata"); reset();
}
void EngramHashState::reset() { position_ = 0; history_size_ = 0; previous_.fill(-1); }
std::vector<std::uint64_t> EngramHashState::append(std::span<const std::uint32_t> ids,
                                                std::span<const std::uint8_t> mask, std::uint64_t start) {
    check(start == position_,"non-contiguous Engram position");
    check(mask.empty() || mask.size() == ids.size(),"mask length mismatch");
    check(ids.size() <= 32768 && ids.size() <= std::numeric_limits<std::uint64_t>::max()-position_,"Engram chunk too large");
    for (auto id : ids) check(id < metadata_->token_map.size(),"token ID out of bounds");
    for (auto live : mask) check(live <= 1,"mask must contain 0 or 1");
    std::vector<std::uint64_t> result; result.reserve(ids.size()*48);
    for (std::size_t t = 0; t < ids.size(); ++t) {
        const auto current = mask.empty() || mask[t] ? std::int64_t(metadata_->token_map[ids[t]]) : -1;
        std::array<std::uint64_t,4> tokens{};
        bool blocked = false;
        for (std::size_t back = 0; back < 4; ++back) {
            const auto source = back == 0 ? current : back <= history_size_ ? previous_[back-1] : -1;
            blocked = blocked || source < 0;
            tokens[back] = blocked ? metadata_->pad_id : std::uint64_t(source);
        }
        for (std::size_t l = 0; l < 2; ++l) {
            auto rolling = tokens[0]*metadata_->multipliers[l][0];
            for (std::size_t n = 1; n < 4; ++n) {
                rolling ^= tokens[n]*metadata_->multipliers[l][n];
                for (std::size_t h = 0; h < 8; ++h) {
                    const auto col = (n-1)*8+h;
                    result.push_back(rolling%metadata_->primes[l][col]+metadata_->offsets[l][col]);
                }
            }
        }
        previous_[2] = previous_[1]; previous_[1] = previous_[0]; previous_[0] = current;
        history_size_ = std::min(std::size_t(3),history_size_+1); ++position_;
    }
    return result;
}
}
