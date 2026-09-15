#pragma once
#include <array>
#include <cstdint>
#include <span>
#include <vector>
namespace dsv41 {
// Per-request reference state: already quantized/dequantized BF16 latent rows.
// No global KV, no complete SWA history, no disk backing.
class SwaReferenceState {
public:
 static constexpr std::size_t window=128,dim=512;
 void reset();
 void append(std::span<const std::uint16_t> rows,std::uint64_t start_position);
 std::uint64_t position() const{return position_;}
 std::vector<std::uint16_t> chronological_rows() const;
 std::vector<std::int32_t> official_decode_indices() const;
 std::span<const std::uint16_t> physical_rows() const{return ring_;}
private:
 std::array<std::uint16_t,window*dim> ring_{};
 std::uint64_t position_=0;
};
}
