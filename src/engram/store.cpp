#include "dsv41/engram.hpp"
#include <bit>
#include <cmath>
#include <cstring>
#include <set>
#include <stdexcept>
#include <unistd.h>

namespace dsv41 {
namespace {
void check(bool ok, const char* msg) { if (!ok) throw std::runtime_error(msg); }
}
EngramStore::EngramStore(WeightCatalog& catalog, std::uint64_t layer, std::uint64_t rows, EngramReadMode mode)
    : rows_(rows), mode_(mode) {
    const auto prefix = "layers."+std::to_string(layer)+".engram.embed.";
    weight_ = catalog.tensor(prefix+"weight"); scale_ = catalog.tensor(prefix+"scale");
    check(weight_.dtype == "F8_E4M3" && scale_.dtype == "F8_E8M0" &&
          weight_.shape == std::vector<std::uint64_t>({rows,256}) &&
          scale_.shape == std::vector<std::uint64_t>({rows,8}),"Engram storage layout mismatch");
    if (mode == EngramReadMode::Mmap) { mapped_weight_ = weight_.map(); mapped_scale_ = scale_.map(); }
}
PackedEngramRows EngramStore::gather(std::span<const std::uint64_t> ids) const {
    check(ids.size() <= 4096,"Engram gather exceeds bounded staging capacity");
    for (auto row : ids) check(row < rows_,"Engram row out of bounds");
    PackedEngramRows result; result.values.resize(ids.size()*256); result.scales.resize(ids.size()*8);
    if (mode_ == EngramReadMode::Mmap) { mapped_weight_.check_unchanged(); mapped_scale_.check_unchanged(); }
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (mode_ == EngramReadMode::Mmap) {
            std::memcpy(result.values.data()+i*256,mapped_weight_.bytes(ids[i]*256,256).data(),256);
            std::memcpy(result.scales.data()+i*8,mapped_scale_.bytes(ids[i]*8,8).data(),8);
        } else {
            weight_.read(ids[i]*256,{reinterpret_cast<std::byte*>(result.values.data()+i*256),256});
            scale_.read(ids[i]*8,{reinterpret_cast<std::byte*>(result.scales.data()+i*8),8});
        }
    }
    if (mode_ == EngramReadMode::Mmap) { mapped_weight_.check_unchanged(); mapped_scale_.check_unchanged(); }
    return result;
}
std::vector<std::uint64_t> EngramStore::pages(std::span<const std::uint64_t> ids) const {
    check(weight_.shard == scale_.shard,"page telemetry requires weight and scale in same shard");
    const auto page = std::uint64_t(sysconf(_SC_PAGESIZE));
    std::set<std::uint64_t> result;
    for (auto id : ids) {
        check(id < rows_,"Engram row out of bounds");
        for (auto [offset,bytes] : {std::pair{weight_.file_offset+id*256,std::uint64_t(256)},
                                  std::pair{scale_.file_offset+id*8,std::uint64_t(8)}})
            for (auto p = offset/page; p <= (offset+bytes-1)/page; ++p) result.insert(p);
    }
    return {result.begin(),result.end()};
}
std::uint16_t engram_bf16(std::uint8_t code, std::uint8_t scale) {
    const auto exponent = (code>>3)&15, mantissa = code&7;
    if (scale == 255 || (exponent == 15 && mantissa == 7)) return 0x7fc0;
    float value = exponent ? std::ldexp(float(8+mantissa),int(exponent)-10) : std::ldexp(float(mantissa),-9);
    if (code&128) value = -value;
    value *= std::ldexp(1.0f,int(scale)-127);
    auto bits = std::bit_cast<std::uint32_t>(value);
    return std::uint16_t((bits+0x7fff+((bits>>16)&1))>>16);
}
std::vector<std::uint16_t> dequantize_engram(const PackedEngramRows& rows) {
    check(rows.values.size()%256 == 0 && rows.scales.size() == rows.values.size()/32,"bad packed Engram rows");
    std::vector<std::uint16_t> out(rows.values.size());
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = engram_bf16(rows.values[i],rows.scales[i/32]);
    return out;
}
}
