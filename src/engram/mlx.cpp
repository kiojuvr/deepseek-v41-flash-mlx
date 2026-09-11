#include "dsv41/engram_mlx.hpp"
#include "engram_kernel_source.hpp"
#include <mlx/fast.h>
#include <mlx/ops.h>
#include <stdexcept>

namespace dsv41 {
mlx::core::array engram_lookup_mlx(const PackedEngramRows& rows) {
    namespace mx = mlx::core;
    if (rows.values.empty() || rows.values.size()%256 || rows.values.size() > 4096*256 ||
        rows.scales.size() != rows.values.size()/32)
        throw std::runtime_error("invalid/beyond-capacity Engram Metal input");
    const int count = static_cast<int>(rows.values.size()), n = count/256;
    mx::array values(rows.values.begin(),{n,256},mx::uint8);
    mx::array scales(rows.scales.begin(),{n,8},mx::uint8);
    static auto kernel = mx::fast::metal_kernel("dsv41_engram_reference",{"values","scales"},{"output"},
                                               dsv41_engram_kernel_source);
    auto outputs = kernel({values,scales},{{n,256}},{mx::uint16},{count,1,1},{256,1,1},{},std::nullopt,false,mx::Device::gpu);
    return mx::view(outputs.at(0),mx::bfloat16,mx::Device::gpu);
}
}
