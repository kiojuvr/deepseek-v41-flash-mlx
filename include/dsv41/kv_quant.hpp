#pragma once
#include <mlx/mlx.h>
namespace dsv41 {
enum class KVQuantFormat { IndexE8M0, MainE4M3 };
struct PackedKVReference { mlx::core::array packed,scales,decoded; };
// BF16 [1..128,K], K=128 for index or 512 for main KV.
// Reference packing: earlier element in low nibble; scale byte per group.
PackedKVReference kv_quant_reference(const mlx::core::array& input,KVQuantFormat format);
}
