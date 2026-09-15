#include "dsv41/model_entry.hpp"
#include <cmath>
#include <stdexcept>

namespace dsv41 {
namespace mx = mlx::core;
namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
mx::array load_embedding(WeightCatalog& catalog) {
    auto tensor = catalog.tensor("embed.weight");
    check(tensor.dtype == "BF16" && tensor.shape == std::vector<std::uint64_t>({129280,5120}) &&
          tensor.size_bytes == 129280ull*5120*2, "unexpected text embedding layout");
    std::vector<std::uint16_t> data(tensor.size_bytes/2);
    tensor.read(0,{reinterpret_cast<std::byte*>(data.data()),tensor.size_bytes});
    return mx::view(mx::array(data.begin(),{129280,5120},mx::uint16),mx::bfloat16);
}
}
mx::array rms_norm_reference(const mx::array& input, const mx::array& weight, float eps) {
    check(input.dtype() == mx::bfloat16 && input.ndim() >= 2 && input.size() > 0,
          "RMSNorm reference requires nonempty BF16 input with rank >= 2");
    check(weight.dtype() == mx::bfloat16 && weight.ndim() == 1 &&
          weight.shape(0) == input.shape(-1), "RMSNorm weight mismatch");
    check(std::isfinite(eps) && eps > 0, "invalid RMSNorm epsilon");
    auto h = mx::astype(input,mx::float32);
    auto var = mx::mean(mx::square(h),-1,true);
    // Match the official torch.rsqrt: MLX rsqrt differs from 1/sqrt at 1 ULP on some elements.
    auto inv = mx::divide(mx::array(1.0f),mx::sqrt(mx::add(var,mx::array(eps))));
    auto normalized = mx::multiply(h,inv);
    return mx::astype(mx::multiply(mx::astype(weight,mx::float32),normalized),mx::bfloat16);
}
TextEntryReference::TextEntryReference(WeightCatalog& catalog) : embedding_(load_embedding(catalog)) {}
TextEntryResult TextEntryReference::forward(std::span<const std::uint32_t> ids) const {
    check(!ids.empty() && ids.size() <= 128,"text entry reference requires 1..128 tokens");
    for (auto id : ids) {
        check(id < vocab_size,"token ID outside vocabulary");
        check(id != 129264,"image token requires the unimplemented image entry path");
    }
    const int n = static_cast<int>(ids.size());
    auto indices = mx::array(ids.begin(),{n},mx::uint32);
    auto embedded = mx::take(embedding_,indices,0);
    auto hidden = mx::broadcast_to(mx::expand_dims(embedded,1),{n,hc_mult,dim});
    auto pre_mix = mx::broadcast_to(mx::array({1.0f,0.0f,0.0f,0.0f}),{n,hc_mult});
    return {hidden,pre_mix};
}
}
