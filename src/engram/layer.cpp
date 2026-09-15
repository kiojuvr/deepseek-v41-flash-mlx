#include "dsv41/engram_layer.hpp"
#include "layer_kernels.hpp"
#include <mlx/mlx.h>
#include <cmath>
#include <stdexcept>

namespace dsv41 {
namespace mx = mlx::core;
namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
mx::array resident(WeightCatalog& catalog, const std::string& name,
                   const std::vector<std::uint64_t>& shape, bool bf16) {
    auto tensor = catalog.tensor(name);
    const auto dtype = bf16 ? "BF16" : name.ends_with("scale") ? "F8_E8M0" : "F8_E4M3";
    check(tensor.shape == shape && tensor.dtype == dtype, "Engram resident tensor layout mismatch");
    mx::Shape dims; for (auto d : shape) dims.push_back(static_cast<int>(d));
    if (bf16) {
        std::vector<std::uint16_t> bytes(tensor.size_bytes / 2);
        tensor.read(0, {reinterpret_cast<std::byte*>(bytes.data()),tensor.size_bytes});
        return mx::view(mx::array(bytes.begin(),dims,mx::uint16),mx::bfloat16);
    }
    std::vector<std::uint8_t> bytes(tensor.size_bytes);
    tensor.read(0, {reinterpret_cast<std::byte*>(bytes.data()),bytes.size()});
    return mx::array(bytes.begin(),dims,mx::uint8);
}
void hidden_shape(const mx::array& x) {
    check(x.dtype() == mx::bfloat16 && x.ndim() == 3 && x.shape(0) >= 1 &&
          x.shape(0) <= 128 && x.shape(1) == 4 && x.shape(2) == 5120,
          "Engram reference requires BF16 [1..128,4,5120] (batch one)");
}
}
EngramActivation engram_activation_reference(const mx::array& values) {
    check(values.dtype() == mx::bfloat16 && values.ndim() == 2 && values.shape(0) >= 1 &&
          values.shape(0) <= 128 && values.shape(1) == 6144,"bad Engram activation shape/dtype");
    const int tokens = values.shape(0);
    static auto quant = mx::fast::metal_kernel("dsv41_engram_act_reference",{"values"},
        {"quantized","scales"},dsv41_activation_quant,dsv41_fp8_header);
    auto a = quant({values},{{tokens,6144},{tokens,192}},{mx::uint8,mx::uint8},
                   {tokens*192,1,1},{64,1,1},{},std::nullopt,false,mx::Device::gpu);
    return {a[0],a[1]};
}
EngramGateResult engram_gate_reference(const mx::array& hidden, const mx::array& kv,
    const mx::array& q, const mx::array& k, const std::optional<mx::array>& mask, float eps) {
    hidden_shape(hidden);
    const int tokens = hidden.shape(0);
    check(kv.dtype() == mx::bfloat16 && kv.shape() == mx::Shape({tokens,25600}), "bad Engram KV projection");
    check(q.dtype() == mx::bfloat16 && k.dtype() == mx::bfloat16 &&
          q.shape() == mx::Shape({4,5120}) && k.shape() == q.shape(), "bad Engram gate weights");
    check(std::isfinite(eps) && eps > 0, "invalid Engram norm epsilon");
    if (mask) check(mask->dtype() == mx::bool_ && mask->shape() == mx::Shape({tokens}), "bad Engram token mask");
    auto h = mx::astype(hidden,mx::float32);
    auto key = mx::reshape(mx::astype(mx::slice(kv,{0,0},{tokens,20480}),mx::float32),{tokens,4,5120});
    auto value = mx::astype(mx::slice(kv,{0,20480},{tokens,25600}),mx::float32);
    auto weight = mx::multiply(mx::astype(q,mx::float32),mx::astype(k,mx::float32));
    auto rstd = mx::multiply(mx::rsqrt(mx::add(mx::mean(mx::square(h),-1),mx::array(eps))),
                            mx::rsqrt(mx::add(mx::mean(mx::square(key),-1),mx::array(eps))));
    auto dot = mx::multiply(mx::multiply(mx::sum(mx::multiply(mx::multiply(h,weight),key),-1),rstd),
                            mx::array(float(std::pow(5120.0,-0.5))));
    auto magnitude = mx::sqrt(mx::maximum(mx::abs(dot),mx::array(1e-6f)));
    // Preserve the sign of zero too, as torch.copysign does.
    auto negative = mx::greater_equal(mx::view(dot,mx::uint32),mx::array(std::uint32_t(0x80000000)));
    auto gate = mx::sigmoid(mx::where(negative,mx::negative(magnitude),magnitude));
    if (mask) gate = mx::where(mx::expand_dims(*mask,-1),gate,mx::array(0.0f));
    auto output = mx::astype(mx::add(h,mx::multiply(mx::expand_dims(gate,-1),mx::expand_dims(value,-2))),mx::bfloat16);
    return {dot,gate,output};
}
struct EngramLayerReference::Impl {
    EngramStore store;
    mx::array weight, scale, q, k;
    float eps;
    Impl(WeightCatalog& catalog, std::uint64_t layer, std::uint64_t rows, float epsilon, EngramReadMode mode)
        : store(catalog,layer,rows,mode),
          weight(resident(catalog,"layers."+std::to_string(layer)+".engram.wkv.weight",{25600,6144},false)),
          scale(resident(catalog,"layers."+std::to_string(layer)+".engram.wkv.scale",{800,192},false)),
          q(resident(catalog,"layers."+std::to_string(layer)+".engram.q_weight",{4,5120},true)),
          k(resident(catalog,"layers."+std::to_string(layer)+".engram.k_weight",{4,5120},true)), eps(epsilon) {}
};
EngramLayerReference::EngramLayerReference(WeightCatalog& catalog, const EngramMetadata& m,
    std::size_t index, float eps, EngramReadMode mode) {
    check(m.revision == catalog.revision() && index < m.layer_ids.size() &&
          index < m.num_embeddings.size(), "Engram metadata/catalog mismatch");
    check(std::isfinite(eps) && eps > 0,"invalid Engram norm epsilon");
    impl_ = std::make_unique<Impl>(catalog,m.layer_ids[index],m.num_embeddings[index],eps,mode);
}
EngramLayerReference::~EngramLayerReference() = default;
EngramForwardResult EngramLayerReference::forward(const mx::array& h,
    std::span<const std::uint64_t> rows, const std::optional<mx::array>& mask) const {
    hidden_shape(h); const int tokens = h.shape(0);
    check(rows.size() == std::size_t(tokens)*24,"Engram rows/token mismatch");
    if (mask) check(mask->dtype() == mx::bool_ && mask->shape() == mx::Shape({tokens}),"bad Engram token mask");
    auto values = mx::reshape(engram_lookup_mlx(impl_->store.gather(rows)),{tokens,6144});
    auto a = engram_activation_reference(values);
    static auto project = mx::fast::metal_kernel("dsv41_engram_projection_reference",
        {"quantized","scales","weight","weight_scale"},{"output"},dsv41_projection,dsv41_fp8_header);
    auto projected = project({a.quantized,a.scales,impl_->weight,impl_->scale},{{tokens,25600}},{mx::uint16},
                             {25600,tokens,1},{128,1,1},{},std::nullopt,false,mx::Device::gpu);
    auto kv = mx::view(projected[0],mx::bfloat16);
    auto result = engram_gate_reference(h,kv,impl_->q,impl_->k,mask,impl_->eps);
    return {a.quantized,a.scales,kv,result.dot,result.gate,result.output};
}
}
