#include "dsv41/kv_quant.hpp"
#include "layer_kernels.hpp"
#include "kv_quant.hpp"
#include <stdexcept>
namespace dsv41 {
namespace mx=mlx::core;
PackedKVReference kv_quant_reference(const mx::array& input,KVQuantFormat format){
 if(format!=KVQuantFormat::IndexE8M0&&format!=KVQuantFormat::MainE4M3)throw std::runtime_error("unknown KV format");
 int width=format==KVQuantFormat::IndexE8M0?32:16,k=width==32?128:512;
 if(input.dtype()!=mx::bfloat16||input.ndim()!=2||input.shape(0)<1||input.shape(0)>128||input.shape(1)!=k)throw std::runtime_error("invalid KV quantization shape");
 int n=input.shape(0),groups=n*k/width;
 static auto kernel=mx::fast::metal_kernel("dsv41_kv_quant_reference",{"values","count","params"},{"packed","scales","decoded","errors"},dsv41_kv_quant_source,dsv41_fp8_header);
 auto out=kernel({input,mx::array({std::uint32_t(groups)}),mx::array({std::uint32_t(width)})},
  {{n,k/2},{n,k/width},{n,k},{groups}},{mx::uint8,mx::uint8,mx::uint16,mx::uint8},
  {groups,1,1},{64,1,1},{},std::nullopt,false,mx::Device::gpu);
 auto bad=mx::any(out[3]);mx::eval(bad);if(bad.item<bool>())throw std::runtime_error("nonfinite KV or unsupported scale overflow");
 return {out[0],out[1],mx::view(out[2],mx::bfloat16)};
}
}
