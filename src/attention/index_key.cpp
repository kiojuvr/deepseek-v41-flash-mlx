#include "dsv41/index_key.hpp"
#include "dsv41/model_entry.hpp"
#include "dsv41/layer_owner.hpp"
#include <stdexcept>
namespace dsv41 {
namespace mx=mlx::core;
namespace {
mx::array load(WeightCatalog& c,const std::string& name,mx::Shape shape){
 auto t=c.tensor(name);std::vector<std::uint64_t> dims(shape.begin(),shape.end());
 std::size_t n=1;for(auto d:shape)n*=d;
 if(t.dtype!="BF16"||t.shape!=dims||t.size_bytes!=n*2)throw std::runtime_error("invalid index key weight");
 std::vector<std::uint16_t> v(n);t.read(0,{reinterpret_cast<std::byte*>(v.data()),t.size_bytes});
 return mx::view(mx::array(v.begin(),shape,mx::uint16),mx::bfloat16);
}
}
IndexKeyReference::IndexKeyReference(WeightCatalog& c,int layer)
 :weight_(load(c,("layers."+std::to_string(layer))+".attn.indexer.wk.weight",{128,512})),
  norm_(load(c,("layers."+std::to_string(layer))+".attn.indexer.k_norm.weight",{128})){
 if(!is_kv_source_layer(layer))throw std::runtime_error("index key owner requires a kv_source layer");
}
mx::array IndexKeyReference::before_quantization(const mx::array& h,std::span<const std::uint64_t> positions) const{
 if(h.dtype()!=mx::bfloat16||h.ndim()!=2||h.shape(1)!=512||h.shape(0)<1||h.shape(0)>128||positions.size()!=std::size_t(h.shape(0)))throw std::runtime_error("invalid index latent");
 for(auto p:positions)if(p>=1048576)throw std::runtime_error("invalid index key position");
 // Keep the reference's one-row matmul geometry (MLX may select a different
 // reduction schedule for a multi-row lhs), then materialize the full chunk
 // before normalization/RoPE and the caller's quantization commit.
 std::vector<mx::array> rows;rows.reserve(h.shape(0));
 for(int i=0;i<h.shape(0);++i)
  rows.push_back(mx::matmul(mx::slice(h,{i,0},{i+1,512}),mx::transpose(weight_)));
 auto key=rms_norm_reference(rows.size()==1?rows.front():mx::concatenate(rows,0),norm_,1e-20f);
 return compressed_rope_reference(key,positions);
}
}
