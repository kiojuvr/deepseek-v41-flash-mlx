#include "dsv41/mhc.hpp"
#include "dsv41/execution_policy.hpp"
#include "split_sinkhorn.hpp"
#include <stdexcept>
namespace dsv41 {
namespace mx=mlx::core;
namespace {
std::size_t& fused_mhc_counter(){static std::size_t value=0;return value;}
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void hidden_shape(const mx::array& h){check(h.dtype()==mx::bfloat16&&h.ndim()==3&&h.shape(0)>0&&h.shape(0)<=128&&h.shape(1)==4&&h.shape(2)==5120,"mHC requires BF16 [1..128,4,5120]");}
mx::array load(WeightCatalog& catalog,const std::string& name,const mx::Shape& shape){
 auto t=catalog.tensor(name);std::vector<std::uint64_t> expected(shape.begin(),shape.end());
 check(t.dtype=="F32"&&t.shape==expected,"mHC checkpoint layout mismatch");
 std::vector<float> values(t.size_bytes/4);t.read(0,{reinterpret_cast<std::byte*>(values.data()),t.size_bytes});
 return mx::array(values.begin(),shape,mx::float32);
}
}
std::size_t fused_mhc_invocation_count(){return fused_mhc_counter();}
void reset_fused_mhc_invocation_count(){fused_mhc_counter()=0;}
HCReference::HCReference(WeightCatalog& catalog,int layer,const std::string& kind)
 :fn_(mx::array(0)),scale_(mx::array(0)),base_(mx::array(0)) {
 check(layer>=0&&layer<40&&(kind=="attn"||kind=="ffn"),"invalid mHC owner");
 auto prefix="layers."+std::to_string(layer)+".hc_"+kind;
 fn_=load(catalog,prefix+"_fn",{24,20480});scale_=load(catalog,prefix+"_scale",{3});base_=load(catalog,prefix+"_base",{24});
}
HCMixes HCReference::mixes(const mx::array& hidden) const {
 hidden_shape(hidden);const int n=hidden.shape(0);
 auto flat=mx::reshape(mx::astype(hidden,mx::float32),{n,20480});
 std::vector<mx::array> rows;
 for(int i=0;i<n;++i) rows.push_back(mx::matmul(mx::slice(flat,{i,0},{i+1,20480}),mx::transpose(fn_)));
 auto projected=rows.size()==1?rows[0]:mx::concatenate(rows,0);
 auto normalized=mx::multiply(projected,mx::divide(mx::array(1.0f),mx::sqrt(mx::add(mx::mean(mx::square(flat),-1,true),mx::array(1e-20f)))));
 if(runtime_fused_mhc_enabled()){
  ++fused_mhc_counter();
  static auto kernel=mx::fast::metal_kernel("dsv41_mhc_split_sinkhorn",
   {"normalized","scale","base","count"},{"pre","post","comb"},
   dsv41_mhc_split_sinkhorn_source);
  auto outputs=kernel({normalized,scale_,base_,mx::array({std::uint32_t(n)})},
   {{n,4},{n,4},{n,4,4}},{mx::float32,mx::float32,mx::float32},
   {n,1,1},{32,1,1},{},std::nullopt,false,mx::Device::gpu);
  return {outputs[0],outputs[1],outputs[2]};
 }
 auto segment=[&](int first,int last,int scale_index){
  return mx::add(mx::multiply(mx::slice(normalized,{0,first},{n,last}),mx::reshape(mx::slice(scale_,{scale_index},{scale_index+1}),{})),mx::slice(base_,{first},{last}));
 };
 auto pre=mx::add(mx::sigmoid(segment(0,4,0)),mx::array(1e-6f));
 auto post=mx::multiply(mx::sigmoid(segment(4,8,1)),mx::array(2.0f));
 auto comb=mx::reshape(segment(8,24,2),{n,4,4});
 // Official initial softmax + eps, column normalization, then 19 row/column pairs.
 auto exponent=mx::exp(mx::subtract(comb,mx::max(comb,-1,true)));
 comb=mx::add(mx::divide(exponent,mx::sum(exponent,-1,true)),mx::array(1e-6f));
 comb=mx::divide(comb,mx::add(mx::sum(comb,-2,true),mx::array(1e-6f)));
 for(int i=0;i<19;++i){
  comb=mx::divide(comb,mx::add(mx::sum(comb,-1,true),mx::array(1e-6f)));
  comb=mx::divide(comb,mx::add(mx::sum(comb,-2,true),mx::array(1e-6f)));
 }
 return {pre,post,comb};
}
mx::array hc_pre_reference(const mx::array& h,const mx::array& pre){
 hidden_shape(h);check(pre.dtype()==mx::float32&&pre.shape()==mx::Shape({h.shape(0),4}),"invalid pre-mix");
 return mx::astype(mx::sum(mx::multiply(mx::expand_dims(pre,-1),mx::astype(h,mx::float32)),1),mx::bfloat16);
}
mx::array hc_post_reference(const mx::array& x,const mx::array& residual,const HCMixes& m){
 hidden_shape(residual);const int n=residual.shape(0);
 check(x.dtype()==mx::bfloat16&&x.shape()==mx::Shape({n,5120})&&m.post.dtype()==mx::float32&&m.post.shape()==mx::Shape({n,4})&&m.comb.dtype()==mx::float32&&m.comb.shape()==mx::Shape({n,4,4}),"invalid mHC post inputs");
 // comb[source copy,destination copy]; reduce source axis, not destination axis.
 auto expanded=mx::multiply(mx::expand_dims(m.post,-1),mx::expand_dims(mx::astype(x,mx::float32),1));
 auto combined=mx::sum(mx::multiply(mx::expand_dims(m.comb,-1),mx::expand_dims(mx::astype(residual,mx::float32),2)),1);
 return mx::astype(mx::add(expanded,combined),mx::bfloat16);
}
}
