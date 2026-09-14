#include "dsv41/moe.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include "dsv41/layer_owner.hpp"
namespace dsv41 {
namespace mx=mlx::core;
namespace {
std::size_t& tie_counter(){static std::size_t count=0;return count;}
std::vector<RouteTieRecord>& tie_records(){static std::vector<RouteTieRecord> records;return records;}
std::uint64_t& current_token(){static std::uint64_t token=0;return token;}
}
std::size_t route_tie_count(){return tie_counter();}
void reset_route_tie_count(){tie_counter()=0;}
std::vector<RouteTieRecord> route_tie_records(){return tie_records();}
void reset_route_tie_records(){tie_records().clear();}
void set_route_trace_token(std::uint64_t token){current_token()=token;}
std::uint64_t route_trace_token(){return current_token();}
namespace {
std::string prefix(int expert,int layer){
 if(expert < -1||expert>=384)throw std::runtime_error("invalid expert ID");
 return reference_moe_prefix(layer)+".ffn."+(expert==-1?std::string("shared_experts"):"experts."+std::to_string(expert));
}
mx::array load(WeightCatalog& c,const std::string& name,mx::Shape shape,bool bf){
 auto t=c.tensor(name);std::vector<std::uint64_t> expected(shape.begin(),shape.end());
 std::size_t count=1;for(auto d:shape)count*=d;
 if(t.shape!=expected||t.dtype!=(bf?"BF16":"F32")||t.size_bytes!=count*(bf?2:4))throw std::runtime_error("unexpected gate weight");
 if(bf){std::vector<std::uint16_t> v(count);t.read(0,{reinterpret_cast<std::byte*>(v.data()),t.size_bytes});return mx::view(mx::array(v.begin(),shape,mx::uint16),mx::bfloat16);}
 std::vector<float> v(count);t.read(0,{reinterpret_cast<std::byte*>(v.data()),t.size_bytes});return mx::array(v.begin(),shape,mx::float32);
}
}
RouteReference select_routes_reference(const mx::array& scores,const mx::array& bias,bool strict,RouteTieRecord* tie){
 if(scores.dtype()!=mx::float32||scores.shape()!=mx::Shape({384})||bias.dtype()!=mx::float32||bias.shape()!=mx::Shape({384}))throw std::runtime_error("invalid route scores");
 auto corrected=mx::add(scores,bias,mx::Device::cpu);mx::eval(corrected);
 auto finite=mx::all(mx::logical_and(mx::isfinite(scores),mx::greater_equal(scores,mx::array(0.0f))));mx::eval(finite);
 if(!finite.item<bool>())throw std::runtime_error("invalid uncorrected scores");
 const float* p=corrected.data<float>();for(int i=0;i<384;++i)if(!std::isfinite(p[i]))throw std::runtime_error("nonfinite route score");
 std::array<int,384> order;std::iota(order.begin(),order.end(),0);
 std::sort(order.begin(),order.end(),[&](int a,int b){return p[a]!=p[b]?p[a]>p[b]:a<b;});
 bool tied=p[order[5]]==p[order[6]];
 if(tie){tie->sixth_id=order[5];tie->seventh_id=order[6];tie->sixth_score=p[order[5]];tie->seventh_score=p[order[6]];tie->tied=tied;}
 if(tied){
  if(strict)throw std::runtime_error("ambiguous top-6 boundary requires official tie oracle");
  ++tie_counter();
 }
 std::array<int,6> ids;std::copy_n(order.begin(),6,ids.begin());
 auto weights=mx::take(scores,mx::array(ids.begin(),{6},mx::int32));
 weights=mx::multiply(mx::divide(weights,mx::add(mx::sum(weights),mx::array(1e-20f))),mx::array(1.5f));
 return {ids,weights};
}
GateReference::GateReference(WeightCatalog& c,int layer):weight_(load(c,reference_moe_prefix(layer)+".ffn.gate.weight",{384,5120},true)),bias_(load(c,reference_moe_prefix(layer)+".ffn.gate.bias",{384},false)){}
RouteReference GateReference::forward(const mx::array& x,RouteTieRecord* tie) const{
 if(x.dtype()!=mx::bfloat16||x.shape()!=mx::Shape({1,5120}))throw std::runtime_error("gate requires one BF16 token");
 auto z=mx::reshape(mx::matmul(mx::astype(x,mx::float32),mx::transpose(mx::astype(weight_,mx::float32))),{384});
 // PyTorch softplus beta=1, threshold=20. gate_temp=1 for this checkpoint.
 auto scores=mx::sqrt(mx::where(mx::greater(z,mx::array(20.0f)),z,mx::log1p(mx::exp(z))));
 return select_routes_reference(scores,bias_,false,tie);
}
GateDiagnostic GateReference::diagnose(const mx::array& x,RouteTieRecord* tie) const {
 if(x.dtype()!=mx::bfloat16||x.shape()!=mx::Shape({1,5120})) throw std::runtime_error("gate requires one BF16 token");
 auto z=mx::reshape(mx::matmul(mx::astype(x,mx::float32),mx::transpose(mx::astype(weight_,mx::float32))),{384});
 auto scores=mx::sqrt(mx::where(mx::greater(z,mx::array(20.0f)),z,mx::log1p(mx::exp(z))));
 auto corrected=mx::add(scores,bias_,mx::Device::cpu); mx::eval(scores,corrected);
 return {scores,corrected,select_routes_reference(scores,bias_,false,tie)};
}
mx::array expert_activation_reference(const mx::array& gate,const mx::array& up,const mx::array& weight){
 if(gate.dtype()!=mx::bfloat16||up.dtype()!=mx::bfloat16||gate.shape()!=up.shape()||gate.ndim()!=2||gate.shape(0)!=1||
    gate.shape(1)!=2304||weight.dtype()!=mx::float32||weight.size()!=1)throw std::runtime_error("invalid expert activation");
 auto g=mx::minimum(mx::astype(gate,mx::float32),mx::array(10.0f));
 auto u=mx::clip(mx::astype(up,mx::float32),mx::array(-10.0f),mx::array(10.0f));
 auto silu=mx::multiply(g,mx::sigmoid(g));
 return mx::astype(mx::multiply(mx::multiply(silu,u),mx::reshape(weight,{})),mx::bfloat16);
}
ExpertReference::ExpertReference(WeightCatalog& c,int e,int layer):w1_(c,prefix(e,layer)+".w1"),w2_(c,prefix(e,layer)+".w2"),w3_(c,prefix(e,layer)+".w3"){
 int bits=e==-1?8:4;
 if(w1_.input_dims()!=5120||w1_.output_dims()!=2304||w3_.input_dims()!=5120||w3_.output_dims()!=2304||w2_.input_dims()!=2304||w2_.output_dims()!=5120||w1_.bits()!=bits||w2_.bits()!=bits||w3_.bits()!=bits)throw std::runtime_error("unexpected expert geometry");
}
mx::array ExpertReference::forward(const mx::array& x,const mx::array& weight) const{
 if(x.dtype()!=mx::bfloat16||x.shape()!=mx::Shape({1,5120}))throw std::runtime_error("expert requires one BF16 token");
 return w2_.forward(expert_activation_reference(w1_.forward(x),w3_.forward(x),weight));
}
MoEReference::MoEReference(WeightCatalog& c,int layer):catalog_(&c),layer_(layer),gate_(c,layer),shared_(c,-1,layer){}
ExpertReference& MoEReference::expert(int id) const{
 auto it=experts_.find(id);
 if(it==experts_.end())it=experts_.emplace(id,std::make_unique<ExpertReference>(*catalog_,id,layer_)).first;
 return *it->second;
}
mx::array MoEReference::forward(const mx::array& x) const{
 RouteTieRecord tie{layer_,route_trace_token(),0,0,0.0f,0.0f,false};
 auto route=gate_.forward(x,&tie);
 if(tie.tied){++tie_count_;tie_records().push_back(tie);}
 auto y=mx::zeros({1,5120},mx::float32);
 // Official sums in ascending expert ID, not top-k score order.
 for(int id=0;id<384;++id)for(int k=0;k<6;++k)if(route.ids[k]==id)
  y=mx::add(y,mx::astype(expert(id).forward(x,mx::take(route.weights,mx::array(k))),mx::float32));
 y=mx::add(y,mx::astype(shared_.forward(x,mx::array(1.0f)),mx::float32));
 return mx::astype(y,mx::bfloat16);
}
}
