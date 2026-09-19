#include "dsv41/swa_attention.hpp"
#include "dsv41/attention_telemetry.hpp"
#include "dsv41/swa_projection.hpp"
#include "dsv41/swa_layer.hpp"
#include "dsv41/moe.hpp"
#include "dsv41/compressor.hpp"
#include "dsv41/index_key.hpp"
#include "dsv41/kv_quant.hpp"
#include "dsv41/global_kv.hpp"
#include "dsv41/index_query.hpp"
#include "dsv41/compressed_layer.hpp"
#include "dsv41/execution_policy.hpp"
#include "dsv41/reused_layer.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
namespace mx=mlx::core;
void equal(const mx::array& a,const mx::array& b,const char* message,bool bits=true){
 if(a.shape()!=b.shape()||a.dtype()!=mx::bfloat16||b.dtype()!=mx::bfloat16)throw std::runtime_error(message);
 auto e=mx::all(bits?mx::equal(mx::view(a,mx::uint16),mx::view(b,mx::uint16)):mx::equal(a,b));mx::eval(e);if(!e.item<bool>())throw std::runtime_error(message);
}
void rms_close(const mx::array& candidate,const mx::array& reference,const char* message){
 if(candidate.shape()!=reference.shape()||candidate.dtype()!=reference.dtype())throw std::runtime_error(message);
 auto c=mx::astype(candidate,mx::float32),r=mx::astype(reference,mx::float32),d=mx::subtract(c,r);
 auto ratio=mx::sqrt(mx::divide(mx::sum(mx::multiply(d,d)),
  mx::maximum(mx::sum(mx::multiply(r,r)),mx::array(1e-30f))));
 auto finite=mx::all(mx::isfinite(c));mx::eval(ratio,finite);
 if(!finite.item<bool>()||ratio.item<float>()>=0.002f)
  throw std::runtime_error(std::string(message)+" relative_rms="+std::to_string(ratio.item<float>()));
}
void rms_report(const mx::array& candidate,const mx::array& reference,const char* message){
 if(candidate.shape()!=reference.shape()||candidate.dtype()!=reference.dtype())throw std::runtime_error(message);
 auto c=mx::astype(candidate,mx::float32),r=mx::astype(reference,mx::float32),d=mx::subtract(c,r);
 auto relative=mx::sqrt(mx::divide(mx::sum(mx::multiply(d,d)),
  mx::maximum(mx::sum(mx::multiply(r,r)),mx::array(1e-30f))));
 auto maximum=mx::max(mx::abs(d)),mean=mx::mean(mx::abs(d));
 auto mismatches=mx::sum(mx::astype(mx::not_equal(candidate,reference),mx::uint32));
 auto finite=mx::logical_and(mx::all(mx::isfinite(c)),mx::all(mx::isfinite(r)));
 mx::eval(relative,maximum,mean,mismatches,finite);
 std::cout<<message<<" relative_rms="<<relative.item<float>()<<" max_abs="<<maximum.item<float>()
          <<" mean_abs="<<mean.item<float>()<<" bit_mismatches="<<mismatches.item<std::uint32_t>()<<"\n";
 if(!finite.item<bool>()||relative.item<float>()>=0.002f)
  throw std::runtime_error(std::string(message)+" semantic gate failed");
}
int main(int argc,char** argv){try{
 mx::set_default_device(mx::Device::gpu);
 {
  auto top=dsv41::index_topk_reference(mx::arange(513,mx::float32),128);
  if(top.size()!=512||top.front()!=129||top.back()!=640)throw std::runtime_error("index top-k selection/position order mismatch");
  if(!dsv41::index_topk_reference(mx::zeros({0}),128).empty())throw std::runtime_error("empty index top-k failed");
  bool tie=false;try{dsv41::index_topk_reference(mx::zeros({513}),128);}catch(const std::exception&){tie=true;}
  if(!tie)throw std::runtime_error("index top-k boundary tie accepted");
 }
 for(auto format:{dsv41::KVQuantFormat::IndexE8M0,dsv41::KVQuantFormat::MainE4M3}){
  int k=format==dsv41::KVQuantFormat::IndexE8M0?128:512;
  std::array<float,8> midpoint{0.25f,0.75f,1.25f,1.75f,2.5f,3.5f,5.0f,6.0f};
  std::array<std::uint8_t,8> codes{0,2,2,4,4,6,6,7};
  std::vector<float> v(k);std::vector<std::uint8_t> expected(k/2);
  for(int j=0;j<k;++j){bool neg=(j%16)>=8;v[j]=(neg?-1:1)*midpoint[j%8];expected[j/2]|=(codes[j%8]|(neg?8:0))<<((j%2)*4);}
  auto input=mx::astype(mx::array(v.begin(),{1,k},mx::float32),mx::bfloat16);
  auto q=dsv41::kv_quant_reference(input,format);
  auto ok=mx::all(mx::equal(q.packed,mx::array(expected.begin(),{1,k/2},mx::uint8)));mx::eval(ok);if(!ok.item<bool>())throw std::runtime_error("FP4 midpoint/packing mismatch");
  ok=mx::all(mx::equal(q.scales,mx::array(format==dsv41::KVQuantFormat::IndexE8M0?127:56)));mx::eval(ok);if(!ok.item<bool>())throw std::runtime_error("FP4 unit scale mismatch");
  auto batch=dsv41::kv_quant_reference(mx::concatenate({input,input},0),format);
  equal(q.decoded,mx::slice(batch.decoded,{1,0},{2,k}),"FP4 chunk bits mismatch");
  auto zero=dsv41::kv_quant_reference(mx::zeros({1,k},mx::bfloat16),format);
  equal(zero.decoded,mx::zeros({1,k},mx::bfloat16),"FP4 zero restoration");
  ok=mx::all(mx::equal(zero.scales,mx::array(1)));mx::eval(ok);if(!ok.item<bool>())throw std::runtime_error("FP4 minimum scale mismatch");
  bool rejected=false;try{dsv41::kv_quant_reference(mx::full({1,k},NAN,mx::bfloat16),format);}catch(const std::exception&){rejected=true;}
  if(!rejected)throw std::runtime_error("FP4 nonfinite accepted");
 }
 {
  const std::array<std::uint64_t,3> positions{0,65536,262143};
  auto x=mx::astype(mx::reshape(mx::sin(mx::arange(3*128,mx::float32)),{3,128}),mx::bfloat16);
  auto full=dsv41::compressed_rope_reference(x,positions);
  for(int i=0;i<3;++i)equal(mx::slice(full,{i,0},{i+1,128}),dsv41::compressed_rope_reference(mx::slice(x,{i,0},{i+1,128}),std::span(positions).subspan(i,1)),"compressed RoPE chunk bits");
  equal(mx::slice(full,{0,0},{1,128}),mx::slice(x,{0,0},{1,128}),"compressed RoPE zero position",false);
  equal(mx::slice(full,{0,0},{3,64}),mx::slice(x,{0,0},{3,64}),"compressed RoPE touched non-rotary channels");
  bool bad=false;const std::array<std::uint64_t,1> invalid{1048576};try{dsv41::compressed_rope_reference(mx::slice(x,{0,0},{1,128}),invalid);}catch(const std::exception&){bad=true;}
  if(!bad)throw std::runtime_error("compressed RoPE position overflow accepted");
 }
 auto route_scores=mx::arange(1,385,mx::float32);
 std::vector<float> bias(384,0);for(int i=0;i<6;++i)bias[i]=1000;
 auto route=dsv41::select_routes_reference(route_scores,mx::array(bias.begin(),{384},mx::float32));
 for(int i=0;i<6;++i)if(route.ids[i]!=5-i)throw std::runtime_error("biased route selection mismatch");
 auto expected_route=mx::multiply(mx::divide(mx::arange(6,0,-1,mx::float32),mx::array(21.0f)),mx::array(1.5f));
 auto route_ok=mx::all(mx::equal(route.weights,expected_route));mx::eval(route_ok);if(!route_ok.item<bool>())throw std::runtime_error("bias contaminated route weights");
 bool tied=false;try{dsv41::select_routes_reference(mx::ones({384}),mx::zeros({384}));}catch(const std::exception&){tied=true;}
 if(!tied)throw std::runtime_error("ambiguous route boundary accepted");
 for(float gate:{-20.0f,0.0f,20.0f})for(float up:{-20.0f,20.0f}){
  float g=std::min(gate,10.0f),u=std::clamp(up,-10.0f,10.0f);
  float e=(g/(1+std::exp(-g))*u)*0.25f;
  equal(dsv41::expert_activation_reference(mx::full({1,2304},gate,mx::bfloat16),mx::full({1,2304},up,mx::bfloat16),mx::array(0.25f)),mx::full({1,2304},e,mx::bfloat16),"expert clamp/weight placement mismatch",false);
 }
 // Independent analytic check of the first complex pair (frequency exactly one).
 std::vector<float> values(512,0.0f);values[0]=3;values[448]=1;
 auto input=mx::astype(mx::array(values.begin(),{1,512},mx::float32),mx::bfloat16);
 for(int position:{0,1,127,128,262143,1048575})for(bool inverse:{false,true}){
  auto expected=values;expected[448]=std::cos(float(position));expected[449]=(inverse?-1:1)*std::sin(float(position));
  equal(dsv41::swa_rope_reference(input,position,inverse),
        mx::astype(mx::array(expected.begin(),{1,512},mx::float32),mx::bfloat16),"RoPE analytic mismatch",false);
 }
 auto varied=mx::astype(mx::reshape(mx::sin(mx::arange(2*64*512,mx::float32)),{2,64,512}),mx::bfloat16);
 auto rotated=dsv41::swa_rope_reference(varied,127);
 for(int i=0;i<2;++i)equal(mx::slice(rotated,{i,0,0},{i+1,64,512}),
  dsv41::swa_rope_reference(mx::slice(varied,{i,0,0},{i+1,64,512}),127+i),"RoPE chunk mismatch");
 bool invalid_position=false;try{dsv41::swa_rope_reference(varied,1048575);}catch(const std::exception&){invalid_position=true;}
 if(!invalid_position)throw std::runtime_error("RoPE accepted position overflow");
 for(int count:{1,63,64,65,127,128}){
  auto q=mx::zeros({64,512},mx::bfloat16),kv=mx::ones({count,512},mx::bfloat16),sink=mx::zeros({64},mx::float32);
  auto y=dsv41::swa_attention_reference(q,kv,sink);
  // Zero logits: count identical unit values and a zero-valued sink with equal mass.
  auto expected=mx::full({64,512},float(count)/float(count+1),mx::bfloat16);
  auto equal=mx::all(mx::equal(y,expected));mx::eval(equal);if(!equal.item<bool>())throw std::runtime_error("sink/block boundary mismatch");
 }
 bool rejected=false;try{dsv41::swa_attention_reference(mx::zeros({64,512},mx::bfloat16),mx::zeros({0,512},mx::bfloat16),mx::zeros({64}));}catch(const std::exception&){rejected=true;}
 if(!rejected)throw std::runtime_error("empty KV accepted");
 for(int live:{1,63,64,65,127,128}){
  auto mask=mx::greater_equal(mx::arange(128,mx::int32),mx::array(128-live));
  auto out=dsv41::swa_attention_masked_reference(mx::zeros({64,512},mx::bfloat16),mx::ones({128,512},mx::bfloat16),mx::zeros({64}),mask);
  equal(out,mx::full({64,512},float(live)/float(live+1),mx::bfloat16),"masked sink mismatch");
 }
 {
  // Minimal compile/semantic fixture for the one-dispatch oMLX topology.
  // Token 0 sees local row 0; token 1 sees rows 0..1. Pooled slots are all
  // invalid, but retain the production fixed-width device work-list shape.
  auto q=varied;
  auto local=mx::astype(mx::reshape(mx::cos(mx::arange(2*512,mx::float32)),{2,512}),mx::bfloat16);
  auto sink=mx::zeros({64},mx::float32);
  auto topk=mx::broadcast_to(mx::array(-1,mx::int32),{2,1});
  auto candidate=dsv41::swa_packed_attention_chunk(q,local,mx::zeros({1,256},mx::uint8),
   mx::ones({1,32},mx::uint8),topk,sink,0,4);
  std::vector<mx::array> reference;
  for(int token=0;token<2;++token)reference.push_back(mx::expand_dims(dsv41::swa_attention_masked_reference(
   mx::reshape(mx::slice(q,{token,0,0},{token+1,64,512}),{64,512}),
   mx::slice(local,{0,0},{token+1,512}),sink,mx::ones({token+1},mx::bool_)),0));
  rms_report(candidate,mx::concatenate(reference,0),"packed fused attention local fixture");
  auto wide=dsv41::swa_wide_attention_chunk(q,local,mx::zeros({1,256},mx::uint8),
   mx::ones({1,32},mx::uint8),topk,sink,0,4);
  rms_report(wide,mx::concatenate(reference,0),"wide fused attention local fixture");
  auto dense_topk=mx::broadcast_to(mx::array(-1,mx::int32),{2,512});
  auto fixed=dsv41::swa_fixed_tile_attention_chunk(q,local,mx::zeros({1,256},mx::uint8),
   mx::ones({1,32},mx::uint8),dense_topk,sink,0,4);
  rms_report(fixed,mx::concatenate(reference,0),"fixed-tile attention local fixture");
  auto fixed_work=dsv41::swa_packed_attention_work_list(local,mx::zeros({1,256},mx::uint8),
   mx::ones({1,32},mx::uint8),dense_topk,0,4,512,true);
  auto ragged=dsv41::swa_attention_fixed_tile_core(q,fixed_work,sink);
  rms_report(ragged,mx::concatenate(reference,0),"ragged-tail QK local fixture");
  auto pooled_source=mx::astype(mx::reshape(mx::sin(mx::arange(512,mx::float32)),{1,512}),mx::bfloat16);
  auto pooled=dsv41::kv_quant_reference(pooled_source,dsv41::KVQuantFormat::MainE4M3);
  auto selected=mx::zeros({2,1},mx::int32);
  candidate=dsv41::swa_packed_attention_chunk(q,local,pooled.packed,pooled.scales,selected,sink,8,4);
  reference.clear();
  for(int token=0;token<2;++token){
   auto ordered=mx::concatenate({mx::slice(local,{0,0},{token+1,512}),pooled.decoded},0);
   reference.push_back(mx::expand_dims(dsv41::swa_attention_masked_reference(
    mx::reshape(mx::slice(q,{token,0,0},{token+1,64,512}),{64,512}),ordered,sink,
    mx::ones({ordered.shape(0)},mx::bool_)),0));
  }
  rms_report(candidate,mx::concatenate(reference,0),"packed fused attention pooled fixture");
  wide=dsv41::swa_wide_attention_chunk(q,local,pooled.packed,pooled.scales,selected,sink,8,4);
  rms_report(wide,mx::concatenate(reference,0),"wide fused attention pooled fixture");
  auto boundary_plan=mx::concatenate({selected,mx::broadcast_to(
   mx::array(-1,mx::int32),{2,511})},1);
  auto boundary_work=dsv41::swa_packed_attention_work_list(
   local,pooled.packed,pooled.scales,boundary_plan,0,1,512,true);
  reference.clear();
  for(int token=0;token<2;++token){
   auto ordered=mx::concatenate({mx::slice(local,{0,0},{token+1,512}),pooled.decoded},0);
   reference.push_back(mx::expand_dims(dsv41::swa_attention_masked_reference(
    mx::reshape(mx::slice(q,{token,0,0},{token+1,64,512}),{64,512}),ordered,sink,
    mx::ones({ordered.shape(0)},mx::bool_)),0));
  }
  auto boundary_candidate=dsv41::swa_attention_fixed_tile_core(q,boundary_work,sink,true);
  auto boundary_reference=mx::concatenate(reference,0);
  equal(mx::slice(boundary_candidate,{0,0,0},{1,64,512}),
        mx::slice(boundary_reference,{0,0,0},{1,64,512}),
        "fixed-tile token-zero boundary class mismatch");
  rms_report(boundary_candidate,boundary_reference,"fixed-tile request-boundary fixture");
  auto class_source=mx::astype(mx::reshape(mx::sin(mx::arange(65*512,mx::float32)),{65,512}),mx::bfloat16);
  auto class_pooled=dsv41::kv_quant_reference(class_source,dsv41::KVQuantFormat::MainE4M3);
  auto class_local=mx::astype(mx::reshape(mx::cos(mx::arange(128*512,mx::float32)),{128,512}),mx::bfloat16);
  // Width 65 regresses the device-class predicate: its ragged tail is width
  // one even though the total selected width is not one.
  for(int width:{1,33,40,63,65}){
   auto ids=mx::broadcast_to(mx::expand_dims(mx::arange(width,mx::int32),0),{2,width});
   auto dense=mx::concatenate({ids,mx::broadcast_to(mx::array(-1,mx::int32),{2,512-width})},1);
   auto work=dsv41::swa_packed_attention_work_list(class_local,class_pooled.packed,
    class_pooled.scales,dense,512,4,512,true);
   auto padded=dsv41::swa_attention_masked_chunk(q,work.ordered,sink,work.valid);
   auto exact_tail=dsv41::swa_attention_fixed_tile_core(q,work,sink);
   rms_report(exact_tail,padded,"ragged-tail QK width-class fixture");
   auto exact_qk_av=dsv41::swa_attention_fixed_tile_core(q,work,sink,true);
   rms_report(exact_qk_av,padded,"ragged-tail QK/AV width-class fixture");
   if(width==1){
    auto diagnostic=dsv41::swa_attention_fixed_tile_width_one_diagnostics(q,work,sink,true);
    rms_report(diagnostic.native_qk,padded,"native width-one QK attribution fixture");
    rms_report(diagnostic.native_av,padded,"native width-one AV attribution fixture");
   }
  }
 }
 {
  dsv41::reset_attention_telemetry();
  auto q=varied;
  auto kv=mx::reshape(mx::concatenate({varied,varied},0),{2,128,512});
  auto sink=mx::zeros({64},mx::float32);
  auto valid=mx::greater_equal(mx::reshape(mx::arange(128,mx::int32),{1,128}),
                               mx::reshape(mx::array({127,126},mx::int32),{2,1}));
  auto candidate=dsv41::swa_attention_masked_chunk(q,kv,sink,valid);

  auto first=dsv41::swa_attention_masked_reference(mx::reshape(mx::slice(q,{0,0,0},{1,64,512}),{64,512}),
   mx::reshape(mx::slice(kv,{0,0,0},{1,128,512}),{128,512}),sink,
   mx::greater_equal(mx::arange(128,mx::int32),mx::array(127)));
  auto second=dsv41::swa_attention_masked_reference(mx::reshape(mx::slice(q,{1,0,0},{2,64,512}),{64,512}),
   mx::reshape(mx::slice(kv,{1,0,0},{2,128,512}),{128,512}),sink,
   mx::greater_equal(mx::arange(128,mx::int32),mx::array(126)));
  auto expected=mx::concatenate({mx::expand_dims(first,0),mx::expand_dims(second,0)},0);
  rms_report(candidate,expected,"chunk attention core");
  const auto telemetry=dsv41::attention_telemetry();
  if(dsv41::runtime_batched_splitk_qk_enabled()&&
     (telemetry.chunk_batched_splitk_qk_calls!=2||telemetry.chunk_scalar_qk_calls!=0||telemetry.chunk_av_batches!=2))
   throw std::runtime_error("batched split-K QK dispatch telemetry mismatch");
  // Fixed-width qualification: padding after an exact 129-row segment must
  // not silently change the selected tail reduction.
  auto exact_kv=mx::astype(mx::reshape(mx::sin(mx::arange(129*512,mx::float32)),{1,129,512}),mx::bfloat16);
  auto exact_valid=mx::ones({1,129},mx::bool_);
  auto exact_q=mx::slice(q,{0,0,0},{1,64,512});
  auto exact_out=dsv41::swa_attention_masked_chunk(exact_q,exact_kv,sink,exact_valid);
  auto tail=dsv41::swa_attention_tail_diagnostics(exact_q,exact_kv,sink,exact_valid);
  rms_report(tail.padded_qk,exact_out,"padded-tail QK attribution fixture");
  rms_report(tail.padded_av,exact_out,"padded-tail AV attribution fixture");
  auto padded_kv=mx::concatenate({exact_kv,mx::zeros({1,511,512},mx::bfloat16)},1);
  auto padded_valid=mx::concatenate({exact_valid,mx::zeros({1,511},mx::bool_)},1);
  rms_report(dsv41::swa_attention_masked_chunk(exact_q,padded_kv,sink,padded_valid),
             exact_out,"fixed-tile 129-to-640 padding diagnostic");
  // Request token zero has exactly two live rows, but the fixed work list
  // places them on opposite sides of the 128-row local/pooled boundary.
  // Keep this sparse topology distinct from the all-live 129-row padding
  // diagnostic above: it exercises online-softmax combination across blocks.
  auto sparse_ids=mx::arange(640,mx::int32);
  auto sparse_valid=mx::expand_dims(mx::logical_or(
   mx::equal(sparse_ids,mx::array(127)),mx::equal(sparse_ids,mx::array(128))),0);
  auto compact_kv=mx::slice(padded_kv,{0,127,0},{1,129,512});
  auto sparse_out=dsv41::swa_attention_masked_chunk(exact_q,padded_kv,sink,sparse_valid);
  auto compact_out=dsv41::swa_attention_masked_chunk(
   exact_q,compact_kv,sink,mx::ones({1,2},mx::bool_));
  rms_report(sparse_out,compact_out,"fixed-tile sparse boundary topology diagnostic");
  for(int live:{63,64,65,128}){
   auto full_valid=mx::broadcast_to(mx::greater_equal(mx::arange(128,mx::int32),mx::array(128-live)),{2,128});
   auto full_candidate=dsv41::swa_attention_masked_chunk(q,kv,sink,full_valid);
   std::vector<mx::array> reference;
   for(int token=0;token<2;++token)reference.push_back(mx::expand_dims(dsv41::swa_attention_masked_reference(
    mx::reshape(mx::slice(q,{token,0,0},{token+1,64,512}),{64,512}),
    mx::reshape(mx::slice(kv,{token,0,0},{token+1,128,512}),{128,512}),sink,
    mx::greater_equal(mx::arange(128,mx::int32),mx::array(128-live))),0));
   rms_report(full_candidate,mx::concatenate(reference,0),"chunk attention live-row ladder");
  }
  if(dsv41::runtime_batched_splitk_qk_enabled())for(int width:{1,31,32,33,39,40,63,64,65,127,128,129,640}){
   dsv41::reset_attention_telemetry();
   auto width_kv=mx::astype(mx::reshape(mx::sin(mx::arange(2*width*512,mx::float32)),{2,width,512}),mx::bfloat16);
   auto width_valid=mx::ones({2,width},mx::bool_);
   auto width_candidate=dsv41::swa_attention_masked_chunk(q,width_kv,sink,width_valid);
   std::vector<mx::array> reference;
   for(int token=0;token<2;++token)reference.push_back(mx::expand_dims(dsv41::swa_attention_masked_reference(
    mx::reshape(mx::slice(q,{token,0,0},{token+1,64,512}),{64,512}),
    mx::reshape(mx::slice(width_kv,{token,0,0},{token+1,width,512}),{width,512}),sink,
    mx::ones({width},mx::bool_)),0));
   rms_report(width_candidate,mx::concatenate(reference,0),"chunk attention split-K shape ladder");
   const auto width_telemetry=dsv41::attention_telemetry();
   if(width_telemetry.chunk_batched_splitk_qk_calls!=std::size_t(width/64)||
      width_telemetry.chunk_scalar_qk_calls!=std::size_t(width%64?2:0))
    throw std::runtime_error("chunk attention full-block/tail QK dispatch mismatch");
  }
  if(dsv41::runtime_batched_splitk_qk_enabled())for(int token_count:{1,3,63,64,65,127,128}){
   // Backbone groups are usually non-zero-offset slices of a 128-token Q tensor.
   // Exercise both their retained input stride and a large grid-z token axis.
   auto parent_q=mx::astype(mx::reshape(mx::sin(mx::arange(130*64*512,mx::float32)),{130,64,512}),mx::bfloat16);
   auto sliced_q=mx::slice(parent_q,{1,0,0},{token_count+1,64,512});
   auto token_kv=mx::astype(mx::reshape(mx::cos(mx::arange(token_count*128*512,mx::float32)),
                                        {token_count,128,512}),mx::bfloat16);
   auto token_valid=mx::ones({token_count,128},mx::bool_);
   auto token_candidate=dsv41::swa_attention_masked_chunk(sliced_q,token_kv,sink,token_valid);
   std::vector<mx::array> reference;reference.reserve(token_count);
   for(int token=0;token<token_count;++token)reference.push_back(mx::expand_dims(dsv41::swa_attention_masked_reference(
    mx::reshape(mx::slice(sliced_q,{token,0,0},{token+1,64,512}),{64,512}),
    mx::reshape(mx::slice(token_kv,{token,0,0},{token+1,128,512}),{128,512}),sink,
    mx::ones({128},mx::bool_)),0));
   rms_report(token_candidate,mx::concatenate(reference,0),"chunk attention split-K token ladder");
  }
 }
 if(argc!=1&&argc!=3)throw std::runtime_error("usage: dsv41-swa-attention-test [checkpoint m1-summary]");
 if(argc==3){
  dsv41::WeightCatalog catalog(argv[1],argv[2]);
  if(std::getenv("DSV41_CHECK_FIXED_TILE_ISOLATION")){
   if(!dsv41::runtime_fixed_tile_attention_enabled()||
      !dsv41::runtime_batched_splitk_qk_enabled())
    throw std::runtime_error("fixed-tile isolation requires fixed tiles and batched split-K QK");
   dsv41::CompressedLayerReference producer(catalog,2);
   dsv41::ReusedLayerReference consumer(catalog,3);
   dsv41::CompressedLayerState producer_serial,producer_candidate;
   dsv41::ReusedLayerState consumer_serial,consumer_candidate;
   dsv41::reset_attention_telemetry();
   for(int chunk=0;chunk<2;++chunk){
    const int start=chunk*128;
    auto values=mx::arange(128*5120,mx::float32);
    auto input=mx::astype(mx::reshape(chunk?mx::cos(values):mx::sin(values),{128,5120}),mx::bfloat16);
    std::vector<dsv41::SharedAttentionReference> serial_publications;
    std::vector<mx::array> producer_rows,consumer_rows;
    serial_publications.reserve(128);producer_rows.reserve(128);consumer_rows.reserve(128);
    for(int token=0;token<128;++token){
     auto row=mx::slice(input,{token,0},{token+1,5120});
     producer_rows.push_back(producer.forward(row,producer_serial,start+token));
     serial_publications.push_back(*producer_serial.publication());
     consumer_rows.push_back(consumer.forward(row,consumer_serial,
                                               serial_publications.back(),start+token));
    }
    std::vector<dsv41::SharedAttentionReference> candidate_publications;
    auto producer_output=producer.forward_chunk(input,producer_candidate,start,&candidate_publications);
    auto consumer_output=consumer.forward_chunk(input,consumer_candidate,candidate_publications,start);
    rms_report(producer_output,mx::concatenate(producer_rows,0),
               chunk?"fixed-tile producer chunk 1":"fixed-tile producer chunk 0");
    rms_report(consumer_output,mx::concatenate(consumer_rows,0),
               chunk?"fixed-tile first reuse chunk 1":"fixed-tile first reuse chunk 0");
    equal(producer_candidate.window(),producer_serial.window(),"fixed-tile producer window bits");
    equal(consumer_candidate.window(),consumer_serial.window(),"fixed-tile consumer window bits");
    if(producer_candidate.position()!=producer_serial.position()||
       consumer_candidate.position()!=consumer_serial.position()||candidate_publications.size()!=128)
     throw std::runtime_error("fixed-tile isolation state/publication mismatch");
    for(int token=0;token<128;++token){
     const auto pos=std::uint64_t(start+token);const int offset=pos==0?1:128;
     auto a=serial_publications[token].device_indices(3,pos,offset);
     auto b=candidate_publications[token].device_indices(3,pos,offset);
     if(a.shape()!=b.shape())throw std::runtime_error("fixed-tile publication row shape mismatch");
     auto same=mx::all(mx::equal(a,b));mx::eval(same);
     if(!same.item<bool>())throw std::runtime_error("fixed-tile publication row mismatch");
    }
   }
   const auto telemetry=dsv41::read_attention_telemetry();
   if(telemetry.fixed_tile_attention_calls!=4||telemetry.chunk_scalar_qk_calls!=0)
    throw std::runtime_error("fixed-tile isolation topology mismatch");
   std::cout<<"PASS: official layer 2 producer and layer 3 reuse fixed-tile isolation; "
            <<"publication/window/position exact; downstream mHC/MoE excluded\n";
   return 0;
  }
  {
   dsv41::CompressedLayerReference producer(catalog,2);dsv41::ReusedLayerReference consumer(catalog,3);
   dsv41::CompressedLayerState source;dsv41::ReusedLayerState target;
   auto inputs=mx::astype(mx::reshape(mx::sin(mx::arange(3*5120,mx::float32)),{3,5120}),mx::bfloat16);
   std::vector<dsv41::SharedAttentionReference> publications;std::vector<mx::array> outputs;
   for(int i=0;i<3;++i){auto x=mx::slice(inputs,{i,0},{i+1,5120});producer.forward(x,source,i);
    publications.push_back(*source.publication());auto snapshot=publications.back().cache().main_bytes();
    outputs.push_back(consumer.forward(x,target,publications.back(),i));
    auto intact=mx::all(mx::equal(snapshot,publications.back().cache().main_bytes()));mx::eval(intact);if(!intact.item<bool>())throw std::runtime_error("consumer changed producer bytes");}
   dsv41::ReusedLayerState chunk_target;auto chunk_publications=publications;
   auto chunk_output=consumer.forward_chunk(inputs,chunk_target,chunk_publications,0);
   if(dsv41::runtime_chunk_attention_enabled()||dsv41::runtime_packed_chunk_attention_enabled()||
      dsv41::runtime_wide_attention_enabled()||dsv41::runtime_fixed_tile_attention_enabled())
    rms_close(chunk_output,mx::concatenate(outputs,0),"consumer chunk output tolerance");
   else equal(chunk_output,mx::concatenate(outputs,0),"consumer chunk output bits");
   equal(chunk_target.window(),target.window(),"consumer chunk window bits");
   if(chunk_target.position()!=target.position())throw std::runtime_error("consumer chunk position mismatch");
   auto saved=target;target.reset();
   for(int i=0;i<3;++i)equal(consumer.forward(mx::slice(inputs,{i,0},{i+1,5120}),target,publications[i],i),outputs[i],"consumer snapshot replay bits");
   equal(target.window(),saved.window(),"consumer window replay bits");
   bool stale=false;auto x=mx::slice(inputs,{0,0},{1,5120});try{consumer.forward(x,target,publications[0],3);}catch(const std::exception&){stale=true;}
   if(!stale||target.position()!=3)throw std::runtime_error("consumer stale publication accepted");equal(target.window(),saved.window(),"consumer rejection state");
   producer.forward(x,source,3);auto fork=target;
   equal(consumer.forward(x,fork,*source.publication(),3),consumer.forward(x,target,*source.publication(),3),"consumer fork bits");
   if(saved.position()!=3||fork.position()!=4)throw std::runtime_error("consumer fork position");
   auto finite=mx::all(mx::isfinite(outputs.back()));mx::eval(finite);if(!finite.item<bool>())throw std::runtime_error("nonfinite consumer");
   std::cout<<"Layer 3 attention: real layer 2 snapshots, replay/reset/fork bits, stale rejection passed (not Block/oracle qualification)\n";
   for(int layer=4;layer<=7;++layer){
    dsv41::ReusedLayerReference reader(catalog,layer);dsv41::ReusedLayerState a,b;
    for(int i=0;i<3;++i){auto x=mx::slice(inputs,{i,0},{i+1,5120});
     equal(reader.forward(x,a,publications[i],i),reader.forward(x,b,publications[i],i),"consumer 4..7 repeat mismatch");}
    equal(a.window(),b.window(),"consumer 4..7 window mismatch");
   }
   bool invalid_owner=false;try{dsv41::ReusedLayerReference wrong(catalog,8);}catch(const std::exception&){invalid_owner=true;}
   if(!invalid_owner)throw std::runtime_error("layer 8 accepted as layer 2 consumer");
   std::cout<<"Layers 4..7 real consumer attention repeat/window bits and layer 8 rejection passed\n";
   if(std::getenv("DSV41_CHECK_CHUNK_ATTENTION_128")){
    dsv41::CompressedLayerState long_source;dsv41::ReusedLayerState long_serial,long_chunk;
    auto long_input=mx::astype(mx::reshape(mx::sin(mx::arange(128*5120,mx::float32)),{128,5120}),mx::bfloat16);
    std::vector<dsv41::SharedAttentionReference> long_publications;std::vector<mx::array> long_outputs;
    long_publications.reserve(128);long_outputs.reserve(128);
    for(int i=0;i<128;++i){auto x=mx::slice(long_input,{i,0},{i+1,5120});producer.forward(x,long_source,i);
     long_publications.push_back(*long_source.publication());long_outputs.push_back(consumer.forward(x,long_serial,long_publications.back(),i));}
    auto candidate_publications=long_publications;
    auto candidate=consumer.forward_chunk(long_input,long_chunk,candidate_publications,0);
    rms_report(candidate,mx::concatenate(long_outputs,0),"layer 3 128-token attention");
    equal(long_chunk.window(),long_serial.window(),"layer 3 128-token window bits");
    if(long_chunk.position()!=long_serial.position())throw std::runtime_error("layer 3 128-token position mismatch");
    auto second_input=mx::astype(mx::reshape(mx::cos(mx::arange(128*5120,mx::float32)),{128,5120}),mx::bfloat16);
    auto second_chunk=long_serial;long_publications.clear();long_outputs.clear();
    for(int i=0;i<128;++i){const int pos=128+i;auto x=mx::slice(second_input,{i,0},{i+1,5120});
     producer.forward(x,long_source,pos);long_publications.push_back(*long_source.publication());
     long_outputs.push_back(consumer.forward(x,long_serial,long_publications.back(),pos));}
    candidate_publications=long_publications;
    candidate=consumer.forward_chunk(second_input,second_chunk,candidate_publications,128);
    rms_report(candidate,mx::concatenate(long_outputs,0),"layer 3 position 128 attention");
    equal(second_chunk.window(),long_serial.window(),"layer 3 position 128 window bits");
    if(second_chunk.position()!=long_serial.position())throw std::runtime_error("layer 3 position 128 mismatch");
   }
  }
  {
   dsv41::CompressedLayerReference attention(catalog,2);dsv41::CompressedLayerState chunk,serial;
   auto input=mx::astype(mx::reshape(mx::sin(mx::arange(5*5120,mx::float32)),{5,5120}),mx::bfloat16);
   std::vector<dsv41::SharedAttentionReference> production_publications;
   auto batch=attention.forward_chunk(input,chunk,0,&production_publications);
   if(!chunk.publication())throw std::runtime_error("layer 2 publication missing");
   auto published=*chunk.publication();
   for(int layer=3;layer<=7;++layer){
    if(dsv41::runtime_index_diagnostics_enabled()){
     if(published.indices(layer,4,128)!=std::vector<std::int32_t>{128,129})throw std::runtime_error("reuse consumer mismatch");
    }else{
     auto rows=published.device_indices(layer,4,128);mx::eval(rows);
     const auto* p=rows.data<std::int32_t>();
     if(rows.dtype()!=mx::int32||rows.shape()!=mx::Shape({2})||p[0]!=0||p[1]!=1)
      throw std::runtime_error("device reuse consumer mismatch");
    }
   }
   for(int layer:{2,8}){bool bad=false;try{
    if(dsv41::runtime_index_diagnostics_enabled())published.indices(layer,4,128);
    else published.device_indices(layer,4,128);
   }catch(const std::exception&){bad=true;}if(!bad)throw std::runtime_error("wrong reuse source accepted");}
   bool stale=false;try{
    if(dsv41::runtime_index_diagnostics_enabled())published.indices(3,5,128);
    else published.device_indices(3,5,128);
   }catch(const std::exception&){stale=true;}if(!stale)throw std::runtime_error("stale publication accepted");
   for(int i=0;i<5;++i)equal(attention.forward(mx::slice(input,{i,0},{i+1,5120}),serial,i),mx::slice(batch,{i,0},{i+1,5120}),"compressed attention chunk bits");
   auto same_state=[&](const dsv41::CompressedLayerState& a,const dsv41::CompressedLayerState& b){
    if(a.position()!=b.position()||a.global().rows()!=b.global().rows())throw std::runtime_error("compressed state position mismatch");
    equal(a.window(),b.window(),"compressed window bits");
    auto bytes=[](const mx::array& x,const mx::array& y){if(x.shape()!=y.shape())throw std::runtime_error("compressed state shape mismatch");auto ok=mx::all(mx::equal(x,y));mx::eval(ok);if(!ok.item<bool>())throw std::runtime_error("compressed global bytes mismatch");};
    bytes(a.global().main_bytes(),b.global().main_bytes());bytes(a.global().main_scales(),b.global().main_scales());
    bytes(a.global().index_bytes(),b.global().index_bytes());bytes(a.global().index_scales(),b.global().index_scales());
    bytes(mx::view(a.global().compressor().pending_kv(),mx::uint32),mx::view(b.global().compressor().pending_kv(),mx::uint32));
    bytes(mx::view(a.global().compressor().pending_scores(),mx::uint32),mx::view(b.global().compressor().pending_scores(),mx::uint32));
   };
   same_state(chunk,serial);auto saved=chunk,fork=chunk;auto token=mx::slice(input,{0,0},{1,5120});
   equal(attention.forward(token,fork,5),attention.forward(token,serial,5),"compressed continuation bits");same_state(fork,serial);same_state(chunk,saved);
   if(published.cache().position()!=5||published.cache().rows()!=2||fork.publication()->cache().rows()!=3)throw std::runtime_error("published snapshot mutated");
   bool rejected=false;try{attention.forward(token,chunk,0);}catch(const std::exception&){rejected=true;}
   if(!rejected)throw std::runtime_error("compressed invalid position accepted");same_state(chunk,saved);
   rejected=false;try{attention.forward(mx::full({1,5120},NAN,mx::bfloat16),chunk,5);}catch(const std::exception&){rejected=true;}
   if(!rejected)throw std::runtime_error("compressed NaN accepted");same_state(chunk,saved);
   chunk.reset();equal(attention.forward(input,chunk,0),batch,"compressed reset bits");same_state(chunk,saved);
   std::cout<<"Layer 2 attention: window/global output and state chunk bits, continuation/fork/reset/rejection passed (not oracle qualification)\n";
  }
  {
   dsv41::GlobalKVProducerReference producer(catalog,2);dsv41::GlobalKVState chunk,serial;
   auto input=mx::astype(mx::reshape(mx::sin(mx::arange(5*5120,mx::float32)),{5,5120}),mx::bfloat16);
   auto bytes=[](const mx::array& a,const mx::array& b){
    if(a.shape()!=b.shape()||a.dtype()!=b.dtype())throw std::runtime_error("global KV shape mismatch");
    auto ok=mx::all(mx::equal(a,b));mx::eval(ok);if(!ok.item<bool>())throw std::runtime_error("global KV bytes mismatch");};
   auto same_cache=[&](const dsv41::GlobalKVState& a,const dsv41::GlobalKVState& b){
    if(a.position()!=b.position()||a.rows()!=b.rows())throw std::runtime_error("global KV position mismatch");
    bytes(a.main_bytes(),b.main_bytes());bytes(a.main_scales(),b.main_scales());bytes(a.index_bytes(),b.index_bytes());bytes(a.index_scales(),b.index_scales());
    bytes(mx::view(a.compressor().pending_kv(),mx::uint32),mx::view(b.compressor().pending_kv(),mx::uint32));
    bytes(mx::view(a.compressor().pending_scores(),mx::uint32),mx::view(b.compressor().pending_scores(),mx::uint32));};
   producer.append(input,chunk,0);
   dsv41::IndexQueryReference query(catalog,2);
   auto x=mx::slice(input,{4,0},{5,5120});
   auto qr=mx::astype(mx::reshape(mx::sin(mx::arange(1280,mx::float32)),{1,1280}),mx::bfloat16);
   auto selected=query.forward(x,qr,chunk,4,128);
   if(selected.rows!=std::vector<std::int32_t>{128,129})throw std::runtime_error("real index reachable positions mismatch");
   auto restored=dsv41::restore_index_reference(chunk);
   auto finite=mx::all(mx::isfinite(restored));mx::eval(finite);if(!finite.item<bool>())throw std::runtime_error("nonfinite restored index cache");
   bool stale=false;try{query.forward(x,qr,chunk,3,128);}catch(const std::exception&){stale=true;}
   if(!stale)throw std::runtime_error("index accepted future cache");
   for(int i=0;i<5;++i){producer.append(mx::slice(input,{i,0},{i+1,5120}),serial,i);if(serial.rows()!=std::size_t((i+1)/2))throw std::runtime_error("premature global KV publication");}
   same_cache(chunk,serial);auto saved=chunk,fork=chunk;auto token=mx::slice(input,{0,0},{1,5120});
   producer.append(token,fork,5);producer.append(token,serial,5);same_cache(fork,serial);same_cache(chunk,saved);
   if(fork.rows()!=3||chunk.rows()!=2)throw std::runtime_error("global KV fork failed");
   bool rejected=false;try{producer.append(token,chunk,0);}catch(const std::exception&){rejected=true;}
   if(!rejected)throw std::runtime_error("global KV invalid position accepted");same_cache(chunk,saved);
   rejected=false;try{producer.append(mx::full({1,5120},NAN,mx::bfloat16),chunk,5);}catch(const std::exception&){rejected=true;}
   if(!rejected)throw std::runtime_error("global KV NaN accepted");same_cache(chunk,saved);
   chunk.reset();producer.append(input,chunk,0);same_cache(chunk,saved);
   std::cout<<"Layer 2 packed global KV publication: byte-exact chunk/token, partial groups, continuation, fork/reset/rejection passed\n";
  }
  {
   dsv41::CompressorReference compressor(catalog,2);dsv41::CompressorState chunk,serial;
   auto h=mx::astype(mx::reshape(mx::sin(mx::arange(3*5120,mx::float32)),{3,5120}),mx::bfloat16);
   auto batch=compressor.forward(h,chunk,0);
   dsv41::IndexKeyReference index(catalog,2);
   auto key=index.before_quantization(batch.values,batch.positions);
   auto repeated_key=index.before_quantization(batch.values,batch.positions);
   equal(key,repeated_key,"index key repeat bits");
   auto key_finite=mx::all(mx::isfinite(key));mx::eval(key_finite);if(!key_finite.item<bool>())throw std::runtime_error("nonfinite index key");
   if(batch.positions!=std::vector<std::uint64_t>{0}||chunk.position()!=3||chunk.pending_kv().shape()!=mx::Shape({1,512}))throw std::runtime_error("compressor group boundary mismatch");
   for(int i=0;i<3;++i){auto one=compressor.forward(mx::slice(h,{i,0},{i+1,5120}),serial,i);
    if(i==1)equal(one.values,batch.values,"compressor chunk bits mismatch");
    else if(one.values.shape()!=mx::Shape({0,512})||!one.positions.empty())throw std::runtime_error("incomplete compressor emitted latent");}
   auto state_bits=[&](const mx::array& a,const mx::array& b){auto ok=mx::all(mx::equal(mx::view(a,mx::uint32),mx::view(b,mx::uint32)));mx::eval(ok);if(!ok.item<bool>())throw std::runtime_error("compressor pending bits mismatch");};
   state_bits(chunk.pending_kv(),serial.pending_kv());state_bits(chunk.pending_scores(),serial.pending_scores());
   auto fork=chunk;auto token=mx::slice(h,{0,0},{1,5120});auto next=compressor.forward(token,fork,3);
   equal(next.values,compressor.forward(token,serial,3).values,"compressor continuation mismatch");
   if(next.positions!=std::vector<std::uint64_t>{2}||chunk.position()!=3||fork.position()!=4||fork.pending_kv().shape(0)!=0)throw std::runtime_error("compressor fork mismatch");
   auto saved=chunk;bool rejected=false;try{compressor.forward(token,chunk,0);}catch(const std::exception&){rejected=true;}
   if(!rejected||chunk.position()!=3)throw std::runtime_error("compressor invalid position accepted");
   rejected=false;try{compressor.forward(mx::full({1,5120},NAN,mx::bfloat16),chunk,3);}catch(const std::exception&){rejected=true;}
   if(!rejected)throw std::runtime_error("compressor NaN accepted");state_bits(saved.pending_kv(),chunk.pending_kv());state_bits(saved.pending_scores(),chunk.pending_scores());
   chunk.reset();equal(compressor.forward(h,chunk,0).values,batch.values,"compressor reset mismatch");
   chunk.reset();auto zero=compressor.forward(mx::zeros({2,5120},mx::bfloat16),chunk,0);
   equal(zero.values,mx::zeros({1,512},mx::bfloat16),"compressor zero analytic mismatch",false);
   std::cout<<"Layer 2 compressor: groups/positions, chunk bits, pending state, fork/reset/rejection and zero analytic passed (not oracle qualification)\n";
  }
  bool unsupported=false;try{dsv41::SwaProjectionReference bad(catalog,2);}catch(const std::exception&){unsupported=true;}
  if(!unsupported)throw std::runtime_error("compressed layer accepted as pure SWA");
  {
   dsv41::SwaProjectionReference layer1(catalog,1);
   auto input1=mx::astype(mx::reshape(mx::sin(mx::arange(2*5120,mx::float32)),{2,5120}),mx::bfloat16);
   auto qkv1=layer1.forward(input1,127);
   for(int i=0;i<2;++i){auto one=layer1.forward(mx::slice(input1,{i,0},{i+1,5120}),127+i);
    equal(one.query,mx::slice(qkv1.query,{i,0,0},{i+1,64,512}),"layer 1 query chunk mismatch");
    equal(one.kv,mx::slice(qkv1.kv,{i,0},{i+1,512}),"layer 1 KV chunk mismatch");}
   dsv41::GateReference gate1(catalog,1);auto token1=mx::slice(input1,{0,0},{1,5120});auto route1=gate1.forward(token1);
   dsv41::ExpertReference expert1(catalog,route1.ids[0],1);
   auto out1=expert1.forward(token1,mx::take(route1.weights,mx::array(0)));
   equal(out1,expert1.forward(token1,mx::take(route1.weights,mx::array(0))),"layer 1 expert repeat mismatch");
   auto valid1=mx::all(mx::isfinite(out1));mx::eval(valid1);if(!valid1.item<bool>())throw std::runtime_error("nonfinite layer 1 expert");
   std::cout<<"Layer 1 Q/KV chunk bits and selected expert repeat/finite passed; compressed-layer rejection passed\n";
  }
  dsv41::GateReference gate(catalog);
  auto expert_input=mx::astype(mx::reshape(mx::sin(mx::arange(5120,mx::float32)),{1,5120}),mx::bfloat16);
  auto selected=gate.forward(expert_input);auto repeated=gate.forward(expert_input);
  if(selected.ids!=repeated.ids)throw std::runtime_error("router repeat mismatch");
  for(int expert:{-1,selected.ids[0]}){
   dsv41::ExpertReference ffn(catalog,expert);
   auto weight=expert==-1?mx::array(1.0f):mx::take(selected.weights,mx::array(0));
   auto output=ffn.forward(expert_input,weight);
   equal(output,ffn.forward(expert_input,weight),"expert repeat mismatch");
   equal(ffn.forward(expert_input,mx::array(0.0f)),mx::zeros({1,5120},mx::bfloat16),"zero route expert output",false);
   auto finite_output=mx::all(mx::isfinite(output));mx::eval(finite_output);if(!finite_output.item<bool>())throw std::runtime_error("nonfinite expert output");
  }
  std::cout<<"MoE selection/activation analytic and real shared/selected-expert finite/repeat checks passed (not full MoE)\n";
  auto owner=std::make_unique<dsv41::SwaProjectionReference>(catalog);
  auto h=mx::astype(mx::reshape(mx::sin(mx::arange(2*5120,mx::float32)),{2,5120}),mx::bfloat16);
  auto batch=owner->forward(h,127);
  for(int i=0;i<2;++i){auto one=owner->forward(mx::slice(h,{i,0},{i+1,5120}),127+i);
   equal(one.query,mx::slice(batch.query,{i,0,0},{i+1,64,512}),"Q projection chunk mismatch");
   equal(one.kv,mx::slice(batch.kv,{i,0},{i+1,512}),"KV projection chunk mismatch");}
  auto deferred=owner->forward(h,127);owner.reset();
  equal(deferred.query,batch.query,"Q owner lifetime mismatch");equal(deferred.kv,batch.kv,"KV owner lifetime mismatch");
  auto finite=mx::all(mx::isfinite(batch.query));mx::eval(finite);if(!finite.item<bool>())throw std::runtime_error("nonfinite query");
  finite=mx::all(mx::isfinite(batch.kv));mx::eval(finite);if(!finite.item<bool>())throw std::runtime_error("nonfinite KV");
  std::cout<<"Real layer 0 Q/KV projection: finite, chunk/token exact, owner lifetime passed (not oracle qualification)\n";
  dsv41::SwaLayerReference layer(catalog);dsv41::SwaLayerState chunk,serial;
  auto y=layer.forward(h,chunk,0);
  for(int i=0;i<2;++i)equal(layer.forward(mx::slice(h,{i,0},{i+1,5120}),serial,i),mx::slice(y,{i,0},{i+1,5120}),"layer chunk mismatch");
  equal(chunk.rows(),serial.rows(),"layer state chunk mismatch");
  auto fork=chunk;auto saved=chunk.rows();auto token=mx::slice(h,{0,0},{1,5120});
  auto continuation=layer.forward(token,fork,2);(void)continuation;
  if(chunk.position()!=2||fork.position()!=3)throw std::runtime_error("layer fork position mismatch");
  equal(chunk.rows(),saved,"layer fork mutated source");
  bool wrong=false;try{layer.forward(token,chunk,0);}catch(const std::exception&){wrong=true;}
  if(!wrong||chunk.position()!=2)throw std::runtime_error("layer invalid position accepted");
  wrong=false;try{layer.forward(mx::full({1,5120},NAN,mx::bfloat16),chunk,2);}catch(const std::exception&){wrong=true;}
  if(!wrong||chunk.position()!=2)throw std::runtime_error("layer nonfinite input accepted");
  equal(chunk.rows(),saved,"layer rejection mutated state");chunk.reset();
  equal(layer.forward(h,chunk,0),y,"layer reset mismatch");
  dsv41::SwaLayerState wrapped;
  auto prefix=mx::broadcast_to(token,{128,5120});layer.forward(prefix,wrapped,0);
  if(wrapped.position()!=128||wrapped.rows().shape()!=mx::Shape({128,512}))throw std::runtime_error("layer window fill mismatch");
  auto before_wrap=wrapped.rows();auto wrap_copy=wrapped;
  auto after=layer.forward(token,wrapped,128);
  equal(after,layer.forward(token,wrap_copy,128),"layer wrapped fork mismatch");
  if(wrapped.position()!=129||wrapped.rows().shape()!=mx::Shape({128,512}))throw std::runtime_error("layer window overflow");
  equal(mx::slice(before_wrap,{1,0},{128,512}),mx::slice(wrapped.rows(),{0,0},{127,512}),"layer oldest row eviction mismatch");
  dsv41::SwaProjectionReference projection(catalog);
  equal(projection.forward(token,128).kv,mx::slice(wrapped.rows(),{127,0},{128,512}),"layer newest row mismatch");
  std::cout<<"Real layer 0 attention: chunk/token and state bitwise exact; continuation/fork/reset/rejection passed (not oracle qualification)\n";
 }
 std::cout<<"SWA RoPE analytic/chunk/position and sink/64-row boundary checks passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
