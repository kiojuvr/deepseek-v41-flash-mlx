#include "dsv41/compressed_layer.hpp"
#include "dsv41/attention_telemetry.hpp"
#include "dsv41/reused_layer.hpp"
#include "dsv41/swa_attention.hpp"
#include "dsv41/model_entry.hpp"
#include "dsv41/engram.hpp"
#include "dsv41/execution_policy.hpp"
#include "dsv41/trace.hpp"
#include <iostream>
#include <stdexcept>
#include "dsv41/layer_owner.hpp"
namespace dsv41 {
namespace mx=mlx::core;
namespace {
mx::array grouped_weight(WeightCatalog& c,int layer){
 auto w=c.tensor(("layers."+std::to_string(layer))+".attn.wo_a.weight"),s=c.tensor(("layers."+std::to_string(layer))+".attn.wo_a.scale");
 if(w.dtype!="F8_E4M3"||w.shape!=std::vector<std::uint64_t>{8192,4096}||w.size_bytes!=8192ull*4096||
    s.dtype!="F8_E8M0"||s.shape!=std::vector<std::uint64_t>{256,128}||s.size_bytes!=256*128)
  throw std::runtime_error("unexpected SWA wo_a layout");
 std::vector<std::uint8_t> weights(w.size_bytes),scales(s.size_bytes);
 w.read(0,{reinterpret_cast<std::byte*>(weights.data()),weights.size()});
 s.read(0,{reinterpret_cast<std::byte*>(scales.data()),scales.size()});
 std::vector<std::uint16_t> decoded(weights.size());
 for(std::size_t i=0;i<weights.size();++i){
  auto code=weights[i],scale=scales[(i/4096/32)*128+(i%4096)/32];
  if((code&127)==127||scale==255)throw std::runtime_error("nonfinite SWA wo_a weight");
  decoded[i]=engram_bf16(code,scale);
 }
 return mx::view(mx::array(decoded.begin(),{8,1024,4096},mx::uint16),mx::bfloat16);
}
mx::array sink(WeightCatalog& c,int layer){
 auto t=c.tensor(("layers."+std::to_string(layer))+".attn.attn_sink");
 if(t.dtype!="F32"||t.shape!=std::vector<std::uint64_t>{64}||t.size_bytes!=256)throw std::runtime_error("unexpected SWA sink");
 std::vector<float> v(64);t.read(0,{reinterpret_cast<std::byte*>(v.data()),256});return mx::array(v.begin(),{64},mx::float32);
}
}
namespace {
mx::array norm(WeightCatalog& c,const std::string& name,int size,int layer){
 auto t=c.tensor("layers."+std::to_string(layer)+".attn."+name);
 if(t.dtype!="BF16"||t.shape!=std::vector<std::uint64_t>{std::uint64_t(size)}||t.size_bytes!=2ull*size)throw std::runtime_error("compressed norm layout mismatch");
 std::vector<std::uint16_t> v(size);t.read(0,{reinterpret_cast<std::byte*>(v.data()),t.size_bytes});return mx::view(mx::array(v.begin(),{size},mx::uint16),mx::bfloat16);
}
mx::array main_rows(const GlobalKVState& state,const std::vector<std::int32_t>& selected,int offset){
 std::vector<int> ids;for(auto index:selected){int row=index-offset;if(row<0||std::size_t(row)>=state.rows())throw std::runtime_error("invalid selected global row");ids.push_back(row);}
 auto indices=mx::array(ids.begin(),{int(ids.size())},mx::int32);
 auto p=mx::take(state.main_bytes(),indices,0),s=mx::take(state.main_scales(),indices,0);
 auto low=mx::bitwise_and(p,mx::array(15,mx::uint8)),high=mx::right_shift(p,mx::array(4,mx::uint8));
 auto codes=mx::reshape(mx::stack({low,high},-1),{int(ids.size()),512});
 auto levels=mx::array({0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f});
 auto v=mx::take(levels,mx::astype(mx::bitwise_and(codes,mx::array(7,mx::uint8)),mx::int32));
 v=mx::where(mx::greater_equal(codes,mx::array(8,mx::uint8)),mx::negative(v),v);
 auto exponent=mx::astype(mx::right_shift(s,mx::array(3,mx::uint8)),mx::float32);
 auto mantissa=mx::astype(mx::bitwise_and(s,mx::array(7,mx::uint8)),mx::float32);
 auto scale=mx::where(mx::equal(exponent,mx::array(0.0f)),mx::multiply(mantissa,mx::array(0x1p-9f)),mx::multiply(mx::add(mantissa,mx::array(8.0f)),mx::power(mx::array(2.0f),mx::subtract(exponent,mx::array(10.0f)))));
 return mx::astype(mx::reshape(mx::multiply(mx::reshape(v,{int(ids.size()),32,16}),mx::expand_dims(scale,-1)),{int(ids.size()),512}),mx::bfloat16);
}
mx::array main_rows(const GlobalKVState& state,const mx::array& selected){
 if(selected.dtype()!=mx::int32||selected.ndim()!=1||selected.size()>512)
  throw std::runtime_error("invalid device selected global rows");
 auto p=mx::take(state.main_bytes(),selected,0),s=mx::take(state.main_scales(),selected,0);
 auto low=mx::bitwise_and(p,mx::array(15,mx::uint8)),high=mx::right_shift(p,mx::array(4,mx::uint8));
 auto codes=mx::reshape(mx::stack({low,high},-1),{int(selected.size()),512});
 auto levels=mx::array({0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f});
 auto v=mx::take(levels,mx::astype(mx::bitwise_and(codes,mx::array(7,mx::uint8)),mx::int32));
 v=mx::where(mx::greater_equal(codes,mx::array(8,mx::uint8)),mx::negative(v),v);
 auto exponent=mx::astype(mx::right_shift(s,mx::array(3,mx::uint8)),mx::float32);
 auto mantissa=mx::astype(mx::bitwise_and(s,mx::array(7,mx::uint8)),mx::float32);
 auto scale=mx::where(mx::equal(exponent,mx::array(0.0f)),mx::multiply(mantissa,mx::array(0x1p-9f)),
  mx::multiply(mx::add(mantissa,mx::array(8.0f)),mx::power(mx::array(2.0f),mx::subtract(exponent,mx::array(10.0f)))));
 return mx::astype(mx::reshape(mx::multiply(mx::reshape(v,{int(selected.size()),32,16}),
  mx::expand_dims(scale,-1)),{int(selected.size()),512}),mx::bfloat16);
}
mx::array dense_topk(const std::vector<mx::array>& selected){
 std::vector<mx::array> rows;rows.reserve(selected.size());
 for(const auto& row:selected){
  if(row.dtype()!=mx::int32||row.ndim()!=1||row.size()>512)
   throw std::runtime_error("invalid dense attention selected rows");
  auto padded=row.size()==512?row:mx::concatenate(
   {row,mx::broadcast_to(mx::array(-1,mx::int32),{512-int(row.size())})},0);
  rows.push_back(mx::expand_dims(padded,0));
 }
 return mx::concatenate(rows,0);
}
void report_fixed_tile_rms(const mx::array& candidate,const mx::array& reference,
 const char* label){
 auto c=mx::astype(candidate,mx::float32),r=mx::astype(reference,mx::float32);
 auto difference=mx::subtract(c,r);
 auto relative=mx::sqrt(mx::divide(mx::sum(mx::multiply(difference,difference)),
  mx::maximum(mx::sum(mx::multiply(r,r)),mx::array(1e-30f))));
 auto maximum=mx::max(mx::abs(difference));
 auto mismatches=mx::sum(mx::astype(mx::not_equal(candidate,reference),mx::uint32));
 mx::eval(relative,maximum,mismatches);
 std::cout<<label<<" relative_rms="<<relative.item<float>()
          <<" max_abs="<<maximum.item<float>()
          <<" bit_mismatches="<<mismatches.item<std::uint32_t>()<<std::endl;
}
void report_fixed_tile_mismatch_tokens(const mx::array& candidate,const mx::array& reference,
 const char* label){
 auto token_mask=mx::any(mx::not_equal(candidate,reference),std::vector<int>{1,2});
 const int tokens=candidate.shape(0);
 auto ids=mx::arange(tokens,mx::int32);
 auto count=mx::sum(mx::astype(token_mask,mx::uint32));
 auto first=mx::min(mx::where(token_mask,ids,mx::array(tokens,mx::int32)));
 auto last=mx::max(mx::where(token_mask,ids,mx::array(-1,mx::int32)));
 mx::eval(count,first,last);
 std::cout<<label<<" mismatch_tokens="<<count.item<std::uint32_t>()
          <<" first="<<first.item<std::int32_t>()<<" last="<<last.item<std::int32_t>()<<std::endl;
}
}
CompressedLayerReference::CompressedLayerReference(WeightCatalog& c,int layer):layer_(checked_producer_layer(layer)),ratio_(layer_compress_ratio(layer)),
 qa_(c,("layers."+std::to_string(layer_))+".attn.wq_a"),qb_(c,("layers."+std::to_string(layer_))+".attn.wq_b"),kv_(c,("layers."+std::to_string(layer_))+".attn.wkv"),output_(c,("layers."+std::to_string(layer_))+".attn.wo_b"),
 qnorm_(norm(c,"q_norm.weight",1280,layer_)),kvnorm_(norm(c,"kv_norm.weight",512,layer_)),grouped_(grouped_weight(c,layer_)),sink_(sink(c,layer_)),producer_(c,layer_),index_(c,layer_,layer_==20,layer_>20){
 if(qa_.input_dims()!=5120||qa_.output_dims()!=1280||qb_.input_dims()!=1280||qb_.output_dims()!=32768||kv_.input_dims()!=5120||kv_.output_dims()!=512||output_.input_dims()!=8192||output_.output_dims()!=5120||qa_.bits()!=8||qb_.bits()!=8||kv_.bits()!=8||output_.bits()!=8)throw std::runtime_error("compressed projection layout mismatch");
 mx::eval(grouped_,sink_);
}
void CompressedLayerReference::prepare_chunk(const mx::array& h,CompressedLayerState& state,
 std::uint64_t start) const{
 if(h.dtype()!=mx::bfloat16||h.ndim()!=2||h.shape(0)<1||h.shape(0)>128||h.shape(1)!=5120||
    state.position()!=start||start>=1048576||std::uint64_t(h.shape(0))>1048576-start)
  throw std::runtime_error("invalid compressed attention preparation input/state");
 std::vector<std::uint64_t> positions;positions.reserve(h.shape(0));
 for(int i=0;i<h.shape(0);++i)positions.push_back(start+std::uint64_t(i));
 auto kv=linear_activation_reference(compressed_rope_reference(
  rms_norm_reference(kv_.forward(h),kvnorm_,1e-20f),positions)).decoded;
 auto next=state;
 producer_.append(h,next.global_,start);
 auto window=mx::concatenate({state.window_,kv},0);
 next.window_=mx::slice(window,{std::max(0,window.shape(0)-128),0},{window.shape(0),512});
 next.publication_.reset();
 mx::eval(next.window_);
 state=std::move(next);
}
mx::array CompressedLayerReference::forward(const mx::array& h,CompressedLayerState& state,std::uint64_t start) const{
 return forward_chunk(h,state,start,nullptr);
}
mx::array CompressedLayerReference::forward_chunk(const mx::array& h,CompressedLayerState& state,
 std::uint64_t start,std::vector<SharedAttentionReference>* publications) const{
 if(h.dtype()!=mx::bfloat16||h.ndim()!=2||h.shape(0)<1||h.shape(0)>128||h.shape(1)!=5120||state.position()!=start||start>=1048576||std::uint64_t(h.shape(0))>1048576-start)throw std::runtime_error("invalid compressed attention input/state");
 std::vector<std::uint64_t> positions;positions.reserve(h.shape(0));
 for(int i=0;i<h.shape(0);++i)positions.push_back(start+std::uint64_t(i));
 auto qr=rms_norm_reference(qa_.forward(h),qnorm_,1e-20f);
 auto q=compressed_rope_reference(mx::reshape(qb_.forward(qr),{h.shape(0),64,512}),positions);
 auto kv=linear_activation_reference(compressed_rope_reference(
  rms_norm_reference(kv_.forward(h),kvnorm_,1e-20f),positions)).decoded;
 const bool trace_arithmetic=active_trace_sink()&&layer_==20;
 const std::string trace_prefix=(layer_<20?"encoder.layer":"decoder.layer")+
  std::to_string(layer_)+".";
 if(trace_arithmetic){trace_record(trace_prefix+"attn_qr",qr);trace_record(trace_prefix+"attn_q",q);
  trace_record(trace_prefix+"attn_kv",kv);}
 auto next=state;std::vector<mx::array> projected_rows;projected_rows.reserve(h.shape(0));
 const bool wide_chunk=runtime_wide_attention_enabled()&&publications&&h.shape(0)>1;
 const bool fixed_tile=runtime_fixed_tile_attention_enabled()&&publications&&h.shape(0)>1;
 if(wide_chunk&&fixed_tile)throw std::runtime_error("wide and fixed-tile attention are mutually exclusive");
 const bool packed_chunk=(runtime_packed_chunk_attention_enabled()||wide_chunk||fixed_tile)&&publications&&h.shape(0)>1;
 std::vector<mx::array> packed_rows;if(packed_chunk)packed_rows.reserve(h.shape(0));
 auto all_window=mx::concatenate({state.window_,kv},0);
 { std::lock_guard l(attention_telemetry_mutex()); auto& t=attention_telemetry(); ++t.concat_calls; t.concat_input_bytes+=(state.window_.size()+kv.size())*2; t.concat_output_bytes+=all_window.size()*2; t.cumulative_bytes_copied+=all_window.size()*2; }
 std::vector<SharedAttentionReference> pending_publications;
 if(publications)pending_publications.reserve(h.shape(0));
 auto cache_prefixes=producer_.append_chunk(h,next.global_,start);
 // `forward()` is the diagnostic/reference entry point; production block
 // execution requests every per-token publication through `publications`.
 auto selections=index_.forward_chunk(h,qr,cache_prefixes,start,nullptr,nullptr,publications==nullptr);
 for(int i=0;i<h.shape(0);++i){
  const std::uint64_t pos=start+std::uint64_t(i);auto x=mx::slice(h,{i,0},{i+1,5120});
  auto end=state.window_.shape(0)+i+1; auto begin=std::max(0,end-128);
  auto window=mx::slice(all_window,{begin,0},{end,512}); { std::lock_guard l(attention_telemetry_mutex()); auto& t=attention_telemetry(); t.attention_rows++; t.logical_tokens++;if(!packed_chunk)t.token_serial_attention_calls++; }
  int padding=pos==0?0:128-window.shape(0);
  auto ordered=padding?mx::concatenate({mx::zeros({padding,512},mx::bfloat16),window},0):window;
  int offset=ordered.shape(0);
  auto selection=std::move(selections[i]);
  auto publication=SharedAttentionReference(cache_prefixes[i],selection.device_rows,
   selection.device_candidates,pos,offset,layer_,ratio_,std::move(selection.rows),
   std::move(selection.candidates));
  if(packed_chunk){
   auto row=selection.device_rows;
   packed_rows.push_back(row);
  }else{
   if(selection.device_rows.size())ordered=mx::concatenate(
    {ordered,main_rows(cache_prefixes[i],selection.device_rows)},0);
   auto valid=mx::greater_equal(mx::arange(ordered.shape(0),mx::int32),mx::array(padding));
   auto token_q=mx::reshape(mx::slice(q,{i,0,0},{i+1,64,512}),{64,512});
   auto o=swa_attention_masked_reference(token_q,ordered,sink_,valid);
   if(trace_arithmetic)trace_record(trace_prefix+"attn_core",mx::reshape(o,{1,64,512}));
   o=compressed_rope_reference(mx::reshape(o,{1,64,512}),std::span(&pos,1),true);
   if(trace_arithmetic)trace_record(trace_prefix+"attn_inverse_rope",o);
   auto projected=mx::matmul(mx::reshape(o,{8,1,4096}),mx::transpose(grouped_,{0,2,1}));
   auto projected_row=mx::reshape(projected,{1,8192});
   projected_rows.push_back(projected_row);
  }
  next.window_=window;next.publication_=publication;
  if(publications)pending_publications.push_back(std::move(publication));
 }
 if(packed_chunk){
  std::vector<mx::array> attention_groups;
  if(wide_chunk){
   auto plan=dense_topk(packed_rows);
   for(auto& publication:pending_publications)
    publication.publish_chunk_plan(layer_,plan,start,h.shape(0));
   auto o=swa_wide_attention_chunk(q,all_window,cache_prefixes.back().main_bytes(),
    cache_prefixes.back().main_scales(),plan,sink_,start,ratio_);
   attention_groups.push_back(o);
  }else if(fixed_tile){
   auto plan=dense_topk(packed_rows);
   for(auto& publication:pending_publications)
    publication.publish_chunk_plan(layer_,plan,start,h.shape(0));
   auto fixed_work=swa_packed_attention_work_list(all_window,cache_prefixes.back().main_bytes(),
    cache_prefixes.back().main_scales(),plan,start,ratio_,512,true);
   if(trace_arithmetic)trace_record(trace_prefix+"attn_widths",fixed_work.widths);
   auto fixed_output=runtime_ragged_tail_qk_enabled()?swa_attention_fixed_tile_core(
    q,fixed_work,sink_,runtime_ragged_tail_av_enabled()):
    swa_attention_masked_chunk(q,fixed_work.ordered,sink_,fixed_work.valid);
   attention_groups.push_back(fixed_output);
   if(trace_arithmetic&&runtime_ragged_tail_qk_enabled()){
    auto diagnostic=swa_attention_fixed_tile_width_one_diagnostics(
     q,fixed_work,sink_,runtime_ragged_tail_av_enabled());
    trace_record(trace_prefix+"attn_core_native_width1_qk",diagnostic.native_qk);
    trace_record(trace_prefix+"attn_core_native_width1_av",diagnostic.native_av);
    // The first request token owns one local row at slot 127 and one selected
    // pooled row at slot 128.  Compact only those two live rows into one
    // reduction block to distinguish fixed-schedule online-softmax topology
    // from the already isolated width-one QK and AV kernels.
    if(start==0&&h.shape(0)>1&&packed_rows.front().size()==1){
     auto compact_rows=mx::concatenate({
      mx::slice(fixed_work.ordered,{0,127,0},{1,128,512}),
      mx::slice(fixed_work.ordered,{0,128,0},{1,129,512})},1);
     auto compact_valid=mx::concatenate({
      mx::slice(fixed_work.valid,{0,127},{1,128}),
      mx::slice(fixed_work.valid,{0,128},{1,129})},1);
     auto compact_first=swa_attention_masked_chunk(
      mx::slice(q,{0,0,0},{1,64,512}),compact_rows,sink_,compact_valid);
     auto compact_candidate=mx::concatenate({compact_first,
      mx::slice(fixed_output,{1,0,0},{h.shape(0),64,512})},0);
     trace_record(trace_prefix+"attn_core_compact_token0",compact_candidate);
    }
   }
   if(runtime_fixed_tile_attention_diagnostics_enabled()){
    const auto production_telemetry=read_attention_telemetry();
    std::vector<mx::array> exact_outputs,content_outputs,qk_padded_outputs,av_padded_outputs;
    auto run_group=[&](int first,int end,int selected_count){
     auto local=mx::slice(all_window,{0,0},{state.window_.shape(0)+end,512});
     mx::array rows=mx::broadcast_to(mx::array(-1,mx::int32),{end-first,1});
     if(selected_count){
      std::vector<mx::array> group_rows;group_rows.reserve(end-first);
      for(int i=first;i<end;++i)group_rows.push_back(mx::expand_dims(packed_rows[i],0));
      rows=mx::concatenate(group_rows,0);
     }
     auto exact_work=swa_packed_attention_work_list(local,cache_prefixes.back().main_bytes(),
      cache_prefixes.back().main_scales(),rows,start+first,ratio_,selected_count);
     auto group_q=mx::slice(q,{first,0,0},{end,64,512});
     exact_outputs.push_back(swa_attention_masked_chunk(
      group_q,exact_work.ordered,sink_,exact_work.valid));
     AttentionTailDiagnostics tails{mx::array(0.0f),mx::array(0.0f)};
     try{tails=swa_attention_tail_diagnostics(
      group_q,exact_work.ordered,sink_,exact_work.valid);}
     catch(const std::exception& e){throw std::runtime_error("tail attribution group first="+
      std::to_string(first)+" end="+std::to_string(end)+" selected="+
      std::to_string(selected_count)+": "+e.what());}
     qk_padded_outputs.push_back(tails.padded_qk);
     av_padded_outputs.push_back(tails.padded_av);
     const int raw_width=start+std::uint64_t(first)==0?1:128;
     const int dense_first=raw_width==1?127:0;
     auto dense_local=mx::slice(fixed_work.ordered,{first,dense_first,0},
                                {end,dense_first+raw_width,512});
     auto dense_valid=mx::slice(fixed_work.valid,{first,dense_first},
                                {end,dense_first+raw_width});
     if(selected_count){
      dense_local=mx::concatenate({dense_local,mx::slice(fixed_work.ordered,
       {first,128,0},{end,128+selected_count,512})},1);
      dense_valid=mx::concatenate({dense_valid,mx::slice(fixed_work.valid,
       {first,128},{end,128+selected_count})},1);
     }
     content_outputs.push_back(swa_attention_masked_chunk(group_q,dense_local,sink_,dense_valid));
    };
    for(int first=0;first<h.shape(0);){
     const int raw_width=start+std::uint64_t(first)==0?1:128;
     const int selected_count=packed_rows[first].size();int end=first+1;
     while(end<h.shape(0)&&(start+std::uint64_t(end)==0?1:128)==raw_width&&
           int(packed_rows[end].size())==selected_count)++end;
     run_group(first,end,selected_count);first=end;
    }
    auto exact=exact_outputs.size()==1?exact_outputs.front():mx::concatenate(exact_outputs,0);
    auto content=content_outputs.size()==1?content_outputs.front():mx::concatenate(content_outputs,0);
    auto qk_padded=qk_padded_outputs.size()==1?qk_padded_outputs.front():mx::concatenate(qk_padded_outputs,0);
    auto av_padded=av_padded_outputs.size()==1?av_padded_outputs.front():mx::concatenate(av_padded_outputs,0);
    report_fixed_tile_rms(content,exact,"fixed-tile producer dense-content/exact-shape");
    report_fixed_tile_rms(qk_padded,exact,"fixed-tile producer padded-tail-QK/exact-shape");
    report_fixed_tile_rms(av_padded,exact,"fixed-tile producer padded-tail-AV/exact-shape");
    report_fixed_tile_mismatch_tokens(av_padded,exact,"fixed-tile producer padded-tail-AV tokens");
    report_fixed_tile_rms(fixed_output,exact,"fixed-tile producer dense-reduction/exact-shape");
    // Qualification-only exact/content graphs must not appear in production
    // dispatch telemetry.
    { std::lock_guard l(attention_telemetry_mutex());attention_telemetry()=production_telemetry; }
   }
  }else{
  auto run_group=[&](int first,int end,int selected_count){
   auto local=mx::slice(all_window,{0,0},{state.window_.shape(0)+end,512});
   mx::array rows=mx::broadcast_to(mx::array(-1,mx::int32),{end-first,1});
   if(selected_count){
    std::vector<mx::array> group_rows;group_rows.reserve(end-first);
    for(int i=first;i<end;++i)group_rows.push_back(mx::expand_dims(packed_rows[i],0));
    rows=mx::concatenate(group_rows,0);
   }
   auto work=swa_packed_attention_work_list(local,cache_prefixes.back().main_bytes(),
    cache_prefixes.back().main_scales(),rows,start+first,ratio_,selected_count);
   attention_groups.push_back(swa_attention_masked_chunk(
    mx::slice(q,{first,0,0},{end,64,512}),work.ordered,sink_,work.valid));
  };
  for(int first=0;first<h.shape(0);){
   const int raw_width=start+std::uint64_t(first)==0?1:128;
   const int selected_count=packed_rows[first].size();int end=first+1;
   while(end<h.shape(0)&&(start+std::uint64_t(end)==0?1:128)==raw_width&&
         int(packed_rows[end].size())==selected_count)++end;
   run_group(first,end,selected_count);first=end;
  }
  }
  auto o=attention_groups.size()==1?attention_groups.front():mx::concatenate(attention_groups,0);
  if(trace_arithmetic)trace_record(trace_prefix+"attn_core",o);
  o=compressed_rope_reference(o,positions,true);
  if(trace_arithmetic)trace_record(trace_prefix+"attn_inverse_rope",o);
  auto grouped=mx::transpose(grouped_,{0,2,1});
  for(int i=0;i<h.shape(0);++i){
   auto token_o=mx::reshape(mx::slice(o,{i,0,0},{i+1,64,512}),{8,1,4096});
   projected_rows.push_back(mx::reshape(mx::matmul(token_o,grouped),{1,8192}));
  }
  { std::lock_guard l(attention_telemetry_mutex());
    if(wide_chunk)++attention_telemetry().wide_attention_calls;
    else if(fixed_tile)++attention_telemetry().fixed_tile_attention_calls;
    else ++attention_telemetry().packed_chunk_attention_calls; }
 }
 auto projected=projected_rows.size()==1?projected_rows.front():mx::concatenate(projected_rows,0);
 if(trace_arithmetic)trace_record(trace_prefix+"attn_grouped",projected);
 next.window_=mx::slice(all_window,{std::max(0,all_window.shape(0)-128),0},{all_window.shape(0),512});
 auto result=output_.forward(projected);
 if(trace_arithmetic)trace_record(trace_prefix+"attn_linear",result);
 if(runtime_layer_finite_checks_enabled()){
  auto ok=mx::logical_and(mx::all(mx::isfinite(result)),mx::all(mx::isfinite(next.window_)));
  mx::eval(result,next.window_,ok);if(!ok.item<bool>())throw std::runtime_error("nonfinite compressed attention output/state");
 }
 if(publications)*publications=std::move(pending_publications);
 state=std::move(next);return result;
}
ReusedLayerReference::ReusedLayerReference(WeightCatalog& c,int layer):layer_(checked_reused_layer(layer)),ratio_(layer_compress_ratio(layer_)),is_index_source_(is_index_source_layer(layer_)),uses_candidates_(layer_>20),qa_(c,("layers."+std::to_string(layer_))+".attn.wq_a"),qb_(c,("layers."+std::to_string(layer_))+".attn.wq_b"),kv_(c,("layers."+std::to_string(layer_))+".attn.wkv"),output_(c,("layers."+std::to_string(layer_))+".attn.wo_b"),qnorm_(norm(c,"q_norm.weight",1280,layer_)),kvnorm_(norm(c,"kv_norm.weight",512,layer_)),grouped_(grouped_weight(c,layer_)),sink_(sink(c,layer_)){
 if(qa_.input_dims()!=5120||qa_.output_dims()!=1280||qb_.input_dims()!=1280||qb_.output_dims()!=32768||kv_.input_dims()!=5120||kv_.output_dims()!=512||output_.input_dims()!=8192||output_.output_dims()!=5120||qa_.bits()!=8||qb_.bits()!=8||kv_.bits()!=8||output_.bits()!=8)throw std::runtime_error("compressed projection layout mismatch");
 if(is_index_source_)index_=std::make_unique<IndexQueryReference>(c,layer_,false,uses_candidates_);
 mx::eval(grouped_,sink_);
}
mx::array ReusedLayerReference::forward(const mx::array& x,ReusedLayerState& state,SharedAttentionReference& publication,std::uint64_t pos) const{
 if(x.dtype()!=mx::bfloat16||x.shape()!=mx::Shape({1,5120})||state.position()!=pos||pos>=1048576)throw std::runtime_error("invalid reuse layer input/state");
 int offset=pos==0?1:128;
 if(runtime_layer_finite_checks_enabled()){
  auto finite=mx::all(mx::isfinite(x));mx::eval(finite);if(!finite.item<bool>())throw std::runtime_error("nonfinite reuse layer input");
 }
 auto qr=rms_norm_reference(qa_.forward(x),qnorm_,1e-20f);
 const bool trace_arithmetic=active_trace_sink()&&(layer_==3||layer_==4||layer_==21);
 const std::string trace_prefix=(layer_<20?"encoder.layer":"decoder.layer")+
  std::to_string(layer_)+".";
 if(trace_arithmetic)trace_record(trace_prefix+"attn_qr",qr);
 std::vector<std::int32_t> selected;
 if(is_index_source_){
  auto selection=index_->forward(x,qr,publication.cache(),pos,offset,uses_candidates_?&publication.candidates():nullptr);
  publication.republish(layer_,selection.rows,std::move(selection.candidates));
  selected=publication.indices(layer_,pos,offset);
 }else{
  selected=publication.indices(layer_,pos,offset);
 }
 auto q=compressed_rope_reference(mx::reshape(qb_.forward(qr),{1,64,512}),std::span(&pos,1));
 auto kv=linear_activation_reference(compressed_rope_reference(rms_norm_reference(kv_.forward(x),kvnorm_,1e-20f),std::span(&pos,1))).decoded;
 if(trace_arithmetic){trace_record(trace_prefix+"attn_q",q);trace_record(trace_prefix+"attn_kv",kv);}
 auto window=mx::concatenate({state.window_,kv},0); { std::lock_guard l(attention_telemetry_mutex()); auto& t=attention_telemetry(); ++t.concat_calls; t.concat_input_bytes+=(state.window_.size()+kv.size())*2; t.concat_output_bytes+=window.size()*2; t.cumulative_bytes_copied+=window.size()*2; }
 if(window.shape(0)>128)window=mx::slice(window,{window.shape(0)-128,0},{window.shape(0),512});
 int padding=offset-window.shape(0);
 auto ordered=padding?mx::concatenate({mx::zeros({padding,512},mx::bfloat16),window},0):window;
 if(!selected.empty())ordered=mx::concatenate({ordered,main_rows(publication.cache(),selected,offset)},0);
 auto valid=mx::greater_equal(mx::arange(ordered.shape(0),mx::int32),mx::array(padding));
 auto o=swa_attention_masked_reference(mx::reshape(q,{64,512}),ordered,sink_,valid);
 if(trace_arithmetic)trace_record(trace_prefix+"attn_core",mx::reshape(o,{1,64,512}));
 o=compressed_rope_reference(mx::reshape(o,{1,64,512}),std::span(&pos,1),true);
 if(trace_arithmetic)trace_record(trace_prefix+"attn_inverse_rope",o);
 auto projected=mx::matmul(mx::reshape(o,{8,1,4096}),mx::transpose(grouped_,{0,2,1}));
 if(trace_arithmetic)trace_record(trace_prefix+"attn_grouped",mx::reshape(projected,{1,8192}));
 auto y=output_.forward(mx::reshape(projected,{1,8192}));
 if(trace_arithmetic)trace_record(trace_prefix+"attn_linear",y);
 if(runtime_layer_finite_checks_enabled()){
  auto ok=mx::logical_and(mx::all(mx::isfinite(y)),mx::all(mx::isfinite(window)));mx::eval(y,window,ok);
  if(!ok.item<bool>())throw std::runtime_error("nonfinite reuse layer output/state");
 }
 state.window_=window;state.position_=pos+1;return y;
}
ReusedLayerState ReusedLayerReference::seed_window(const mx::array& x,std::uint64_t start) const{
 if(x.dtype()!=mx::bfloat16||x.ndim()!=2||x.shape(0)<1||x.shape(0)>128||x.shape(1)!=5120||
    start>=1048576||std::uint64_t(x.shape(0))>1048576-start)
  throw std::runtime_error("invalid reused attention seed input");
 std::vector<std::uint64_t> positions;positions.reserve(x.shape(0));
 for(int i=0;i<x.shape(0);++i)positions.push_back(start+std::uint64_t(i));
 auto kv=linear_activation_reference(compressed_rope_reference(
  rms_norm_reference(kv_.forward(x),kvnorm_,1e-20f),positions)).decoded;
 ReusedLayerState seeded;
 seeded.window_=kv.shape(0)>128?mx::slice(kv,{kv.shape(0)-128,0},{kv.shape(0),512}):kv;
 seeded.position_=start+x.shape(0);
 mx::eval(seeded.window_);
 return seeded;
}
mx::array ReusedLayerReference::forward_chunk(const mx::array& x,ReusedLayerState& state,
 std::vector<SharedAttentionReference>& publications,std::uint64_t start) const{
 if(x.dtype()!=mx::bfloat16||x.ndim()!=2||x.shape(0)<1||x.shape(0)>128||x.shape(1)!=5120||
    state.position()!=start||publications.size()!=std::size_t(x.shape(0))||start>=1048576||
    std::uint64_t(x.shape(0))>1048576-start)
  throw std::runtime_error("invalid reuse layer chunk input/state");
 if(runtime_layer_finite_checks_enabled()){
  auto finite=mx::all(mx::isfinite(x));mx::eval(finite);
  if(!finite.item<bool>())throw std::runtime_error("nonfinite reuse layer chunk input");
 }
 std::vector<std::uint64_t> positions;positions.reserve(x.shape(0));
 for(int i=0;i<x.shape(0);++i)positions.push_back(start+std::uint64_t(i));
 auto qr=rms_norm_reference(qa_.forward(x),qnorm_,1e-20f);
 auto q=compressed_rope_reference(mx::reshape(qb_.forward(qr),{x.shape(0),64,512}),positions);
 auto kv=linear_activation_reference(compressed_rope_reference(
  rms_norm_reference(kv_.forward(x),kvnorm_,1e-20f),positions)).decoded;
 const bool trace_arithmetic=active_trace_sink()&&(layer_==3||layer_==4||layer_==21);
 const std::string trace_prefix=(layer_<20?"encoder.layer":"decoder.layer")+
  std::to_string(layer_)+".";
 if(trace_arithmetic){trace_record(trace_prefix+"attn_qr",qr);trace_record(trace_prefix+"attn_q",q);
  trace_record(trace_prefix+"attn_kv",kv);}
 auto next=state;std::vector<mx::array> projected_rows;projected_rows.reserve(x.shape(0));
 auto all_window=mx::concatenate({state.window_,kv},0);
 std::vector<IndexSelection> selections;
 if(is_index_source_){
  std::vector<GlobalKVState> caches;std::vector<std::vector<std::uint8_t>> candidates;
  std::vector<mx::array> device_candidates;
  caches.reserve(publications.size());candidates.reserve(publications.size());
  device_candidates.reserve(publications.size());
  for(const auto& publication:publications){
   caches.push_back(publication.cache());
   if(uses_candidates_){candidates.push_back(publication.candidates());
    device_candidates.push_back(publication.device_candidates());}
  }
  selections=index_->forward_chunk(x,qr,caches,start,uses_candidates_?&candidates:nullptr,
                                   uses_candidates_?&device_candidates:nullptr);
 }
 std::vector<mx::array> selected_rows,windows;selected_rows.reserve(x.shape(0));windows.reserve(x.shape(0));
 for(int i=0;i<x.shape(0);++i){
  const std::uint64_t pos=start+std::uint64_t(i);const int offset=pos==0?1:128;
  if(is_index_source_){
   auto selection=std::move(selections[i]);
   publications[i].republish(layer_,selection.device_rows,selection.device_candidates,
    std::move(selection.rows),std::move(selection.candidates));
  }
  auto selected=publications[i].device_indices(layer_,pos,offset);
  selected_rows.push_back(selected);
  auto end=state.window_.shape(0)+i+1; auto begin=std::max(0,end-128);
  windows.push_back(mx::slice(all_window,{begin,0},{end,512}));
 }
 { std::lock_guard l(attention_telemetry_mutex());auto& t=attention_telemetry();
   t.logical_tokens+=x.shape(0);t.attention_rows+=x.shape(0); }
 if((runtime_packed_chunk_attention_enabled()||runtime_wide_attention_enabled()||
     runtime_fixed_tile_attention_enabled())&&x.shape(0)>1){
  const bool wide_chunk=runtime_wide_attention_enabled();
  const bool fixed_tile=runtime_fixed_tile_attention_enabled();
  if(wide_chunk&&fixed_tile)throw std::runtime_error("wide and fixed-tile attention are mutually exclusive");
  const auto& pooled=publications.back().cache();
  std::vector<mx::array> attention_groups;
  if(wide_chunk||fixed_tile){
   mx::array plan=mx::array(0);
   if(is_index_source_){
   plan=dense_topk(selected_rows);
   for(auto& publication:publications)
     publication.publish_chunk_plan(layer_,plan,start,x.shape(0));
   }else if(publications.front().chunk_plan_matches(layer_,start,x.shape(0)))
    plan=publications.front().device_chunk_indices(layer_,start,x.shape(0));
   else
    // Deferred decoder suffixes shift the chunk frontier by 127 rows per
    // layer. Repack immutable device selections at the new boundary without
    // recomputing the index query or reading rows back to the host.
    plan=dense_topk(selected_rows);
   if(wide_chunk)attention_groups.push_back(swa_wide_attention_chunk(q,all_window,pooled.main_bytes(),
    pooled.main_scales(),plan,sink_,start,ratio_));
   else{
    auto work=swa_packed_attention_work_list(all_window,pooled.main_bytes(),pooled.main_scales(),
                                             plan,start,ratio_,512,true);
    if(trace_arithmetic)trace_record(trace_prefix+"attn_widths",work.widths);
    if(runtime_ragged_tail_qk_enabled()){
     attention_groups.push_back(swa_attention_fixed_tile_core(
      q,work,sink_,runtime_ragged_tail_av_enabled()));
     if(trace_arithmetic&&layer_==3){
      auto diagnostic=swa_attention_fixed_tile_width_one_diagnostics(
       q,work,sink_,runtime_ragged_tail_av_enabled());
      trace_record(trace_prefix+"attn_core_native_width1_qk",diagnostic.native_qk);
      trace_record(trace_prefix+"attn_core_native_width1_av",diagnostic.native_av);
     }
    }else attention_groups.push_back(swa_attention_masked_chunk(q,work.ordered,sink_,work.valid));
   }
  }else{
  auto run_group=[&](int first,int end,int selected_count){
   auto local=mx::slice(all_window,{0,0},{state.window_.shape(0)+end,512});
   mx::array rows=mx::broadcast_to(mx::array(-1,mx::int32),{end-first,1});
   if(selected_count){
    std::vector<mx::array> group_rows;group_rows.reserve(end-first);
    for(int i=first;i<end;++i)group_rows.push_back(mx::expand_dims(selected_rows[i],0));
    rows=mx::concatenate(group_rows,0);
   }
   auto work=swa_packed_attention_work_list(local,pooled.main_bytes(),pooled.main_scales(),
                                            rows,start+first,ratio_,selected_count);
   attention_groups.push_back(swa_attention_masked_chunk(
    mx::slice(q,{first,0,0},{end,64,512}),work.ordered,sink_,work.valid));
  };
  for(int first=0;first<x.shape(0);){
   const int raw_width=start+std::uint64_t(first)==0?1:128;
   const int selected_count=selected_rows[first].size();int end=first+1;
   while(end<x.shape(0)&&(start+std::uint64_t(end)==0?1:128)==raw_width&&
         int(selected_rows[end].size())==selected_count)++end;
   run_group(first,end,selected_count);first=end;
  }
  }
  auto o=attention_groups.size()==1?attention_groups.front():mx::concatenate(attention_groups,0);
  if(trace_arithmetic)trace_record(trace_prefix+"attn_core",o);
  o=compressed_rope_reference(o,positions,true);
  if(trace_arithmetic)trace_record(trace_prefix+"attn_inverse_rope",o);
  auto grouped=mx::transpose(grouped_,{0,2,1});
  for(int i=0;i<x.shape(0);++i){
   auto token_o=mx::reshape(mx::slice(o,{i,0,0},{i+1,64,512}),{8,1,4096});
   projected_rows.push_back(mx::reshape(mx::matmul(token_o,grouped),{1,8192}));
  }
  { std::lock_guard l(attention_telemetry_mutex());
    if(wide_chunk)++attention_telemetry().wide_attention_calls;
    else if(fixed_tile)++attention_telemetry().fixed_tile_attention_calls;
    else ++attention_telemetry().packed_chunk_attention_calls; }
 }else if(runtime_chunk_attention_enabled()&&x.shape(0)>1){
  auto grouped=mx::transpose(grouped_,{0,2,1});
  for(int first=0;first<x.shape(0);){
   const auto first_pos=start+std::uint64_t(first);
   const int raw_width=first_pos==0?1:128;
   const int selected_count=int(selected_rows[first].size());
   int end=first+1;
   while(end<x.shape(0)&&(start+std::uint64_t(end)==0?1:128)==raw_width&&
         int(selected_rows[end].size())==selected_count)++end;
   std::vector<mx::array> ordered_rows,valid_rows;ordered_rows.reserve(end-first);valid_rows.reserve(end-first);
   for(int i=first;i<end;++i){
    const int padding=raw_width-windows[i].shape(0);
    auto raw=padding?mx::concatenate({mx::zeros({padding,512},mx::bfloat16),windows[i]},0):windows[i];
    auto global=selected_count?main_rows(publications[i].cache(),selected_rows[i]):mx::zeros({0,512},mx::bfloat16);
    auto ordered=selected_count?mx::concatenate({raw,global},0):raw;
    auto valid=mx::concatenate({mx::greater_equal(mx::arange(raw_width,mx::int32),mx::array(padding)),
                                mx::ones({selected_count},mx::bool_)},0);
    ordered_rows.push_back(mx::expand_dims(ordered,0));valid_rows.push_back(mx::expand_dims(valid,0));
   }
   auto ordered=mx::concatenate(ordered_rows,0),valid=mx::concatenate(valid_rows,0);
   auto group_q=mx::slice(q,{first,0,0},{end,64,512});
   auto o=swa_attention_masked_chunk(group_q,ordered,sink_,valid);
   o=compressed_rope_reference(o,std::span(positions).subspan(first,end-first),true);
   for(int i=0;i<end-first;++i){
    auto token_o=mx::reshape(mx::slice(o,{i,0,0},{i+1,64,512}),{8,1,4096});
    projected_rows.push_back(mx::reshape(mx::matmul(token_o,grouped),{1,8192}));
   }
   { std::lock_guard l(attention_telemetry_mutex());++attention_telemetry().chunk_attention_calls; }
   first=end;
  }
 }else for(int i=0;i<x.shape(0);++i){
  const std::uint64_t pos=start+std::uint64_t(i);const int offset=pos==0?1:128;
  auto selected=selected_rows[i];auto window=windows[i];
  const int padding=offset-window.shape(0);
  auto ordered=padding?mx::concatenate({mx::zeros({padding,512},mx::bfloat16),window},0):window;
  if(selected.size())ordered=mx::concatenate({ordered,main_rows(publications[i].cache(),selected)},0);
  auto valid=mx::greater_equal(mx::arange(ordered.shape(0),mx::int32),mx::array(padding));
  auto token_q=mx::reshape(mx::slice(q,{i,0,0},{i+1,64,512}),{64,512});
  auto o=swa_attention_masked_reference(token_q,ordered,sink_,valid);
  o=compressed_rope_reference(mx::reshape(o,{1,64,512}),std::span(&pos,1),true);
  auto projected=mx::matmul(mx::reshape(o,{8,1,4096}),mx::transpose(grouped_,{0,2,1}));
  projected_rows.push_back(mx::reshape(projected,{1,8192}));
  { std::lock_guard l(attention_telemetry_mutex());++attention_telemetry().token_serial_attention_calls; }
 }
 auto projected=projected_rows.size()==1?projected_rows.front():mx::concatenate(projected_rows,0);
 if(trace_arithmetic)trace_record(trace_prefix+"attn_grouped",projected);
 next.window_=mx::slice(all_window,{std::max(0,all_window.shape(0)-128),0},{all_window.shape(0),512});
 next.position_=start+x.shape(0);
 auto result=output_.forward(projected);
 if(trace_arithmetic)trace_record(trace_prefix+"attn_linear",result);
 if(runtime_layer_finite_checks_enabled()){
  auto ok=mx::logical_and(mx::all(mx::isfinite(result)),mx::all(mx::isfinite(next.window_)));
  mx::eval(result,next.window_,ok);if(!ok.item<bool>())throw std::runtime_error("nonfinite reuse layer chunk output/state");
 }
 state=std::move(next);return result;
}
}
