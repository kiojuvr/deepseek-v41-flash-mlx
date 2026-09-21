// Checkpoint-free micro-probe for the one-token routed-expert path.
// Reproduces the exact shapes/dtypes of PackedExpertBank::forward_batch_expert_major
// with synthetic mxfp4 weights, so the per-op cost can be isolated without a
// 340 GB model load. Diagnostic only; it is not runtime qualification.
#include "dsv41/linear.hpp"
#include <mlx/mlx.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace mx=mlx::core;
using Clock=std::chrono::steady_clock;

static double elapsed(Clock::time_point start){
 return std::chrono::duration<double>(Clock::now()-start).count();
}
static double median(std::vector<double> values){
 std::sort(values.begin(),values.end());return values[values.size()/2];
}
template <typename Fn>
static double time_ms(Fn&& fn,int iterations){
 fn();fn();std::vector<double> samples; // warm up
 for(int i=0;i<iterations;++i){const auto started=Clock::now();fn();samples.push_back(elapsed(started)*1000.0);}
 return median(std::move(samples));
}

static const char* reduce_source=R"reduce(uint element = thread_position_in_grid.x;
uint total = params[0] * 5120;
if (element >= total) return;
uint token = element / 5120;
uint column = element - token * 5120;
float sum = 0.0f;
for (uint rank = 0; rank < 6; ++rank) {
    uint assignment = reduction_slots[token * 6 + rank];
    sum += float(weighted[assignment * 5120 + column]);
}
accumulated[element] = sum;
)reduce";

int main(int argc,char** argv){try{
 mx::set_default_device(mx::Device::gpu);
 const int experts=384;
 const std::size_t footprint_gib=argc>1?std::stoull(argv[1]):0;
 // Non-zero packed weights and scales to rule out a zeros fast path.
 auto w1=mx::full({experts,2304,640},mx::array(std::uint32_t{0x11111111u}),mx::uint32);
 auto s1=mx::full({experts,2304,160},mx::array(std::uint8_t{127}),mx::uint8);
 auto w3=mx::full({experts,2304,640},mx::array(std::uint32_t{0x22222222u}),mx::uint32);
 auto s3=mx::full({experts,2304,160},mx::array(std::uint8_t{127}),mx::uint8);
 auto w2=mx::full({experts,5120,288},mx::array(std::uint32_t{0x33333333u}),mx::uint32);
 auto s2=mx::full({experts,5120,72},mx::array(std::uint8_t{127}),mx::uint8);
 auto x=mx::astype(mx::reshape(mx::sin(mx::arange(5120,mx::float32)),{1,5120}),mx::bfloat16);
 auto ids=mx::array(std::vector<std::uint32_t>{0,1,2,3,4,5}.data(),{6},mx::uint32);
 mx::eval(w1,s1,w3,s3,w2,s2,x,ids);
 // Optional dummy footprint to mimic the 271 GB resident atlas and its effect
 // on page tables / allocator pools. Kept out of the timed regions.
 std::vector<mx::array> ballast;
 if(footprint_gib>0){
  const std::size_t bytes=footprint_gib<<30;
  ballast.push_back(mx::zeros({static_cast<int>(bytes/4)},mx::float32));
  mx::eval(ballast.back());
  std::printf("ballast footprint %.0f GiB\n",double(bytes)/1073741824.0);
 }

 // Cold vs warm latency of the custom FP8 quant/decode metal kernel. A large
 // cold cost means the in-model slowdown is JIT compilation, not execution.
 {
  const auto t0=Clock::now();
  auto a=dsv41::linear_activation_reference(x).decoded;mx::eval(a);mx::synchronize();
  const auto t1=Clock::now();
  auto b=dsv41::linear_activation_reference(x).decoded;mx::eval(b);mx::synchronize();
  const auto t2=Clock::now();
  std::printf("FP8 quant cold %.3f ms, warm %.3f ms\n",
   elapsed(t0)*1000.0,elapsed(t1)*1000.0);
 }

 const int iters=100;
 const double quant_ms=time_ms([&]{
  auto a=dsv41::linear_activation_reference(x);mx::eval(a.decoded);mx::synchronize();
 },iters);

 const double gateup_ms=time_ms([&]{
  auto a=dsv41::linear_activation_reference(x).decoded;
  auto gate=mx::gather_qmm(mx::reshape(a,{1,1,5120}),w1,s1,std::nullopt,ids,ids,true,32,4,"mxfp4",false,mx::Device::gpu);
  auto up=mx::gather_qmm(mx::reshape(a,{1,1,5120}),w3,s3,std::nullopt,ids,ids,true,32,4,"mxfp4",false,mx::Device::gpu);
  mx::eval(gate,up);mx::synchronize();
 },iters);

 auto gate=mx::gather_qmm(mx::reshape(dsv41::linear_activation_reference(x).decoded,{1,1,5120}),w1,s1,std::nullopt,ids,ids,true,32,4,"mxfp4",false,mx::Device::gpu);
 auto up=mx::gather_qmm(mx::reshape(dsv41::linear_activation_reference(x).decoded,{1,1,5120}),w3,s3,std::nullopt,ids,ids,true,32,4,"mxfp4",false,mx::Device::gpu);
 auto g=mx::clip(mx::astype(gate,mx::float32),mx::array(-1e30f),mx::array(10.0f));
 auto u=mx::clip(mx::astype(up,mx::float32),mx::array(-10.0f),mx::array(10.0f));
 auto activation=mx::astype(mx::multiply(mx::multiply(g,mx::sigmoid(g)),u),mx::bfloat16);
 mx::eval(activation);

 const double swiglu_ms=time_ms([&]{
  auto a=mx::astype(mx::multiply(mx::multiply(g,mx::sigmoid(g)),u),mx::bfloat16);mx::eval(a);mx::synchronize();
 },iters);

 const double down_quant_ms=time_ms([&]{
  auto a=dsv41::linear_activation_reference(mx::reshape(activation,{6,2304})).decoded;mx::eval(a);mx::synchronize();
 },iters);

 const double down_ms=time_ms([&]{
  auto a=dsv41::linear_activation_reference(mx::reshape(activation,{6,2304})).decoded;
  auto d=mx::gather_qmm(mx::reshape(a,{6,1,2304}),w2,s2,std::nullopt,ids,ids,true,32,4,"mxfp4",false,mx::Device::gpu);
  mx::eval(d);mx::synchronize();
 },iters);

 auto flat_rhs=mx::array(std::vector<std::uint32_t>{5,4,3,2,1,0}.data(),{6},mx::uint32);
 const double prep_ms=time_ms([&]{
  auto order=mx::argsort(flat_rhs);auto rhs=mx::take(flat_rhs,order);mx::eval(rhs);mx::synchronize();
 },iters);

 const double full_ms=time_ms([&]{
  auto a=dsv41::linear_activation_reference(x).decoded;
  auto gt=mx::gather_qmm(mx::reshape(a,{1,1,5120}),w1,s1,std::nullopt,ids,ids,true,32,4,"mxfp4",false,mx::Device::gpu);
  auto ut=mx::gather_qmm(mx::reshape(a,{1,1,5120}),w3,s3,std::nullopt,ids,ids,true,32,4,"mxfp4",false,mx::Device::gpu);
  auto gg=mx::clip(mx::astype(gt,mx::float32),mx::array(-1e30f),mx::array(10.0f));
  auto uu=mx::clip(mx::astype(ut,mx::float32),mx::array(-10.0f),mx::array(10.0f));
  auto act=mx::astype(mx::multiply(mx::multiply(gg,mx::sigmoid(gg)),uu),mx::bfloat16);
  auto di=dsv41::linear_activation_reference(mx::reshape(act,{6,2304})).decoded;
  auto dn=mx::gather_qmm(mx::reshape(di,{6,1,2304}),w2,s2,std::nullopt,ids,ids,true,32,4,"mxfp4",false,mx::Device::gpu);
  mx::eval(dn);mx::synchronize();
 },iters);

 // Full routed path including route weighting and the device reduce kernel,
 // matching PackedExpertBank::forward_batch_expert_major at tokens=1.
 auto reduce_kernel=mx::fast::metal_kernel("dsv41_route_reduce_expert_major",
  {"weighted","reduction_slots","params"},{"accumulated"},reduce_source);
 auto route_weights=mx::array(std::vector<float>{0.1f,0.2f,0.3f,0.4f,0.5f,0.6f}.data(),{6},mx::float32);
 auto reduction_slots=mx::array(std::vector<std::uint32_t>{0,1,2,3,4,5}.data(),{1,6},mx::uint32);
 mx::eval(route_weights,reduction_slots);
 const double routed_full_ms=time_ms([&]{
  auto a=dsv41::linear_activation_reference(x).decoded;
  auto gt=mx::gather_qmm(mx::reshape(a,{1,1,5120}),w1,s1,std::nullopt,ids,ids,true,32,4,"mxfp4",false,mx::Device::gpu);
  auto ut=mx::gather_qmm(mx::reshape(a,{1,1,5120}),w3,s3,std::nullopt,ids,ids,true,32,4,"mxfp4",false,mx::Device::gpu);
  auto gg=mx::clip(mx::astype(gt,mx::float32),mx::array(-1e30f),mx::array(10.0f));
  auto uu=mx::clip(mx::astype(ut,mx::float32),mx::array(-10.0f),mx::array(10.0f));
  auto act=mx::astype(mx::multiply(mx::multiply(gg,mx::sigmoid(gg)),uu),mx::bfloat16);
  auto di=dsv41::linear_activation_reference(mx::reshape(act,{6,2304})).decoded;
  auto dn=mx::gather_qmm(mx::reshape(di,{6,1,2304}),w2,s2,std::nullopt,ids,ids,true,32,4,"mxfp4",false,mx::Device::gpu);
  auto w=mx::astype(mx::multiply(mx::astype(dn,mx::float32),
   mx::reshape(route_weights,{6,1,1})),mx::bfloat16);
  auto result=reduce_kernel({w,reduction_slots,mx::array({std::uint32_t(1)})},
   {{1,5120}},{mx::float32},{5120,1,1},{256,1,1},{},std::nullopt,false,mx::Device::gpu).front();
  mx::eval(result);mx::synchronize();
 },iters);

 // JIT-cache thrash: compile many distinct activation shapes, then re-time the
 // canonical K=5120 call. If it jumps, the in-model slowdown is cache eviction.
 for(int k=32;k<=992;k+=32){
  auto a=dsv41::linear_activation_reference(mx::zeros({1,k},mx::bfloat16)).decoded;
  mx::eval(a);mx::synchronize();
 }
 const double quant_after_thrash=time_ms([&]{
  auto a=dsv41::linear_activation_reference(x).decoded;mx::eval(a);mx::synchronize();
 },20);

 std::printf("iters=%d (median ms/call)\n",iters);
 std::printf("  input FP8 quant+decode   %.4f\n",quant_ms);
 std::printf("  input FP8 after thrash   %.4f\n",quant_after_thrash);
 std::printf("  gate+up gather_qmm       %.4f\n",gateup_ms);
 std::printf("  swiglu                   %.4f\n",swiglu_ms);
 std::printf("  down-input FP8 quant     %.4f\n",down_quant_ms);
 std::printf("  down gather_qmm          %.4f\n",down_ms);
 std::printf("  prep argsort/take        %.4f\n",prep_ms);
 std::printf("  full routed (no reduce)  %.4f\n",full_ms);
 std::printf("  full routed (+reduce)    %.4f\n",routed_full_ms);
 std::printf("  sum of parts             %.4f\n",
  quant_ms+gateup_ms+swiglu_ms+down_quant_ms+down_ms+prep_ms);
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
 return 0;
}
