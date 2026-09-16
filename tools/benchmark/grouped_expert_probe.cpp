#include "dsv41/moe.hpp"
#include <mlx/mlx.h>
#include <array>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace mx=mlx::core;
using Clock=std::chrono::steady_clock;

void same(const mx::array& a,const mx::array& b,const char* name){
 if(a.shape()!=b.shape()||a.dtype()!=b.dtype())throw std::runtime_error(std::string(name)+" shape/dtype mismatch");
 auto dtype=a.dtype()==mx::bfloat16?mx::uint16:mx::uint32;
 auto ok=mx::all(mx::equal(mx::view(a,dtype),mx::view(b,dtype)));mx::eval(ok);
 if(!ok.item<bool>())throw std::runtime_error(std::string(name)+" bit mismatch");
}

mx::array bank(const std::vector<std::unique_ptr<dsv41::ExpertReference>>& experts,
 const dsv41::PackedLinearReference& (dsv41::ExpertReference::*projection)() const,bool scales){
 std::vector<mx::array> rows;rows.reserve(experts.size());
 for(const auto& expert:experts){const auto& linear=(expert.get()->*projection)();rows.push_back(scales?linear.scales():linear.packed_weight());}
 return mx::stack(rows,0);
}

int main(int argc,char** argv){try{
 if(argc!=3)throw std::runtime_error("usage: dsv41-grouped-expert-probe checkpoint m1-summary");
 mx::set_default_device(mx::Device::gpu);
 dsv41::WeightCatalog catalog(argv[1],argv[2]);
 std::vector<std::unique_ptr<dsv41::ExpertReference>> experts;
 for(int id=0;id<6;++id)experts.push_back(std::make_unique<dsv41::ExpertReference>(catalog,id,0));
 auto w1=bank(experts,&dsv41::ExpertReference::gate_projection,false);
 auto s1=bank(experts,&dsv41::ExpertReference::gate_projection,true);
 auto w3=bank(experts,&dsv41::ExpertReference::up_projection,false);
 auto s3=bank(experts,&dsv41::ExpertReference::up_projection,true);
 auto w2=bank(experts,&dsv41::ExpertReference::down_projection,false);
 auto s2=bank(experts,&dsv41::ExpertReference::down_projection,true);
 mx::eval(w1,s1,w3,s3,w2,s2);

 auto x=mx::astype(mx::reshape(mx::sin(mx::arange(5120,mx::float32)),{1,5120}),mx::bfloat16);
 auto input=dsv41::linear_activation_reference(x).decoded;
 auto ids=mx::arange(6,mx::uint32),lhs=mx::zeros({6},mx::uint32);
 auto gather=[&](const mx::array& value,const mx::array& weight,const mx::array& scale,
                 const mx::array& left){
  return mx::gather_qmm(value,weight,scale,std::nullopt,left,ids,true,32,4,"mxfp4",false,mx::Device::gpu);
 };
 auto gate=gather(mx::reshape(input,{1,1,5120}),w1,s1,lhs);
 auto up=gather(mx::reshape(input,{1,1,5120}),w3,s3,lhs);
 auto g=mx::clip(mx::astype(gate,mx::float32),mx::array(-1e30f),mx::array(10.0f));
 auto u=mx::clip(mx::astype(up,mx::float32),mx::array(-10.0f),mx::array(10.0f));
 auto activation=mx::astype(mx::multiply(mx::multiply(g,mx::sigmoid(g)),u),mx::bfloat16);
 auto down_input=dsv41::linear_activation_reference(mx::reshape(activation,{6,2304})).decoded;
 auto down=gather(mx::reshape(down_input,{6,1,2304}),w2,s2,ids);
 mx::eval(gate,up,activation,down);
 const std::array<float,6> route_weights{0.05f,0.1f,0.15f,0.2f,0.25f,0.3f};
 for(int id=0;id<6;++id){
  auto component=experts[id]->components(x,mx::array(route_weights[id]));
  same(mx::slice(gate,{id,0,0},{id+1,1,2304}),mx::reshape(component.gate,{1,1,2304}),"gate");
  same(mx::slice(up,{id,0,0},{id+1,1,2304}),mx::reshape(component.up,{1,1,2304}),"up");
  same(mx::slice(activation,{id,0,0},{id+1,1,2304}),mx::reshape(component.activation,{1,1,2304}),"activation");
  auto weighted=mx::astype(mx::multiply(mx::astype(mx::slice(down,{id,0,0},{id+1,1,5120}),mx::float32),mx::array(route_weights[id])),mx::bfloat16);
  same(weighted,mx::reshape(component.output,{1,1,5120}),"down");
 }
 auto grouped=[&]{
  auto a=dsv41::linear_activation_reference(x).decoded;
  auto ga=gather(mx::reshape(a,{1,1,5120}),w1,s1,lhs);
  auto ua=gather(mx::reshape(a,{1,1,5120}),w3,s3,lhs);
  auto gf=mx::clip(mx::astype(ga,mx::float32),mx::array(-1e30f),mx::array(10.0f));
  auto uf=mx::clip(mx::astype(ua,mx::float32),mx::array(-10.0f),mx::array(10.0f));
  auto act=mx::astype(mx::multiply(mx::multiply(gf,mx::sigmoid(gf)),uf),mx::bfloat16);
  auto da=dsv41::linear_activation_reference(mx::reshape(act,{6,2304})).decoded;
  auto out=gather(mx::reshape(da,{6,1,2304}),w2,s2,ids);
  return mx::astype(mx::multiply(mx::astype(out,mx::float32),
   mx::reshape(mx::array(route_weights.begin(),{6},mx::float32),{6,1,1})),mx::bfloat16);
 };
 auto individual=[&]{
  std::vector<mx::array> out;for(int id=0;id<6;++id)out.push_back(experts[id]->components(x,mx::array(route_weights[id])).output);
  return mx::stack(out,0);
 };
 auto timed=[](auto&& fn){auto start=Clock::now();auto out=fn();mx::eval(out);mx::synchronize();return std::chrono::duration<double>(Clock::now()-start).count();};
 auto grouped_warm=grouped(),individual_warm=individual();mx::eval(grouped_warm,individual_warm);mx::synchronize();
 same(grouped_warm,individual_warm,"weighted grouped output");
 std::vector<double> grouped_times,individual_times;
 for(int round=0;round<9;++round){
  if(round%2==0){individual_times.push_back(timed(individual));grouped_times.push_back(timed(grouped));}
  else{grouped_times.push_back(timed(grouped));individual_times.push_back(timed(individual));}
 }
 std::sort(grouped_times.begin(),grouped_times.end());std::sort(individual_times.begin(),individual_times.end());
 std::cout<<"individual_median_seconds="<<individual_times[4]<<" grouped_median_seconds="<<grouped_times[4]
          <<" speedup="<<individual_times[4]/grouped_times[4]<<"\n";
 std::cout<<"PASS: six-expert gather_qmm bank matches individual gate/up/activation/down bits; diagnostic only\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
