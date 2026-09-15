#include "dsv41/sampling.hpp"
#include <array>
#include <iostream>
#include <map>
#include <stdexcept>
namespace mx=mlx::core;
int main(){
 try{
  // Greedy picks the maximum and is deterministic.
  mx::array logits=mx::array({0.5f,-2.0f,3.0f,1.0f},mx::float32);
  if(dsv41::greedy_reference(logits)!=2)throw std::runtime_error("greedy argmax mismatch");
  std::uint64_t zero_state=0;
  if(dsv41::sample_reference(logits,0.0f,zero_state)!=2)throw std::runtime_error("temperature 0 must be greedy");

  // Frozen Exp(1) inputs exercise the official FP32 formula independently of RNG.
  mx::array oracle_logits=mx::array({0.25f,-1.0f,2.0f,0.5f},mx::float32);
  const std::array<float,4> exponentials{0.8f,0.2f,100.0f,0.01f};
  const auto fixed=dsv41::sample_from_exponentials_reference(oracle_logits,0.7f,exponentials);
  if(fixed!=3)
   throw std::runtime_error("fixed-exponential formula mismatch");
  const auto floored=dsv41::sample_from_exponentials_reference(oracle_logits,1e-8f,exponentials);
  if(floored!=2)
   throw std::runtime_error("temperature floor mismatch");

  // A dominant logit must be sampled almost surely at low temperature.
  mx::array sharp=mx::array({-10.0f,50.0f,-10.0f},mx::float32);
  for(std::uint64_t seed=0;seed<64;++seed){std::uint64_t s=seed;if(dsv41::sample_reference(sharp,1.0f,s)!=1)throw std::runtime_error("dominant logit not sampled");}

  // The same seed reproduces the same token stream.
  mx::array flat=mx::array({0.0f,0.0f,0.0f,0.0f},mx::float32);
  std::uint64_t a=12345,b=12345;std::vector<std::uint32_t> ta,tb;
  for(int i=0;i<200;++i){ta.push_back(dsv41::sample_reference(flat,1.0f,a));tb.push_back(dsv41::sample_reference(flat,1.0f,b));}
  if(ta!=tb)throw std::runtime_error("same seed diverged");
  bool varied=false;for(auto t:ta)if(t!=ta[0])varied=true;
  if(!varied)throw std::runtime_error("sampling produced no variation on a uniform distribution");

  // A finite temperature must cover every category eventually.
  std::map<std::uint32_t,int> counts;std::uint64_t c=7;
  for(int i=0;i<2000;++i)counts[dsv41::sample_reference(flat,1.0f,c)]++;
  if(counts.size()!=4)throw std::runtime_error("not all categories sampled");

  // Non-finite logits are rejected.
  bool rejected=false;try{dsv41::greedy_reference(mx::array({0.0f,std::nanf("")},mx::float32));}catch(const std::exception&){rejected=true;}
  if(!rejected)throw std::runtime_error("nonfinite logits accepted");
  rejected=false;try{std::uint64_t s=0;(void)dsv41::sample_reference(logits,std::nanf(""),s);}catch(const std::exception&){rejected=true;}
  if(!rejected)throw std::runtime_error("nonfinite temperature accepted");
  rejected=false;try{dsv41::sample_from_exponentials_reference(oracle_logits,0.7f,{exponentials.data(),3});}catch(const std::exception&){rejected=true;}
  if(!rejected)throw std::runtime_error("wrong exponential shape accepted");
  auto invalid=exponentials;invalid[1]=0.0f;rejected=false;
  try{dsv41::sample_from_exponentials_reference(oracle_logits,0.7f,invalid);}catch(const std::exception&){rejected=true;}
  if(!rejected)throw std::runtime_error("invalid exponential accepted");

  std::cout<<"PASS: greedy, fixed-exponential FP32 formula/floor, RNG reproducibility/coverage, invalid-input rejection; fixed="<<fixed<<" floor="<<floored<<"\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
