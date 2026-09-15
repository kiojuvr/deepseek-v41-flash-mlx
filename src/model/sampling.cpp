#include "dsv41/sampling.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
namespace dsv41 {
namespace mx=mlx::core;
namespace {
std::uint64_t splitmix64(std::uint64_t& state){
 state+=0x9E3779B97F4A7C15ULL;
 std::uint64_t z=state;
 z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;
 z=(z^(z>>27))*0x94D049BB133111EBULL;
 return z^(z>>31);
}
double uniform_open(std::uint64_t& state){
 // (0,1) to keep -log(u) finite.
 const double u=(splitmix64(state)>>11)*(1.0/9007199254740992.0);
 return u>0.0?u:std::numeric_limits<double>::min();
}
}
std::uint32_t greedy_reference(const mx::array& logits){
 if(logits.size()==0||logits.ndim()<1)throw std::runtime_error("greedy requires nonempty logits");
 auto cpu=mx::astype(logits,mx::float32,mx::Device::cpu);mx::eval(cpu);
 const float* p=cpu.data<float>();int n=int(cpu.size());
 for(int i=0;i<n;++i)if(!std::isfinite(p[i]))throw std::runtime_error("nonfinite logits");
 int best=0;for(int i=1;i<n;++i)if(p[i]>p[best])best=i;
 return std::uint32_t(best);
}
std::uint32_t sample_reference(const mx::array& logits,float temperature,std::uint64_t& rng_state){
 if(!std::isfinite(temperature))throw std::runtime_error("sampling requires finite temperature");
 if(!(temperature>0.0f))return greedy_reference(logits);
 if(logits.size()==0||logits.ndim()<1)throw std::runtime_error("sampling requires nonempty logits");
 std::vector<float> exponentials(logits.size());
 for(auto& value:exponentials)value=static_cast<float>(-std::log(uniform_open(rng_state)));
 return sample_from_exponentials_reference(logits,temperature,exponentials);
}
std::uint32_t sample_from_exponentials_reference(const mx::array& logits,float temperature,
 std::span<const float> exponentials){
 if(!(temperature>0.0f) || !std::isfinite(temperature))
  throw std::runtime_error("fixed-exponential sampling requires finite positive temperature");
 if(logits.size()==0||logits.ndim()<1 || exponentials.size()!=logits.size())
  throw std::runtime_error("fixed-exponential sampling shape mismatch");
 auto cpu=mx::astype(logits,mx::float32,mx::Device::cpu);mx::eval(cpu);
 const float* p=cpu.data<float>();int n=int(cpu.size());
 const float effective=std::max(temperature,1e-5f);
 std::vector<float> scaled(n),probs(n);
 float maximum=-std::numeric_limits<float>::infinity();
 for(int i=0;i<n;++i){
  if(!std::isfinite(p[i]))throw std::runtime_error("nonfinite logits");
  if(!(exponentials[i]>0.0f) || !std::isfinite(exponentials[i]))
   throw std::runtime_error("invalid exponential variate");
  scaled[i]=p[i]/effective;maximum=std::max(maximum,scaled[i]);
 }
 // Official semantic order/dtype: FP32 softmax(logits/max(T,1e-5)),
 // division by same-shape FP32 Exp(1), then argmax. Reduction order here is
 // the explicit local CPU reference schedule, not a CUDA bitwise claim.
 float sum=0.0f;
 for(int i=0;i<n;++i){probs[i]=std::exp(scaled[i]-maximum);sum+=probs[i];}
 int best=0;float best_score=-1.0f;
 for(int i=0;i<n;++i){
  probs[i]/=sum;
  const float score=probs[i]/exponentials[i];
  if(score>best_score){best_score=score;best=i;}
 }
 return std::uint32_t(best);
}
}
