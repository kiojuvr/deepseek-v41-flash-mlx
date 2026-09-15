#include "dsv41/compressed_rope.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
namespace dsv41 {
namespace mx=mlx::core;
mx::array compressed_rope_reference(const mx::array& input,std::span<const std::uint64_t> positions,bool inverse){
 if(input.dtype()!=mx::bfloat16||(input.ndim()!=2&&input.ndim()!=3)||(input.shape(-1)!=512&&input.shape(-1)!=128)||
    input.shape(0)<1||input.shape(0)>128||(input.ndim()==3&&input.shape(1)!=64&&input.shape(1)!=32)||
    positions.size()!=std::size_t(input.shape(0)))
  throw std::runtime_error("invalid compressed RoPE input/position");
 for(auto p:positions)if(p>=1048576)throw std::runtime_error("invalid compressed RoPE position");
 const int tail_start=input.shape(-1)-64;
 auto begin=mx::Shape(input.ndim(),0),end=input.shape();begin.back()=tail_start;
 auto tail=mx::astype(mx::slice(input,begin,end),mx::float32);
 auto pair_shape=tail.shape();pair_shape.back()=32;pair_shape.push_back(2);
 auto pairs=mx::reshape(tail,pair_shape);
 auto real=mx::take(pairs,mx::array(0),-1),imag=mx::take(pairs,mx::array(1),-1);
 auto frequency=mx::divide(mx::array(1.0f),mx::power(mx::array(160000.0f),mx::divide(mx::arange(0,64,2,mx::float32),mx::array(64.0f))));
 // Official YaRN corrected-dimension bounds: dim=64, original=65536, beta=32/1.
 const auto corrected=[](double rotations){return 64.0*std::log(65536.0/(rotations*2.0*std::acos(-1.0)))/(2.0*std::log(160000.0));};
 const float low=float(std::max(0.0,std::floor(corrected(32)))),high=float(std::min(63.0,std::ceil(corrected(1))));
 auto ramp=mx::clip(mx::divide(mx::subtract(mx::arange(32,mx::float32),mx::array(low)),mx::array(std::max(high-low,1e-3f))),mx::array(0.0f),mx::array(1.0f));
 auto smooth=mx::subtract(mx::array(1.0f),ramp);
 frequency=mx::add(mx::multiply(mx::divide(frequency,mx::array(16.0f)),mx::subtract(mx::array(1.0f),smooth)),mx::multiply(frequency,smooth));
 std::vector<float> p;for(auto position:positions)p.push_back(float(position));
 auto angle=mx::multiply(mx::array(p.begin(),{input.shape(0),1},mx::float32),frequency);
 if(input.ndim()==3)angle=mx::expand_dims(angle,1);
 auto c=mx::cos(angle),s=mx::sin(angle);if(inverse)s=mx::negative(s);
 auto rotated=mx::stack({mx::subtract(mx::multiply(real,c),mx::multiply(imag,s)),
                        mx::add(mx::multiply(real,s),mx::multiply(imag,c))},-1);
 begin.back()=0;end.back()=tail_start;
 return mx::concatenate({mx::slice(input,begin,end),mx::astype(mx::reshape(rotated,tail.shape()),mx::bfloat16)},-1);
}
}
