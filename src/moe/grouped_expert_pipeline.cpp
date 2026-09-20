// Derived from oMLX v0.7.0.dev2 deepseek_v41_grouped_expert.cpp and
// activation.py (MIT). Adapted for the repository's official FP8 roundtrip.
#include "dsv41/moe_pipeline.hpp"
#include "layer_kernels.hpp"

#include <mlx/backend/metal/device.h>
#include <mlx/primitives.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace dsv41 {
namespace mx=mlx::core;
namespace {
using namespace mlx::core;
class GroupedExpertPipeline final:public mx::Primitive {
 public:
  GroupedExpertPipeline(mx::Stream stream,const mx::array& gate,const mx::array& up,
                        const mx::array& activation,const mx::array& down)
   :Primitive(stream),operations_{gate.primitive_ptr(),up.primitive_ptr(),
                                  activation.primitive_ptr(),down.primitive_ptr()},
    intermediate_shape_(gate.shape()),gate_inputs_(gate.inputs().size()),
    up_inputs_(up.inputs().size()){}
  void eval_cpu(const std::vector<mx::array>&,std::vector<mx::array>&) override{
   throw std::runtime_error("grouped expert pipeline requires GPU execution");
  }
  void eval_gpu(const std::vector<mx::array>& input,std::vector<mx::array>& output) override{
   std::vector<mx::array> gate{mx::array(intermediate_shape_,mx::bfloat16,nullptr,{})};
   std::vector<mx::array> up{mx::array(intermediate_shape_,mx::bfloat16,nullptr,{})};
   std::vector<mx::array> activation{mx::array(intermediate_shape_,mx::bfloat16,nullptr,{})};
   auto& encoder=mx::metal::get_command_encoder(stream());
   const auto up_begin=gate_inputs_;
   const auto activation_begin=gate_inputs_+up_inputs_;
   const auto down_begin=activation_begin+2;
   operations_[0]->eval_gpu(
    std::vector<mx::array>(input.begin(),input.begin()+up_begin),gate);
   encoder.add_temporary(gate[0]);
   operations_[1]->eval_gpu(
    std::vector<mx::array>(input.begin()+up_begin,input.begin()+activation_begin),up);
   encoder.add_temporary(up[0]);
   operations_[2]->eval_gpu({gate[0],up[0],input[activation_begin],input[activation_begin+1]},
                             activation);
   encoder.add_temporary(activation[0]);
   std::vector<mx::array> down_inputs{activation[0]};
   down_inputs.insert(down_inputs.end(),input.begin()+down_begin,input.end());
   operations_[3]->eval_gpu(down_inputs,output);
  }
  DEFINE_NAME(GroupedExpertPipeline)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive&) const override{return false;}
 private:
  std::vector<std::shared_ptr<Primitive>> operations_;
  mx::Shape intermediate_shape_;
  std::size_t gate_inputs_,up_inputs_;
};
}

mx::array fused_swiglu_fp8_roundtrip(const mx::array& gate,const mx::array& up){
 if(gate.dtype()!=mx::bfloat16||up.dtype()!=mx::bfloat16||gate.shape()!=up.shape()||
    gate.ndim()!=3||gate.shape(0)<1||gate.shape(0)>48||gate.shape(1)!=1||
    gate.shape(2)!=2304)
  throw std::invalid_argument("unsupported fused SwiGLU FP8 shape");
 static const std::string source=R"metal(
const uint i=thread_position_in_grid.x;
const uint n=params[0];
float g=float(gate[i]),u=float(up[i]);
g=min(g,10.0f);u=clamp(u,-10.0f,10.0f);
const float negative_sigmoid=1.0f/(1.0f+exp(abs(g)));
const float sigmoid_value=g<0.0f?negative_sigmoid:1.0f-negative_sigmoid;
const float value=float(bfloat16_t((g*sigmoid_value)*u));
const bool finite=simd_all(isfinite(value));
const float amax=max(simd_max(abs(value)),1e-4f);
uint bits=as_type<uint>(amax*(1.0f/448.0f));
int power=int((bits>>23)&255u)-127+int((bits&0x7fffffu)!=0);
float scale=as_type<float>(uint(power+127)<<23);
activation[i]=finite?bfloat16_t(dsv41_e4m3(dsv41_quant_e4m3(value/scale))*scale):
                     bfloat16_t(NAN);
)metal";
 static auto kernel=mx::fast::metal_kernel("dsv41_fused_swiglu_fp8_roundtrip",
  {"gate","up","params","limit"},{"activation"},source,dsv41_fp8_header);
 auto output=kernel({gate,up,mx::array({std::uint32_t(gate.size())}),mx::array({10.0f})},
  {gate.shape()},{mx::bfloat16},{gate.size(),1,1},{256,1,1},{},std::nullopt,false,mx::Device::gpu);
 return output.front();
}

mx::array grouped_expert_pipeline(const mx::array& gate,const mx::array& up,
                                  const mx::array& activation,const mx::array& down){
 for(const auto* node:{&gate,&up,&down}){
  if(!node->has_primitive()||std::string(node->primitive().name())!="GatherQMM"||
     (node->inputs().size()!=5&&node->inputs().size()!=6)||node->dtype()!=mx::bfloat16)
   throw std::invalid_argument("expected unevaluated BF16 GatherQMM nodes");
 }
 if(!activation.has_primitive()||std::string(activation.primitive().name())!="CustomKernel"||
    activation.inputs().size()!=4||activation.dtype()!=mx::bfloat16||
    activation.inputs()[0].id()!=gate.id()||activation.inputs()[1].id()!=up.id()||
    down.inputs()[0].id()!=activation.id()||gate.shape()!=up.shape()||
    gate.shape()!=activation.shape()||gate.ndim()!=3||gate.shape(1)!=1||
    down.ndim()!=3||down.shape(0)!=gate.shape(0)||down.shape(1)!=1)
  throw std::invalid_argument("unsupported grouped expert graph");
 auto stream=gate.primitive().stream();
 if(stream.device!=mx::Device::gpu)throw std::invalid_argument("grouped expert requires GPU");
 for(const auto* node:{&up,&activation,&down})if(node->primitive().stream()!=stream)
  throw std::invalid_argument("grouped expert nodes must share one GPU stream");
 std::vector<mx::array> inputs=gate.inputs();
 inputs.insert(inputs.end(),up.inputs().begin(),up.inputs().end());
 inputs.push_back(activation.inputs()[2]);inputs.push_back(activation.inputs()[3]);
 inputs.insert(inputs.end(),down.inputs().begin()+1,down.inputs().end());
 return mx::array(down.shape(),down.dtype(),
  std::make_shared<GroupedExpertPipeline>(stream,gate,up,activation,down),std::move(inputs));
}
}
