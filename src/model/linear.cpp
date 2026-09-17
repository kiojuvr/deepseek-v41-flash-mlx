#include "dsv41/linear.hpp"
#include "dsv41/execution_policy.hpp"
#include "layer_kernels.hpp"
#include "engram_kernel_source.hpp"
#include <algorithm>
#include <bit>
#include <climits>
#include <cstring>
#include <stdexcept>

namespace dsv41 {
namespace mx=mlx::core;
namespace {
void check(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }
void activation_shape(const mx::array& x) {
    check(x.dtype()==mx::bfloat16&&x.ndim()==2&&x.shape(0)>0&&x.shape(0)<=1024&&
          x.shape(1)>0&&x.shape(1)%32==0&&x.size()<=INT_MAX,
          "linear reference requires BF16 [1..1024,K], positive K divisible by 32");
}
std::vector<std::uint8_t> read(const TensorFile& tensor) {
    std::vector<std::uint8_t> data(tensor.size_bytes);
    tensor.read(0,{reinterpret_cast<std::byte*>(data.data()),data.size()}); return data;
}
}
LinearActivation linear_activation_reference(const mx::array& input) {
    activation_shape(input);
    const int m=input.shape(0),k=input.shape(1),groups=m*(k/32);
    const std::string guard="if (thread_position_in_grid.x >= count[0]) return;\n";
    static auto quant=mx::fast::metal_kernel("dsv41_linear_act_reference",{"values","count"},
        {"quantized","scales"},guard+dsv41_activation_quant,dsv41_fp8_header);
    auto q=quant({input,mx::array({std::uint32_t(groups)})},{{m,k},{m,k/32}},
                 {mx::uint8,mx::uint8},{groups,1,1},{64,1,1},{},std::nullopt,false,mx::Device::gpu);
    static auto decode=mx::fast::metal_kernel("dsv41_linear_act_decode",{"values","scales","count"},
        {"output"},guard+dsv41_engram_kernel_source);
    auto decoded=decode({q[0],q[1],mx::array({std::uint32_t(m*k)})},{{m,k}},{mx::uint16},
                        {m*k,1,1},{128,1,1},{},std::nullopt,false,mx::Device::gpu);
    return {q[0],q[1],mx::view(decoded[0],mx::bfloat16,mx::Device::gpu)};
}
struct PackedLinearReference::Impl {
    mx::array weight, scale;
    int n,k,bits;
    Impl(mx::array w,mx::array s,int outputs,int inputs,int width)
        : weight(std::move(w)),scale(std::move(s)),n(outputs),k(inputs),bits(width) {}
};
PackedLinearReference::PackedLinearReference(WeightCatalog& catalog,const std::string& prefix) {
    static_assert(std::endian::native==std::endian::little);
    auto w=catalog.tensor(prefix+".weight"),s=catalog.tensor(prefix+".scale");
    const int width=w.dtype=="F8_E4M3"?8:(w.dtype=="I8"||w.dtype=="U8")?4:0;
    check(width&&w.shape.size()==2&&w.shape[0]>0&&w.shape[1]>0&&
          w.shape[0]<=INT_MAX&&w.shape[1]<=std::uint64_t(INT_MAX)/(8/width)&&w.shape[1]%4==0,
          "unsupported packed linear weight layout");
    const int n=int(w.shape[0]),k=int(w.shape[1])*(8/width);
    check(k%32==0&&s.dtype=="F8_E8M0"&&s.shape==std::vector<std::uint64_t>({
          std::uint64_t(width==8?(std::uint64_t(n)+31)/32:n),std::uint64_t(k/32)}),
          "packed linear scale layout mismatch");
    auto wb=read(w),sb=read(s);
    check(wb.size()==std::uint64_t(n)*w.shape[1]&&sb.size()==s.shape[0]*s.shape[1],"packed linear byte size mismatch");
    check(std::find(sb.begin(),sb.end(),255)==sb.end(),"NaN weight scale");
    if(width==8) check(std::none_of(wb.begin(),wb.end(),[](auto code){return (code&127)==127;}),"NaN FP8 weight");
    std::vector<std::uint32_t> packed(wb.size()/4);
    std::memcpy(packed.data(),wb.data(),wb.size());
    std::vector<std::uint8_t> expanded(std::size_t(n)*(k/32));
    for(int row=0;row<n;++row)
        std::copy_n(sb.begin()+std::size_t(width==8?row/32:row)*(k/32),k/32,expanded.begin()+std::size_t(row)*(k/32));
    impl_=std::make_unique<Impl>(mx::array(packed.begin(),{n,int(w.shape[1]/4)},mx::uint32),
                                 mx::array(expanded.begin(),{n,k/32},mx::uint8),n,k,width);
}
PackedLinearReference::~PackedLinearReference()=default;
const mx::array& PackedLinearReference::packed_weight() const { return impl_->weight; }
const mx::array& PackedLinearReference::scales() const { return impl_->scale; }
int PackedLinearReference::input_dims() const { return impl_->k; }
int PackedLinearReference::output_dims() const { return impl_->n; }
int PackedLinearReference::bits() const { return impl_->bits; }
mx::array PackedLinearReference::forward(const mx::array& input) const {
    activation_shape(input); check(input.shape(1)==impl_->k,"linear input width mismatch");
    auto activation=linear_activation_reference(input);
    if(runtime_batched_dense_qmm_enabled()&&input.shape(0)>1)
        return project_batch_diagnostic(activation);
    return project_quantized(activation);
}
mx::array PackedLinearReference::project_quantized(const LinearActivation& a) const {
    activation_shape(a.decoded);
    check(a.decoded.shape(1)==impl_->k&&a.values.dtype()==mx::uint8&&a.scales.dtype()==mx::uint8&&
          a.values.shape()==a.decoded.shape()&&a.scales.shape()==mx::Shape({a.decoded.shape(0),impl_->k/32}),
          "linear quantized activation layout mismatch");
    // Fix the reduction dispatch at one row, independent of prefill/decode chunk size.
    std::vector<mx::array> rows; rows.reserve(a.decoded.shape(0));
    for(int row=0;row<a.decoded.shape(0);++row)
        rows.push_back(mx::quantized_matmul(mx::slice(a.decoded,{row,0},{row+1,impl_->k}),
            impl_->weight,impl_->scale,std::nullopt,true,32,impl_->bits,
            impl_->bits==8?"mxfp8":"mxfp4",mx::Device::gpu));
    return rows.size()==1?rows.front():mx::concatenate(rows,0,mx::Device::gpu);
}
mx::array PackedLinearReference::project_batch_diagnostic(const LinearActivation& a) const {
    activation_shape(a.decoded); check(a.decoded.shape(1)==impl_->k,"linear input width mismatch");
    return mx::quantized_matmul(a.decoded,impl_->weight,impl_->scale,std::nullopt,true,32,impl_->bits,
                               impl_->bits==8?"mxfp8":"mxfp4",mx::Device::gpu);
}
}
