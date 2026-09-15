#include "dsv41/linear.hpp"
#include "dsv41/checkpoint_atlas.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
namespace {
namespace mx=mlx::core; using J=nlohmann::json;
void check(bool ok,const std::string& m) {if(!ok) throw std::runtime_error(m);}
std::string read_bytes(const std::filesystem::path& p) {
    check(std::filesystem::file_size(p)<=64*1024*1024,"fixture too large");
    std::ifstream f(p,std::ios::binary); check(bool(f),"cannot read fixture"); return {std::istreambuf_iterator<char>(f),{}};
}
template<class T> std::vector<T> read(const std::filesystem::path& p) {
    auto b=read_bytes(p); check(!b.empty()&&b.size()%sizeof(T)==0,"bad fixture size");
    std::vector<T> result(b.size()/sizeof(T)); std::memcpy(result.data(),b.data(),b.size()); return result;
}
template<class T> std::vector<T> host(const mx::array& a) {
    mx::eval(a); mx::synchronize(); return {a.data<T>(),a.data<T>()+a.size()};
}
std::vector<std::uint16_t> bits(const mx::array& a) {return host<std::uint16_t>(mx::view(a,mx::uint16,mx::Device::cpu));}
J compare(const std::vector<std::uint16_t>& a,const std::vector<std::uint16_t>& e) {
    check(a.size()==e.size(),"comparison shape mismatch"); std::size_t count=0; unsigned ulp=0; double error=0; J first=nullptr;
    auto order=[](std::uint16_t b) {return b&0x8000?0x8000-int(b&0x7fff):0x8000+int(b);};
    for(std::size_t i=0;i<a.size();++i) {
        float av=std::bit_cast<float>(std::uint32_t(a[i])<<16),ev=std::bit_cast<float>(std::uint32_t(e[i])<<16);
        check(std::isfinite(av)&&std::isfinite(ev),"nonfinite projection");
        if(a[i]!=e[i]) {++count; ulp=std::max(ulp,unsigned(std::abs(order(a[i])-order(e[i]))));
            error=std::max(error,std::abs(double(av)-ev));
            if(first.is_null()) first={{"index",i},{"actual_bits",a[i]},{"expected_bits",e[i]}};}
    }
    return {{"elements",a.size()},{"bit_mismatches",count},{"max_bf16_ulp",ulp},{"max_abs",error},{"first_mismatch",first}};
}
template<class F> void rejects(F f) {bool failed=false;try{f();}catch(const std::exception&){failed=true;}check(failed,"invalid input accepted");}
}
int main(int argc,char** argv) {
    using namespace dsv41;
    try {
        std::filesystem::path checkpoint,summary,fixtures,output;
        for(int i=1;i<argc;++i) {
            std::string flag=argv[i]; check(i+1<argc,"missing option value"); auto v=argv[++i];
            if(flag=="--checkpoint") checkpoint=v; else if(flag=="--m1-summary") summary=v;
            else if(flag=="--fixtures") fixtures=v; else if(flag=="--output") output=v; else check(false,"unknown option");
        }
        check(!checkpoint.empty()&&!summary.empty()&&!fixtures.empty()&&!output.empty(),"all paths required");
        auto relative=std::filesystem::weakly_canonical(output).lexically_relative(std::filesystem::canonical(checkpoint));
        check(!relative.empty()&&*relative.begin()=="..","checkpoint read-only");
        auto m=read_json_file(fixtures/"manifest.json"),o=read_json_file(fixtures/"omlx-comparison.json");
        check(o.at("manifest_sha256")==sha256_text(read_bytes(fixtures/"manifest.json")),"omlx fixture mismatch");
        for(const auto* hashes:{&m.at("fixture_sha256"),&o.at("generated_sha256")})
            for(const auto& [name,h]:hashes->items()) {
                check(std::filesystem::path(name).filename()==name,"unsafe fixture name");
                check(sha256_text(read_bytes(fixtures/name))==h.get<std::string>(),"fixture digest mismatch");
            }
        for(const auto& [name,h]:m.at("source_sha256").items()) {
            check(name=="inference/model.py"||name=="inference/kernel.py","unexpected oracle path");
            check(sha256_text(read_bytes(checkpoint/name))==h.get<std::string>(),"official source changed");
        }
        WeightCatalog catalog(checkpoint,summary); check(m.at("revision")==catalog.revision(),"revision mismatch");
        mx::set_default_device(mx::Device::gpu); J runs=J::array(); bool cpu_exact=true,local_exact=true;
        for(const auto& shape:{mx::Shape{1,32},mx::Shape{3,96}}) {
            auto tail=linear_activation_reference(mx::ones(shape,mx::bfloat16));
            check(bits(tail.decoded)==std::vector<std::uint16_t>(std::size_t(shape[0]*shape[1]),0x3f80),"partial threadgroup mismatch");
        }
        for(const auto& c:m.at("cases")) {
            const int t=c.at("tokens"),k=c.at("input_dims"),n=c.at("output_dims"); const std::string tag=c.at("tag");
            check(t==10&&k>0&&n>0&&std::filesystem::path(tag).filename()==tag,"bad case");
            auto input=read<std::uint16_t>(fixtures/(tag+"-input.bf16")); check(input.size()==std::size_t(t)*k,"input size mismatch");
            auto x=mx::view(mx::array(input.begin(),{t,k},mx::uint16),mx::bfloat16);
            auto layer=std::make_unique<PackedLinearReference>(catalog,c.at("prefix"));
            check(layer->input_dims()==k&&layer->output_dims()==n&&layer->bits()==c.at("bits"),"matrix shape mismatch");
            auto weight=host<std::uint32_t>(layer->packed_weight()); auto scale=host<std::uint8_t>(layer->scales());
            check(sha256_text(std::string(reinterpret_cast<const char*>(weight.data()),weight.size()*4))==c.at("packed_weight_sha256").get<std::string>(),"weight repack differs");
            check(sha256_text(std::string(reinterpret_cast<const char*>(scale.data()),scale.size()))==c.at("expanded_scale_sha256").get<std::string>(),"scale expansion differs");
            rejects([&]{layer->forward(mx::zeros({1,k+32},mx::bfloat16));});
            rejects([&]{layer->forward(mx::astype(x,mx::float32));});
            rejects([&]{layer->forward(mx::zeros({0,k},mx::bfloat16));});
            auto a=linear_activation_reference(x);
            check(host<std::uint8_t>(a.values)==read<std::uint8_t>(fixtures/(tag+"-quant.u8")),"activation bytes differ");
            check(host<std::uint8_t>(a.scales)==read<std::uint8_t>(fixtures/(tag+"-activation-scale.u8")),"activation scales differ");
            check(bits(a.decoded)==read<std::uint16_t>(fixtures/(tag+"-decoded.bf16")),"activation decoded differs");
            auto projected=layer->project_quantized(a),batched=layer->project_batch_diagnostic(a);
            std::vector<std::uint16_t> incremental;
            for(int token=0;token<t;++token) {
                auto part=bits(layer->forward(mx::slice(x,{token,0},{token+1,k})));
                incremental.insert(incremental.end(),part.begin(),part.end());
            }
            layer.reset(); auto actual=bits(projected);
            auto same_backend=compare(bits(batched),read<std::uint16_t>(fixtures/(tag+"-omlx-qmm.bf16")));
            check(same_backend.at("bit_mismatches")==0,"native/Python QMM mismatch with identical inputs");
            auto cpu=compare(actual,read<std::uint16_t>(fixtures/(tag+"-cpu-projection.bf16")));
            auto chunk=compare(incremental,actual);
            cpu_exact=cpu_exact&&cpu.at("bit_mismatches")==0; local_exact=local_exact&&chunk.at("bit_mismatches")==0;
            runs.push_back({{"prefix",c.at("prefix")},{"repack_exact",true},{"activation_exact",true},
                {"same_input_omlx_batched_qmm",same_backend},{"cpu_block_reference",cpu},{"tokenwise_vs_reference_chunk",chunk},
                {"batched_qmm_vs_reference",compare(bits(batched),actual)},
                {"owner_destroyed_before_evaluation",true},{"invalid_inputs_rejected",true}});
        }
        J result={{"status",cpu_exact&&local_exact?"reference_fixtures_exact":"numerical_mismatch"},
            {"manifest_sha256",sha256_text(read_bytes(fixtures/"manifest.json"))},
            {"omlx_comparison_sha256",sha256_text(read_bytes(fixtures/"omlx-comparison.json"))},
            {"official_cuda_executed",false},{"full_model_qualified",false},{"optimized_path_promoted",false},
            {"reference_schedule","one-row QMM, independent of input chunk length"},{"batched_qmm_qualified",false},
            {"partial_threadgroups_exact",true},{"runs",runs}};
        std::filesystem::create_directories(output.parent_path()); std::ofstream out(output); check(bool(out),"output open failed");
        out<<result.dump(2)<<'\n'; out.close();check(bool(out),"output write failed"); std::cout<<result.dump(2)<<'\n';
        return cpu_exact&&local_exact?0:2;
    } catch(const std::exception& e) {std::cerr<<"Linear probe: "<<e.what()<<'\n';return 1;}
}
