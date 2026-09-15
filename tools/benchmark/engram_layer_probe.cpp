#include "dsv41/engram_layer.hpp"
#include "dsv41/checkpoint_atlas.hpp"
#include <mlx/mlx.h>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
namespace mx=mlx::core;
using J=nlohmann::json;
void check(bool ok,const std::string& message) { if(!ok) throw std::runtime_error(message); }
std::string bytes(const std::filesystem::path& p) {
    check(std::filesystem::file_size(p)<=64*1024*1024,"fixture/source too large");
    std::ifstream f(p,std::ios::binary); check(bool(f),"fixture read failed");
    return {std::istreambuf_iterator<char>(f),{}};
}
template<class T> std::vector<T> read(const std::filesystem::path& p) {
    static_assert(std::endian::native==std::endian::little);
    auto b=bytes(p); check(!b.empty()&&b.size()%sizeof(T)==0,"bad binary fixture");
    std::vector<T> result(b.size()/sizeof(T)); std::memcpy(result.data(),b.data(),b.size()); return result;
}
mx::array bf16(const std::filesystem::path& p,const mx::Shape& shape) {
    auto data=read<std::uint16_t>(p); std::size_t count=1; for(auto n:shape) count*=n;
    check(count==data.size(),"BF16 fixture shape mismatch");
    return mx::view(mx::array(data.begin(),shape,mx::uint16),mx::bfloat16);
}
mx::array fp32(const std::filesystem::path& p,const mx::Shape& shape) {
    auto data=read<float>(p); std::size_t count=1; for(auto n:shape) count*=n;
    check(count==data.size(),"FP32 fixture shape mismatch");
    return mx::array(data.begin(),shape,mx::float32);
}
template<class T> std::vector<T> host(const mx::array& a) {
    auto x=mx::astype(a,a.dtype(),mx::Device::cpu); mx::eval(x); mx::synchronize();
    return {x.data<T>(),x.data<T>()+x.size()};
}
std::vector<std::uint16_t> bits(const mx::array& a) { return host<std::uint16_t>(mx::view(a,mx::uint16,mx::Device::cpu)); }
J compare(const std::vector<std::uint16_t>& actual,const std::vector<std::uint16_t>& expected) {
    check(actual.size()==expected.size(),"comparison size mismatch");
    std::size_t mismatches=0; double max_abs=0,max_relative=0; unsigned max_ulp=0;
    J first=nullptr;
    auto ordered=[](std::uint16_t b){return (b&0x8000)?0x8000-int(b&0x7fff):0x8000+int(b);};
    for(std::size_t i=0;i<actual.size();++i) {
        const float a=std::bit_cast<float>(std::uint32_t(actual[i])<<16), e=std::bit_cast<float>(std::uint32_t(expected[i])<<16);
        check(std::isfinite(a)&&std::isfinite(e),"non-finite comparison value");
        if(actual[i]!=expected[i]) {
            ++mismatches; if(first.is_null()) first={{"index",i},{"actual_bits",actual[i]},{"expected_bits",expected[i]}};
            const double error=std::abs(double(a)-double(e)); max_abs=std::max(max_abs,error);
            max_relative=std::max(max_relative,error/std::max(std::abs(double(e)),1e-30));
            max_ulp=std::max(max_ulp,unsigned(std::abs(ordered(actual[i])-ordered(expected[i]))));
        }
    }
    return {{"elements",actual.size()},{"bit_mismatches",mismatches},{"max_abs",max_abs},
            {"max_relative",max_relative},{"max_bf16_ulp",max_ulp},{"first_mismatch",first}};
}
J float_compare(const mx::array& value,const std::filesystem::path& path) {
    auto actual=host<float>(value), expected=read<float>(path);
    check(actual.size()==expected.size(),"FP32 comparison size mismatch");
    std::size_t count=0; double max_abs=0; std::uint64_t max_ulp=0;
    auto ordered=[](std::uint32_t b)->std::int64_t {return (b&0x80000000u)?0x80000000ll-(b&0x7fffffffu):0x80000000ll+b;};
    for(std::size_t i=0;i<actual.size();++i) {
        check(std::isfinite(actual[i])&&std::isfinite(expected[i]),"non-finite FP32 diagnostic");
        auto a=std::bit_cast<std::uint32_t>(actual[i]), e=std::bit_cast<std::uint32_t>(expected[i]);
        count+=a!=e; max_abs=std::max(max_abs,std::abs(double(actual[i])-expected[i]));
        max_ulp=std::max(max_ulp,std::uint64_t(std::abs(ordered(a)-ordered(e))));
    }
    return {{"bit_mismatches",count},{"max_abs",max_abs},{"max_fp32_ulp",max_ulp},
            {"actual",actual},{"expected",expected}};
}
// Diagnostic graph only: never used by the production/reference forward.
J gate_trace(const mx::array& hidden,const mx::array& kv,const mx::array& q,
             const mx::array& k,const mx::array& oracle_dot,float eps) {
    const int n=hidden.shape(0);
    auto h=mx::astype(hidden,mx::float32);
    auto key=mx::reshape(mx::astype(mx::slice(kv,{0,0},{n,20480}),mx::float32),{n,4,5120});
    auto weight=mx::multiply(mx::astype(q,mx::float32),mx::astype(k,mx::float32));
    auto hm=mx::mean(mx::square(h),-1), km=mx::mean(mx::square(key),-1);
    auto hr=mx::rsqrt(mx::add(hm,mx::array(eps))), kr=mx::rsqrt(mx::add(km,mx::array(eps)));
    auto rstd=mx::multiply(hr,kr);
    auto sum=mx::sum(mx::multiply(mx::multiply(h,weight),key),-1);
    auto dot=mx::multiply(mx::multiply(sum,rstd),mx::array(float(std::pow(5120.0,-0.5))));
    auto signed_root=[](const mx::array& d) {
        auto mag=mx::sqrt(mx::maximum(mx::abs(d),mx::array(1e-6f)));
        auto neg=mx::greater_equal(mx::view(d,mx::uint32),mx::array(std::uint32_t(0x80000000)));
        return mx::where(neg,mx::negative(mag),mag);
    };
    auto root=signed_root(dot), injected_root=signed_root(oracle_dot);
    J trace;
    for(const auto& [name,value]:std::vector<std::pair<std::string,mx::array>>{
        {"hidden_mean_square",hm},{"key_mean_square",km},{"hidden_rsqrt",hr},
        {"key_rsqrt",kr},{"rstd",rstd},{"weighted_sum",sum},{"dot",dot},
        {"signed_root",root},{"gate_unmasked",mx::sigmoid(root)},
        {"oracle_dot_signed_root",injected_root},{"oracle_dot_gate",mx::sigmoid(injected_root)}}) {
        trace[name]=host<float>(value);
    }
    return trace;
}
template<class F> void rejects(F action) {
    bool rejected=false; try { action(); } catch(const std::exception&) {rejected=true;}
    check(rejected,"invalid input was accepted");
}
}
int main(int argc,char** argv) {
    try {
        using namespace dsv41;
        std::filesystem::path checkpoint,summary,rows,fixtures,output;
        for(int i=1;i<argc;++i) {
            const std::string flag=argv[i]; check(i+1<argc,"missing argument");
            const auto value=argv[++i];
            if(flag=="--checkpoint") checkpoint=value; else if(flag=="--m1-summary") summary=value;
            else if(flag=="--row-fixtures") rows=value; else if(flag=="--fixtures") fixtures=value;
            else if(flag=="--output") output=value; else throw std::runtime_error("unknown option");
        }
        check(!checkpoint.empty()&&!summary.empty()&&!rows.empty()&&!fixtures.empty()&&!output.empty(),"all paths required");
        auto relative=std::filesystem::weakly_canonical(output).lexically_relative(std::filesystem::canonical(checkpoint));
        check(!relative.empty()&&*relative.begin()=="..","output must be outside checkpoint");
        auto manifest=read_json_file(fixtures/"manifest.json");
        check(manifest.at("schema_version")==1,"unsupported fixture schema");
        for(const auto& [name,h]:manifest.at("fixture_sha256").items()) {
            check(std::filesystem::path(name).filename()==name,"unsafe fixture name");
            check(sha256_text(bytes(fixtures/name))==h.get<std::string>(),"fixture digest mismatch");
        }
        for(const auto& [name,h]:manifest.at("source_sha256").items()) {
            const std::filesystem::path p(name); check(!p.is_absolute(),"unsafe source path");
            for(const auto& part:p) check(part!="..","unsafe source path");
            check(sha256_text(bytes(checkpoint/p))==h.get<std::string>(),"oracle source changed");
        }
        check(sha256_text(bytes(rows/"fixture-provenance.json"))==manifest.at("row_fixture_provenance_sha256").get<std::string>(),"row fixture identity changed");
        auto row_manifest=read_json_file(rows/"fixture-provenance.json");
        check(sha256_text(bytes(rows/"metadata.json"))==row_manifest.at("fixture_sha256").at("metadata.json").get<std::string>(),"metadata changed");
        WeightCatalog catalog(checkpoint,summary); auto metadata=EngramMetadata::load(rows/"metadata.json");
        check(catalog.revision()==manifest.at("revision").get<std::string>()&&metadata->revision==catalog.revision(),"revision mismatch");
        const float eps=manifest.at("norm_eps").get<float>();
        mx::set_default_device(mx::Device::gpu);
        auto finite=engram_activation_reference(bf16(fixtures/"finite-bf16-input.bf16",{11,6144}));
        check(host<std::uint8_t>(finite.quantized)==read<std::uint8_t>(fixtures/"finite-bf16-quant.u8"),"finite BF16 activation pattern mismatch");
        check(host<std::uint8_t>(finite.scales)==read<std::uint8_t>(fixtures/"finite-bf16-scales.u8"),"finite BF16 scale pattern mismatch");
        rejects([&]{engram_activation_reference(mx::zeros({1,6143},mx::bfloat16));});
        J results=J::array(); bool exact=true;
        for(const auto& c:manifest.at("cases")) {
            const int n=c.at("tokens"); check(n==3,"expected 3-token fixture");
            const std::string tag=c.at("tag"); check(std::filesystem::path(tag).filename()==tag,"unsafe fixture tag");
            auto ids=c.at("rows").get<std::vector<std::uint64_t>>(); check(ids.size()==std::size_t(n)*24,"row count mismatch");
            const auto layer_index=c.at("layer_index").get<std::size_t>();
            check(layer_index<metadata->layer_ids.size()&&metadata->layer_ids[layer_index]==c.at("layer_id").get<std::uint64_t>(),"layer mismatch");
            auto h=bf16(fixtures/(tag+"-hidden.bf16"),{n,4,5120});
            auto kv=bf16(fixtures/(tag+"-kv.bf16"),{n,25600});
            auto q=bf16(fixtures/(tag+"-q.bf16"),{4,5120}); auto k=bf16(fixtures/(tag+"-k.bf16"),{4,5120});
            auto mask_values=c.at("mask").get<std::vector<bool>>(); check(mask_values.size()==std::size_t(n),"mask size mismatch");
            mx::array mask(mask_values.begin(),{n},mx::bool_);
            const auto expected=read<std::uint16_t>(fixtures/(tag+"-expected.bf16"));
            auto isolated=engram_gate_reference(h,kv,q,k,mask,eps);
            const auto isolated_compare=compare(bits(isolated.output),expected);
            const auto dot_compare=float_compare(isolated.dot,fixtures/(tag+"-dot.f32"));
            const auto gate_compare=float_compare(isolated.gate,fixtures/(tag+"-gate.f32"));
            auto trace=gate_trace(h,kv,q,k,fp32(fixtures/(tag+"-dot.f32"),{n,4}),eps);
            check(trace.at("dot")==dot_compare.at("actual"),"diagnostic graph differs from native dot");
            // Boundary injection diagnoses the discrepancy without changing the runtime path.
            auto oracle_gate=fp32(fixtures/(tag+"-gate.f32"),{n,4});
            auto value=mx::astype(mx::slice(kv,{0,20480},{n,25600}),mx::float32);
            auto injected=mx::astype(mx::add(mx::astype(h,mx::float32),
                mx::multiply(mx::expand_dims(oracle_gate,-1),mx::expand_dims(value,-2))),mx::bfloat16);
            auto injected_compare=compare(bits(injected),expected);
            check(injected_compare.at("bit_mismatches")==0,"oracle gate injection residual mismatch");
            exact=exact&&isolated_compare.at("bit_mismatches")==0&&
                  dot_compare.at("bit_mismatches")==0&&gate_compare.at("bit_mismatches")==0;
            std::vector<std::uint16_t> baseline;
            for(auto mode:{EngramReadMode::Mmap,EngramReadMode::Pread}) {
                const auto started=std::chrono::steady_clock::now();
                // Destroy the layer before evaluation: the returned graph must own its inputs.
                auto layer=std::make_unique<EngramLayerReference>(catalog,*metadata,layer_index,eps,mode);
                rejects([&]{layer->forward(h,std::span(ids).first(ids.size()-1),mask);});
                rejects([&]{layer->forward(mx::astype(h,mx::float32),ids,mask);});
                rejects([&]{layer->forward(h,ids,mx::zeros({1},mx::bool_));});
                auto invalid=ids; invalid[0]=metadata->num_embeddings[layer_index];
                rejects([&]{layer->forward(h,invalid,mask);});
                auto r=layer->forward(h,ids,mask);
                std::vector<std::uint16_t> incremental, incremental_kv;
                for(int t=0;t<n;++t) {
                    auto part=layer->forward(mx::slice(h,{t,0,0},{t+1,4,5120}),std::span(ids).subspan(t*24,24),
                                             mx::slice(mask,{t},{t+1}));
                    auto values=bits(part.output), projected=bits(part.kv);
                    incremental.insert(incremental.end(),values.begin(),values.end());
                    incremental_kv.insert(incremental_kv.end(),projected.begin(),projected.end());
                }
                layer.reset();
                auto actual=bits(r.output);
                check(actual==incremental&&bits(r.kv)==incremental_kv,"batch/tokenwise local exactness mismatch");
                check(host<std::uint8_t>(r.quantized)==read<std::uint8_t>(fixtures/(tag+"-quant.u8")),"activation quant bytes mismatch");
                check(host<std::uint8_t>(r.scales)==read<std::uint8_t>(fixtures/(tag+"-scales.u8")),"activation scale bytes mismatch");
                auto projection_compare=compare(bits(r.kv),read<std::uint16_t>(fixtures/(tag+"-kv.bf16")));
                auto output_compare=compare(actual,expected);
                if(mode==EngramReadMode::Mmap) baseline=actual;
                else check(actual==baseline,"mmap/pread full local output mismatch");
                auto hidden_bits=bits(h);
                for(int t=0;t<n;++t) if(!mask_values[t])
                    check(std::equal(actual.begin()+t*20480,actual.begin()+(t+1)*20480,hidden_bits.begin()+t*20480),"masked token changed");
                exact=exact&&projection_compare.at("bit_mismatches")==0&&output_compare.at("bit_mismatches")==0;
                results.push_back({{"layer",c.at("layer_id")},{"mode",mode==EngramReadMode::Mmap?"mmap":"pread"},
                    {"activation_bytes_exact",true},{"activation_scales_exact",true},{"masked_residual_exact",true},
                    {"batch_tokenwise_exact",true},{"invalid_inputs_rejected",true},
                    {"isolated_official_gate",isolated_compare},{"cpu_transcribed_projection",projection_compare},
                    {"official_dot_fp32",dot_compare},{"official_gate_fp32",gate_compare},{"gate_trace",trace},
                    {"injected_official_gate_residual",injected_compare},
                    {"full_residual",output_compare},
                    {"load_and_forward_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()}});
                std::cout<<results.back().dump()<<std::endl;
            }
        }
        J result={{"status",exact?"fixtures_bitwise_exact":"numerical_mismatch"},
            {"manifest_sha256",sha256_text(bytes(fixtures/"manifest.json"))},{"revision",catalog.revision()},
            {"official_cuda_projection_executed",false},{"full_model_qualified",false},
            {"finite_bf16_activation_patterns_exact",65280},
            {"optimized_path_promoted",false},{"runs",results}};
        std::filesystem::create_directories(output.parent_path());
        std::ofstream f(output); check(bool(f),"cannot write result"); f<<result.dump(2)<<'\n'; f.close(); check(bool(f),"result write failed");
        return exact?0:2;
    } catch(const std::exception& e) { std::cerr<<"Engram layer probe: "<<e.what()<<'\n'; return 1; }
}
