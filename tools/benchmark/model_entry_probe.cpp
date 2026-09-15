#include "dsv41/model_entry.hpp"
#include "dsv41/checkpoint_atlas.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

namespace {
namespace mx=mlx::core;
using J=nlohmann::json;
void check(bool ok,const char* msg) { if(!ok) throw std::runtime_error(msg); }
std::string bytes(const std::filesystem::path& p) {
    check(std::filesystem::file_size(p)<64*1024*1024,"fixture too large");
    std::ifstream f(p,std::ios::binary); check(bool(f),"fixture read failed");
    return {std::istreambuf_iterator<char>(f),{}};
}
std::vector<std::uint16_t> raw(const std::filesystem::path& p) {
    auto b=bytes(p); check(!b.empty()&&b.size()%2==0,"bad BF16 fixture");
    std::vector<std::uint16_t> v(b.size()/2); std::memcpy(v.data(),b.data(),b.size()); return v;
}
mx::array bf16(const std::filesystem::path& p,const mx::Shape& shape) {
    auto v=raw(p); std::size_t n=1; for(int x:shape) n*=x;
    check(n==v.size(),"fixture shape mismatch");
    return mx::view(mx::array(v.begin(),shape,mx::uint16),mx::bfloat16);
}
std::vector<std::uint16_t> bits(const mx::array& x) {
    auto b=mx::contiguous(mx::view(x,mx::uint16,mx::Device::cpu),false,mx::Device::cpu); mx::eval(b); mx::synchronize();
    return {b.data<std::uint16_t>(),b.data<std::uint16_t>()+b.size()};
}
J compare(const std::vector<std::uint16_t>& a,const std::vector<std::uint16_t>& e) {
    check(a.size()==e.size(),"comparison size mismatch");
    std::size_t count=0; int ulp=0; J first=nullptr;
    auto ordered=[](unsigned b){return b&0x8000?0x8000-int(b&0x7fff):0x8000+int(b);};
    for(std::size_t i=0;i<a.size();++i) {
        check((a[i]&0x7f80)!=0x7f80&&(e[i]&0x7f80)!=0x7f80,"nonfinite norm result");
        if(a[i]!=e[i]) {
            ++count; ulp=std::max(ulp,std::abs(ordered(a[i])-ordered(e[i])));
            if(first.is_null()) first={{"index",i},{"actual_bits",a[i]},{"expected_bits",e[i]}};
        }
    }
    return {{"elements",a.size()},{"bit_mismatches",count},{"max_bf16_ulp",ulp},{"first_mismatch",first}};
}
template<class F> void rejects(F f) {
    bool rejected=false; try { f(); } catch(const std::exception&) { rejected=true; }
    check(rejected,"invalid input accepted");
}
}
int main(int argc,char** argv) {
    try {
        using namespace dsv41;
        std::filesystem::path checkpoint,summary,fixtures,output;
        for(int i=1;i<argc;++i) {
            std::string flag=argv[i]; check(i+1<argc,"missing argument"); auto v=argv[++i];
            if(flag=="--checkpoint") checkpoint=v; else if(flag=="--m1-summary") summary=v;
            else if(flag=="--fixtures") fixtures=v; else if(flag=="--output") output=v;
            else throw std::runtime_error("unknown argument");
        }
        check(!checkpoint.empty()&&!summary.empty()&&!fixtures.empty()&&!output.empty(),"all paths required");
        auto rel=std::filesystem::weakly_canonical(output).lexically_relative(std::filesystem::canonical(checkpoint));
        check(!rel.empty()&&*rel.begin()=="..","checkpoint is read-only");
        auto manifest=read_json_file(fixtures/"manifest.json"); check(manifest.at("schema_version")==1,"bad schema");
        for(const auto& [name,hash]:manifest.at("fixture_sha256").items()) {
            check(std::filesystem::path(name).filename()==name,"unsafe fixture name");
            check(sha256_text(bytes(fixtures/name))==hash.get<std::string>(),"fixture hash mismatch");
        }
        check(sha256_text(bytes(checkpoint/"inference/model.py"))==manifest.at("source_sha256").at("inference/model.py").get<std::string>(),"source changed");
        WeightCatalog catalog(checkpoint,summary); check(catalog.revision()==manifest.at("revision").get<std::string>(),"revision mismatch");
        mx::set_default_device(mx::Device::gpu);
        auto ids=manifest.at("token_ids").get<std::vector<std::uint32_t>>(); const int n=ids.size();
        auto entry=std::make_unique<TextEntryReference>(catalog);
        auto result=entry->forward(ids); auto expected=raw(fixtures/"embedding.bf16");
        check(expected.size()==ids.size()*5120,"embedding fixture shape");
        std::vector<std::uint16_t> expected_hidden,incremental;
        for(int t=0;t<n;++t) {
            for(int copy=0;copy<4;++copy) expected_hidden.insert(expected_hidden.end(),expected.begin()+t*5120,expected.begin()+(t+1)*5120);
            auto one=bits(entry->forward(std::span(ids).subspan(t,1)).hidden);
            incremental.insert(incremental.end(),one.begin(),one.end());
        }
        rejects([&]{entry->forward({});});
        for(auto invalid:{129280u,129264u,0xffffffffu}) rejects([&]{entry->forward(std::span(&invalid,1));});
        auto too_many=std::vector<std::uint32_t>(129,0); rejects([&]{entry->forward(too_many);});
        entry.reset(); // Evaluate a fresh graph after its model owner has been destroyed.
        auto actual=bits(result.hidden); check(actual==expected_hidden&&actual==incremental,"embedding/copies/tokenwise mismatch");
        auto mix=mx::contiguous(result.pre_mix,false,mx::Device::cpu); mx::eval(mix); mx::synchronize();
        for(int i=0;i<n*4;++i) check(mix.data<float>()[i]==(i%4==0?1.0f:0.0f),"initial mix mismatch");
        int nt=manifest.at("norm_tokens"); const float eps=manifest.at("norm_eps");
        auto input=bf16(fixtures/"norm-input.bf16",{nt,5120}), weight=bf16(fixtures/"norm-weight.bf16",{5120});
        auto norm=rms_norm_reference(input,weight,eps); auto norm_bits=bits(norm);
        std::vector<std::uint16_t> norm_incremental;
        for(int t=0;t<nt;++t) {
            auto v=bits(rms_norm_reference(mx::slice(input,{t,0},{t+1,5120}),weight,eps));
            norm_incremental.insert(norm_incremental.end(),v.begin(),v.end());
        }
        check(norm_bits==norm_incremental,"norm batch/tokenwise mismatch");
        rejects([&]{rms_norm_reference(input,weight,0);});
        rejects([&]{rms_norm_reference(input,weight,NAN);});
        rejects([&]{rms_norm_reference(mx::astype(input,mx::float32),weight,eps);});
        rejects([&]{rms_norm_reference(input,mx::zeros({1},mx::bfloat16),eps);});
        auto norm_comparison=compare(norm_bits,raw(fixtures/"norm-expected.bf16"));
        const bool exact=norm_comparison.at("bit_mismatches")==0;
        J report={{"status",exact?"fixtures_bitwise_exact":"numerical_mismatch"},{"revision",catalog.revision()},
            {"manifest_sha256",sha256_text(bytes(fixtures/"manifest.json"))},{"embedding_and_copies_exact",true},
            {"initial_pre_mix_exact",true},{"batch_tokenwise_exact",true},{"invalid_inputs_rejected",true},
            {"owner_destroyed_before_evaluation",true},{"official_cpu_rmsnorm",norm_comparison},
            {"embedding_resident_payload_bytes",129280ull*5120*2},{"full_model_qualified",false},{"optimized_path_promoted",false}};
        std::filesystem::create_directories(output.parent_path()); std::ofstream f(output); check(bool(f),"result open failed");
        f<<report.dump(2)<<'\n'; f.close(); check(bool(f),"result write failed"); std::cout<<report.dump()<<'\n';
        return exact?0:2;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
