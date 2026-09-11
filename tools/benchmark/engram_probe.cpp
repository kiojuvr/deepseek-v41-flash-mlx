#include "dsv41/checkpoint_atlas.hpp"
#include "dsv41/engram.hpp"
#ifdef DSV41_HAVE_MLX
#include "dsv41/engram_mlx.hpp"
#include <mlx/mlx.h>
#endif
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <libproc.h>
#include <numeric>
#include <set>
#include <stdexcept>
#include <sys/resource.h>
#include <unistd.h>

namespace {
using J=nlohmann::json;
using Clock=std::chrono::steady_clock;
void check(bool ok,const std::string& msg) { if(!ok) throw std::runtime_error(msg); }
std::string read(const std::filesystem::path& p) {
    check(std::filesystem::file_size(p)<=64*1024*1024,"fixture too large");
    std::ifstream f(p,std::ios::binary); check(bool(f),"cannot read fixture");
    return {std::istreambuf_iterator<char>(f),{}};
}
template<class T> std::vector<T> binary(const std::filesystem::path& p) {
    static_assert(std::endian::native==std::endian::little);
    auto bytes=read(p); check(bytes.size()%sizeof(T)==0,"unaligned fixture");
    std::vector<T> result(bytes.size()/sizeof(T)); std::memcpy(result.data(),bytes.data(),bytes.size()); return result;
}
J telemetry() {
    struct rusage_info_v4 info{};
    check(proc_pid_rusage(getpid(),RUSAGE_INFO_V4,reinterpret_cast<rusage_info_t*>(&info))==0,"cannot query process telemetry");
    struct rusage usage{}; check(getrusage(RUSAGE_SELF,&usage)==0,"getrusage failed");
    return {{"phys_footprint",info.ri_phys_footprint},{"resident_size",info.ri_resident_size},
            {"diskio_bytesread",info.ri_diskio_bytesread},{"pageins",info.ri_pageins},
            {"minor_faults",usage.ru_minflt},{"major_faults",usage.ru_majflt}};
}
std::vector<std::uint16_t> gpu_decode(const dsv41::PackedEngramRows& packed) {
#ifdef DSV41_HAVE_MLX
    namespace mx=mlx::core;
    auto a=dsv41::engram_lookup_mlx(packed);
    auto bits=mx::view(a,mx::uint16,mx::Device::cpu);
    mx::eval(bits); mx::synchronize();
    return {bits.data<std::uint16_t>(),bits.data<std::uint16_t>()+bits.size()};
#else
    (void)packed; throw std::runtime_error("build with DSV41_ENABLE_MLX=ON for --mlx");
#endif
}
}

int main(int argc,char** argv) {
    using namespace dsv41;
    try {
        std::filesystem::path checkpoint,summary,fixtures,output;
        std::string mode="mmap"; bool mlx=false,verify_only=false;
        std::size_t benchmark_tokens=0;
        for(int i=1;i<argc;++i) {
            std::string flag=argv[i];
            if(flag=="--mlx") { mlx=true; continue; }
            if(flag=="--verify-only") { verify_only=true; continue; }
            check(i+1<argc,"missing option value");
            if(flag=="--checkpoint") checkpoint=argv[++i];
            else if(flag=="--m1-summary") summary=argv[++i];
            else if(flag=="--fixtures") fixtures=argv[++i];
            else if(flag=="--output") output=argv[++i];
            else if(flag=="--mode") mode=argv[++i];
            else if(flag=="--benchmark-tokens") {
                const std::string value=argv[++i]; std::size_t consumed=0;
                benchmark_tokens=std::stoull(value,&consumed);
                check(consumed==value.size() && benchmark_tokens>=1 && benchmark_tokens<=32768,"invalid benchmark token limit");
            }
            else throw std::runtime_error("unknown option: "+flag);
        }
        check(!checkpoint.empty()&&!summary.empty()&&!fixtures.empty()&&!output.empty(),"checkpoint, m1-summary, fixtures and output required");
        check(mode=="mmap"||mode=="pread","mode must be mmap or pread");
        auto canonical_checkpoint=std::filesystem::canonical(checkpoint);
        auto relative=std::filesystem::weakly_canonical(output).lexically_relative(canonical_checkpoint);
        check(!relative.empty()&&*relative.begin()=="..","output must be outside checkpoint");
        auto fixture_identity=read_json_file(fixtures/"fixture-provenance.json");
        for(const auto& [name,digest] : fixture_identity.at("fixture_sha256").items()) {
            check(std::filesystem::path(name).filename()==name,"unsafe fixture path");
            check(sha256_text(read(fixtures/name))==digest.get<std::string>(),"fixture digest mismatch: "+name);
        }
        auto metadata=EngramMetadata::load(fixtures/"metadata.json");
        WeightCatalog catalog(checkpoint,summary);
        check(metadata->revision==catalog.revision()&&metadata->revision==fixture_identity.at("revision").get<std::string>(),"snapshot mismatch");
        for(const auto& [name,digest] : fixture_identity.at("source_sha256").items()) {
            const std::filesystem::path source(name);
            check(!source.is_absolute(),"unsafe source path");
            for(const auto& part : source) check(part!="..","unsafe source path");
            check(sha256_text(read(checkpoint/source))==digest.get<std::string>(),"oracle input changed: "+name);
        }
        auto allbits=binary<std::uint16_t>(fixtures/"dequantize-all-bits.bf16");
        check(allbits.size()==65536,"bad exhaustive fixture size");
        PackedEngramRows packed_all;
        for(unsigned s=0;s<256;++s) {
            for(unsigned v=0;v<256;++v) packed_all.values.push_back(v);
            for(unsigned g=0;g<8;++g) packed_all.scales.push_back(s);
        }
        check(dequantize_engram(packed_all)==allbits,"CPU exhaustive BF16 oracle mismatch");
        if(mlx) {
#ifdef DSV41_HAVE_MLX
            mlx::core::set_default_device(mlx::core::Device::gpu);
#endif
            check(gpu_decode(packed_all)==allbits,"Metal exhaustive BF16 oracle mismatch");
        }
        auto traces=read_json_file(fixtures/"traces.json").at("cases");
        std::uint64_t row_comparisons=0;
        for(const auto& trace : traces) {
            const auto ids=trace.at("token_ids").get<std::vector<std::uint32_t>>();
            const auto mask=trace.at("token_mask").get<std::vector<std::uint8_t>>();
            const auto expected=binary<std::uint64_t>(fixtures/trace.at("expected_rows_file").get<std::string>());
            EngramHashState state(metadata);
            check(state.append(ids,mask,0)==expected,"official full hash mismatch");
            state.reset(); std::vector<std::uint64_t> incremental;
            for(std::size_t i=0;i<ids.size();++i) {
                auto rows=state.append(std::span(ids).subspan(i,1),std::span(mask).subspan(i,1),i);
                incremental.insert(incremental.end(),rows.begin(),rows.end());
            }
            check(incremental==expected,"official tokenwise hash mismatch"); row_comparisons+=expected.size();
        }
        std::vector<EngramStore> stores;
        for(std::size_t l=0;l<2;++l) stores.emplace_back(catalog,metadata->layer_ids[l],metadata->num_embeddings[l],
                                      mode=="mmap"?EngramReadMode::Mmap:EngramReadMode::Pread);
        const auto samples=read_json_file(fixtures/"samples.json");
        const auto expected_samples=binary<std::uint16_t>(fixtures/"samples.bf16");
        std::size_t cursor=0;
        for(std::size_t l=0;l<2;++l) {
            check(samples.at(l).at("layer_id")==metadata->layer_ids[l],"sample layer mismatch");
            const auto ids=samples[l].at("row_ids").get<std::vector<std::uint64_t>>();
            const auto packed=stores[l].gather(ids); const auto decoded=dequantize_engram(packed);
            check(decoded.size()<=expected_samples.size()-cursor,"sample count overflow");
            check(std::equal(decoded.begin(),decoded.end(),expected_samples.begin()+cursor),"real row oracle mismatch");
            if(mlx) check(gpu_decode(packed)==decoded,"real row native Metal mismatch");
            cursor+=decoded.size();
        }
        check(cursor==expected_samples.size(),"unused sample values");
        J result={{"schema_version",1},{"revision",metadata->revision},{"metadata_identity",metadata->identity},
                  {"fixture_provenance_sha256",sha256_text(read(fixtures/"fixture-provenance.json"))},
                  {"mode",mode},{"page_size",sysconf(_SC_PAGESIZE)},
                  {"verified_row_ids",row_comparisons},{"exhaustive_dequant_cases",65536},
                  {"verified_checkpoint_rows",cursor/256},{"metal_exact",mlx?J(true):J(nullptr)},
                  {"full_model_qualified",false},{"resident_backbone_loaded",false},
                  {"scope","Engram hash + gather + CPU BF16 reference lookup only; not decode TPT"},
                  {"cache_condition","natural OS cache; no purge/prefetch/mlock/F_NOCACHE; oracle samples pre-read"},
                  {"runs",J::array()}};
        if(!verify_only) for(const auto index : {0,1,0}) {
            const auto& trace=traces.at(index);
            auto ids=trace.at("token_ids").get<std::vector<std::uint32_t>>();
            auto mask=trace.at("token_mask").get<std::vector<std::uint8_t>>();
            if(benchmark_tokens) { ids.resize(std::min(ids.size(),benchmark_tokens)); mask.resize(ids.size()); }
            check(!ids.empty(),"empty benchmark trace");
            EngramHashState state(metadata);
            std::set<std::pair<std::size_t,std::uint64_t>> unique_rows,unique_pages;
            std::vector<double> times; times.reserve(ids.size());
            auto before=telemetry(); std::uint64_t checksum=0;
            for(std::size_t t=0;t<ids.size();++t) {
                const auto start=Clock::now();
                auto hashes=state.append(std::span(ids).subspan(t,1),std::span(mask).subspan(t,1),t);
                std::array<std::vector<std::uint16_t>,2> decoded;
                for(std::size_t l=0;l<2;++l)
                    decoded[l]=dequantize_engram(stores[l].gather(std::span(hashes).subspan(l*24,24)));
                times.push_back(std::chrono::duration<double,std::milli>(Clock::now()-start).count());
                // Diagnostic accounting is outside measured lookup wall time.
                for(std::size_t l=0;l<2;++l) {
                    for(auto value : decoded[l]) checksum=checksum*131+value;
                    for(auto row : std::span(hashes).subspan(l*24,24)) unique_rows.emplace(l,row);
                    for(auto page : stores[l].pages(std::span(hashes).subspan(l*24,24))) unique_pages.emplace(l,page);
                }
            }
            auto after=telemetry(); auto sorted=times; std::sort(sorted.begin(),sorted.end());
            auto percentile=[&](double p){return sorted.at(std::min(sorted.size()-1,std::size_t(p*double(sorted.size()-1))));};
            result["runs"].push_back({{"trace",trace.at("name")},{"input_tokens",ids.size()},
                {"mean_lookup_ms",std::accumulate(times.begin(),times.end(),0.0)/times.size()},
                {"p50_ms",percentile(.5)},{"p95_ms",percentile(.95)},{"p99_ms",percentile(.99)},{"max_ms",sorted.back()},
                {"logical_row_bytes",ids.size()*48*264},{"unique_rows",unique_rows.size()},
                {"unique_file_pages",unique_pages.size()},
                {"addressed_page_bytes",unique_pages.size()*std::uint64_t(sysconf(_SC_PAGESIZE))},
                {"checksum",checksum},{"telemetry_before",before},{"telemetry_after",after},{"lookup_ms",times}});
            std::cerr<<mode<<' '<<trace.at("name")<<": "<<result["runs"].back().at("mean_lookup_ms")<<" ms/input token\n";
        }
        result["status"]="passed";
        std::filesystem::create_directories(output.parent_path());
        std::ofstream out(output); check(bool(out),"cannot write result"); out<<result.dump(2)<<'\n'; out.close(); check(bool(out),"result write failed");
        std::cout<<"Engram oracle checks passed; "<<row_comparisons<<" hash IDs, 65536 dequant cases, "<<cursor/256<<" real rows\n";
    } catch(const std::exception& e) { std::cerr<<"Engram probe: "<<e.what()<<'\n'; return 1; }
}
