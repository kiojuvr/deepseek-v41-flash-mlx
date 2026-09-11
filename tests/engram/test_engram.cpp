#include "dsv41/engram.hpp"
#include "dsv41/checkpoint_atlas.hpp"
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

namespace {
void check(bool ok) { if (!ok) throw std::runtime_error("Engram test failed"); }
template<class F> void rejects(F f) { bool failed = false; try { f(); } catch (const std::exception&) { failed = true; } check(failed); }
}
int main() {
    using namespace dsv41;
    using J = nlohmann::json;
    try {
        auto m = std::make_shared<EngramMetadata>();
        m->pad_id=0; m->token_map={0,1,2,3,4,5}; m->layer_ids={1,14};
        m->multipliers={{{1,3,5,7}},{{1,3,5,7}}};
        std::array<std::uint64_t,24> primes{}, offsets{};
        for (std::size_t i=0;i<24;++i) { primes[i]=17; offsets[i]=i*17; }
        m->primes={primes,primes}; m->offsets={offsets,offsets};
        EngramHashState all(m), chunked(m);
        const std::array<std::uint32_t,4> ids{1,2,3,4};
        const auto expected=all.append(ids,{},0);
        const auto first=chunked.append(std::span(ids).first(2),{},0);
        const auto second=chunked.append(std::span(ids).last(2),{},2);
        auto joined=first; joined.insert(joined.end(),second.begin(),second.end()); check(joined==expected);
        for (std::size_t l=0;l<2;++l) for (std::size_t h=0;h<8;++h) {
            check(expected[3*48+l*24+h]==13+offsets[h]);
            check(expected[3*48+l*24+8+h]==7+offsets[8+h]);
            check(expected[3*48+l*24+16+h]==offsets[16+h]);
        }
        const auto saved=chunked;
        const std::array<std::uint32_t,2> invalid{2,99};
        rejects([&]{chunked.append(invalid,{},4);}); check(chunked.position()==4);
        rejects([&]{chunked.append(ids,{},0);}); check(chunked.position()==4);
        const std::array<std::uint8_t,1> dead{0};
        const std::array<std::uint32_t,1> single{5};
        auto dead_rows=chunked.append(single,dead,4);
        auto after_dead=chunked.append(single,{},5);
        for (std::size_t i=0;i<48;++i) { check(dead_rows[i]==offsets[i%24]); check(after_dead[i]==5+offsets[i%24]); }
        chunked=saved; check(chunked.position()==4);
        all.reset(); check(all.append(ids,{},0)==expected);
        check(engram_bf16(0x38,127)==0x3f80 && engram_bf16(0x7e,127)==0x43e0);
        check(engram_bf16(0x80,127)==0x8000 && engram_bf16(0x38,0)==0x0040);
        check(engram_bf16(0xff,127)==0x7fc0 && engram_bf16(0x38,255)==0x7fc0);
        check(engram_bf16(0x7e,254)==0x7f80);
        rejects([]{dequantize_engram(PackedEngramRows{{1},{1}});});

        char name[]="/private/tmp/dsv41-engram-test-XXXXXX";
        auto* directory=mkdtemp(name); check(directory!=nullptr);
        const std::filesystem::path root(directory);
        struct Cleanup { std::filesystem::path p; ~Cleanup(){std::filesystem::remove_all(p);} } cleanup{root};
        const auto path=root/"one.safetensors";
        J h=J::object();
        h["layers.1.engram.embed.weight"]={{"dtype","F8_E4M3"},{"shape",{2,256}},{"data_offsets",{0,512}}};
        h["layers.1.engram.embed.scale"]={{"dtype","F8_E8M0"},{"shape",{2,8}},{"data_offsets",{512,528}}};
        const auto text=h.dump();
        std::ofstream f(path,std::ios::binary);
        const std::uint64_t len=text.size(); f.write(reinterpret_cast<const char*>(&len),8); f<<text;
        std::vector<std::uint8_t> bytes(528,127);
        std::fill(bytes.begin(),bytes.begin()+256,0x38); std::fill(bytes.begin()+256,bytes.begin()+512,0x40);
        f.write(reinterpret_cast<const char*>(bytes.data()),bytes.size()); f.close();
        J map=J::object(); for (const auto& [key,_] : h.items()) map[key]="one.safetensors";
        std::ofstream(root/"model.safetensors.index.json")<<J({{"weight_map",map}}).dump();
        auto header=read_safetensors_header(path); header.erase("tensors");
        std::ofstream(root/"summary.json")<<J({{"status","verified"},{"revision","test"},{"shards",{{"one.safetensors",header}}}}).dump();
        MappedTensor surviving;
        {
            WeightCatalog catalog(root,root/"summary.json");
            auto tensor=catalog.tensor("layers.1.engram.embed.weight"); surviving=tensor.map();
            rejects([&]{tensor.read(512,std::span<std::byte>(reinterpret_cast<std::byte*>(bytes.data()),1));});
            rejects([&]{catalog.tensor("unknown");});
            rejects([&]{EngramStore wrong(catalog,1,3,EngramReadMode::Mmap);});
            EngramStore mapped(catalog,1,2,EngramReadMode::Mmap), positional(catalog,1,2,EngramReadMode::Pread);
            const std::array<std::uint64_t,4> selected{1,0,1,1};
            auto a=mapped.gather(selected), b=positional.gather(selected);
            check(a.values==b.values && a.scales==b.scales);
            const auto decoded=dequantize_engram(a);
            check(decoded.size()==1024 && decoded[0]==0x4000 && decoded[256]==0x3f80);
            check(mapped.gather({}).values.empty());
            const std::array<std::uint64_t,1> beyond{2};
            rejects([&]{mapped.gather(beyond);});
            rejects([&]{positional.gather(beyond);});
            rejects([&]{mapped.gather(std::vector<std::uint64_t>(4097));});
            check(!mapped.pages(selected).empty());
        }
        check(surviving.bytes(0,1)[0]==std::byte(0x38)); // mapping outlives catalog and TensorFile
        rejects([&]{surviving.bytes(511,2);});
        std::filesystem::resize_file(path,8);
        rejects([&]{surviving.check_unchanged();}); // detect mutation without touching invalid mapping
        std::cout<<"Engram hash/mask/rollback, numeric edges, mmap/pread, bounds and ownership passed\n";
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
