#include "dsv41/text_backbone.hpp"
#include "dsv41/checkpoint_atlas.hpp"
#include "dsv41/execution_policy.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <nlohmann/json.hpp>
#include <array>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
namespace mx=mlx::core;using J=nlohmann::json;
namespace {
std::string hex(const unsigned char* p,std::size_t n){std::ostringstream s;s<<std::hex<<std::setfill('0');for(std::size_t i=0;i<n;++i)s<<std::setw(2)<<unsigned(p[i]);return s.str();}
J digest(const mx::array& value){auto bytes=mx::contiguous(mx::view(value,mx::uint8,mx::Device::cpu),false,mx::Device::cpu);mx::eval(bytes);mx::synchronize();unsigned char out[CC_SHA256_DIGEST_LENGTH];CC_SHA256(bytes.data<std::uint8_t>(),bytes.size(),out);return {{"shape",value.shape()},{"dtype",int(value.dtype().val())},{"sha256",hex(out,sizeof(out))}};}
J producer(const dsv41::CompressedLayerState& s,int consumer,std::uint64_t pos){J j={{"position",s.position()},{"window",digest(s.window())},{"main",digest(s.global().main_bytes())},{"main_scales",digest(s.global().main_scales())},{"index",digest(s.global().index_bytes())},{"index_scales",digest(s.global().index_scales())},{"pending_kv",digest(s.global().compressor().pending_kv())},{"pending_scores",digest(s.global().compressor().pending_scores())}};if(auto p=s.publication()){j["publication"]={{"source_layer",p->source_layer()},{"index_source_layer",p->index_source_layer()},{"indices",digest(p->device_indices(consumer,pos,pos?128:1))},{"candidates",digest(p->device_candidates())}};}return j;}
J state(const dsv41::TextBackboneState& s){J j={{"revision",s.revision()},{"hash_position",s.encoder.hash.position()}};for(const auto& x:s.encoder.swa)j["encoder_swa"].push_back({{"position",x.position()},{"rows",digest(x.rows())}});for(int i=0;i<3;++i)j["encoder_producer"].push_back(producer(s.encoder.producer[i],3+6*i,s.encoder.producer[i].position()-1));for(const auto& x:s.encoder.reuse)j["encoder_reuse"].push_back({{"position",x.position()},{"window",digest(x.window())}});j["decoder_producer"]=producer(s.decoder.producer,39,s.decoder.producer.position()-1);for(const auto& x:s.decoder.reuse)j["decoder_reuse"].push_back({{"position",x.position()},{"window",digest(x.window())}});return j;}
J ties(){J out=J::array();for(const auto& x:dsv41::route_tie_records())out.push_back({{"layer",x.layer},{"token",x.token},{"sixth_id",x.sixth_id},{"seventh_id",x.seventh_id},{"sixth_score",x.sixth_score},{"seventh_score",x.seventh_score},{"tied",x.tied}});return out;}
}
int main(int argc,char** argv){try{
 if(argc!=5)throw std::runtime_error("usage: dsv41-backbone-digest checkpoint summary metadata output-json");mx::set_default_device(mx::Device::gpu);dsv41::WeightCatalog catalog(argv[1],argv[2]);auto metadata=dsv41::EngramMetadata::load(argv[3]);dsv41::TextBackboneReference model(catalog,metadata);dsv41::TextBackboneState current(metadata);dsv41::reset_route_tie_records();
 std::vector<std::uint32_t> prompt(129);for(std::size_t i=0;i<prompt.size();++i)prompt[i]=std::uint32_t((i*7919)%129263);J report={{"schema_version",1},{"status","exactness_digest_requires_offline_comparison"},{"wired_limit_bytes",model.wired_limit_bytes()},{"expert_backing_file_backed",model.expert_backing_file_backed()},{"resident_expert_atlas",dsv41::runtime_resident_expert_atlas_enabled()},{"compact_expert_bank",dsv41::runtime_compact_expert_bank_enabled()},{"steps",J::array()}};
 auto record=[&](std::uint64_t start,const dsv41::BlockResult& result){report["steps"].push_back({{"start",start},{"hidden",digest(result.hidden)},{"pre_mix",digest(result.pre_mix)},{"logits",digest(model.logits(result))},{"state",state(current)}});};
 record(0,model.forward_packed_sweep(prompt,current,0));for(std::uint64_t pos=129;pos<133;++pos){const std::array<std::uint32_t,1> token{std::uint32_t(pos*17)};record(pos,model.forward_packed_sweep(token,current,pos));}
 report["route_ties"]=ties();auto before=state(current);auto saved=current;const std::array<std::uint32_t,1> invalid{129264};bool rejected=false;try{model.forward_packed_sweep(invalid,current,133);}catch(const std::exception&){rejected=true;}auto after=state(current);if(!rejected||before!=after||current.revision()!=saved.revision())throw std::runtime_error("invalid request changed digest state");report["invalid_request_atomicity"]=true;auto hash=current.encoder.hash;const std::array<std::uint32_t,1> suffix{42};report["hash_continuation"]=hash.append(suffix,{},133);
 std::ofstream out(argv[4],std::ios::binary|std::ios::trunc);if(!out)throw std::runtime_error("cannot create digest output");out<<report.dump(2)<<'\n';out.close();std::cout<<"PASS: emitted full bounded backbone digest; offline comparison required\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
