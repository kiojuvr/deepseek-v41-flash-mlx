#include "dsv41/text_backbone.hpp"
#include "dsv41/checkpoint_atlas.hpp"
#include "dsv41/generation_loop.hpp"
#include "dsv41/moe.hpp"
#include <bit>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
namespace mx=mlx::core;
namespace {
struct ArrayInfo { std::string name,dtype; std::vector<int> shape; };
// Accumulates per-token records and concatenates them along axis 0 at finalize, so sub-boundary
// recordings from inside a token loop become the same [tokens, ...] layout as the oracle trace.
class NpyTraceSink : public dsv41::TraceSink {
public:
 explicit NpyTraceSink(std::filesystem::path dir):dir_(std::move(dir)){}
 void record(const std::string& name,const mx::array& value) override{
  if(name.find('/')!=std::string::npos||name.find("..")!=std::string::npos)throw std::runtime_error("invalid trace name");
  if(!parts_.count(name))order_.push_back(name);
  parts_[name].push_back(value);
 }
 void finalize(){
  for(const auto& name:order_){
   auto& parts=parts_[name];
   auto joined=parts.size()==1?parts.front():mx::concatenate(parts,0);
   mx::eval(joined);
   mx::save((dir_/(name+".npy")).string(),joined);
   std::vector<int> shape;for(int i=0;i<int(joined.ndim());++i)shape.push_back(int(joined.shape(i)));
   arrays_.push_back({name,dtype_name(joined.dtype()),shape});
  }
 }
 const std::vector<ArrayInfo>& arrays() const{return arrays_;}
private:
 static std::string dtype_name(mx::Dtype d){
  if(d==mx::bfloat16)return "bfloat16";
  if(d==mx::float32)return "float32";
  if(d==mx::uint16)return "uint16";
  if(d==mx::uint8)return "uint8";
  return "other";
 }
 std::filesystem::path dir_;
 std::vector<std::string> order_;
 std::map<std::string,std::vector<mx::array>> parts_;
 std::vector<ArrayInfo> arrays_;
};
std::string json_escape(const std::string& s){
 std::string out;for(char c:s){if(c=='"'||c=='\\')out+='\\';out+=c;}return out;
}
std::vector<std::uint32_t> read_tokens(const std::string& path){
 std::ifstream file(path);if(!file)throw std::runtime_error("cannot open tokens file");
 std::vector<std::uint32_t> tokens;std::uint64_t value;
 while(file>>value){if(value>=129280||value==129264)throw std::runtime_error("token file must contain legal text token IDs");tokens.push_back(std::uint32_t(value));}
 if(tokens.empty()||tokens.size()>256)throw std::runtime_error("token file must contain 1..256 text tokens");
 return tokens;
}
}
int main(int argc,char** argv){try{
 if(argc<5||argc>6)throw std::runtime_error("usage: dsv41-text-trace checkpoint m1-summary metadata output-dir [tokens-file]");
 mx::set_default_device(mx::Device::gpu);
 dsv41::WeightCatalog catalog(argv[1],argv[2]);
 auto metadata=dsv41::EngramMetadata::load(argv[3]);
 std::filesystem::path out(argv[4]);
 if(std::filesystem::exists(out)&&!std::filesystem::is_directory(out))throw std::runtime_error("output path is not a directory");
 std::filesystem::create_directories(out);
 auto proof=dsv41::read_json_file("artifacts/engram/fixture-provenance.json");
 std::ifstream meta(argv[3],std::ios::binary);std::string raw{std::istreambuf_iterator<char>(meta),{}};
 if(dsv41::sha256_text(raw)!=proof.at("fixture_sha256").at("metadata.json").get<std::string>())throw std::runtime_error("Engram metadata identity mismatch");
 std::cout<<"Loading full backbone layers 0..39 on-demand experts and Engram 1/14 mmap backing"<<std::endl;
 dsv41::TextBackboneReference model(catalog,metadata);
 const std::vector<std::uint32_t> ids=argc==6?read_tokens(argv[5]):std::vector<std::uint32_t>{0,42,1000,42};
 dsv41::TextBackboneState state(metadata);
 NpyTraceSink sink(out);
 dsv41::reset_route_tie_records();
 std::vector<mx::array> logits_parts;
 dsv41::run_prefill_chunks(ids.size(),128,[&](std::size_t offset,std::size_t count){
  auto final_hidden=model.forward(std::span(ids).subspan(offset,count),state,offset,&sink);
  logits_parts.push_back(model.logits(final_hidden,&sink));
 });
 auto logits=logits_parts.size()==1?logits_parts.front():mx::concatenate(logits_parts,0);
 mx::eval(logits);
 sink.finalize();
 // argmax per token on CPU for a quick manifest summary.
 auto cpu=mx::astype(logits,mx::float32,mx::Device::cpu);mx::eval(cpu);
 std::vector<int> argmax;for(int t=0;t<cpu.shape(0);++t){
  const float* row=cpu.data<float>()+std::size_t(t)*129280;int best=0;for(int i=1;i<129280;++i)if(row[i]>row[best])best=i;argmax.push_back(best);
 }
 auto ties=dsv41::route_tie_records();
 std::ofstream manifest(out/"manifest.json");
 manifest<<"{\n";
 manifest<<"  \"schema_version\": 1,\n";
 manifest<<"  \"token_ids\": [";for(std::size_t i=0;i<ids.size();++i){if(i)manifest<<", ";manifest<<ids[i];}manifest<<"],\n";
 manifest<<"  \"logits_argmax\": [";for(std::size_t i=0;i<argmax.size();++i){if(i)manifest<<", ";manifest<<argmax[i];}manifest<<"],\n";
 manifest<<"  \"route_tie_count\": "<<ties.size()<<",\n";
 manifest<<"  \"route_ties\": [";
 for(std::size_t i=0;i<ties.size();++i){
  if(i)manifest<<", ";
  manifest<<"{\"layer\": "<<ties[i].layer<<", \"token\": "<<ties[i].token
          <<", \"sixth_id\": "<<ties[i].sixth_id<<", \"seventh_id\": "<<ties[i].seventh_id
          <<", \"sixth_score\": "<<ties[i].sixth_score<<", \"seventh_score\": "<<ties[i].seventh_score
          <<", \"sixth_score_bits\": "<<std::bit_cast<std::uint32_t>(ties[i].sixth_score)
          <<", \"seventh_score_bits\": "<<std::bit_cast<std::uint32_t>(ties[i].seventh_score)<<"}";
 }
 manifest<<"],\n";
 manifest<<"  \"arrays\": [\n";
 const auto& arrays=sink.arrays();
 for(std::size_t i=0;i<arrays.size();++i){
  manifest<<"    {\"name\": \""<<json_escape(arrays[i].name)<<"\", \"dtype\": \""<<arrays[i].dtype<<"\", \"shape\": [";
  for(std::size_t j=0;j<arrays[i].shape.size();++j){if(j)manifest<<", ";manifest<<arrays[i].shape[j];}
  manifest<<"]}";
  manifest<<(i+1<arrays.size()?",\n":"\n");
 }
 manifest<<"  ],\n";
 manifest<<"  \"scope\": \"Native reference backbone trace for fixed teacher-forced tokens; oracle comparison not yet performed.\"\n";
 manifest<<"}\n";
 std::cout<<"PASS: wrote "<<arrays.size()<<" boundary arrays and "<<ties.size()<<" route ties to "<<out<<std::endl;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
