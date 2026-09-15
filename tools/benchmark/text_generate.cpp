#include "dsv41/text_generate.hpp"
#include "dsv41/checkpoint_atlas.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
namespace mx=mlx::core;
int main(int argc,char** argv){try{
 if(argc<5||argc>8)throw std::runtime_error("usage: dsv41-text-generate checkpoint m1-summary metadata max-new [temperature] [seed] [tokens-file]");
 mx::set_default_device(mx::Device::gpu);
 dsv41::WeightCatalog catalog(argv[1],argv[2]);
 auto metadata=dsv41::EngramMetadata::load(argv[3]);
 const std::size_t max_new=std::stoul(argv[4]);
 float temperature=argc>5?std::stof(argv[5]):0.0f;
 std::uint64_t seed=argc>6?std::stoull(argv[6]):0;
 auto proof=dsv41::read_json_file("artifacts/engram/fixture-provenance.json");
 std::ifstream meta(argv[3],std::ios::binary);std::string raw{std::istreambuf_iterator<char>(meta),{}};
 if(dsv41::sha256_text(raw)!=proof.at("fixture_sha256").at("metadata.json").get<std::string>())throw std::runtime_error("Engram metadata identity mismatch");
 std::cout<<"Loading full backbone layers 0..39 on-demand experts and Engram 1/14 mmap backing"<<std::endl;
 dsv41::TextGenerationReference generator(catalog,metadata);
 std::vector<std::uint32_t> prompt;
 if(argc>7){std::ifstream in(argv[7]); if(!in)throw std::runtime_error("cannot open tokens file"); std::uint64_t token; while(in>>token){if(token>129279)throw std::runtime_error("token out of range"); prompt.push_back(static_cast<std::uint32_t>(token));} if(prompt.empty())throw std::runtime_error("tokens file is empty");}
 else prompt={0,42,1000,42};
 dsv41::SamplingConfig config{temperature,seed};
 auto result=generator.generate(prompt,max_new,config);
 if(std::getenv("DSV41_CHECK_GENERATION_LIFECYCLE")){
  if(max_new<2)throw std::runtime_error("lifecycle check requires max-new >= 2");
  auto require=[](bool ok){if(!ok)throw std::runtime_error("model generation lifecycle mismatch");};
  dsv41::GenerationControl cancelled;
  cancelled.is_cancelled=[]{return true;};
  auto early=generator.generate(prompt,max_new,config,{},cancelled);
  require(early.cancelled && early.tokens.empty() && early.next_position==prompt.size());
  cancelled.is_cancelled={};
  cancelled.on_token=[&](auto token,auto index){require(index==0 && token==result.tokens.at(0));return false;};
  auto partial=generator.generate(prompt,max_new,config,{},cancelled);
  require(partial.cancelled && partial.tokens.size()==1 && partial.next_position==prompt.size()+1);
  std::vector<std::uint32_t> observed;
  dsv41::GenerationControl streamed;
  streamed.on_token=[&](auto token,auto index){require(index==observed.size());observed.push_back(token);return true;};
  auto replay=generator.generate(prompt,max_new,config,{},streamed);
  require(observed==result.tokens && replay.tokens==result.tokens && !replay.cancelled && !replay.stopped && replay.next_position==result.next_position);
  std::cout<<"PASS: full-model callback parity, cancellation, and fresh request after cancellation"<<std::endl;
 }
 std::cout<<"prompt_tokens: ";for(std::size_t i=0;i<prompt.size();++i)std::cout<<(i?" ":"")<<prompt[i];std::cout<<std::endl;
 std::cout<<"generated:";for(auto token:result.tokens)std::cout<<" "<<token;std::cout<<std::endl;
 std::cout<<"next_position: "<<result.next_position<<" stopped: "<<(result.stopped?"true":"false")<<std::endl;
 std::cout<<"PASS: greedy/temperature generation produced "<<result.tokens.size()<<" committed tokens (RNG not oracle-verified; not M2 qualification)"<<std::endl;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
