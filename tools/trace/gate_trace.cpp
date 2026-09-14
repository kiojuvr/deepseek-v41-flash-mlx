#include "dsv41/moe.hpp"
#include "dsv41/checkpoint_atlas.hpp"
#include <fstream>
#include <iostream>
#include <vector>
namespace mx=mlx::core;
int main(int argc,char** argv){try{
 if(argc!=5) throw std::runtime_error("usage: dsv41-gate-trace checkpoint summary native-trace output-directory");
 std::filesystem::path out=std::filesystem::weakly_canonical(argv[4]); if(std::filesystem::exists(out)) throw std::runtime_error("output must be fresh");
 auto manifest=dsv41::read_json_file(std::filesystem::path(argv[3])/"manifest.json"); auto n=manifest.at("token_ids").size(); if(n<1||n>128) throw std::runtime_error("requires 1..128 tokens");
 auto input=mx::load((std::filesystem::path(argv[3])/"encoder.layer0.ffn_in.npy").string()); if(input.dtype()!=mx::bfloat16||input.shape()!=mx::Shape{int(n),5120}) throw std::runtime_error("invalid ffn input");
 mx::set_default_device(mx::Device::gpu); dsv41::WeightCatalog cat(argv[1],argv[2]); dsv41::GateReference gate(cat,0);
 std::vector<mx::array> raw,corrected; std::vector<int> selected; std::vector<float> margins;
 for(int t=0;t<int(n);++t){auto row=mx::slice(input,{t,0},{t+1,5120}); dsv41::RouteTieRecord tie{0,(std::uint64_t)t,0,0,0,0,false}; auto d=gate.diagnose(row,&tie); raw.push_back(d.raw_scores); corrected.push_back(d.corrected_scores); selected.insert(selected.end(),d.route.ids.begin(),d.route.ids.end()); margins.push_back(tie.sixth_score-tie.seventh_score);}
 std::filesystem::create_directories(out); auto save=[&](const char* name,const mx::array& a){mx::eval(a);mx::save((out/(std::string(name)+".npy")).string(),a);};
 save("encoder.layer0.gate_raw_scores",mx::concatenate(raw,0)); save("encoder.layer0.gate_corrected_scores",mx::concatenate(corrected,0));
 std::ofstream f(out/"manifest.json"); f<<nlohmann::json{{"schema_version",1},{"token_ids",manifest.at("token_ids")},{"arrays",{{{"name","encoder.layer0.gate_raw_scores"},{"dtype","float32"},{"shape",{n,384}}},{{"name","encoder.layer0.gate_corrected_scores"},{"dtype","float32"},{"shape",{n,384}}}}},{"selected_experts",selected},{"boundary_margin_sixth_minus_seventh",margins},{"scope","Native layer0 gate diagnostic; no oracle or full-model qualification"}}.dump(2)<<'\n';
 std::cout<<"Completed native gate trace\n"; return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
