#include "dsv41/moe.hpp"
#include "dsv41/checkpoint_atlas.hpp"
#include <fstream>
#include <iostream>
#include <vector>
namespace mx=mlx::core;
int main(int argc,char** argv){try{
 if(argc!=7) throw std::runtime_error("usage: dsv41-expert-trace checkpoint summary native-trace gate-trace output expert-id");
 int expert_id=std::stoi(argv[6]); if(expert_id<0||expert_id>=384) throw std::runtime_error("invalid expert id");
 auto dir=std::filesystem::path(argv[3]); auto gm=dsv41::read_json_file(std::filesystem::path(argv[4])/"manifest.json"); auto m=dsv41::read_json_file(dir/"manifest.json"); auto n=m.at("token_ids").size(); if(gm.at("token_ids")!=m.at("token_ids")) throw std::runtime_error("token IDs mismatch");
 auto input=mx::load((dir/"encoder.layer0.ffn_in.npy").string()); if(input.dtype()!=mx::bfloat16||input.shape()!=mx::Shape{int(n),5120}) throw std::runtime_error("invalid ffn input");
 auto selected=gm.at("selected_experts").get<std::vector<int>>(); auto weights=mx::load((std::filesystem::path(argv[4])/"encoder.layer0.route_weights.npy").string()); if(weights.dtype()!=mx::float32||weights.size()!=n*6) throw std::runtime_error("invalid route weights"); weights=mx::reshape(weights,{int(n),6});
 mx::set_default_device(mx::Device::gpu); dsv41::WeightCatalog cat(argv[1],argv[2]); dsv41::MoEReference moe(cat,0); std::vector<mx::array> rows; int occurrences=0;
 for(int t=0;t<int(n);++t){auto row=mx::slice(input,{t,0},{t+1,5120}); auto out=mx::zeros({1,5120},mx::bfloat16); for(int k=0;k<6;++k) if(selected[t*6+k]==expert_id){out=moe.expert_contribution(row,expert_id,mx::slice(weights,{t,k},{t+1,k+1}));++occurrences;} rows.push_back(out);}
 std::filesystem::path out=std::filesystem::weakly_canonical(argv[5]); if(std::filesystem::exists(out)) throw std::runtime_error("output must be fresh"); std::filesystem::create_directories(out); auto value=mx::concatenate(rows,0);mx::eval(value);mx::save((out/"encoder.layer0.expert_contribution.npy").string(),value);
 std::ofstream f(out/"manifest.json"); f<<nlohmann::json{{"schema_version",1},{"token_ids",m.at("token_ids")},{"expert_id",expert_id},{"occurrences",occurrences},{"arrays",{{{"name","encoder.layer0.expert_contribution"},{"dtype","bfloat16"},{"shape",{n,5120}}}}},{"scope","Native routed expert contribution diagnostic; no oracle or full-model qualification"}}.dump(2)<<'\n'; std::cout<<"Completed expert trace id="<<expert_id<<" occurrences="<<occurrences<<'\n'; return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
