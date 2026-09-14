#include "dsv41/moe.hpp"
#include "dsv41/checkpoint_atlas.hpp"
#include <fstream>
#include <iostream>
#include <vector>
namespace mx=mlx::core;
int main(int argc,char** argv){try{
 if(argc!=5) throw std::runtime_error("usage: dsv41-moe-trace checkpoint summary native-trace output-directory");
 std::filesystem::path out=std::filesystem::weakly_canonical(argv[4]); if(std::filesystem::exists(out)) throw std::runtime_error("output must be fresh");
 auto input_dir=std::filesystem::path(argv[3]); auto manifest=dsv41::read_json_file(input_dir/"manifest.json"); auto n=manifest.at("token_ids").size(); if(n<1||n>128) throw std::runtime_error("requires 1..128 tokens");
 auto input=mx::load((input_dir/"encoder.layer0.ffn_in.npy").string()); if(input.dtype()!=mx::bfloat16||input.shape()!=mx::Shape{int(n),5120}) throw std::runtime_error("invalid ffn input");
 mx::set_default_device(mx::Device::gpu); dsv41::WeightCatalog cat(argv[1],argv[2]); dsv41::MoEReference moe(cat,0); dsv41::GateReference gate(cat,0); std::vector<mx::array> shared,routed,total,slots; std::vector<int> route_ids;
 for(int t=0;t<int(n);++t){dsv41::set_route_trace_token(t); auto row=mx::slice(input,{t,0},{t+1,5120}); auto c=moe.forward_components(row); auto r=gate.forward(row); std::vector<mx::array> slot_rows; for(int k=0;k<6;++k){route_ids.push_back(r.ids[k]); auto one=moe.expert_contribution(row,r.ids[k],mx::take(r.weights,mx::array(k))); slot_rows.push_back(one);} auto slot=mx::concatenate(slot_rows,0); shared.push_back(c.shared); routed.push_back(c.routed); total.push_back(c.total); slots.push_back(slot); if((t+1)%8==0) std::cout<<"Completed token "<<t+1<<"/"<<n<<'\n';}
 std::filesystem::create_directories(out); auto save=[&](const char* name,const std::vector<mx::array>& rows){auto a=mx::concatenate(rows,0);mx::eval(a);mx::save((out/name),a);}; save("encoder.layer0.moe_shared.npy",shared); save("encoder.layer0.moe_routed.npy",routed); save("encoder.layer0.moe_out.npy",total); save("encoder.layer0.moe_slots.npy",slots);
 std::ofstream f(out/"manifest.json"); f<<nlohmann::json{{"schema_version",2},{"token_ids",manifest.at("token_ids")},{"route_ids",route_ids},{"arrays",{{{"name","encoder.layer0.moe_shared"},{"dtype","bfloat16"},{"shape",{n,5120}}},{{"name","encoder.layer0.moe_routed"},{"dtype","bfloat16"},{"shape",{n,5120}}},{{"name","encoder.layer0.moe_out"},{"dtype","bfloat16"},{"shape",{n,5120}}},{{"name","encoder.layer0.moe_slots"},{"dtype","bfloat16"},{"shape",{n,6,5120}}}}},{"route_tie_count",moe.tie_count()},{"scope","Native MLX layer0 MoE component and per-route replay on frozen ffn_in; no oracle or full-model qualification"}}.dump(2)<<'\n';
 std::cout<<"Completed native MoE trace\n"; return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
