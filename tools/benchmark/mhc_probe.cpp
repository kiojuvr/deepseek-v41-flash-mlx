#include "dsv41/mhc.hpp"
#include "dsv41/checkpoint_atlas.hpp"
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
namespace {
namespace mx=mlx::core;using J=nlohmann::json;
void check(bool b,const char* m){if(!b)throw std::runtime_error(m);}
std::string raw(const std::filesystem::path& p){check(std::filesystem::file_size(p)<64000000,"fixture too large");std::ifstream f(p,std::ios::binary);check(bool(f),"read failed");return {std::istreambuf_iterator<char>(f),{}};}
mx::array load(const std::filesystem::path& p,mx::Shape shape,bool bf){
 auto bytes=raw(p);std::size_t size=1;for(auto n:shape)size*=n;check(bytes.size()==size*(bf?2:4),"shape mismatch");
 if(bf){std::vector<std::uint16_t> v(size);std::memcpy(v.data(),bytes.data(),bytes.size());return mx::view(mx::array(v.begin(),shape,mx::uint16),mx::bfloat16);}
 std::vector<float> v(size);std::memcpy(v.data(),bytes.data(),bytes.size());return mx::array(v.begin(),shape,mx::float32);
}
std::string data(const mx::array& a){auto b=mx::view(a,mx::uint8,mx::Device::cpu);mx::eval(b);mx::synchronize();return {reinterpret_cast<const char*>(b.data<std::uint8_t>()),b.size()};}
J compare(const mx::array& a,const mx::array& e){
 check(a.shape()==e.shape()&&a.dtype()==e.dtype(),"comparison shape mismatch");auto ab=data(a),eb=data(e);std::size_t mismatches=0;double error=0;
 int width=a.dtype()==mx::bfloat16?2:4;
 for(std::size_t i=0;i<ab.size();i+=width){std::uint32_t x=0,y=0;std::memcpy(&x,ab.data()+i,width);std::memcpy(&y,eb.data()+i,width);mismatches+=x!=y;if(width==2){x<<=16;y<<=16;}float xf=std::bit_cast<float>(x),yf=std::bit_cast<float>(y);check(std::isfinite(xf)&&std::isfinite(yf),"nonfinite output");error=std::max(error,std::abs(double(xf)-yf));}
 return {{"elements",a.size()},{"bit_mismatches",mismatches},{"max_abs",error}};
}
}
int main(int argc,char** argv){try{
 using namespace dsv41;std::filesystem::path checkpoint,summary,fixtures,output;
 for(int i=1;i<argc;++i){std::string f=argv[i];check(i+1<argc,"missing option");auto v=argv[++i];if(f=="--checkpoint")checkpoint=v;else if(f=="--summary")summary=v;else if(f=="--fixtures")fixtures=v;else if(f=="--output")output=v;else check(false,"unknown option");}
 check(!checkpoint.empty()&&!summary.empty()&&!fixtures.empty()&&!output.empty(),"paths required");auto relative=std::filesystem::weakly_canonical(output).lexically_relative(std::filesystem::canonical(checkpoint));check(!relative.empty()&&*relative.begin()=="..","checkpoint read only");
 auto m=read_json_file(fixtures/"manifest.json");for(const auto& [name,h]:m.at("fixture_sha256").items()){check(std::filesystem::path(name).filename()==name,"unsafe fixture");check(sha256_text(raw(fixtures/name))==h.get<std::string>(),"fixture changed");}
 for(const auto& [name,h]:m.at("source_sha256").items()){check(name=="inference/model.py"||name=="inference/kernel.py"||name=="inference/config.json","unsafe source");check(sha256_text(raw(checkpoint/name))==h.get<std::string>(),"source changed");}
 WeightCatalog catalog(checkpoint,summary);check(m.at("revision")==catalog.revision(),"revision mismatch");mx::set_default_device(mx::Device::gpu);
 auto h=load(fixtures/"hidden.bf16",{2,4,5120},true),x=load(fixtures/"sublayer.bf16",{2,5120},true);J runs=J::array();bool exact=true;
 for(const std::string kind:{"attn","ffn"}){
  auto owner=std::make_unique<HCReference>(catalog,0,kind);auto mixes=owner->mixes(h);
  auto collapsed=hc_pre_reference(h,mixes.pre),expanded=hc_post_reference(x,h,mixes);
  for(int i=0;i<2;++i){auto hi=mx::slice(h,{i,0,0},{i+1,4,5120});auto mi=owner->mixes(hi);
   check(data(mi.pre)==data(mx::slice(mixes.pre,{i,0},{i+1,4}))&&data(mi.post)==data(mx::slice(mixes.post,{i,0},{i+1,4}))&&data(mi.comb)==data(mx::slice(mixes.comb,{i,0,0},{i+1,4,4})),"mHC chunk mismatch");}
  owner.reset();auto pre=load(fixtures/(kind+"-pre.f32"),{2,4},false),post=load(fixtures/(kind+"-post.f32"),{2,4},false),comb=load(fixtures/(kind+"-comb.f32"),{2,4,4},false);
  J r={{"kind",kind},{"pre",compare(mixes.pre,pre)},{"post",compare(mixes.post,post)},{"comb",compare(mixes.comb,comb)},
   {"collapsed",compare(collapsed,load(fixtures/(kind+"-collapsed.bf16"),{2,5120},true))},{"expanded",compare(expanded,load(fixtures/(kind+"-expanded.bf16"),{2,4,5120},true))},
   {"tokenwise_exact",true},{"owner_lifetime_checked",true}};
  for(auto key:{"pre","post","comb","collapsed","expanded"})exact=exact&&r[key]["bit_mismatches"]==0;runs.push_back(r);
 }
 J result={{"status",exact?"fixtures_exact":"numerical_mismatch"},{"manifest_sha256",sha256_text(raw(fixtures/"manifest.json"))},{"runs",runs},{"official_cuda_executed",false},{"full_block_connected",false},{"full_model_qualified",false}};
 std::filesystem::create_directories(output.parent_path());std::ofstream f(output);f<<result.dump(2)<<'\n';f.close();check(bool(f),"output failed");std::cout<<result.dump(2)<<'\n';return exact?0:2;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
