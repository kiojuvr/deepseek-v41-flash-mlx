#include "dsv41/expert_backing.hpp"
#include "dsv41/checkpoint_atlas.hpp"
#include "dsv41/moe.hpp"
#include "dsv41/layer_owner.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <nlohmann/json.hpp>
#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
namespace dsv41 {
namespace fs=std::filesystem;namespace mx=mlx::core;using J=nlohmann::json;
namespace {
constexpr std::size_t page=16384,chunk=8<<20;
void check(bool ok,const std::string& what){if(!ok)throw std::runtime_error(what+": "+std::strerror(errno));}
std::string hex(const unsigned char* p,std::size_t n){std::ostringstream s;s<<std::hex<<std::setfill('0');for(std::size_t i=0;i<n;++i)s<<std::setw(2)<<unsigned(p[i]);return s.str();}
std::string prefix(int layer,int expert,const char* name){return reference_moe_prefix(layer)+".ffn.experts."+std::to_string(expert)+"."+name;}
std::string filename(int layer,const char* projection,const char* kind){char out[80];std::snprintf(out,sizeof(out),"layer-%02d-%s-%s.bin",layer,projection,kind);return out;}
std::uint64_t rounded(std::uint64_t n){return (n+page-1)/page*page;}
J identity(const struct stat& s){return {{"device",s.st_dev},{"inode",s.st_ino},{"size",s.st_size},{"mtime_sec",s.st_mtimespec.tv_sec},{"mtime_nsec",s.st_mtimespec.tv_nsec},{"ctime_sec",s.st_ctimespec.tv_sec},{"ctime_nsec",s.st_ctimespec.tv_nsec}};}
bool same_identity(const J& j,const struct stat& s){return j.at("device")==s.st_dev&&j.at("inode")==s.st_ino&&j.at("size")==s.st_size&&j.at("mtime_sec")==s.st_mtimespec.tv_sec&&j.at("mtime_nsec")==s.st_mtimespec.tv_nsec&&j.at("ctime_sec")==s.st_ctimespec.tv_sec&&j.at("ctime_nsec")==s.st_ctimespec.tv_nsec;}
void write_json_atomic(const fs::path& path,const J& value){
 auto temp=path.string()+".tmp";{std::ofstream out(temp,std::ios::binary|std::ios::trunc);if(!out)throw std::runtime_error("cannot write backing manifest");out<<value.dump(2)<<'\n';out.flush();if(!out)throw std::runtime_error("cannot flush backing manifest");}
 int fd=open(temp.c_str(),O_RDONLY|O_NOFOLLOW);check(fd>=0,"open manifest temp");check(fsync(fd)==0,"fsync manifest");close(fd);check(rename(temp.c_str(),path.c_str())==0,"publish manifest");
}
struct Mapping {int fd=-1;void* ptr=MAP_FAILED;std::size_t bytes=0;~Mapping(){if(ptr!=MAP_FAILED)munmap(ptr,bytes);if(fd>=0)close(fd);}};
mx::array map_array(const fs::path& path,const J& entry,mx::Shape shape,mx::Dtype dtype){
 auto owner=std::make_shared<Mapping>();owner->fd=open(path.c_str(),O_RDONLY|O_NOFOLLOW);check(owner->fd>=0,"open expert backing");struct stat s{};check(fstat(owner->fd,&s)==0&&S_ISREG(s.st_mode),"stat expert backing");
 if((s.st_mode&0222)||!same_identity(entry.at("stat"),s))throw std::runtime_error("expert backing identity or read-only mode mismatch: "+path.string());
 owner->bytes=entry.at("mapped_bytes").get<std::size_t>();owner->ptr=mmap(nullptr,owner->bytes,PROT_READ,MAP_PRIVATE,owner->fd,0);check(owner->ptr!=MAP_FAILED,"mmap expert backing");
 auto buffer=mx::allocator::make_buffer(owner->ptr,owner->bytes);if(!buffer.ptr())throw std::runtime_error("MLX rejected no-copy expert backing");
 return mx::array(buffer,std::move(shape),dtype,[owner](mx::allocator::Buffer b){mx::allocator::release(b);});
}
J make_payload(WeightCatalog& catalog,const fs::path& root,int layer,const char* projection,const char* kind,int n,int k){
 const bool weight=std::string_view(kind)=="weight";const std::uint64_t per=weight?std::uint64_t(n)*k/2:std::uint64_t(n)*k/32;
 const auto name=filename(layer,projection,kind);const fs::path temp=root/(name+".tmp"),final=root/name;
 int fd=open(temp.c_str(),O_CREAT|O_TRUNC|O_RDWR|O_NOFOLLOW,0600);check(fd>=0,"create expert backing payload");
 const auto logical=per*384,mapped=rounded(logical);check(ftruncate(fd,mapped)==0,"size expert backing payload");
 CC_SHA256_CTX sha;CC_SHA256_Init(&sha);std::vector<std::byte> buffer(std::min<std::uint64_t>(chunk,per));J sources=J::array();std::uint64_t output=0;
 try{
  for(int expert=0;expert<384;++expert){auto tensor=catalog.tensor(prefix(layer,expert,projection)+"."+kind);
   const std::vector<std::uint64_t> expected_shape=weight?
    std::vector<std::uint64_t>{std::uint64_t(n),std::uint64_t(k/2)}:
    std::vector<std::uint64_t>{std::uint64_t(n),std::uint64_t(k/32)};
   if(tensor.size_bytes!=per||tensor.shape!=expected_shape||
      (weight?(tensor.dtype!="I8"&&tensor.dtype!="U8"):tensor.dtype!="F8_E8M0"))
    throw std::runtime_error("unexpected source tensor for expert backing");
   sources.push_back({{"name",tensor.name},{"shard",tensor.shard},{"file_offset",tensor.file_offset},{"bytes",tensor.size_bytes}});
   for(std::uint64_t at=0;at<per;){const auto count=std::size_t(std::min<std::uint64_t>(buffer.size(),per-at));tensor.read(at,{buffer.data(),count});CC_SHA256_Update(&sha,buffer.data(),CC_LONG(count));
    std::size_t done=0;while(done<count){auto wrote=pwrite(fd,buffer.data()+done,count-done,off_t(output+done));if(wrote<0&&errno==EINTR)continue;check(wrote>0,"write expert backing payload");done+=std::size_t(wrote);}at+=count;output+=count;}
  }
  check(fsync(fd)==0,"fsync expert backing payload");check(fchmod(fd,0444)==0,"make expert backing read-only");close(fd);fd=-1;check(rename(temp.c_str(),final.c_str())==0,"publish expert backing payload");
  struct stat s{};check(lstat(final.c_str(),&s)==0&&S_ISREG(s.st_mode),"stat published expert backing");
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];CC_SHA256_Final(digest,&sha);
  return {{"file",name},{"layer",layer},{"projection",projection},{"kind",kind},{"n",n},{"k",k},{"experts",384},{"logical_bytes",logical},{"mapped_bytes",mapped},{"sha256",hex(digest,sizeof(digest))},{"sources",std::move(sources)},{"stat",identity(s)}};
 }catch(...){if(fd>=0)close(fd);unlink(temp.c_str());throw;}
}
}
struct ExpertBackingStore::Impl {fs::path root;J manifest;};
ExpertBackingStore::ExpertBackingStore(const fs::path& root,const WeightCatalog& catalog):impl_(std::make_unique<Impl>()){
 impl_->root=fs::canonical(root);struct stat ds{},ms{};auto manifest_path=impl_->root/"manifest.json";
 if(lstat(impl_->root.c_str(),&ds)!=0||!S_ISDIR(ds.st_mode)||(ds.st_mode&0222)||
    lstat(manifest_path.c_str(),&ms)!=0||!S_ISREG(ms.st_mode)||(ms.st_mode&0222))
  throw std::runtime_error("expert backing root/manifest must be real and read-only");
 impl_->manifest=read_json_file(manifest_path);
 if(impl_->manifest.at("status")!="complete"||impl_->manifest.at("schema_version")!=1||impl_->manifest.at("checkpoint_revision")!=catalog.revision())throw std::runtime_error("expert backing manifest mismatch");
}
ExpertBackingStore::~ExpertBackingStore()=default;
ExpertBackingArrays ExpertBackingStore::projection(int layer,const char* name,int n,int k) const{
 const auto& entries=impl_->manifest.at("payloads");const auto& we=entries.at(filename(layer,name,"weight"));const auto& se=entries.at(filename(layer,name,"scale"));
 auto w=map_array(impl_->root/we.at("file").get<std::string>(),we,{384,n,k/8},mx::uint32);
 auto s=map_array(impl_->root/se.at("file").get<std::string>(),se,{384,n,k/32},mx::uint8);
 return {std::move(w),std::move(s),we.at("logical_bytes").get<std::size_t>()+se.at("logical_bytes").get<std::size_t>()};
}
void repair_expert_backing_identity(const fs::path& root){
 auto canonical=fs::canonical(root),manifest_path=canonical/"manifest.json";J manifest=read_json_file(manifest_path);
 if(manifest.at("status")!="complete"||manifest.at("schema_version")!=1||manifest.at("payloads").size()!=240)throw std::runtime_error("cannot repair incomplete expert backing");
 check(chmod(canonical.c_str(),0755)==0,"open backing root for identity repair");check(chmod(manifest_path.c_str(),0644)==0,"open backing manifest for identity repair");
 try{
  for(auto& [name,entry]:manifest["payloads"].items()){auto path=canonical/name;struct stat s{};check(lstat(path.c_str(),&s)==0&&S_ISREG(s.st_mode)&&!(s.st_mode&0222),"stat repair payload");const auto& old=entry.at("stat");
   if(old.at("device")!=s.st_dev||old.at("inode")!=s.st_ino||old.at("size")!=s.st_size||old.at("mtime_sec")!=s.st_mtimespec.tv_sec||old.at("mtime_nsec")!=s.st_mtimespec.tv_nsec)
    throw std::runtime_error("expert backing changed beyond rename ctime: "+name);
   entry["stat"]=identity(s);
  }
  write_json_atomic(manifest_path,manifest);check(chmod(manifest_path.c_str(),0444)==0,"seal repaired backing manifest");check(chmod(canonical.c_str(),0555)==0,"seal repaired backing root");
 }catch(...){chmod(manifest_path.c_str(),0444);chmod(canonical.c_str(),0555);throw;}
}
void prepare_expert_backing(WeightCatalog& catalog,const fs::path& destination){
 if(fs::exists(destination))throw std::runtime_error("expert backing destination already exists");
 if(!destination.parent_path().empty())fs::create_directories(destination.parent_path());
 auto building=fs::path(destination.string()+".building");fs::create_directories(building);
 check(chmod(building.c_str(),0755)==0,"make building root writable");
 auto progress_path=building/"progress.json",complete_path=building/"manifest.json";
 if(fs::exists(complete_path))check(chmod(complete_path.c_str(),0644)==0,"resume backing manifest");
 J progress=fs::exists(progress_path)?read_json_file(progress_path):fs::exists(complete_path)?read_json_file(complete_path):
  J{{"schema_version",1},{"status","building"},{"checkpoint_revision",catalog.revision()},{"payloads",J::object()}};
 if(progress.at("checkpoint_revision")!=catalog.revision())throw std::runtime_error("expert backing resume revision mismatch");
 const std::array<std::tuple<const char*,int,int>,3> projections{{{"w1",2304,5120},{"w3",2304,5120},{"w2",5120,2304}}};
 for(int layer=0;layer<40;++layer)for(auto [name,n,k]:projections)for(auto kind:{"weight","scale"}){
  auto file=filename(layer,name,kind);if(progress["payloads"].contains(file)){struct stat s{};auto path=building/file;if(stat(path.c_str(),&s)==0&&same_identity(progress["payloads"][file].at("stat"),s))continue;fs::remove(path);progress["payloads"].erase(file);}
  auto entry=make_payload(catalog,building,layer,name,kind,n,k);progress["payloads"][file]=std::move(entry);write_json_atomic(progress_path,progress);
  std::cout<<"prepared "<<progress["payloads"].size()<<"/240 "<<file<<std::endl;
 }
 progress["status"]="complete";write_json_atomic(complete_path,progress);fs::remove(progress_path);
 check(chmod(complete_path.c_str(),0444)==0,"make backing manifest read-only");
 check(chmod(building.c_str(),0555)==0,"make backing root read-only");
 int dfd=open(building.c_str(),O_RDONLY);check(dfd>=0,"open backing directory");check(fsync(dfd)==0,"fsync backing directory");close(dfd);
 check(rename(building.c_str(),destination.c_str())==0,"publish expert backing root");
 auto parent=destination.parent_path().empty()?fs::path("."):destination.parent_path();dfd=open(parent.c_str(),O_RDONLY);check(dfd>=0,"open backing parent");check(fsync(dfd)==0,"fsync backing parent");close(dfd);
}
}
