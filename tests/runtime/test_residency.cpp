#include "dsv41/runtime_residency.hpp"
#include <mlx/mlx.h>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <array>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
namespace mx=mlx::core;
void require(bool ok){if(!ok)throw std::runtime_error("residency assertion failed");}
void env(const char* value){require(setenv("DSV41_RUNTIME_WIRED_LIMIT_BYTES",value,1)==0);}
void rejected(){bool failed=false;try{dsv41::RuntimeResidencyLease lease;}catch(const std::exception&){failed=true;}require(failed);}
int main(){try{
 mx::set_default_device(mx::Device::gpu);
 // Public MLX no-copy API accepts a page-aligned, read-only file mapping.
 char path[]="/tmp/dsv41-file-backed-buffer-XXXXXX";const int fd=mkstemp(path);require(fd>=0);
 require(unlink(path)==0);constexpr std::size_t mapped_bytes=16384;
 require(ftruncate(fd,mapped_bytes)==0);std::array<float,4> values{1,2,3,4};
 require(pwrite(fd,values.data(),sizeof(values),0)==ssize_t(sizeof(values)));
 void* mapping=mmap(nullptr,mapped_bytes,PROT_READ,MAP_PRIVATE,fd,0);require(mapping!=MAP_FAILED);
 auto buffer=mx::allocator::make_buffer(mapping,mapped_bytes);require(buffer.ptr()!=nullptr);
 {mx::array file_values(buffer,{4},mx::float32,[=](mx::allocator::Buffer b){
    mx::allocator::release(b);munmap(mapping,mapped_bytes);close(fd);
   });
  require(mx::sum(file_values).item<float>()==10.0f);
 }
 for(auto value:{"", "-1", "1x", "1", "18446744073709551616"}){env(value);rejected();}
 env("0");{dsv41::RuntimeResidencyLease a,b;require(a.requested_bytes()==0);
  env("274877906944");rejected();}
 env("549755813888");rejected(); // leaves no OS reserve on the target machine
 const auto original=mx::set_wired_limit(0);
 mx::set_wired_limit(original);
 env("274877906944");
 try{
  dsv41::RuntimeResidencyLease lease;require(lease.requested_bytes()==274877906944ULL);
  auto x=mx::arange(4096,mx::float32);auto y=mx::add(x,mx::array(1.0f));lease.activate();mx::eval(y);
  require(mx::max(y).item<float>()==4096.0f);
  rejected();env("0");rejected();
  throw std::runtime_error("test constructor-unwind equivalent");
 }catch(const std::exception& e){
  require(std::string(e.what())=="test constructor-unwind equivalent");
 }
 const auto restored=mx::set_wired_limit(original);require(restored==original);
 env("0");{dsv41::RuntimeResidencyLease lease;}
 std::cout<<"PASS: residency policy parsing, exclusivity, GPU execution, unwind and restoration plus read-only file-backed no-copy buffer; no model qualification\n";
 return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
