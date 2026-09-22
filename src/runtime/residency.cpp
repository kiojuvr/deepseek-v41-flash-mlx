#include "dsv41/runtime_residency.hpp"
#include <mlx/mlx.h>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string_view>
namespace dsv41 {
namespace {
namespace mx=mlx::core;
constexpr std::size_t GiB=std::size_t(1)<<30;
std::mutex mutex;
std::size_t owners=0,budget=0,previous=0;bool reserved=false;
std::size_t requested_budget(){
 const char* text=std::getenv("DSV41_RUNTIME_WIRED_LIMIT_BYTES");
 if(!text)return 0;
 const std::string_view value(text);std::size_t result=0;
 auto parsed=std::from_chars(value.data(),value.data()+value.size(),result);
 if(value.empty()||parsed.ec!=std::errc{}||parsed.ptr!=value.data()+value.size())
  throw std::runtime_error("invalid DSV41_RUNTIME_WIRED_LIMIT_BYTES");
 if(result&&result<256*GiB)
  throw std::runtime_error("residency candidate requires at least 256 GiB for routed-expert residency");
 return result;
}
}
RuntimeResidencyLease::RuntimeResidencyLease():requested_(requested_budget()){
 std::lock_guard lock(mutex);
 if(owners&&(reserved||requested_))
  throw std::runtime_error("residency candidate requires exclusive backbone ownership");
 if(requested_){
  if(!__builtin_available(macOS 15.0, *))
   throw std::runtime_error("residency candidate requires macOS 15+");
  if(!mx::is_available(mx::Device::gpu))throw std::runtime_error("residency requires Metal GPU");
  const auto& info=mx::device_info(mx::Device::gpu);
  const auto memory=std::get<std::size_t>(info.at("memory_size"));
  const auto cap=std::get<std::size_t>(info.at("max_recommended_working_set_size"));
  const auto reserve=std::max(64*GiB,memory/10);
  if(memory<=reserve||requested_>memory-reserve||requested_>cap)
   throw std::runtime_error("residency budget exceeds device cap or OS reserve; no automatic clamp/sysctl change");
  // Do not retroactively admit a foreign model or a large idle buffer pool.
  mx::synchronize();
  if(mx::get_active_memory()>GiB)
   throw std::runtime_error("residency must be acquired before model allocation (active memory >1 GiB)");
  mx::clear_cache();
 }
 if(requested_)reserved=true;++owners;
}
void RuntimeResidencyLease::activate(){
 std::lock_guard lock(mutex);if(!requested_||active_)return;
 mx::synchronize();previous=mx::set_wired_limit(requested_);budget=requested_;active_=true;
 std::fprintf(stderr,"residency: requested=%zu applied=%zu previous=%zu bytes after atlas mapping (policy, not physical-page proof)\n",requested_,budget,previous);
}
void RuntimeResidencyLease::synchronize() const{
 if(active_)mx::synchronize();
}
RuntimeResidencyLease::~RuntimeResidencyLease() noexcept{
 std::lock_guard lock(mutex);
 if(active_){
  try{
   mx::synchronize();
   mx::set_wired_limit(previous);
   std::fprintf(stderr,"residency: restored=%zu bytes\n",previous);
   budget=0;previous=0;
  }catch(...){
   // Continuing with an unknown process-global policy is unsafe.
   std::fputs("fatal: residency budget restoration failed\n",stderr);
   std::terminate();
  }
 }
 if(requested_)reserved=false;--owners;
}
}
