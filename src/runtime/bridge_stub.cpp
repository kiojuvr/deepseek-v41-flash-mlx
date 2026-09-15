#include "bridge_executor.hpp"
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>

namespace {
// MLX reference models are not qualified for concurrent host graph construction.
std::atomic_flag model_busy=ATOMIC_FLAG_INIT;
struct ModelLease {
 bool held=!model_busy.test_and_set(std::memory_order_acquire);
 ~ModelLease(){if(held)model_busy.clear(std::memory_order_release);}
};
void diagnostic(char* out,size_t capacity,const char* message){
 if(out && capacity){auto n=std::min(capacity-1,std::strlen(message));std::memcpy(out,message,n);out[n]=0;}
}
}
struct dsv41_bridge {
 std::atomic<uint64_t> next_request{1};
 std::unique_ptr<dsv41::BridgeExecutor> executor;
 std::mutex mutex;
 uint64_t active_id=0;
 bool cancelled=false;
};
namespace dsv41 {
dsv41_bridge_t* bridge_from_executor(std::unique_ptr<BridgeExecutor> executor){
 auto bridge=std::make_unique<dsv41_bridge>();
 bridge->executor=std::move(executor);
 return bridge.release();
}
#ifndef DSV41_BRIDGE_MLX
std::unique_ptr<BridgeExecutor> make_bridge_executor(const dsv41_model_config_t&){
 throw std::runtime_error("native runtime bridge is not connected");
}
#endif
}
extern "C" dsv41_bridge_t* dsv41_bridge_create(uint32_t version){
 if(version!=DSV41_BRIDGE_ABI_VERSION)return nullptr;
 try{return dsv41::bridge_from_executor(nullptr);}catch(...){return nullptr;}
}
extern "C" dsv41_bridge_t* dsv41_bridge_create_model(const dsv41_model_config_t* config,char* error,size_t capacity){
 diagnostic(error,capacity,"");
 try{
  if(!config || config->abi_version!=DSV41_BRIDGE_ABI_VERSION ||
     !config->checkpoint_path || !config->summary_path || !config->metadata_path ||
     !config->provenance_path || (!config->stop_tokens && config->stop_token_count) ||
     config->stop_token_count>129280)throw std::runtime_error("invalid model configuration");
  for(size_t i=0;i<config->stop_token_count;++i)
   if(config->stop_tokens[i]>=129280)throw std::runtime_error("stop token out of range");
  ModelLease lease;
  if(!lease.held)throw std::runtime_error("native model operation is busy");
  return dsv41::bridge_from_executor(dsv41::make_bridge_executor(*config));
 }catch(const std::exception& e){diagnostic(error,capacity,e.what());}
 catch(...){diagnostic(error,capacity,"unknown model initialization failure");}
 return nullptr;
}
extern "C" void dsv41_bridge_destroy(dsv41_bridge_t* bridge){delete bridge;}

extern "C" int dsv41_bridge_submit(dsv41_bridge_t* bridge,const dsv41_request_t* request,
 dsv41_event_callback callback,void* context,uint64_t* id_out){
 return dsv41_bridge_submit_cancellable(bridge,request,callback,context,nullptr,id_out);
}
extern "C" int dsv41_bridge_submit_cancellable(dsv41_bridge_t* bridge,const dsv41_request_t* request,
 dsv41_event_callback callback,void* context,dsv41_cancel_poll poll,uint64_t* id_out){
 if(id_out)*id_out=0;
 if(!bridge || !request || request->abi_version!=DSV41_BRIDGE_ABI_VERSION ||
    !request->input_tokens || !callback || !request->input_token_count || !request->max_new_tokens ||
    request->input_token_count>DSV41_BRIDGE_MAX_NEW_TOKENS ||
    request->max_new_tokens>DSV41_BRIDGE_MAX_NEW_TOKENS-request->input_token_count ||
    !std::isfinite(request->temperature) || request->temperature<0)return DSV41_BRIDGE_INVALID_ARGUMENT;
 for(size_t i=0;i<request->input_token_count;++i)
  if(request->input_tokens[i]>=129280)return DSV41_BRIDGE_INVALID_ARGUMENT;
 ModelLease lease;
 if(!lease.held)return DSV41_BRIDGE_BUSY;
 const auto id=bridge->next_request.fetch_add(1,std::memory_order_relaxed);
 if(id_out)*id_out=id;
 uint64_t committed=0;
 bool terminal_sent=false;
 // No lock is held across a callback; cancel() is safe from it or another thread.
 struct ActiveRequest {
  dsv41_bridge* bridge;
  ActiveRequest(dsv41_bridge* b,uint64_t id):bridge(b){std::lock_guard lock(b->mutex);b->active_id=id;b->cancelled=false;}
  ~ActiveRequest(){std::lock_guard lock(bridge->mutex);bridge->active_id=0;bridge->cancelled=false;}
 };
 auto terminal=[&](dsv41_event_kind_t kind,const char* reason,const char* code,const char* message){
  terminal_sent=true;
  dsv41_event_t event{kind,id,committed,0,reason,code,message};
  callback(&event,context);
 };
 try{
  ActiveRequest active(bridge,id);
  if(!bridge->executor){
   terminal(DSV41_EVENT_ERROR,nullptr,"runtime_unavailable","native runtime bridge is not connected");
   return DSV41_BRIDGE_UNAVAILABLE;
  }
  dsv41::GenerationControl control;
  control.is_cancelled=[&]{
   const bool external=poll && poll(context);
   std::lock_guard lock(bridge->mutex);
   bridge->cancelled=bridge->cancelled || external;
   return bridge->cancelled;
  };
  control.on_token=[&](uint32_t token,uint64_t index){
   if(index!=committed || committed>=request->max_new_tokens || token>=129280)
    throw std::runtime_error("invalid committed token event");
   ++committed;
   dsv41_event_t event{DSV41_EVENT_TOKEN,id,index,token,nullptr,nullptr,nullptr};
   if(callback(&event,context)){
    std::lock_guard lock(bridge->mutex);bridge->cancelled=true;return false;
   }
   return true;
  };
  auto result=bridge->executor->generate(*request,control);
  if(result.tokens.size()!=committed || result.next_position!=request->input_token_count+committed)
   throw std::runtime_error("generation accounting mismatch");
  bool cancelled;
  {std::lock_guard lock(bridge->mutex);cancelled=bridge->cancelled || result.cancelled;bridge->active_id=0;}
  if(!cancelled && !result.stopped && committed!=request->max_new_tokens)
   throw std::runtime_error("generation ended before its token budget without a stop reason");
  terminal(DSV41_EVENT_FINISHED,cancelled?"cancelled":result.stopped?"stop":"length",nullptr,nullptr);
  return DSV41_BRIDGE_OK;
 }catch(const std::exception& e){
  if(!terminal_sent){try{terminal(DSV41_EVENT_ERROR,nullptr,"runtime_error",e.what());}catch(...){}}
 }catch(...){
  if(!terminal_sent){try{terminal(DSV41_EVENT_ERROR,nullptr,"runtime_error","unknown native exception");}catch(...){}}
 }
 return DSV41_BRIDGE_RUNTIME_ERROR;
}
extern "C" int dsv41_bridge_cancel(dsv41_bridge_t* bridge,uint64_t id){
 if(!bridge || !id)return DSV41_BRIDGE_INVALID_ARGUMENT;
 try{
  std::lock_guard lock(bridge->mutex);
  if(!bridge->executor)return DSV41_BRIDGE_UNAVAILABLE;
  if(bridge->active_id!=id)return DSV41_BRIDGE_NOT_FOUND;
  bridge->cancelled=true;
  return DSV41_BRIDGE_OK;
 }catch(...){return DSV41_BRIDGE_RUNTIME_ERROR;}
}
