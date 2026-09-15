#include "bridge_executor.hpp"
#include <iostream>
#include <thread>
#include <future>

static void require(bool ok){if(!ok)throw std::runtime_error("bridge lifecycle assertion failed");}
struct Executor final:dsv41::BridgeExecutor {
 bool fail=false,stop=false;
 std::function<void()> prefill;
 dsv41::GenerationResult generate(const dsv41_request_t& r,const dsv41::GenerationControl& control) override{
  unsigned sample=0;
  const uint32_t stops[]={11};
  return dsv41::run_generation_loop(r.input_token_count,r.max_new_tokens,
   stop?std::span<const uint32_t>(stops):std::span<const uint32_t>{},control,
   [&]{if(prefill)prefill();},[&]{return 10u+sample++;},[&](auto,auto){if(fail)throw std::runtime_error("injected model error");});
 }
};
struct Seen {
 std::vector<uint32_t> tokens;
 std::string reason,error;
 uint64_t id=0,count=0;
 int terminals=0;
 bool valid=true;
 std::function<int(const dsv41_event_t&)> action;
};
static int event(const dsv41_event_t* e,void* ctx){
 auto& s=*static_cast<Seen*>(ctx);
 s.valid=s.valid && !s.terminals && e->request_id!=0 && (!s.id || s.id==e->request_id);
 s.id=e->request_id;
 if(e->kind==DSV41_EVENT_TOKEN){
  s.valid=s.valid && e->committed_index==s.tokens.size();s.tokens.push_back(e->token_id);
 }else{
  ++s.terminals;s.count=e->committed_index;
  s.reason=e->finish_reason?e->finish_reason:"";s.error=e->error_code?e->error_code:"";
 }
 return s.action?s.action(*e):0;
}
int main(){
 auto executor=std::make_unique<Executor>();auto* impl=executor.get();
 auto* bridge=dsv41::bridge_from_executor(std::move(executor));
 const uint32_t tokens[]={0,42};
 dsv41_request_t request{1,tokens,2,4,0,0};
 uint64_t id=0;
 Seen seen;
 auto submit=[&]{return dsv41_bridge_submit(bridge,&request,event,&seen,&id);};
 require(submit()==0 && seen.valid && seen.terminals==1 && seen.reason=="length" && seen.count==4 && seen.id==id);
 require(dsv41_bridge_cancel(bridge,id)==DSV41_BRIDGE_NOT_FOUND);
 seen={};seen.action=[](const auto& e){return e.kind==DSV41_EVENT_TOKEN?1:0;};
 require(submit()==0 && seen.valid && seen.terminals==1 && seen.reason=="cancelled" && seen.count==1);
 seen={};seen.action=[&](const auto& e){
  if(e.kind==DSV41_EVENT_TOKEN){
   uint64_t nested=99;
   seen.valid=seen.valid && dsv41_bridge_submit(bridge,&request,event,nullptr,&nested)==DSV41_BRIDGE_BUSY && nested==0;
   seen.valid=seen.valid && dsv41_bridge_cancel(bridge,e.request_id)==0;
  }return 0;
 };
 require(submit()==0 && seen.valid && seen.reason=="cancelled" && seen.count==1);
 impl->fail=true;seen={};
 require(submit()==DSV41_BRIDGE_RUNTIME_ERROR && seen.valid && seen.terminals==1 && seen.error=="runtime_error" && seen.count==1);
 impl->fail=false;impl->stop=true;seen={};
 require(submit()==0 && seen.valid && seen.reason=="stop" && seen.count==2);
 impl->stop=false;seen={};
 require(submit()==0 && seen.valid && seen.reason=="length" && seen.count==4);
 // Publish request ID with a future, not an unsynchronised read of id_out.
 std::promise<uint64_t> started;auto ready=started.get_future();
 std::promise<void> cancelled;auto done=cancelled.get_future();
 impl->prefill=[&]{started.set_value(id);done.wait();};
 seen={};int status=99;
 std::thread worker([&]{status=submit();});
 auto active=ready.get();
 auto cancel_status=dsv41_bridge_cancel(bridge,active);
 cancelled.set_value();worker.join();
 require(cancel_status==0 && status==0 && seen.valid && seen.reason=="cancelled" && seen.count==0);
 impl->prefill={};
 seen={};
 impl->prefill=[]{throw std::runtime_error("prefill must not run after cancellation");};
 require(dsv41_bridge_submit_cancellable(bridge,&request,event,&seen,[](void*){return 1;},&id)==0);
 require(seen.valid && seen.reason=="cancelled" && seen.count==0 && seen.terminals==1);
 impl->prefill={};
 const uint32_t invalid[]={129280};request.input_tokens=invalid;request.input_token_count=1;
 seen={};id=9;
 require(submit()==DSV41_BRIDGE_INVALID_ARGUMENT && id==0 && seen.terminals==0);
 char error[8];
 require(!dsv41_bridge_create_model(nullptr,error,sizeof(error)) && error[7]==0);
 dsv41_bridge_destroy(bridge);
 std::cout<<"PASS: ABI events, cancellation, reentry, errors and reuse (checkpoint-free)\n";
}
