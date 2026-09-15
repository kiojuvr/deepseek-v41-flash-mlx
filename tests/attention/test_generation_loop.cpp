#include "dsv41/generation_loop.hpp"
#include <iostream>
#include <limits>
#include <utility>

static void require(bool ok){if(!ok)throw std::runtime_error("generation lifecycle check failed");}
int main(){
 using namespace dsv41;
 // Every context boundary is covered exactly once with contiguous <=128 spans.
 for(auto prompt:{std::size_t(1),std::size_t(127),std::size_t(128),std::size_t(129),
                  std::size_t(255),std::size_t(256),std::size_t(257),std::size_t(262144)}){
  std::size_t covered=0,chunks=0;
  run_prefill_chunks(prompt,128,[&](std::size_t offset,std::size_t count){
   require(offset==covered && count>0 && count<=128 && count<=prompt-offset);
   covered+=count;++chunks;
  });
  require(covered==prompt && chunks==(prompt+127)/128);
 }
 for(auto sizes:{std::pair<std::size_t,std::size_t>{0,128},{1,0}}){
  bool rejected=false;
  try{run_prefill_chunks(sizes.first,sizes.second,[](auto,auto){});}
  catch(const std::runtime_error&){rejected=true;}
  require(rejected);
 }
 int prefill=0,samples=0,advances=0;
 bool cancel=false;
 std::vector<std::uint32_t> events;
 GenerationControl control{[&](auto token,auto index){
  require(index==events.size());events.push_back(token);return true;
 },[&]{return cancel;}};
 auto run=[&](std::size_t count,std::span<const std::uint32_t> stops={}){
  prefill=samples=advances=0;events.clear();
  return run_generation_loop(4,count,stops,control,[&]{++prefill;},
   [&]{return std::uint32_t(10+samples++);},[&](auto token,auto pos){
    require(token==std::uint32_t(10+advances));require(pos==std::uint64_t(4+advances));++advances;
   });
 };
 auto r=run(3);
 require(r.tokens==std::vector<std::uint32_t>({10,11,12}) && events==r.tokens);
 require(prefill==1 && samples==3 && advances==2 && r.next_position==7 && !r.cancelled && !r.stopped);
 const std::uint32_t stop[]={11};r=run(4,stop);
 require(r.stopped && !r.cancelled && r.tokens.size()==2 && events==r.tokens && advances==1);
 control.on_token=[&](auto token,auto index){require(index==0 && token==10);return false;};
 r=run(3);require(r.cancelled && r.tokens.size()==1 && advances==0 && samples==1);
 const std::uint32_t first_stop[]={10};r=run(3,first_stop);
 require(r.cancelled && r.stopped && r.tokens.size()==1 && advances==0);
 cancel=true;r=run(3);require(r.cancelled && r.tokens.empty() && prefill==0 && r.next_position==4);
 cancel=false;r=run(0);require(!r.cancelled && prefill==0 && samples==0);
 r=run_generation_loop(4,3,{},control,[&]{cancel=true;},[]{throw std::runtime_error("sample after cancel");return 0u;},[](auto,auto){});
 require(r.cancelled && r.tokens.empty());
 cancel=false;
 r=run_generation_loop(4,3,{},control,[]{},[&]{cancel=true;return 10u;},[](auto,auto){throw std::runtime_error("advance after cancel");});
 require(r.cancelled && r.tokens.empty());
 // Invalid admission must not enter any model callback, including overflow.
 for(auto count:{std::size_t(262141),std::numeric_limits<std::size_t>::max()}){
  bool rejected=false;
  try{run(count);}catch(const std::runtime_error&){rejected=true;}
  require(rejected && prefill==0 && samples==0);
 }
 cancel=false;control.on_token={};r=run(2);
 require(r.tokens==std::vector<std::uint32_t>({10,11}) && !r.cancelled);
 bool propagated=false;
 try{run_generation_loop(4,1,{},control,[]{throw std::runtime_error("model failure");},[]{return 0u;},[](auto,auto){});}
 catch(const std::runtime_error& e){propagated=std::string(e.what())=="model failure";}
 require(propagated);
 bool callback_error=false;
 control.on_token=[](auto,auto)->bool{throw std::runtime_error("sink failure");};
 try{run(2);}catch(const std::runtime_error& e){callback_error=std::string(e.what())=="sink failure";}
 require(callback_error && advances==0);
 control.on_token={};
 r=run_generation_loop(262143,1,{},control,[]{},[]{return 5u;},[](auto,auto){throw std::runtime_error("unexpected final decode");});
 require(r.next_position==262144 && r.tokens.size()==1);
 bool empty_rejected=false;
 try{run_generation_loop(0,1,{},control,[]{},[]{return 5u;},[](auto,auto){});}
 catch(const std::runtime_error&){empty_rejected=true;}
 require(empty_rejected);
 std::cout<<"PASS: generation lifecycle and bounded prefill schedule (no model/checkpoint qualification)\n";
}
