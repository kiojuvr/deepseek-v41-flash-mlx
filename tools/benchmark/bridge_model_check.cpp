#include "dsv41/runtime_bridge.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok){if(!ok)throw std::runtime_error("model bridge lifecycle mismatch");}
struct Events {
 std::vector<uint32_t> tokens;
 uint64_t id=0,count=0;
 unsigned terminals=0;
 bool valid=true,cancel=false;
 std::string reason,error;
};
static int receive(const dsv41_event_t* e,void* context){
 auto& out=*static_cast<Events*>(context);
 out.valid=out.valid && !out.terminals && e->request_id && (!out.id || out.id==e->request_id);
 out.id=e->request_id;
 if(e->kind==DSV41_EVENT_TOKEN){
  out.valid=out.valid && e->committed_index==out.tokens.size();
  out.tokens.push_back(e->token_id);
  return out.cancel?1:0;
 }
 ++out.terminals;out.count=e->committed_index;
 out.valid=out.valid && e->kind==DSV41_EVENT_FINISHED;
 out.reason=e->finish_reason?e->finish_reason:"";
 out.error=e->error_message?e->error_message:"";
 return 0;
}
static std::vector<uint32_t> read_tokens(const char* path){
 std::ifstream file(path);if(!file)throw std::runtime_error("cannot open token file");
 std::vector<uint32_t> tokens;uint64_t token;
 while(file>>token){if(token>=129280)throw std::runtime_error("invalid text token");tokens.push_back(uint32_t(token));}
 if(!file.eof() || tokens.empty())throw std::runtime_error("invalid token file");
 return tokens;
}
int main(int argc,char** argv){try{
 if(argc!=9)throw std::runtime_error("usage: bridge-model-check checkpoint summary metadata provenance prompt expected temperature seed");
 auto prompt=read_tokens(argv[5]),expected=read_tokens(argv[6]);
 if(expected.size()<2 || expected.size()>262144)throw std::runtime_error("expected token count must be 2..262144");
 dsv41_model_config_t config{1,argv[1],argv[2],argv[3],argv[4],nullptr,0};
 char error[1024];
 std::cout<<"Loading native bridge model"<<std::endl;
 std::unique_ptr<dsv41_bridge_t,decltype(&dsv41_bridge_destroy)> bridge(
  dsv41_bridge_create_model(&config,error,sizeof(error)),dsv41_bridge_destroy);
 if(!bridge)throw std::runtime_error(error);
 dsv41_request_t request{1,prompt.data(),prompt.size(),uint32_t(expected.size()),std::stof(argv[7]),std::stoull(argv[8])};
 uint64_t previous=0;
 for(unsigned pass=0;pass<3;++pass){
  Events events;events.cancel=pass==1;uint64_t id=0;
  int status=dsv41_bridge_submit(bridge.get(),&request,receive,&events,&id);
  if(!events.error.empty())std::cerr<<events.error<<std::endl;
  require(status==0 && id>previous && events.id==id && events.valid && events.terminals==1 && events.count==events.tokens.size());
  previous=id;
  require(events.reason==(events.cancel?"cancelled":"length"));
  require(events.tokens==(events.cancel?std::vector<uint32_t>{expected.front()}:expected));
  require(dsv41_bridge_cancel(bridge.get(),id)==DSV41_BRIDGE_NOT_FOUND);
  std::cout<<"PASS: bridge request "<<id<<" reason="<<events.reason<<" committed="<<events.count<<std::endl;
 }
 std::cout<<"PASS: C ABI tokens match direct native generation; cancellation and fresh request accounting match (not API/256K qualification)"<<std::endl;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
