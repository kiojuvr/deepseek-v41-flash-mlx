#include "dsv41/moe.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>

namespace mx=mlx::core;

int main(){try{
 mx::set_default_device(mx::Device::cpu);
 bool strict_rejected=false;
 try{dsv41::select_routes_reference(mx::ones({384}),mx::zeros({384}));}
 catch(const std::exception&){strict_rejected=true;}
 if(!strict_rejected)throw std::runtime_error("strict route policy accepted a boundary tie");

 std::vector<float> scores(384,5.0f);
 for(int i=0;i<5;++i)scores[i]=10.0f-i;
 dsv41::reset_route_tie_count();
 dsv41::RouteTieRecord record{};
 auto route=dsv41::select_routes_reference(
  mx::array(scores.begin(),{384},mx::float32),mx::zeros({384},mx::float32),false,&record);
 for(int i=0;i<6;++i)if(route.ids[i]!=i)throw std::runtime_error("lowest-ID route tie policy mismatch");
 if(!record.tied||record.sixth_id!=5||record.seventh_id!=6||dsv41::route_tie_count()!=1)
  throw std::runtime_error("route tie record mismatch");
 std::cout<<"PASS: strict rejection and deterministic lowest-ID route tie policy\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
