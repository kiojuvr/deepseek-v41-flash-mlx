#include "dsv41/moe.hpp"
#include "dsv41/index_query.hpp"
#include <iostream>
#include <algorithm>
#include <limits>
#include <numeric>
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

 std::vector<float> batch_scores(3*384);
 for(int token=0;token<3;++token)for(int id=0;id<384;++id)
  batch_scores[token*384+id]=0.25f+float((id*37+token*101)%997)/997.0f;
 // Exercise the deterministic boundary policy in one row without making all
 // selected values equal.
 batch_scores[384+17]=2.0f;batch_scores[384+23]=1.9f;batch_scores[384+29]=1.8f;
 batch_scores[384+31]=1.7f;batch_scores[384+37]=1.6f;
 batch_scores[384+41]=1.5f;batch_scores[384+43]=1.5f;
 auto batch_array=mx::array(batch_scores.begin(),{3,384},mx::float32);
 auto batch=dsv41::select_routes_batch_reference(batch_array,mx::zeros({384},mx::float32),false,9,1000);
 if(batch.ids.size()!=3||batch.weights.shape()!=mx::Shape({3,6})||batch.ties.size()!=1||
    batch.ties.front().layer!=9||batch.ties.front().token!=1001||
    batch.ties.front().sixth_id!=41||batch.ties.front().seventh_id!=43)
  throw std::runtime_error("batched route metadata mismatch");
 std::vector<mx::array> serial_weights;
 for(int token=0;token<3;++token){
  dsv41::RouteTieRecord serial_tie{};
  auto serial=dsv41::select_routes_reference(
   mx::reshape(mx::slice(batch_array,{token,0},{token+1,384}),{384}),
   mx::zeros({384},mx::float32),false,&serial_tie);
  if(serial.ids!=batch.ids[token])throw std::runtime_error("batched route ID mismatch");
  serial_weights.push_back(serial.weights);
 }
 auto serial_weight_array=mx::stack(serial_weights,0);
 auto weight_bits_equal=mx::all(mx::equal(mx::view(batch.weights,mx::uint32),
                                          mx::view(serial_weight_array,mx::uint32)));
 mx::eval(weight_bits_equal);
 if(!weight_bits_equal.item<bool>())throw std::runtime_error("batched route weight bit mismatch");
 auto device_batch=dsv41::select_routes_batch_device(
  batch_array,mx::zeros({384},mx::float32),false,9,1000);
 if(device_batch.ids!=batch.ids||device_batch.ties.size()!=batch.ties.size()||
    device_batch.ties.front().sixth_id!=batch.ties.front().sixth_id||
    device_batch.ties.front().seventh_id!=batch.ties.front().seventh_id||
    device_batch.ties.front().sixth_score!=batch.ties.front().sixth_score||
    device_batch.ties.front().seventh_score!=batch.ties.front().seventh_score)
  throw std::runtime_error("device batched route metadata mismatch");
 auto device_weight_bits_equal=mx::all(mx::equal(mx::view(device_batch.weights,mx::uint32),
                                                 mx::view(batch.weights,mx::uint32)));
 mx::eval(device_weight_bits_equal);
 if(!device_weight_bits_equal.item<bool>())
  throw std::runtime_error("device batched route weight bit mismatch");
 auto device_only=dsv41::select_routes_batch_device(
  batch_array,mx::zeros({384},mx::float32),false,9,1000,false);
 if(!device_only.ids.empty()||!device_only.ties.empty())
  throw std::runtime_error("device-only route path published host diagnostics");
 auto device_only_ids_equal=mx::all(mx::equal(device_only.device_ids,device_batch.device_ids));
 auto device_only_weights_equal=mx::all(mx::equal(mx::view(device_only.weights,mx::uint32),
                                                  mx::view(device_batch.weights,mx::uint32)));
 mx::eval(device_only_ids_equal,device_only_weights_equal);
 if(!device_only_ids_equal.item<bool>()||!device_only_weights_equal.item<bool>())
  throw std::runtime_error("device-only route result mismatch");

 std::vector<float> index_scores(513);
 for(int i=0;i<513;++i)index_scores[i]=513.0f-i;
 index_scores[512]=index_scores[511];
 auto index_array=mx::array(index_scores.begin(),{513},mx::float32);
 strict_rejected=false;
 try{dsv41::index_topk_reference(index_array,128);}
 catch(const std::exception&){strict_rejected=true;}
 if(!strict_rejected)throw std::runtime_error("strict index policy accepted a boundary tie");
 dsv41::reset_index_tie_count();dsv41::reset_index_tie_records();
 dsv41::IndexTieRecord index_record{2,1025};
 auto selected=dsv41::index_topk_reference(index_array,128,false,&index_record);
 if(selected.size()!=512||selected.front()!=128||selected.back()!=639)
  throw std::runtime_error("lowest-ID index tie policy mismatch");
 if(!index_record.tied||index_record.selected_id!=511||index_record.excluded_id!=512||
    dsv41::index_tie_count()!=1||dsv41::index_tie_records().size()!=1)
  throw std::runtime_error("index tie record mismatch");

 std::vector<float> large(4097);for(int i=0;i<int(large.size());++i)large[i]=float((i*7919)%10007)+float(i)*1e-5f;
 std::vector<int> full(large.size());std::iota(full.begin(),full.end(),0);
 std::sort(full.begin(),full.end(),[&](int a,int b){return large[a]!=large[b]?large[a]>large[b]:a<b;});
 full.resize(512);std::sort(full.begin(),full.end());
 auto partial=dsv41::index_topk_reference(mx::array(large.begin(),{int(large.size())},mx::float32),37,false);
 for(int i=0;i<512;++i)if(partial[i]!=full[i]+37)throw std::runtime_error("partial index top-k differs from full sort");

 std::vector<float> candidate_logits(32768);for(int i=0;i<int(candidate_logits.size());++i)candidate_logits[i]=float((i*37)%1009);
 const int block_size=8,topk_blocks=128,num_blocks=int(candidate_logits.size())/block_size;
 std::vector<float> block_scores(num_blocks,-std::numeric_limits<float>::infinity());
 for(int b=0;b<num_blocks;++b)for(int i=0;i<block_size;++i)block_scores[b]=std::max(block_scores[b],candidate_logits[b*block_size+i]);
 block_scores[(32752-1)/block_size]=std::numeric_limits<float>::infinity();
 std::vector<int> block_order(num_blocks);std::iota(block_order.begin(),block_order.end(),0);
 std::sort(block_order.begin(),block_order.end(),[&](int a,int b){return block_scores[a]!=block_scores[b]?block_scores[a]>block_scores[b]:a<b;});
 std::vector<std::uint8_t> expected_mask(candidate_logits.size(),0);
 for(int i=0;i<topk_blocks;++i)for(int j=0;j<block_size;++j)expected_mask[block_order[i]*block_size+j]=1;
 auto partial_mask=dsv41::select_candidate_blocks_reference(candidate_logits,32752,topk_blocks,block_size);
 if(partial_mask!=expected_mask)throw std::runtime_error("partial candidate block top-k differs from full sort");
 std::cout<<"PASS: strict rejection and deterministic lowest-ID route/index tie policies\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
