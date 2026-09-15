#pragma once
#include <stdexcept>
#include <string>
namespace dsv41 {
// Backbone layer count from the fixed config (encoder 0..19, decoder 20..39).
inline constexpr int kBackboneLayers=40;
// Config kv_source_layers / index_source_layers.
inline bool is_kv_source_layer(int layer){
 return layer==2||layer==8||layer==14||layer==20;
}
inline bool is_index_source_layer(int layer){
 return layer==2||layer==8||layer==14||layer==20||layer==24||layer==28||layer==32||layer==36;
}
// compress_ratios for the fixed config: 0 for 0..1, 2 for 2..19, 1 for 20..39.
inline int layer_compress_ratio(int layer){
 if(layer<0||layer>=kBackboneLayers)throw std::runtime_error("layer out of backbone range");
 if(layer<2)return 0;
 return layer<20?2:1;
}
// The kv_source layer whose compressed cache a consumer reads. Consumers 3..7 read 2,
// 9..13 read 8, 15..19 read 14, 21..39 read 20. A source layer itself returns its own id.
inline int kv_source_for_layer(int layer){
 if(layer<0||layer>=kBackboneLayers)throw std::runtime_error("layer out of backbone range");
 if(layer<2)throw std::runtime_error("layer 0/1 has no compressed KV source");
 if(layer<8)return 2;
 if(layer<14)return 8;
 if(layer<20)return 14;
 return 20;
}
// The index_source layer whose top-k a consumer reuses. Index sources own an indexer;
// all other compressed layers reuse the nearest preceding index source.
inline int index_source_for_layer(int layer){
 if(layer<0||layer>=kBackboneLayers)throw std::runtime_error("layer out of backbone range");
 if(is_index_source_layer(layer))return layer;
 if(layer<8)return 2;
 if(layer<14)return 8;
 if(layer<20)return 14;
 if(layer<24)return 20;
 if(layer<28)return 24;
 if(layer<32)return 28;
 if(layer<36)return 32;
 return 36;
}
// Reuse consumers are layers that read a preceding source; source layers are producers.
inline int checked_reused_layer(int layer){
 if(layer<3||layer>=kBackboneLayers||is_kv_source_layer(layer))throw std::runtime_error("reuse consumer must be a non-source layer 3..39");
 return layer;
}
inline int checked_producer_layer(int layer){
 if(!is_kv_source_layer(layer))throw std::runtime_error("producer must be a kv_source layer");
 return layer;
}
inline std::string reference_moe_prefix(int layer){
 if(layer<0||layer>=kBackboneLayers)throw std::runtime_error("MoE reference supports layer 0..39 only");
 return "layers."+std::to_string(layer);
}
inline int checked_pure_swa_layer(int layer){
 if(layer!=0&&layer!=1)throw std::runtime_error("only pure SWA layer 0/1 is implemented");
 return layer;
}
inline std::string pure_swa_prefix(int layer){
 return "layers."+std::to_string(checked_pure_swa_layer(layer));
}
}
