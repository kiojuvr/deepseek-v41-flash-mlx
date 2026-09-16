#pragma once
#include "dsv41/compressor.hpp"
#include "dsv41/index_key.hpp"
namespace dsv41 {
class GlobalKVProducerReference;
// A kv_source layer's owned global cache. Persistent arrays contain packed bytes/scales only.
class GlobalKVState {
public:
 GlobalKVState();
 void reset();
 std::uint64_t position() const{return compressor_.position();}
 std::size_t rows() const{return main_.shape(0);}
 const mlx::core::array& main_bytes() const{return main_;}
 const mlx::core::array& main_scales() const{return main_scale_;}
 const mlx::core::array& index_bytes() const{return index_;}
 const mlx::core::array& index_scales() const{return index_scale_;}
 const CompressorState& compressor() const{return compressor_;}
private:
 CompressorState compressor_;
 mlx::core::array main_,main_scale_,index_,index_scale_;
 friend class GlobalKVProducerReference;
};
class GlobalKVProducerReference {
public:
 explicit GlobalKVProducerReference(WeightCatalog& catalog,int layer);
 // Input is this producer layer's normalized attention hidden, not raw embedding/layer output.
 void append(const mlx::core::array& hidden,GlobalKVState& state,std::uint64_t start) const;
 // Atomically appends a chunk and returns immutable cache prefixes for each query token.
 std::vector<GlobalKVState> append_chunk(const mlx::core::array& hidden,GlobalKVState& state,
                                         std::uint64_t start) const;
private:
 int layer_,ratio_;
 CompressorReference compressor_;
 IndexKeyReference index_;
};
}
