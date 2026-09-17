// MLX custom-kernel body. One threadgroup owns one token and eight heads.
// Architecture adapted from DwarfStar's MIT-licensed V4.1 batch attention:
// stage each KV row once per head group and keep every token/head on device.
const uint group=threadgroup_position_in_grid.x;
const uint token=group/8u;
const uint head_group=group-token*8u;
const uint tid=thread_index_in_threadgroup;
const uint lane=thread_index_in_simdgroup;
const uint sg=simdgroup_index_in_threadgroup;
const uint head=head_group*8u+sg;
const uint tokens=params[0],rows=params[1];
if(token>=tokens||head>=64u)return;

threadgroup float shared_k[512];
threadgroup float block_scores[8*64];
float accum[16];
for(uint i=0;i<16u;++i)accum[i]=0.0f;
float maximum=-1.0e30f,denominator=0.0f;
const uint q_base=(token*64u+head)*512u;

for(uint first=0;first<rows;first+=64u){
 const uint count=min(64u,rows-first);
 for(uint r=0;r<count;++r){
  const bool live=bool(valid[token*rows+first+r]);
  const uint kv_base=(token*rows+first+r)*512u;
  shared_k[tid]=live?float(kv[kv_base+tid]):0.0f;
  shared_k[tid+256u]=live?float(kv[kv_base+tid+256u]):0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float partial=0.0f;
  for(uint i=0;i<16u;++i){
   const uint col=lane+i*32u;
   partial+=float(q[q_base+col])*shared_k[col];
  }
  const float score=simd_sum(partial)*0.04419417382415922f;
  if(lane==0u)block_scores[sg*64u+r]=live?score:-INFINITY;
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 float block_max=-INFINITY;
 for(uint r=0;r<count;++r)block_max=max(block_max,block_scores[sg*64u+r]);
 const float next_max=max(maximum,block_max);
 const float rescale=exp(maximum-next_max);
 denominator*=rescale;
 for(uint i=0;i<16u;++i)accum[i]*=rescale;
 for(uint r=0;r<count;++r){
  if(!bool(valid[token*rows+first+r]))continue;
  const float exponent=exp(block_scores[sg*64u+r]-next_max);
  denominator+=exponent;
  const uint bits=as_type<uint>(exponent);
  const ushort rounded_bits=ushort((bits+0x7fffu+((bits>>16u)&1u))>>16u);
  const float rounded=as_type<float>(uint(rounded_bits)<<16u);
  const uint kv_base=(token*rows+first+r)*512u;
  for(uint i=0;i<16u;++i){
   const uint col=lane+i*32u;
   accum[i]+=rounded*float(kv[kv_base+col]);
  }
 }
 maximum=next_max;
 threadgroup_barrier(mem_flags::mem_threadgroup);
}
denominator+=exp(sink[head]-maximum);
const float inverse=1.0f/denominator;
const uint out_base=(token*64u+head)*512u;
for(uint i=0;i<16u;++i){
 const uint col=lane+i*32u;
 output[out_base+col]=bfloat16_t(accum[i]*inverse);
}
