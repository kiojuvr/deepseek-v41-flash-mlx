// Attention execution topology adapted from DwarfStar's
// kernel_dsv4_indexed_mixed_attention_heads8 (MIT license).
// Copyright notices and permission text: third_party/DWARFSTAR-LICENSE.
//
// This adaptation retains the official DeepSeek-V4.1-Flash BF16 Q/local-KV
// boundary and decodes the checkpoint's packed FP4/E4M3 pooled rows in place.
// One dispatch owns the complete layer chunk: one threadgroup handles eight
// heads for one token and scans raw and selected rows in semantic order.
using T = bfloat16_t;
constexpr int DSV41_WIDE_HEADS = 64;
constexpr int DSV41_WIDE_DIM = 512;

const uint token = threadgroup_position_in_grid.x;
const uint head_group = threadgroup_position_in_grid.y;
const uint lane = thread_index_in_simdgroup;
const uint simd = simdgroup_index_in_threadgroup;
const uint tid = thread_index_in_threadgroup;
const uint head = head_group * 8u + simd;
if (token >= uint(meta[0]) || head >= DSV41_WIDE_HEADS) return;

threadgroup T shared_kv[DSV41_WIDE_DIM];
const int local_rows = meta[1];
const int pooled_rows = meta[2];
const int start = meta[3];
const int ratio = meta[4];
const int topk_width = meta[5];
const int local_offset = local_rows - meta[0];
const int local_end = metal::min(local_rows, local_offset + int(token) + 1);
const int local_begin = metal::max(0, local_end - 128);
const int pooled_visible = metal::min(pooled_rows, (start + int(token) + 1) / ratio);
const device T* qrow = queries + (size_t(token) * DSV41_WIDE_HEADS + head) * DSV41_WIDE_DIM;
const auto selected = topk + size_t(token) * topk_width;

float qv[16];
float out[16];
#pragma clang loop unroll(full)
for (uint i = 0; i < 16u; ++i) {
  qv[i] = float(qrow[lane + i * 32u]);
  out[i] = 0.0f;
}
float maximum = -1e30f;
float denominator = 0.0f;

const int total = (local_end - local_begin) + topk_width;
for (int sequence = 0; sequence < total; ++sequence) {
  int source = -1;
  if (sequence < local_end - local_begin) {
    source = local_begin + sequence;
  } else {
    const int row = selected[sequence - (local_end - local_begin)];
    if (row >= 0 && row < pooled_visible) source = local_rows + row;
  }
  if (source < 0) continue;

  for (uint d = tid; d < DSV41_WIDE_DIM; d += 256u) {
    if (source < local_rows) {
      shared_kv[d] = local_kv[size_t(source) * DSV41_WIDE_DIM + d];
    } else {
      const int row = source - local_rows;
      shared_kv[d] = T(dsv41_pooled_value(
          pooled_values + size_t(row) * 256,
          pooled_scales + size_t(row) * 32, int(d)));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float partial = 0.0f;
  #pragma clang loop unroll(full)
  for (uint i = 0; i < 16u; ++i)
    partial += qv[i] * float(shared_kv[lane + i * 32u]);
  const float score = simd_sum(partial) * scale;
  const float next_maximum = metal::max(maximum, score);
  const float rescale = metal::exp(maximum - next_maximum);
  const float exponent = metal::exp(score - next_maximum);
  const float rounded = float(T(exponent));
  denominator = denominator * rescale + exponent;
  #pragma clang loop unroll(full)
  for (uint i = 0; i < 16u; ++i)
    out[i] = out[i] * rescale + rounded * float(shared_kv[lane + i * 32u]);
  maximum = next_maximum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

denominator += metal::exp(sinks[head] - maximum);
const float inverse = denominator == 0.0f ? 0.0f : 1.0f / denominator;
device T* dst = output + (size_t(token) * DSV41_WIDE_HEADS + head) * DSV41_WIDE_DIM;
#pragma clang loop unroll(full)
for (uint i = 0; i < 16u; ++i) dst[lane + i * 32u] = T(out[i] * inverse);
