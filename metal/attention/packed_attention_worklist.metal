// Device-resident packed KV work-list materialization. This preserves the
// qualified reference reduction while removing host shape grouping and
// per-token pooled gather/decode graphs.
using T = bfloat16_t;
const int d = int(thread_position_in_grid.x);
const int slot = int(thread_position_in_grid.y);
const int token = int(thread_position_in_grid.z);
const int tokens = meta[0];
const int localL = meta[1];
const int pooledL = meta[2];
const int q_offset = meta[3];
const int compress_ratio = meta[4];
const int topkN = meta[5];
const int window_slots = meta[6] ? 128 : (q_offset == 0 ? metal::min(128, tokens) : 128);
const int rows = window_slots + topkN;
if (d >= 512 || slot >= rows || token >= tokens) return;

if (d == 0 && slot == 0) {
  int selected = 0;
  for (int i = 0; i < topkN; ++i) selected += topk[token * topkN + i] >= 0;
  widths[token] = selected;
}

const int local_offset = localL - tokens;
const int local_end = metal::min(localL, local_offset + token + 1);
const int local_start = metal::max(0, local_end - 128);
// Exact-shape chunk zero uses a one-row token-zero group followed by 128-row
// groups with leading causal padding.  A fixed 128-row plan must therefore
// right-align every live prefix, including chunk zero; otherwise token 1..127
// observe the right rows in a different chronological slot layout.
const int window_first = meta[6] ? local_end - window_slots
                                 : (q_offset == 0 ? local_start : local_end - window_slots);
const int pooled_valid = metal::min(pooledL, (q_offset + token + 1) / compress_ratio);
int source = -1;
bool pooled = false;
if (slot < window_slots) {
  const int row = window_first + slot;
  if (row >= 0 && row < local_end) source = row;
} else {
  const int row = int(topk[token * topkN + slot - window_slots]);
  if (row >= 0 && row < pooled_valid) { source = row; pooled = true; }
}
if (d == 0) valid[token * rows + slot] = source >= 0;
T value = T(0);
if (source >= 0) {
  if (!pooled) {
    value = local_kv[size_t(source) * 512 + d];
  } else {
    constexpr float levels[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
    const uchar packed = pooled_values[size_t(source) * 256 + d / 2];
    const uint code = (packed >> ((d % 2) * 4)) & 15;
    const uchar scale_code = pooled_scales[size_t(source) * 32 + d / 16];
    const uint a = scale_code & 127, exponent = a >> 3, mantissa = a & 7;
    const float magnitude = exponent == 0 ? float(mantissa) * 0x1p-9f
      : as_type<float>(((exponent + 120u) << 23) | (mantissa << 20));
    const float scale_value = scale_code & 128 ? -magnitude : magnitude;
    value = T(levels[code & 7] * ((code & 8) ? -1.0f : 1.0f) * scale_value);
  }
}
ordered[(size_t(token) * rows + slot) * 512 + d] = value;
