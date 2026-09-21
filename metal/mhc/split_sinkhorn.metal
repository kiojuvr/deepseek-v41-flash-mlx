// mHC post-projection split + Sinkhorn normalization, fused into one dispatch.
//
// Ported from DwarfStar / ds4 `kernel_dsv4_hc_split_sinkhorn` and
// `hc_split_sinkhorn_one` (https://github.com/antirez/ds4, commit 8db1d1d1),
// which is MIT licensed. oMLX v0.7.0.dev2 `deepseek_v41_sinkhorn`
// (Apache-2.0) was consulted only as a design reference for fusing the
// normalization loop; no oMLX source is reproduced here.
//
// The arithmetic below reproduces src/mhc/reference.cpp HCReference::mixes
// exactly:
//   pre[i]  = sigmoid(n[i]*scale[0] + base[i]) + 1e-6
//   post[i] = sigmoid(n[4+i]*scale[1] + base[4+i]) * 2
//   c       = n[8..24]*scale[2] + base[8..24]                     (4x4)
//   c       = exp(c - max(c, -1)) / sum(c, -1) + 1e-6             (softmax)
//   c       = c / (sum(c, -2) + 1e-6)
//   repeat 19x: c = c / (sum(c,-1)+1e-6); c = c / (sum(c,-2)+1e-6)
// On this MLX 0.32.2 build both `mx::exp` and `mx::sigmoid` resolve to
// `metal::precise::exp`, and every sum runs in the reference's sequential
// order. Division is used (not reciprocal multiplication) to preserve the
// last bit. The fused kernel is bit-identical to the reference path.
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)

const uint row = thread_position_in_grid.x;
if (row >= count[0]) return;

const device float* n = normalized + row * 24;
device float* pre_out = pre + row * 4;
device float* post_out = post + row * 4;
device float* c = comb + row * 16;

const float scale0 = scale[0];
const float scale1 = scale[1];
const float scale2 = scale[2];

for (int i = 0; i < 4; ++i) {
  const float z = n[i] * scale0 + base[i];
  const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(z)));
  pre_out[i] = ((z < 0.0f) ? y : 1.0f - y) + 1e-6f;
}
for (int i = 0; i < 4; ++i) {
  const float z = n[4 + i] * scale1 + base[4 + i];
  const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(z)));
  post_out[i] = ((z < 0.0f) ? y : 1.0f - y) * 2.0f;
}
for (int i = 0; i < 16; ++i) c[i] = n[8 + i] * scale2 + base[8 + i];

// Initial row softmax, then column normalization.
for (int r = 0; r < 4; ++r) {
  float maximum = c[r * 4];
  for (int j = 1; j < 4; ++j) maximum = metal::max(maximum, c[r * 4 + j]);
  float total = 0.0f;
  for (int j = 0; j < 4; ++j) {
    c[r * 4 + j] = metal::precise::exp(c[r * 4 + j] - maximum);
    total += c[r * 4 + j];
  }
  for (int j = 0; j < 4; ++j) c[r * 4 + j] = c[r * 4 + j] / total + 1e-6f;
}
for (int j = 0; j < 4; ++j) {
  float total = 0.0f;
  for (int r = 0; r < 4; ++r) total += c[r * 4 + j];
  const float denom = total + 1e-6f;
  for (int r = 0; r < 4; ++r) c[r * 4 + j] = c[r * 4 + j] / denom;
}

// Nineteen row/column normalization pairs.
for (int iteration = 0; iteration < 19; ++iteration) {
  for (int r = 0; r < 4; ++r) {
    float total = 0.0f;
    for (int j = 0; j < 4; ++j) total += c[r * 4 + j];
    const float denom = total + 1e-6f;
    for (int j = 0; j < 4; ++j) c[r * 4 + j] = c[r * 4 + j] / denom;
  }
  for (int j = 0; j < 4; ++j) {
    float total = 0.0f;
    for (int r = 0; r < 4; ++r) total += c[r * 4 + j];
    const float denom = total + 1e-6f;
    for (int r = 0; r < 4; ++r) c[r * 4 + j] = c[r * 4 + j] / denom;
  }
}
