// Copyright © 2023-2024 Apple Inc.
// SPDX-License-Identifier: MIT
// Specialization of MLX 0.32.2 GEMVKernel<float,4,1,1,32,4,4,false>.
// fast::metal_kernel already prepends MLX's common utility preamble, so this
// file deliberately contains no MLX includes or duplicate utility types.
struct DSV41WidthOneGEMV {
  static METAL_FUNC void run(
      const device float* mat,
      const device float* in_vec,
      device float* out_vec,
      uint3 tid,
      uint simd_gid,
      uint simd_lid) {
    thread float result[4] = {0};
    thread float inter[4];
    thread float v_coeff[4];

    int bn = int(simd_lid) * 4;
    const int out_row = int(tid.x) * 16 + int(simd_gid) * 4;
    mat += size_t(out_row) * 512;

    // The native M=64, N=1, K=512 selector uses four 128-wide iterations.
    for (int i = 0; i < 4; ++i) {
#pragma clang loop unroll(full)
      for (int tn = 0; tn < 4; ++tn)
        v_coeff[tn] = in_vec[bn + tn];

      int mat_offset = 0;
#pragma clang loop unroll(full)
      for (int tm = 0; tm < 4; ++tm) {
#pragma clang loop unroll(full)
        for (int tn = 0; tn < 4; ++tn)
          inter[tn] = mat[mat_offset + bn + tn];
#pragma clang loop unroll(full)
        for (int tn = 0; tn < 4; ++tn)
          result[tm] += inter[tn] * v_coeff[tn];
        mat_offset += 512;
      }
      bn += 128;
    }

#pragma clang loop unroll(full)
    for (int tm = 0; tm < 4; ++tm) {
#pragma clang loop unroll(full)
      for (ushort sn = 16; sn >= 1; sn >>= 1)
        result[tm] += simd_shuffle_down(result[tm], sn);
    }

    if (simd_lid == 0) {
#pragma clang loop unroll(full)
      for (int tm = 0; tm < 4; ++tm)
        out_vec[out_row + tm] = result[tm];
    }
  }
};
