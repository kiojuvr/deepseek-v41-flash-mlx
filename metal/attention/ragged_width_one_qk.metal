// Copyright © 2023-2024 Apple Inc.
// SPDX-License-Identifier: MIT
// Device-token adaptation of the exact MLX 0.32.2 M=64, N=1, K=512 GEMV
// selected before its GEMM dispatcher. One native GEMV threadgroup grid covers
// every token; non-width-one rows are initialized but never selected.
const uint3 group = threadgroup_position_in_grid;
const uint3 local = thread_position_in_threadgroup;
const int token = int(group.z);
if (token >= meta[0]) return;

if ((widths[token] & 63) != 1) {
    const uint linear = local.x + 32 * local.z;
    if (group.x == 0 && linear < 64)
        scores[size_t(token) * 64 + linear] = 0.0f;
    return;
}

const int tail_row = 128 + (widths[token] / 64) * 64;
const device float* matrix = queries + size_t(token) * 64 * 512;
const device float* vector = keys + (size_t(token) * meta[1] + tail_row) * 512;
device float* output = scores + size_t(token) * 64;
DSV41WidthOneGEMV::run(
    matrix, vector, output, group,
    simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
