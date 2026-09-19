// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: MIT
// Indirect token work-list adaptation of MLX 0.32.2 steel_gemm_splitk.
// BN/PARTITIONS/MIN_WIDTH/MAX_WIDTH are the native short-N Steel
// configurations used for widths greater than one.
using gemm_kernel = mlx::steel::GEMMKernel<
    float, float, 32, BN, 16, 2, 2, false, true, false, true>;
using loader_a_t = typename gemm_kernel::loader_a_t;
using loader_b_t = typename gemm_kernel::loader_b_t;
using mma_t = typename gemm_kernel::mma_t;

threadgroup float As[gemm_kernel::tgp_mem_size_a];
threadgroup float Bs[gemm_kernel::tgp_mem_size_b];

const uint lane = thread_index_in_simdgroup;
const uint simd_group = simdgroup_index_in_threadgroup;
const uint3 group = threadgroup_position_in_grid;
const int token = int(group.z) / PARTITIONS;
const int split = int(group.z) % PARTITIONS;
if (token >= meta[0]) return;
const int selected = widths[token];
const int columns = selected & 63;
if (columns < MIN_WIDTH || columns > MAX_WIDTH) return;
const int tiles_n = (columns + BN - 1) / BN;
if (int(group.x) >= tiles_n || int(group.y) >= 2) return;

const int c_row = int(group.y) * 32;
const int c_col = int(group.x) * BN;
const int iterations = 32 / PARTITIONS;
const int partition_size = iterations * 16;
const int k_start = partition_size * split;
const int tail_row = 128 + (selected / 64) * 64;
const device float* A = queries + size_t(token) * 64 * 512 + size_t(c_row) * 512 + k_start;
const device float* B = keys + (size_t(token) * meta[1] + tail_row + c_col) * 512 + k_start;
device float* C = partial + size_t(token) * PARTITIONS * 64 * 64 +
    size_t(split) * 64 * 64 + size_t(c_row) * 64 + c_col;

thread loader_a_t loader_a(A, 512, As, simd_group, lane);
thread loader_b_t loader_b(B, 512, Bs, simd_group, lane);
thread mma_t mma_op(simd_group, lane);
const short tile_columns = min(BN, columns - c_col);
if (tile_columns == BN) {
    gemm_kernel::gemm_loop(As, Bs, iterations, loader_a, loader_b, mma_op,
        short(32), tile_columns, short(0), mlx::steel::LoopAlignment<true, true, true>{});
} else {
    gemm_kernel::gemm_loop(As, Bs, iterations, loader_a, loader_b, mma_op,
        short(32), tile_columns, short(0), mlx::steel::LoopAlignment<true, false, true>{});
}
// Match MLX steel_gemm_splitk exactly: every SIMD group must finish its
// cooperative threadgroup-memory reads before any group stores and exits.
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tile_columns == BN) {
    mma_op.store_result(C, 64);
} else {
    mma_op.store_result_safe(C, 64, short2(tile_columns, 32));
}
