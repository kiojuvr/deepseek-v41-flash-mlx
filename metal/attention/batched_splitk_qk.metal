// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: MIT
// Batched adaptation of MLX 0.32.2 steel_gemm_splitk. Each token keeps the
// exact scalar M=64, K=512 split-K tile and accumulation topology.
using gemm_kernel = mlx::steel::GEMMKernel<
    float, float, 32, BN, 16, 2, 2, false, true, MN_ALIGNED, true>;
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
const int rows = meta[0];
const int columns = meta[1];
const int tiles_n = meta[2];
if (token >= meta[3] || int(group.x) >= tiles_n || int(group.y) >= 2) return;

const int c_row = int(group.y) * 32;
const int c_col = int(group.x) * BN;
const int iterations = 32 / PARTITIONS;
const int partition_size = iterations * 16;
const int k_start = partition_size * split;
const device float* A = queries + size_t(token) * 64 * 512 + size_t(c_row) * 512 + k_start;
const device float* B = keys + size_t(token) * rows * 512 + size_t(c_col) * 512 + k_start;
device float* C = partial + size_t(token) * PARTITIONS * 64 * columns +
    size_t(split) * 64 * columns + size_t(c_row) * columns + c_col;

thread loader_a_t loader_a(A, 512, As, simd_group, lane);
thread loader_b_t loader_b(B, 512, Bs, simd_group, lane);
thread mma_t mma_op(simd_group, lane);
const short tile_rows = min(32, 64 - c_row);
const short tile_columns = min(BN, columns - c_col);
if (MN_ALIGNED || (tile_rows == 32 && tile_columns == BN)) {
    gemm_kernel::gemm_loop(As, Bs, iterations, loader_a, loader_b, mma_op,
        tile_rows, tile_columns, short(0), mlx::steel::LoopAlignment<true, true, true>{});
} else if (tile_columns == BN) {
    gemm_kernel::gemm_loop(As, Bs, iterations, loader_a, loader_b, mma_op,
        tile_rows, tile_columns, short(0), mlx::steel::LoopAlignment<false, true, true>{});
} else if (tile_rows == 32) {
    gemm_kernel::gemm_loop(As, Bs, iterations, loader_a, loader_b, mma_op,
        tile_rows, tile_columns, short(0), mlx::steel::LoopAlignment<true, false, true>{});
} else {
    gemm_kernel::gemm_loop(As, Bs, iterations, loader_a, loader_b, mma_op,
        tile_rows, tile_columns, short(0), mlx::steel::LoopAlignment<false, false, true>{});
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (MN_ALIGNED || (tile_rows == 32 && tile_columns == BN))
    mma_op.store_result(C, columns);
else
    mma_op.store_result_safe(C, columns, short2(tile_columns, tile_rows));
