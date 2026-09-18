// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: MIT
// Device work-list adaptation of the MLX 0.32.2 per-token regular Steel GEMM
// used by the ragged [64,K] x [K,512] AV tail. Ragged K preserves the oracle's
// unaligned-tail-first MMA order; full blocks stay on MLX's batched path.
using gemm_kernel = mlx::steel::GEMMKernel<
    float, float, 64, 32, 32, 2, 2, false, false, true, true>;
using loader_a_t = typename gemm_kernel::loader_a_t;
using loader_b_t = typename gemm_kernel::loader_b_t;
using mma_t = typename gemm_kernel::mma_t;

threadgroup float As[gemm_kernel::tgp_mem_size_a];
threadgroup float Bs[gemm_kernel::tgp_mem_size_b];

const uint lane = thread_index_in_simdgroup;
const uint simd_group = simdgroup_index_in_threadgroup;
const uint3 group = threadgroup_position_in_grid;
const int token = int(group.z);
if (token >= meta[0] || int(group.x) >= 16 || int(group.y) >= 1) return;
const int selected = widths[token];
const int columns = selected & 63;
const int tail_block = selected / 64;
if (columns == 0 || tail_block != meta[2]) return;

const int c_row = int(group.y) * 64;
const int c_col = int(group.x) * 32;
const int tail_row = 128 + tail_block * 64;
const device float* A = probabilities + size_t(token) * 64 * 64 + size_t(c_row) * 64;
const device float* B = keys + (size_t(token) * meta[1] + tail_row) * 512 + c_col;
device float* C = output + size_t(token) * 64 * 512 + size_t(c_row) * 512 + c_col;

thread loader_a_t loader_a(A, 64, As, simd_group, lane);
thread loader_b_t loader_b(B, 512, Bs, simd_group, lane);
thread mma_t mma_op(simd_group, lane);
const int aligned = columns / 32;
const short remainder = short(columns - aligned * 32);

// MLX regular GEMM accumulates the unaligned K suffix first.
if (remainder) {
    const size_t offset = size_t(aligned) * 32;
    loader_a.src += offset;
    loader_b.src += offset * 512;
    loader_a.load_safe(short2(remainder,64));
    loader_b.load_safe(short2(32,remainder));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As,Bs);
    loader_a.src -= offset;
    loader_b.src -= offset * 512;
}
for (int k = 0; k < aligned; ++k) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_a.load_unsafe();
    loader_b.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As,Bs);
    loader_a.next();
    loader_b.next();
}
mma_op.store_result(C,512);
