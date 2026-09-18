// Adapted from oMLX's DeepSeek V4.1 packed attention kernel.
// Copyright © 2026 OpenAI. Licensed under Apache-2.0.
// Modified for the dsv41 split pooled-cache layout and BF16 local state.
using T = bfloat16_t;
using AccumType = float;
constexpr short BK = 64;
constexpr short DC = 32;
constexpr short H = 64;
constexpr short D = 512;
constexpr short WM = 8;
constexpr short kFragSize = 8;
constexpr short padQ = 16 / sizeof(T);
constexpr short padK = 16 / sizeof(T);
constexpr short padV = 16 / sizeof(T);
constexpr short LDQ = DC + padQ;
constexpr short LDK = BK + padK;
constexpr short LDV = DC + padV;
constexpr short TQ = H / (WM * kFragSize);
constexpr short TK = BK / kFragSize;
constexpr short TDC = DC / kFragSize;
constexpr short D_CHUNKS = D / DC;
constexpr short TGP_SIZE = WM * 32;

const int lane = int(simdgroup_index_in_threadgroup * 32 + thread_index_in_simdgroup);
const int token = int(threadgroup_position_in_grid.x);
if (token >= meta[0]) return;

threadgroup T Qs[H * LDQ];
threadgroup T KVs[(BK * LDV > DC * LDK) ? BK * LDV : DC * LDK];
threadgroup int selected[BK];

using MMAFragAcc = BaseMMAFrag<AccumType, kFragSize, kFragSize>;
MMATile<AccumType, TQ, 1, MMAFragAcc> Qtile;
MMATile<AccumType, 1, TK, MMAFragAcc> Ktile;
MMATile<AccumType, TQ, TK, MMAFragAcc> Stile;
MMATile<AccumType, 1, 1, MMAFragAcc> Vtile;
MMATile<AccumType, TQ, D_CHUNKS * TDC, MMAFragAcc> Otile;
Otile.clear();

const short2 simd_coord = MMAFragAcc::get_coord(thread_index_in_simdgroup);
const short sm = simd_coord.y;
const short sn = simd_coord.x;
const short tm = kFragSize * TQ * simdgroup_index_in_threadgroup;
const short Qs_offset = (tm + sm) * LDQ + sn;
const short Ks_offset = sm * LDK + sn;
const short Vs_offset = sm * LDV + sn;
const AccumType score_scale = AccumType(scale);
constexpr short rows_per_thread = decltype(Stile)::kRowsPerThread;
AccumType max_score[rows_per_thread];
AccumType sum_score[rows_per_thread] = {0};

#pragma clang loop unroll(full)
for (short i = 0; i < rows_per_thread; ++i) {
  const int head = int(tm + sm + i * kFragSize);
  max_score[i] = head < H ? AccumType(-1e30f) : Limits<AccumType>::finite_min;
}

const device T* q_base = queries + size_t(token) * H * D;
const int topkN = meta[5];
const auto topk_base = topk + size_t(token) * topkN;
const int localL = meta[1];
const int pooledL = meta[2];
const int q_offset = meta[3];
const int compress_ratio = meta[4];
const int qL = meta[0];
const int local_offset = localL - qL;
const int local_end = metal::min(localL, local_offset + token + 1);
const int local_start = metal::max(0, local_end - 128);
const int window_slots = q_offset == 0 ? metal::min(128, qL) : 128;
const int window_first = q_offset == 0 ? local_start : local_end - window_slots;
const int pooled_valid = metal::min(pooledL, (q_offset + token + 1) / compress_ratio);
const int total_tiles = (window_slots + topkN + BK - 1) / BK;

for (int ktile = 0; ktile < total_tiles; ++ktile) {
  for (int k = lane; k < BK; k += TGP_SIZE) {
    const int slot = ktile * BK + k;
    int k_pos = -1;
    if (slot < window_slots) {
      const int row = window_first + slot;
      if (row >= 0 && row < local_end) k_pos = row;
    } else if (slot < window_slots + topkN) {
      const int row = topk_base[slot - window_slots];
      if (row >= 0 && row < pooled_valid) k_pos = localL + row;
    }
    selected[k] = k_pos;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  Stile.clear();

  #pragma clang loop unroll(full)
  for (short dchunk = 0; dchunk < D_CHUNKS; ++dchunk) {
    const int dbase = int(dchunk) * DC;
    for (int elem = lane; elem < H * DC; elem += TGP_SIZE) {
      const int h = elem / DC;
      const int d = elem - h * DC;
      Qs[h * LDQ + d] = q_base[h * D + dbase + d];
    }
    for (int elem = lane; elem < BK * DC; elem += TGP_SIZE) {
      const int k = elem / DC;
      const int d = elem - k * DC;
      const int k_pos = selected[k];
      T value = T(0);
      if (k_pos >= 0) {
        if (k_pos < localL) {
          value = local_kv[size_t(k_pos) * D + dbase + d];
        } else {
          const int row = k_pos - localL;
          value = T(dsv41_pooled_value(
              pooled_values + size_t(row) * 256,
              pooled_scales + size_t(row) * 32, dbase + d));
        }
      }
      KVs[k + d * LDK] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (short dd = 0; dd < TDC; ++dd) {
      simdgroup_barrier(mem_flags::mem_none);
      Qtile.template load<T, 1, 1, LDQ, 1>(&Qs[Qs_offset + dd * kFragSize]);
      Ktile.template load<T, 1, 1, LDK, 1>(&KVs[Ks_offset + dd * kFragSize * LDK]);
      simdgroup_barrier(mem_flags::mem_none);
      tile_matmad(Stile, Qtile, Ktile, Stile);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  #pragma clang loop unroll(full)
  for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ++ii)
    Stile.elems()[ii] *= score_scale;
  {
    using stile_t = decltype(Stile);
    using selem_t = typename stile_t::elem_type;
    constexpr auto neg_inf = Limits<selem_t>::finite_min;
    #pragma clang loop unroll(full)
    for (short i = 0; i < stile_t::kTileRows; ++i) {
      #pragma clang loop unroll(full)
      for (short j = 0; j < stile_t::kTileCols; ++j) {
        const short col_pos = sn + j * stile_t::kFragCols;
        #pragma clang loop unroll(full)
        for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; ++jj)
          if (selected[col_pos + jj] < 0) Stile.frag_at(i, j)[jj] = neg_inf;
      }
    }
  }

  AccumType new_max[rows_per_thread];
  AccumType factor[rows_per_thread];
  #pragma clang loop unroll(full)
  for (short i = 0; i < rows_per_thread; ++i) new_max[i] = max_score[i];
  Stile.template row_reduce<Dsv41SparseMaxOp>(new_max);
  Stile.template row_bin_op<Dsv41SparseExpSubOp>(new_max);
  #pragma clang loop unroll(full)
  for (short i = 0; i < rows_per_thread; ++i) {
    factor[i] = metal::exp(max_score[i] - new_max[i]);
    max_score[i] = new_max[i];
  }
  AccumType sum_score_tmp[rows_per_thread] = {0};
  Stile.template row_reduce<Dsv41SparseSumOp>(sum_score_tmp);
  #pragma clang loop unroll(full)
  for (short i = 0; i < rows_per_thread; ++i)
    sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
  // Official path keeps the denominator in FP32 and rounds only PV to BF16.
  #pragma clang loop unroll(full)
  for (short i = 0; i < decltype(Stile)::kElemsPerTile; ++i)
    Stile.elems()[i] = AccumType(T(Stile.elems()[i]));
  Otile.template row_bin_op<Dsv41SparseMulOp>(factor);

  #pragma clang loop unroll(full)
  for (short vchunk = 0; vchunk < D_CHUNKS; ++vchunk) {
    const int dbase = int(vchunk) * DC;
    for (int elem = lane; elem < BK * DC; elem += TGP_SIZE) {
      const int k = elem / DC;
      const int d = elem - k * DC;
      const int k_pos = selected[k];
      T value = T(0);
      if (k_pos >= 0) {
        if (k_pos < localL) {
          value = local_kv[size_t(k_pos) * D + dbase + d];
        } else {
          const int row = k_pos - localL;
          value = T(dsv41_pooled_value(
              pooled_values + size_t(row) * 256,
              pooled_scales + size_t(row) * 32, dbase + d));
        }
      }
      KVs[k * LDV + d] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (short iq = 0; iq < TQ; ++iq) {
      #pragma clang loop unroll(full)
      for (short id = 0; id < TDC; ++id) {
        #pragma clang loop unroll(full)
        for (short ik = 0; ik < TK; ++ik) {
          const short kk = ik * kFragSize;
          const short dd = id * kFragSize;
          Vtile.template load<T, 1, 1, LDV, 1>(&KVs[Vs_offset + kk * LDV + dd]);
          MMAFragAcc::mma(Otile.frag_at(iq, vchunk * TDC + id),
              Stile.frag_at(iq, ik), Vtile.frag_at(0, 0),
              Otile.frag_at(iq, vchunk * TDC + id));
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

#pragma clang loop unroll(full)
for (short i = 0; i < rows_per_thread; ++i) {
  const int head = int(tm + sm + i * kFragSize);
  if (head < H) sum_score[i] += metal::exp(AccumType(sinks[head]) - max_score[i]);
}
Otile.template row_bin_op<Dsv41SparseDivOp>(sum_score);
device T* out = output + size_t(token) * H * D + size_t(tm + sm) * D + sn;
Otile.template store<T, 1, 1>(out, D);
