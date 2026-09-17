// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: MIT
// Batched adaptation of MLX 0.32.2 gemm_splitk_accum.
const uint column = thread_position_in_grid.x;
const uint token = thread_position_in_grid.y;
const int elements = meta[0];
const int partitions = meta[1];
if (int(column) < elements && int(token) < meta[2]) {
    const size_t base = size_t(token) * partitions * elements + column;
    float value = 0.0f;
    for (int split = 0; split < partitions; ++split)
        value += partial[base + size_t(split) * elements];
    scores[size_t(token) * elements + column] = value;
}
