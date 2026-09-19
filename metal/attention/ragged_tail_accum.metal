// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: MIT
// Accumulate the native split order into a padded score tile. Non-class and
// non-tail columns are -inf so three fixed class results can be selected.
const uint element = thread_position_in_grid.x;
const uint token = thread_position_in_grid.y;
if (int(token) >= meta[0] || int(element) >= 64 * 64) return;
const int selected = widths[token];
const int columns = selected & 63;
const int column = int(element) & 63;
float value = -metal::numeric_limits<float>::infinity();
if (columns >= MIN_WIDTH && columns <= MAX_WIDTH && column < columns) {
    const int row = int(element) >> 6;
    const size_t base = size_t(token) * PARTITIONS * 64 * PARTIAL_COLUMNS +
        size_t(row) * PARTIAL_COLUMNS + column;
    value = 0.0f;
    for (int split = 0; split < PARTITIONS; ++split)
        value += partial[base + size_t(split) * 64 * PARTIAL_COLUMNS];
}
scores[size_t(token) * 64 * 64 + element] = value;
