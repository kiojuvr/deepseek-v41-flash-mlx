// Original implementation of the wide-prefill execution structure demonstrated
// by MIT-licensed DwarfStar and Apache-2.0 oMLX. This first candidate consumes the runtime's existing
// BF16 cache ABI while preserving the official 64-key maximum/denominator and
// BF16 probability boundary.  One dispatch replaces per-token QK/AV graphs.
const uint lane = thread_index_in_simdgroup;
const uint head = threadgroup_position_in_grid.x;
const uint query = threadgroup_position_in_grid.y;
const int rows = meta[0];
threadgroup float scores[64];
threadgroup float probabilities[64];
float qv[16], accumulated[16];
for (int v = 0; v < 16; ++v) {
    qv[v] = float(queries[(query * 64 + head) * 512 + lane * 16 + v]);
    accumulated[v] = 0.0f;
}
float maximum = -1e30f;
float denominator = 0.0f;
for (int first = 0; first < rows; first += 64) {
    const int count = min(64, rows - first);
    for (int j = 0; j < count; ++j) {
        float dot = 0.0f;
        if (valid[query * rows + first + j]) {
            const size_t base = (size_t(query) * rows + first + j) * 512 + lane * 16;
            for (int v = 0; v < 16; ++v)
                dot = fma(qv[v], float(keys[base + v]), dot);
            dot = simd_sum(dot) * scale[0];
        } else {
            dot = -INFINITY;
        }
        if (lane == 0) scores[j] = dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float lane_max = -INFINITY;
    for (int j = int(lane); j < count; j += 32) lane_max = max(lane_max, scores[j]);
    const float block_maximum = simd_max(lane_max);
    const float next_maximum = max(maximum, block_maximum);
    const float correction = maximum == -1e30f ? 0.0f : exp(maximum - next_maximum);
    denominator *= correction;
    for (int v = 0; v < 16; ++v) accumulated[v] *= correction;
    float lane_denominator = 0.0f;
    for (int j = int(lane); j < count; j += 32) {
        const float probability = exp(scores[j] - next_maximum);
        lane_denominator += probability;
        probabilities[j] = float(T(probability));
    }
    denominator += simd_sum(lane_denominator);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = 0; j < count; ++j) {
        if (!valid[query * rows + first + j]) continue;
        const size_t base = (size_t(query) * rows + first + j) * 512 + lane * 16;
        const float probability = probabilities[j];
        for (int v = 0; v < 16; ++v)
            accumulated[v] = fma(probability, float(keys[base + v]), accumulated[v]);
    }
    maximum = next_maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
denominator += exp(sinks[head] - maximum);
const float inverse = 1.0f / denominator;
const size_t out = (size_t(query) * 64 + head) * 512 + lane * 16;
for (int v = 0; v < 16; ++v) output[out + v] = T(accumulated[v] * inverse);
