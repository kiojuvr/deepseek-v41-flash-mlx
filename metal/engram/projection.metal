// One output per thread, explicit serial FP32 reduction per 32-wide block.
// CUDA tensor-core reduction equivalence requires an external oracle; not assumed.
uint n = thread_position_in_grid.x, m = thread_position_in_grid.y;
float accumulated = 0.0f;
for (uint block = 0; block < 192; ++block) {
    float partial = 0.0f;
    for (uint j = 0; j < 32; ++j) {
        float a = dsv41_e4m3(quantized[m * 6144 + block * 32 + j]);
        float b = dsv41_e4m3(weight[n * 6144 + block * 32 + j]);
        partial = partial + a * b;
    }
    float scaled = partial * dsv41_e8m0(scales[m * 192 + block]);
    scaled = scaled * dsv41_e8m0(weight_scale[(n / 32) * 192 + block]);
    accumulated = accumulated + scaled;
}
uint bits = as_type<uint>(accumulated);
output[m * 25600 + n] = ushort(isnan(accumulated) ? 0x7fc0u :
    (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
