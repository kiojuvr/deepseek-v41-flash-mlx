// One thread per 32-element activation block. Reference, not a tuned kernel.
uint group = thread_position_in_grid.x;
float amax = 1e-4f;
bool finite = true;
for (uint j = 0; j < 32; ++j) {
    float x = float(values[group * 32 + j]);
    finite = finite && isfinite(x);
    amax = max(amax, abs(x));
}
if (!finite) {
    scales[group] = uchar(255);
    for (uint j = 0; j < 32; ++j) quantized[group * 32 + j] = uchar(127);
} else {
    uint bits = as_type<uint>(amax * (1.0f / 448.0f));
    int power = int((bits >> 23) & 255u) - 127 + int((bits & 0x7fffffu) != 0);
    float scale = as_type<float>(uint(power + 127) << 23);
    scales[group] = uchar(power + 127);
    for (uint j = 0; j < 32; ++j)
        quantized[group * 32 + j] = dsv41_quant_e4m3(float(values[group * 32 + j]) / scale);
}
