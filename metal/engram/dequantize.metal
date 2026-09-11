// MLX custom-kernel body. Exact FP8 E4M3 * E8M0 -> BF16 bit representation.
// Integer scaling preserves BF16 subnormals even on GPUs that flush FP32 subnormals.
uint i = thread_position_in_grid.x;
uint code = values[i];
uint scale = scales[i / 32];
uint sign = (code & 128u) << 8;
uint exponent = (code >> 3) & 15u;
uint mantissa = code & 7u;
uint result;
if (scale == 255u || (exponent == 15u && mantissa == 7u)) {
    result = 0x7fc0u;
} else {
    uint significant = exponent ? 8u + mantissa : mantissa;
    int power = (exponent ? int(exponent) - 10 : -9) + int(scale) - 127;
    if (significant == 0) {
        result = sign;
    } else {
        int high = 31 - int(clz(significant));
        int normalized = power + high;
        if (normalized > 127) {
            result = sign | 0x7f80u;
        } else if (normalized >= -126) {
            result = sign | (uint(normalized + 127) << 7) | ((significant << (7-high)) - 128u);
        } else {
            int shift = power + 133;
            uint subnormal = 0;
            if (shift >= 0) {
                subnormal = significant << shift;
            } else if (-shift <= 4) {
                uint right = uint(-shift);
                subnormal = significant >> right;
                uint remainder = significant & ((1u << right) - 1u);
                uint halfway = 1u << (right - 1u);
                subnormal += uint(remainder > halfway || (remainder == halfway && (subnormal & 1u)));
            }
            result = sign | subnormal;
        }
    }
}
output[i] = ushort(result);
