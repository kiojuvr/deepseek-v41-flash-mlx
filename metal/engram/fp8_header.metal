#pragma clang fp contract(off)
inline float dsv41_e4m3(uint code) {
    uint exponent = (code >> 3) & 15u, mantissa = code & 7u;
    if (exponent == 15u && mantissa == 7u) return NAN;
    float x = exponent ? ldexp(float(8u + mantissa), int(exponent) - 10)
                       : ldexp(float(mantissa), -9);
    return (code & 128u) ? -x : x;
}
inline float dsv41_e8m0(uint code) {
    if (code == 255u) return NAN;
    return as_type<float>(code == 0u ? 0x00400000u : code << 23);
}
inline uchar dsv41_quant_e4m3(float x) {
    uint sign = (as_type<uint>(x) >> 24) & 128u;
    float magnitude = min(abs(x), 448.0f);
    uint lo = 0, hi = 126;
    while (lo < hi) {
        uint mid = (lo + hi + 1) / 2;
        if (dsv41_e4m3(mid) <= magnitude) lo = mid; else hi = mid - 1;
    }
    if (lo < 126) {
        float lower = magnitude - dsv41_e4m3(lo);
        float upper = dsv41_e4m3(lo + 1) - magnitude;
        if (upper < lower || (upper == lower && (lo & 1u))) ++lo;
    }
    return uchar(sign | lo);
}
