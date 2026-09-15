uint group = thread_position_in_grid.x;
if (group >= count[0]) return;
uint width = params[0];
bool main_kv = width == 16;
float amax = main_kv ? 6.0f * 0x1p-9f : 6.0f * 0x1p-126f;
bool valid = true;
for (uint j=0;j<width;++j) {
    float x=float(values[group*width+j]);
    valid=valid && isfinite(x); amax=max(amax,abs(x));
}
float scale;
uchar scale_code;
if (main_kv) {
    float raw=amax/6.0f;
    // Reject unsupported overflow rather than silently saturating the scale.
    valid=valid && raw<=448.0f;
    scale_code=dsv41_quant_e4m3(raw); scale=dsv41_e4m3(scale_code);
} else {
    uint bits=as_type<uint>(amax*(1.0f/6.0f));
    uint exponent=((bits>>23)&255u)+uint((bits&0x7fffffu)!=0);
    valid=valid && exponent<255u;
    scale_code=uchar(exponent); scale=as_type<float>(exponent<<23);
}
scales[group]=scale_code; errors[group]=valid?uchar(0):uchar(1);
constexpr float levels[8]={0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};
for(uint j=0;j<width;j+=2){
    uint byte=0;
    for(uint k=0;k<2;++k){
        float x=float(values[group*width+j+k]);
        float z=clamp(x/scale,-6.0f,6.0f),mag=abs(z);
        uint code=0;
        for(uint n=1;n<8;++n){
            float a=abs(mag-levels[code]),b=abs(mag-levels[n]);
            if(b<a || (b==a && (n&1u)==0u))code=n;
        }
        uint sign=(as_type<uint>(z)>>28)&8u;
        byte|=(code|sign)<<(k*4);
        float restored=(sign?-levels[code]:levels[code])*scale;
        uint bits=as_type<uint>(restored);
        decoded[group*width+j+k]=ushort((bits+0x7fffu+((bits>>16)&1u))>>16);
    }
    packed[(group*width+j)/2]=uchar(byte);
}
