"""Diagnose the native RMSNorm 1-ULP gap against the official CPU formula.

Reproduces the native `rms_norm_reference` in MLX and tests variants of the variance /
reciprocal-sqrt step to find one that matches the official formula on the traced entry.
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


def u16_to_f32(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def f32_to_u16(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).copy()
    lsb = (u >> 16) & 1
    rounded = (u + (0x7FFF + lsb)) & np.uint32(0xFFFF0000)
    return (rounded >> 16).astype(np.uint16)


def read_bf16_bits(checkpoint, name):
    import struct
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
    path = checkpoint / index[name]
    with path.open('rb') as f:
        length = struct.unpack('<Q', f.read(8))[0]
        spec = json.loads(f.read(length))[name]
        f.seek(8 + length + spec['data_offsets'][0])
        data = f.read(spec['data_offsets'][1] - spec['data_offsets'][0])
    return np.frombuffer(data, dtype=np.uint16).reshape(spec['shape'])


def bits(a):
    return np.array(a.view(mx.uint16)).view(np.uint16)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--layer', type=int, default=0)
    a = p.parse_args()

    entry_hidden = np.load(a.native / 'encoder.entry.hidden.npy', allow_pickle=False).view(np.uint16)
    entry_pre = np.load(a.native / 'encoder.entry.pre_mix.npy', allow_pickle=False)
    weight_bits = read_bf16_bits(a.checkpoint, f'layers.{a.layer}.attn_norm.weight')

    h = u16_to_f32(entry_hidden).astype(np.float32)
    pre = entry_pre.astype(np.float32)
    w = u16_to_f32(weight_bits).astype(np.float32)

    # Official CPU formula (reference).
    collapsed_cpu = f32_to_u16(np.sum(pre[..., None] * h, axis=1))
    f_cpu = u16_to_f32(collapsed_cpu)
    var = np.mean(np.square(f_cpu), axis=-1, keepdims=True, dtype=np.float32)
    cpu = f32_to_u16(w * (f_cpu * (np.float32(1.0) / np.sqrt(var + np.float32(1e-20)))))

    # Native C++: collapse via MLX, then mean/rsqrt in float32.
    hx = mx.array(entry_hidden).view(mx.bfloat16).astype(mx.float32)
    px = mx.array(pre)
    collapsed = mx.sum(px[..., None] * hx, axis=1).astype(mx.bfloat16)
    fx = collapsed.astype(mx.float32)
    wx = mx.array(weight_bits).view(mx.bfloat16)
    eps = mx.array(1e-20, mx.float32)

    variants = {
        'native_rsqrt': lambda: fx * mx.rsqrt(mx.mean(fx * fx, axis=-1, keepdims=True) + eps),
        'inv_sqrt': lambda: fx * (mx.array(1.0, mx.float32) / mx.sqrt(mx.mean(fx * fx, axis=-1, keepdims=True) + eps)),
        'double_mean_rsqrt': lambda: fx * mx.rsqrt(mx.mean(fx.astype(mx.float32) * fx, axis=-1, keepdims=True) + eps),
    }
    native_actual = np.load(a.native / f'encoder.layer{a.layer}.attn_in.npy', allow_pickle=False).view(np.uint16)
    report = {'boundary': f'encoder.layer{a.layer}.attn_in', 'cpu_collapse_mismatch': int(np.count_nonzero(collapsed_cpu != bits(collapsed)))}
    for name, fn in variants.items():
        out = (wx.astype(mx.float32) * fn()).astype(mx.bfloat16)
        b = bits(out)
        report[name] = {'vs_cpu': int(np.count_nonzero(b != cpu)), 'vs_native': int(np.count_nonzero(b != native_actual))}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
