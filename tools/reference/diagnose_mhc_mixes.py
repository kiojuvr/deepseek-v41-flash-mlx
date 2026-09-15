"""Isolate the native mHC `mixes` 1-ULP coefficient gap.

Compares the NumPy CPU transcription of the official `hc_mixes` projection against an
MLX reproduction of the native path for layer 0's attention mix, and reports whether the
difference is in the mean-square normalization, the projection, or the Sinkhorn split.
"""
import argparse
import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np


def u16_to_f32(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def load_entry(native):
    h = np.load(native / 'encoder.entry.hidden.npy', allow_pickle=False).view(np.uint16)
    return u16_to_f32(h)


def read_f32(checkpoint, name):
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
    path = checkpoint / index[name]
    with path.open('rb') as f:
        length = struct.unpack('<Q', f.read(8))[0]
        spec = json.loads(f.read(length))[name]
        if spec['dtype'] != 'F32':
            raise ValueError(f'{name} is {spec["dtype"]}, expected F32')
        f.seek(8 + length + spec['data_offsets'][0])
        data = f.read(spec['data_offsets'][1] - spec['data_offsets'][0])
    return np.frombuffer(data, dtype=np.float32).reshape(spec['shape'])


def u32(a):
    return np.array(a.view(mx.uint32)).view(np.uint32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--layer', type=int, default=0)
    p.add_argument('--kind', choices=['attn', 'ffn'], default='attn')
    a = p.parse_args()

    hidden = load_entry(a.native)                       # [tokens,4,5120]
    flat = hidden.reshape(hidden.shape[0], -1)          # [tokens,20480]
    fn = read_f32(a.checkpoint, f'layers.{a.layer}.hc_{a.kind}_fn')
    eps = np.float32(1e-20)

    # CPU: F.linear then rsqrt(mean(x^2)+eps)
    mean_cpu = np.mean(flat.astype(np.float32) ** 2, axis=-1, keepdims=True, dtype=np.float32)
    proj_cpu = flat.astype(np.float32) @ fn.T
    norm_cpu = proj_cpu * (np.float32(1.0) / np.sqrt(mean_cpu + eps))

    # MLX native path: per-row matmul, mean, 1/sqrt
    fxm = mx.array(flat)
    fnm = mx.array(fn)
    mean_mlx = mx.mean(mx.square(fxm), axis=-1, keepdims=True)
    rows = [mx.matmul(fxm[i:i + 1], mx.transpose(fnm)) for i in range(flat.shape[0])]
    proj_mlx = mx.concatenate(rows, 0)
    proj_batched = mx.matmul(fxm, mx.transpose(fnm))
    norm_mlx_1 = proj_mlx * (mx.array(1.0, mx.float32) / mx.sqrt(mean_mlx + mx.array(eps)))
    norm_mlx_rsqrt = proj_mlx * mx.rsqrt(mean_mlx + mx.array(eps))
    mx.eval(mean_mlx, proj_mlx, proj_batched, norm_mlx_1, norm_mlx_rsqrt)

    report = {
        'layer': a.layer, 'kind': a.kind,
        'mean_sq_bits_mismatch': int(np.count_nonzero(u32(mean_mlx) != mean_cpu.view(np.uint32))),
        'projection_rowwise_mismatch': int(np.count_nonzero(u32(proj_mlx) != proj_cpu.view(np.uint32))),
        'projection_batched_mismatch': int(np.count_nonzero(u32(proj_batched) != proj_cpu.view(np.uint32))),
        'normalized_1sqrt_mismatch': int(np.count_nonzero(u32(norm_mlx_1) != norm_cpu.view(np.uint32))),
        'normalized_rsqrt_mismatch': int(np.count_nonzero(u32(norm_mlx_rsqrt) != norm_cpu.view(np.uint32))),
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
