"""Diagnose the layer-0 Q RoPE gap.

Native applies the pure-SWA RoPE to all 60 tokens at once; the oMLX harness feeds one token
at a time. This computes the same RoPE both batched and per-token in MLX from the exact
`attn_qb` projection and compares the bit patterns.
"""
import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np


def load_bf16(path):
    raw = np.load(path, allow_pickle=False)
    u = raw.view(np.uint16).astype(np.uint32)
    return (u << 16).view(np.float32)


def bits(a):
    return np.array(a.view(mx.uint16)).view(np.uint16)


def rope(x, positions):
    d = 64
    tail = x[..., -d:].astype(mx.float32)
    pairs = tail.reshape(*tail.shape[:-1], d // 2, 2)
    real, imag = pairs[..., 0], pairs[..., 1]
    freq = 1.0 / mx.power(mx.array(10000.0, mx.float32), mx.arange(0, d, 2).astype(mx.float32) / d)
    angle = positions.astype(mx.float32)[:, None] * freq
    angle = angle.reshape(1, len(positions), *([1] * (x.ndim - 3)), d // 2)
    c, s = mx.cos(angle), mx.sin(angle)
    rot = mx.stack([real * c - imag * s, real * s + imag * c], -1)
    return mx.concatenate([x[..., :-d], rot.reshape(*x.shape[:-1], d).astype(x.dtype)], -1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--layer', type=int, default=0)
    a = p.parse_args()

    qb = load_bf16(a.native / f'encoder.layer{a.layer}.attn_qb.npy').astype(np.float32)
    qb = qb.reshape(qb.shape[0], 64, 512).astype(np.float32)
    x = mx.array(qb).astype(mx.bfloat16).reshape(1, qb.shape[0], 64, 512)  # [1,tokens,64,512]
    n = x.shape[1]

    batched = rope(x, mx.arange(0, n))
    per_token = mx.concatenate([rope(x[:, i:i + 1], mx.array([i])) for i in range(n)], 1)
    mx.eval(batched, per_token)
    batched = batched.reshape(n, 64, 512)
    per_token = per_token.reshape(n, 64, 512)

    native = np.load(a.native / f'encoder.layer{a.layer}.attn_q.npy', allow_pickle=False).view(np.uint16)
    bb = bits(batched)
    pb = bits(per_token)
    print(f'tokens={n}')
    print('batched vs native mismatches:', int(np.count_nonzero(bb != native)))
    print('per_token vs native mismatches:', int(np.count_nonzero(pb != native)))
    print('batched vs per_token mismatches:', int(np.count_nonzero(bb != pb)))


if __name__ == '__main__':
    main()
