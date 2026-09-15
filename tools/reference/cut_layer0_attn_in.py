"""Cut the first native/oMLX divergence with the official-formula CPU transcription.

For a given layer the boundary `attn_in` is `RMSNorm(hc_pre(hidden, pre_mix))`. This script
reads the (bit-identical) `encoder.entry` hidden / pre-mix from a native trace, loads the
layer's `attn_norm.weight`, computes the official formula on NumPy CPU float32 with explicit
round-to-nearest-even bfloat16 casts, and compares the result against both the native and the
oMLX `attn_in` traces. CPU reduction order is not a CUDA oracle; this only tells which runtime
follows the official formula's structure.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def u16_to_f32(u):
    return (u.astype(np.uint32) << 16).view(np.float32)


def f32_to_u16(x):
    """Round-to-nearest-even float32 -> bfloat16 bits."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).copy()
    lsb = (u >> 16) & 1
    rounded = (u + (0x7FFF + lsb)) & np.uint32(0xFFFF0000)
    return (rounded >> 16).astype(np.uint16)


def load_bf16(path):
    raw = np.load(path, allow_pickle=False)
    return u16_to_f32(raw.view(np.uint16))


def read_bf16_bits(checkpoint, name):
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
    shard = index[name]
    path = checkpoint / shard
    with path.open('rb') as f:
        length = struct.unpack('<Q', f.read(8))[0]
        header = f.read(length)
        spec = json.loads(header)[name]
        if spec['dtype'] != 'BF16':
            raise ValueError(f'{name} is {spec["dtype"]}, expected BF16')
        offset = 8 + length + spec['data_offsets'][0]
        f.seek(offset)
        data = f.read(spec['data_offsets'][1] - spec['data_offsets'][0])
    return np.frombuffer(data, dtype=np.uint16).reshape(spec['shape'])


def compare(expected_bits, actual_bits):
    if expected_bits.shape != actual_bits.shape:
        return {'shape_mismatch': True, 'expected': list(expected_bits.shape), 'actual': list(actual_bits.shape)}
    mism = int(np.count_nonzero(expected_bits != actual_bits))
    ef = u16_to_f32(expected_bits).ravel()
    af = u16_to_f32(actual_bits).ravel()
    return {'mismatches': mism, 'max_abs_diff': float(np.abs(ef - af).max())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--oracle', type=Path, required=True)
    p.add_argument('--layer', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()

    prefix = 'encoder' if a.layer < 20 else 'decoder'
    entry_hidden_bits = np.load(a.native / 'encoder.entry.hidden.npy', allow_pickle=False).view(np.uint16)
    entry_hidden = u16_to_f32(entry_hidden_bits)                         # [tokens,4,5120]
    entry_pre = np.load(a.native / 'encoder.entry.pre_mix.npy', allow_pickle=False)  # [tokens,4]

    weight_bits = read_bf16_bits(a.checkpoint, f'layers.{a.layer}.attn_norm.weight')
    weight = u16_to_f32(weight_bits)
    eps = np.float32(1e-20)

    h = entry_hidden.astype(np.float32)
    pre = entry_pre.astype(np.float32)
    collapsed = f32_to_u16(np.sum(pre[..., None] * h, axis=1))           # bf16 bits [tokens,5120]
    f = u16_to_f32(collapsed)
    var = np.mean(np.square(f), axis=-1, keepdims=True, dtype=np.float32)
    normalized = f * (np.float32(1.0) / np.sqrt(var + eps))
    expected = f32_to_u16(weight * normalized)                           # bf16 bits

    native_actual = np.load(a.native / f'{prefix}.layer{a.layer}.attn_in.npy', allow_pickle=False).view(np.uint16)
    oracle_actual = np.load(a.oracle / f'{prefix}.layer{a.layer}.attn_in.npy', allow_pickle=False).view(np.uint16)

    report = {
        'schema_version': 1,
        'layer': a.layer,
        'boundary': f'{prefix}.layer{a.layer}.attn_in',
        'formula': 'RMSNorm(sum(pre_mix.unsqueeze(-1)*hidden.float(), hc)).to(bf16) with bf16 attn_norm.weight, eps=1e-20, NumPy CPU float32, round-to-nearest-even bf16 casts',
        'native_vs_cpu': compare(expected, native_actual),
        'oracle_vs_cpu': compare(expected, oracle_actual),
        'native_trace': str(a.native),
        'oracle_trace': str(a.oracle),
        'source_sha256': {
            'native_entry_hidden': sha(a.native / 'encoder.entry.hidden.npy'),
            'native_entry_pre_mix': sha(a.native / 'encoder.entry.pre_mix.npy'),
            'script': sha(__file__),
        },
        'scope': 'Official-formula CPU transcription of one boundary; NumPy reduction order is not a CUDA/oracle execution.',
    }
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'boundary': report['boundary'],
                      'native_vs_cpu': report['native_vs_cpu'],
                      'oracle_vs_cpu': report['oracle_vs_cpu']}, indent=2))


if __name__ == '__main__':
    main()
